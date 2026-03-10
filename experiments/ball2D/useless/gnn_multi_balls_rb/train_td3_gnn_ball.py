import argparse
from dataclasses import asdict, dataclass
import os

import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from envball_gnn_rb import BallRoboBalletEnvironment, StartState


NUM_BALLS = 4
TARGETS_PER_BALL = 2
NUM_TARGETS = NUM_BALLS * TARGETS_PER_BALL
NODE_COUNT = NUM_BALLS + NUM_TARGETS
OBS_DIM = NUM_BALLS * 6 + NUM_TARGETS * 2 + NUM_TARGETS + 2
ACT_DIM = NUM_BALLS * 2
BALL_POS_END = NUM_BALLS * 2
BALL_VEL_END = BALL_POS_END + NUM_BALLS * 2
BALL_START_END = BALL_VEL_END + NUM_BALLS * 2
TARGET_POS_END = BALL_START_END + NUM_TARGETS * 2
TARGET_DONE_START = TARGET_POS_END
TARGET_DONE_END = TARGET_DONE_START + NUM_TARGETS
CURRENT_SCORE_IDX = TARGET_DONE_END
TIME_FRACTION_IDX = CURRENT_SCORE_IDX + 1
GLOBAL_FEATURE_DIM = 8
RETURN_THRESHOLD = 0.35
RETURN_PROGRESS_HORIZON = 4.0
TARGET_CLUSTER = np.repeat(np.arange(NUM_BALLS, dtype=np.int32), TARGETS_PER_BALL)
TARGET_SLOT = np.tile(np.arange(TARGETS_PER_BALL, dtype=np.int32), NUM_BALLS)
TARGET_CLUSTER_JNP = jnp.asarray(TARGET_CLUSTER, dtype=jnp.float32)
TARGET_SLOT_JNP = jnp.asarray(TARGET_SLOT, dtype=jnp.float32)
CHECKPOINT_ARCHITECTURE = 'rb_graphnet_v1'
HIGH_COMPLETION_HINDSIGHT_MARGIN = 2


@dataclass
class EpisodeStats:
    total_reward: float = 0.0
    final_score: float = 0.0
    num_targets_done: int = 0
    episode_length: int = 0
    total_collision_penalty: float = 0.0
    total_collision_ball_steps: int = 0
    total_mean_acceleration: float = 0.0
    success: int = 0
    all_targets_done: int = 0
    entered_return_phase: int = 0
    return_ready: int = 0
    completion_ratio: float = 0.0
    final_mean_target_score: float = 0.0
    final_return_to_start_score: float = 0.0
    final_return_distance_mean: float = 0.0
    final_return_distance_max: float = 0.0
    min_pairwise_ball_distance: float = 0.0
    is_hindsight_replay: int = 0
    hindsight_target_count: int = 0
    hindsight_targets_changed: int = 0
    hindsight_goal_delta: int = 0


@dataclass
class HindsightMetadata:
    requested_target_count: int = 0
    changed_target_indices: tuple[int, ...] = ()
    source_steps: tuple[int, ...] = ()
    source_ball_indices: tuple[int, ...] = ()
    source_is_completion_event: tuple[int, ...] = ()
    completion_event_candidate_count: int = 0


def _build_graph_topology():
    senders = []
    receivers = []
    for receiver in range(NUM_BALLS):
        for sender in range(NUM_BALLS):
            if sender == receiver:
                continue
            senders.append(sender)
            receivers.append(receiver)
    for target_idx in range(NUM_TARGETS):
        sender = NUM_BALLS + target_idx
        for receiver in range(NUM_BALLS):
            senders.append(sender)
            receivers.append(receiver)
    return np.asarray(senders, dtype=np.int32), np.asarray(receivers, dtype=np.int32)


GRAPH_SENDERS_NP, GRAPH_RECEIVERS_NP = _build_graph_topology()
GRAPH_SENDERS = jnp.asarray(GRAPH_SENDERS_NP, dtype=jnp.int32)
GRAPH_RECEIVERS = jnp.asarray(GRAPH_RECEIVERS_NP, dtype=jnp.int32)
GRAPH_N_NODE = jnp.array([NODE_COUNT], dtype=jnp.int32)
GRAPH_N_EDGE = jnp.array([GRAPH_SENDERS_NP.shape[0]], dtype=jnp.int32)
DEFAULT_GLOBALS = jnp.zeros((1, GLOBAL_FEATURE_DIM), dtype=jnp.float32)


def flatten_obs(obs):
    return np.concatenate(
        [
            np.asarray(obs['ball_pos'], dtype=np.float32).reshape(-1),
            np.asarray(obs['ball_vel'], dtype=np.float32).reshape(-1),
            np.asarray(obs['ball_start_pos'], dtype=np.float32).reshape(-1),
            np.asarray(obs['target_pos'], dtype=np.float32).reshape(-1),
            np.asarray(obs['target_done'], dtype=np.float32).reshape(-1),
            np.asarray([obs['current_score'], obs['time_fraction']], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def flat_obs_to_dict(flat_obs):
    flat_obs = np.asarray(flat_obs, dtype=np.float32)
    return {
        'ball_pos': flat_obs[:BALL_POS_END].reshape(NUM_BALLS, 2),
        'ball_vel': flat_obs[BALL_POS_END:BALL_VEL_END].reshape(NUM_BALLS, 2),
        'ball_start_pos': flat_obs[BALL_VEL_END:BALL_START_END].reshape(NUM_BALLS, 2),
        'target_pos': flat_obs[BALL_START_END:TARGET_POS_END].reshape(NUM_TARGETS, 2),
        'target_done': flat_obs[TARGET_DONE_START:TARGET_DONE_END].reshape(NUM_TARGETS),
        'current_score': float(flat_obs[CURRENT_SCORE_IDX]),
        'time_fraction': float(flat_obs[TIME_FRACTION_IDX]),
        'target_cluster': TARGET_CLUSTER.copy(),
        'target_slot': TARGET_SLOT.copy(),
    }


def normalize_obs_array(obs_array, obs_mean, obs_std):
    return (obs_array - obs_mean) / (obs_std + 1e-8)


def serialize_params_to_uint8(params):
    return np.frombuffer(flax.serialization.to_bytes(params), dtype=np.uint8)


def npz_value_to_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.void):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            scalar = value.item()
            if isinstance(scalar, bytes):
                return scalar
            if isinstance(scalar, str):
                return scalar.encode('utf-8')
            if isinstance(scalar, np.void):
                return bytes(scalar)
        return value.tobytes()
    if isinstance(value, str):
        return value.encode('utf-8')
    return bytes(value)


def build_graph_from_components(nodes, edges, globals_=None):
    if globals_ is None:
        globals_ = DEFAULT_GLOBALS
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=GRAPH_SENDERS,
        receivers=GRAPH_RECEIVERS,
        globals=globals_,
        n_node=GRAPH_N_NODE,
        n_edge=GRAPH_N_EDGE,
    )


def compute_return_progress_from_raw_distances(return_distances):
    parked = (return_distances < RETURN_THRESHOLD).astype(jnp.float32)
    threshold_progress = jnp.clip(
        1.0 - return_distances / (RETURN_PROGRESS_HORIZON * RETURN_THRESHOLD),
        0.0,
        1.0,
    )
    inverse_distance_progress = jnp.clip(
        RETURN_THRESHOLD / jnp.maximum(return_distances, RETURN_THRESHOLD),
        0.0,
        1.0,
    )
    dense_progress = jnp.maximum(threshold_progress, inverse_distance_progress)
    return jnp.maximum(parked, 0.35 * parked + 0.65 * dense_progress)


def flat_obs_batch_to_graph_components(flat_obs_batch, obs_mean, obs_std):
    flat_obs_batch = jnp.asarray(flat_obs_batch, dtype=jnp.float32)
    obs_mean = jnp.asarray(obs_mean, dtype=jnp.float32)
    obs_std = jnp.asarray(obs_std, dtype=jnp.float32)
    batch_size = flat_obs_batch.shape[0]

    ball_pos = flat_obs_batch[:, :BALL_POS_END].reshape(batch_size, NUM_BALLS, 2)
    ball_vel = flat_obs_batch[:, BALL_POS_END:BALL_VEL_END].reshape(batch_size, NUM_BALLS, 2)
    ball_start_pos = flat_obs_batch[:, BALL_VEL_END:BALL_START_END].reshape(batch_size, NUM_BALLS, 2)
    target_pos = flat_obs_batch[:, BALL_START_END:TARGET_POS_END].reshape(batch_size, NUM_TARGETS, 2)
    target_done = flat_obs_batch[:, TARGET_DONE_START:TARGET_DONE_END].reshape(batch_size, NUM_TARGETS)
    current_score = flat_obs_batch[:, CURRENT_SCORE_IDX:CURRENT_SCORE_IDX + 1]
    time_fraction = flat_obs_batch[:, TIME_FRACTION_IDX:TIME_FRACTION_IDX + 1]

    ball_pos_mean = obs_mean[:BALL_POS_END].reshape(NUM_BALLS, 2)
    ball_pos_std = obs_std[:BALL_POS_END].reshape(NUM_BALLS, 2)
    ball_vel_mean = obs_mean[BALL_POS_END:BALL_VEL_END].reshape(NUM_BALLS, 2)
    ball_vel_std = obs_std[BALL_POS_END:BALL_VEL_END].reshape(NUM_BALLS, 2)
    ball_start_mean = obs_mean[BALL_VEL_END:BALL_START_END].reshape(NUM_BALLS, 2)
    ball_start_std = obs_std[BALL_VEL_END:BALL_START_END].reshape(NUM_BALLS, 2)
    target_pos_mean = obs_mean[BALL_START_END:TARGET_POS_END].reshape(NUM_TARGETS, 2)
    target_pos_std = obs_std[BALL_START_END:TARGET_POS_END].reshape(NUM_TARGETS, 2)

    ball_pos_norm = normalize_obs_array(ball_pos, ball_pos_mean[None, :, :], ball_pos_std[None, :, :])
    ball_vel_norm = normalize_obs_array(ball_vel, ball_vel_mean[None, :, :], ball_vel_std[None, :, :])
    ball_start_norm = normalize_obs_array(ball_start_pos, ball_start_mean[None, :, :], ball_start_std[None, :, :])
    target_pos_norm = normalize_obs_array(target_pos, target_pos_mean[None, :, :], target_pos_std[None, :, :])

    rel_target_to_ball = target_pos_norm[:, None, :, :] - ball_pos_norm[:, :, None, :]
    target_distances = jnp.linalg.norm(rel_target_to_ball, axis=-1)
    masked_target_distances = jnp.where(target_done[:, None, :] > 0.5, 1e6, target_distances)
    nearest_target_idx = jnp.argmin(masked_target_distances, axis=-1)
    nearest_rel = jnp.take_along_axis(rel_target_to_ball, nearest_target_idx[..., None, None], axis=2).squeeze(axis=2)
    nearest_dist = jnp.take_along_axis(masked_target_distances, nearest_target_idx[..., None], axis=2).squeeze(axis=2)
    active_mask = (target_done < 0.5).astype(jnp.float32)
    active_count = jnp.sum(active_mask, axis=-1, keepdims=True)
    mean_target_pos = jnp.sum(target_pos_norm * active_mask[..., None], axis=1, keepdims=True) / jnp.maximum(active_count[..., None], 1.0)
    mean_rel = mean_target_pos - ball_pos_norm
    has_active_targets = (active_count > 0).repeat(NUM_BALLS, axis=1)
    nearest_rel = jnp.where(has_active_targets[..., None], nearest_rel, 0.0)
    nearest_dist = jnp.where(has_active_targets, nearest_dist, 0.0)
    mean_rel = jnp.where(has_active_targets[..., None], mean_rel, 0.0)
    active_ratio = active_count / NUM_TARGETS
    return_rel = ball_start_norm - ball_pos_norm
    pairwise_ball_dist = jnp.linalg.norm(ball_pos_norm[:, None, :, :] - ball_pos_norm[:, :, None, :], axis=-1)
    pairwise_ball_dist = pairwise_ball_dist + jnp.eye(NUM_BALLS, dtype=jnp.float32)[None, :, :] * 1e6
    nearest_ball_dist = jnp.min(pairwise_ball_dist, axis=-1, keepdims=True)

    ball_nodes = jnp.concatenate(
        [
            nearest_rel,
            mean_rel,
            ball_vel_norm,
            return_rel,
            nearest_dist[..., None],
            active_ratio.repeat(NUM_BALLS, axis=1)[..., None],
            nearest_ball_dist,
        ],
        axis=-1,
    )

    ball_to_target_dist = jnp.linalg.norm(ball_pos_norm[:, None, :, :] - target_pos_norm[:, :, None, :], axis=-1)
    nearest_ball_idx = jnp.argmin(ball_to_target_dist, axis=-1)
    batch_indices = jnp.arange(batch_size)[:, None]
    nearest_ball_pos = ball_pos_norm[batch_indices, nearest_ball_idx]
    rel_to_nearest_ball = target_pos_norm - nearest_ball_pos
    cluster_norm = (TARGET_CLUSTER_JNP / max(NUM_BALLS - 1, 1)) * 2.0 - 1.0
    slot_norm = (TARGET_SLOT_JNP / max(TARGETS_PER_BALL - 1, 1)) * 2.0 - 1.0
    cluster_norm = jnp.broadcast_to(cluster_norm[None, :, None], (batch_size, NUM_TARGETS, 1))
    slot_norm = jnp.broadcast_to(slot_norm[None, :, None], (batch_size, NUM_TARGETS, 1))
    return_phase = jnp.all(target_done > 0.5, axis=-1, keepdims=True).astype(jnp.float32)
    return_phase = jnp.broadcast_to(return_phase[:, None, :], (batch_size, NUM_TARGETS, 1))
    target_padding = jnp.zeros((batch_size, NUM_TARGETS, 2), dtype=jnp.float32)
    target_nodes = jnp.concatenate(
        [
            target_pos_norm,
            rel_to_nearest_ball,
            target_done[..., None],
            cluster_norm,
            slot_norm,
            (ball_to_target_dist.min(axis=-1, keepdims=True)),
            return_phase,
            target_padding,
        ],
        axis=-1,
    )

    nodes = jnp.concatenate([ball_nodes, target_nodes], axis=1)

    edge_list = []
    for receiver in range(NUM_BALLS):
        for sender in range(NUM_BALLS):
            if sender == receiver:
                continue
            rel_pos = ball_pos_norm[:, sender] - ball_pos_norm[:, receiver]
            rel_vel = ball_vel_norm[:, sender] - ball_vel_norm[:, receiver]
            dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)
            type_flags = jnp.tile(jnp.array([[1.0, 0.0]], dtype=jnp.float32), (batch_size, 1))
            padding = jnp.zeros((batch_size, 1), dtype=jnp.float32)
            edge_list.append(jnp.concatenate([rel_pos, rel_vel, dist, type_flags, nearest_ball_dist[:, receiver], padding], axis=-1))
    for target_idx in range(NUM_TARGETS):
        cluster = TARGET_CLUSTER[target_idx]
        slot = TARGET_SLOT[target_idx]
        for receiver in range(NUM_BALLS):
            rel_pos = target_pos_norm[:, target_idx] - ball_pos_norm[:, receiver]
            rel_vel = -ball_vel_norm[:, receiver]
            dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)
            done_feat = target_done[:, target_idx:target_idx + 1]
            nearest_feat = (nearest_ball_idx[:, target_idx:target_idx + 1] == receiver).astype(jnp.float32)
            cluster_feat = jnp.full((batch_size, 1), -1.0 + 2.0 * cluster / max(NUM_BALLS - 1, 1), dtype=jnp.float32)
            slot_feat = jnp.full((batch_size, 1), -1.0 if slot == 0 else 1.0, dtype=jnp.float32)
            edge_list.append(jnp.concatenate([rel_pos, rel_vel, done_feat, nearest_feat, dist, cluster_feat, slot_feat], axis=-1))
    edges = jnp.stack(edge_list, axis=1)

    completion_ratio = jnp.mean(target_done, axis=-1, keepdims=True)
    mean_nearest_distance = jnp.mean(nearest_dist, axis=-1, keepdims=True)
    max_nearest_distance = jnp.max(nearest_dist, axis=-1, keepdims=True)
    min_ball_distance = jnp.min(pairwise_ball_dist, axis=(-1, -2))[:, None]
    raw_return_distances = jnp.linalg.norm(ball_start_pos - ball_pos, axis=-1)
    return_progress = compute_return_progress_from_raw_distances(raw_return_distances)
    return_score = jnp.mean(return_progress, axis=-1, keepdims=True)
    all_targets_done = jnp.all(target_done > 0.5, axis=-1, keepdims=True).astype(jnp.float32)
    globals_ = jnp.concatenate(
        [completion_ratio, current_score, time_fraction, mean_nearest_distance, max_nearest_distance, min_ball_distance, return_score, all_targets_done],
        axis=-1,
    )
    return nodes, edges, globals_


def obs_to_graph(obs):
    flat_obs = flatten_obs(obs)[None, :]
    nodes, edges, globals_ = flat_obs_batch_to_graph_components(
        flat_obs,
        np.zeros(OBS_DIM, dtype=np.float32),
        np.ones(OBS_DIM, dtype=np.float32),
    )
    return build_graph_from_components(nodes[0], edges[0], globals_[0:1])


class ReplayBuffer:
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return {
            'obs': self.obs_buf[idxs],
            'act': self.act_buf[idxs],
            'rew': self.rew_buf[idxs],
            'next_obs': self.next_obs_buf[idxs],
            'done': self.done_buf[idxs],
        }


class GraphMLP(nn.Module):
    layer_sizes: tuple[int, ...]
    activation: str = 'gelu'
    use_layernorm: bool = True
    last_layer_scale: float = 1.0

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for index, layer_size in enumerate(self.layer_sizes):
            is_last_layer = index == len(self.layer_sizes) - 1
            if is_last_layer:
                kernel_init = nn.initializers.variance_scaling(
                    scale=self.last_layer_scale,
                    mode='fan_in',
                    distribution='truncated_normal',
                )
                x = nn.Dense(layer_size, kernel_init=kernel_init)(x)
            else:
                x = nn.Dense(layer_size)(x)
                if self.use_layernorm:
                    x = nn.LayerNorm()(x)
                if self.activation == 'relu':
                    x = nn.relu(x)
                elif self.activation == 'elu':
                    x = nn.elu(x)
                elif self.activation == 'lrelu':
                    x = nn.leaky_relu(x)
                else:
                    x = nn.gelu(x)
        return x.astype(jnp.float32)


class GraphNetCore(nn.Module):
    hidden_dim: int
    edge_hidden_dim: int
    update_num_layers: int = 3
    edge_update_num_layers: int = 2
    use_layernorm: bool = True
    use_skip_connections: bool = False

    @nn.compact
    def __call__(self, graph):
        update_edge_fn = jraph.concatenated_args(
            GraphMLP(
                layer_sizes=tuple([self.edge_hidden_dim] * self.edge_update_num_layers),
                use_layernorm=self.use_layernorm,
                name='update_edge',
            )
        )
        update_node_fn = jraph.concatenated_args(
            GraphMLP(
                layer_sizes=tuple([self.hidden_dim] * self.update_num_layers),
                use_layernorm=self.use_layernorm,
                name='update_node',
            )
        )
        update_global_fn = jraph.concatenated_args(
            GraphMLP(
                layer_sizes=tuple([self.hidden_dim] * self.update_num_layers),
                use_layernorm=self.use_layernorm,
                name='update_global',
            )
        )
        gnn = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
            aggregate_edges_for_nodes_fn=jraph.segment_mean,
            aggregate_nodes_for_globals_fn=jraph.segment_mean,
            aggregate_edges_for_globals_fn=jraph.segment_mean,
        )
        updated = gnn(graph)
        if self.use_skip_connections:
            updated = updated._replace(
                nodes=graph.nodes + updated.nodes,
                edges=graph.edges + updated.edges,
                globals=graph.globals + updated.globals,
            )
        return updated


class GraphNet(nn.Module):
    hidden_dim: int
    edge_hidden_dim: int
    embedding_num_layers: int = 2
    update_num_layers: int = 3
    edge_update_num_layers: int = 2
    num_message_passing_rounds: int = 2
    use_layernorm: bool = True
    use_skip_connections: bool = False
    shared_message_passing_weights: bool = True

    @nn.compact
    def __call__(self, graph_in):
        embed_graph = jraph.GraphMapFeatures(
            embed_edge_fn=GraphMLP(
                layer_sizes=tuple([self.edge_hidden_dim] * self.embedding_num_layers),
                use_layernorm=self.use_layernorm,
                name='embed_edge',
            ),
            embed_node_fn=GraphMLP(
                layer_sizes=tuple([self.hidden_dim] * self.embedding_num_layers),
                use_layernorm=self.use_layernorm,
                name='embed_node',
            ),
            embed_global_fn=GraphMLP(
                layer_sizes=tuple([self.hidden_dim] * self.embedding_num_layers),
                use_layernorm=self.use_layernorm,
                name='embed_global',
            ),
        )
        graph = embed_graph(graph_in)
        if self.shared_message_passing_weights:
            core = GraphNetCore(
                hidden_dim=self.hidden_dim,
                edge_hidden_dim=self.edge_hidden_dim,
                update_num_layers=self.update_num_layers,
                edge_update_num_layers=self.edge_update_num_layers,
                use_layernorm=self.use_layernorm,
                use_skip_connections=self.use_skip_connections,
                name='shared_core',
            )
            for _ in range(self.num_message_passing_rounds):
                graph = core(graph)
        else:
            for round_idx in range(self.num_message_passing_rounds):
                graph = GraphNetCore(
                    hidden_dim=self.hidden_dim,
                    edge_hidden_dim=self.edge_hidden_dim,
                    update_num_layers=self.update_num_layers,
                    edge_update_num_layers=self.edge_update_num_layers,
                    use_layernorm=self.use_layernorm,
                    use_skip_connections=self.use_skip_connections,
                    name=f'core_{round_idx}',
                )(graph)
        return graph


def add_actions_to_ball_nodes(graph, action):
    action = jnp.asarray(action, dtype=jnp.float32).reshape(NUM_BALLS, 2)
    nodes = graph.nodes
    action_padding = jnp.zeros((nodes.shape[0], action.shape[-1]), dtype=jnp.float32)
    action_padding = action_padding.at[:NUM_BALLS].set(action)
    return graph._replace(nodes=jnp.concatenate([nodes, action_padding], axis=-1))


class GraphTD3Policy(nn.Module):
    action_dim: int
    gnn_hidden: int = 128
    mlp_hidden: int = 256
    edge_hidden: int = 96
    message_passing_rounds: int = 2

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        gnn_out = GraphNet(
            hidden_dim=self.gnn_hidden,
            edge_hidden_dim=self.edge_hidden,
            num_message_passing_rounds=self.message_passing_rounds,
            use_layernorm=True,
            use_skip_connections=False,
            shared_message_passing_weights=True,
            name='policy_graphnet',
        )(graph)
        ball_nodes = gnn_out.nodes[:NUM_BALLS]
        policy_head = GraphMLP(
            layer_sizes=(self.mlp_hidden, self.mlp_hidden, self.action_dim // NUM_BALLS),
            use_layernorm=True,
            last_layer_scale=0.005,
            name='policy_head',
        )
        actions = policy_head(ball_nodes)
        return jnp.tanh(actions.reshape(-1))


class GraphTD3Critic(nn.Module):
    gnn_hidden: int = 128
    mlp_hidden: int = 256
    edge_hidden: int = 96
    message_passing_rounds: int = 2

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, action):
        graph_with_action = add_actions_to_ball_nodes(graph, action)
        gnn_out = GraphNet(
            hidden_dim=self.gnn_hidden,
            edge_hidden_dim=self.edge_hidden,
            num_message_passing_rounds=self.message_passing_rounds,
            use_layernorm=True,
            use_skip_connections=False,
            shared_message_passing_weights=True,
            name='critic_graphnet',
        )(graph_with_action)
        q_value = GraphMLP(
            layer_sizes=(self.mlp_hidden, self.mlp_hidden, 1),
            use_layernorm=True,
            last_layer_scale=0.005,
            name='q_head',
        )(gnn_out.globals).squeeze(-1)
        reward_pred = GraphMLP(
            layer_sizes=(self.mlp_hidden, self.mlp_hidden, 1),
            use_layernorm=True,
            last_layer_scale=0.005,
            name='reward_head',
        )(gnn_out.globals).squeeze(-1)
        return q_value, reward_pred


class TD3Agent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        seed=0,
        policy_lr=5e-5,
        critic_lr=5e-5,
        discount=0.94,
        policy_tau=4e-5,
        critic_tau=8e-5,
        target_sigma=0.2,
        noise_clip=0.4,
        reward_loss_weight=0.0,
        gnn_hidden=96,
        mlp_hidden=256,
        obs_mean=None,
        obs_std=None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discount = float(discount)
        self.policy_tau = float(policy_tau)
        self.critic_tau = float(critic_tau)
        self.target_sigma = float(target_sigma)
        self.noise_clip = float(noise_clip)
        self.reward_loss_weight = float(reward_loss_weight)
        self.gnn_hidden = gnn_hidden
        self.mlp_hidden = mlp_hidden
        self.obs_mean = np.asarray(obs_mean if obs_mean is not None else np.zeros(obs_dim, dtype=np.float32), dtype=np.float32)
        self.obs_std = np.maximum(
            np.asarray(obs_std if obs_std is not None else np.ones(obs_dim, dtype=np.float32), dtype=np.float32),
            1e-2,
        ).astype(np.float32)
        self.obs_mean_jnp = jnp.asarray(self.obs_mean, dtype=jnp.float32)
        self.obs_std_jnp = jnp.asarray(self.obs_std, dtype=jnp.float32)
        self.last_losses = {}

        self.rng = jax.random.PRNGKey(seed)
        self.rng, policy_key, q1_key, q2_key = jax.random.split(self.rng, 4)

        dummy_graph = obs_to_graph(BallRoboBalletEnvironment().reset())
        dummy_action = jnp.zeros((act_dim,), dtype=jnp.float32)

        self.policy = GraphTD3Policy(action_dim=act_dim, gnn_hidden=gnn_hidden, mlp_hidden=mlp_hidden)
        self.q1 = GraphTD3Critic(gnn_hidden=gnn_hidden, mlp_hidden=mlp_hidden)
        self.q2 = GraphTD3Critic(gnn_hidden=gnn_hidden, mlp_hidden=mlp_hidden)

        self.policy_state = train_state.TrainState.create(apply_fn=self.policy.apply, params=self.policy.init(policy_key, dummy_graph), tx=optax.adam(policy_lr))
        self.q1_state = train_state.TrainState.create(apply_fn=self.q1.apply, params=self.q1.init(q1_key, dummy_graph, dummy_action), tx=optax.adam(critic_lr))
        self.q2_state = train_state.TrainState.create(apply_fn=self.q2.apply, params=self.q2.init(q2_key, dummy_graph, dummy_action), tx=optax.adam(critic_lr))
        self.target_policy_params = self.policy_state.params
        self.target_q1_params = self.q1_state.params
        self.target_q2_params = self.q2_state.params

        self._policy_apply_batch_jit = jax.jit(self._policy_apply_batch)
        self._q1_apply_batch_jit = jax.jit(self._q1_apply_batch)
        self._q2_apply_batch_jit = jax.jit(self._q2_apply_batch)
        self._update_step_jit = jax.jit(self._update_step)

    def _policy_apply_batch(self, policy_params, nodes_batch, edges_batch, globals_batch):
        return jax.vmap(lambda nodes, edges, globals_: self.policy.apply(policy_params, build_graph_from_components(nodes, edges, globals_[None, :])))(nodes_batch, edges_batch, globals_batch)

    def _q1_apply_batch(self, q_params, nodes_batch, edges_batch, globals_batch, actions):
        return jax.vmap(lambda nodes, edges, globals_, action: self.q1.apply(q_params, build_graph_from_components(nodes, edges, globals_[None, :]), action))(nodes_batch, edges_batch, globals_batch, actions)

    def _q2_apply_batch(self, q_params, nodes_batch, edges_batch, globals_batch, actions):
        return jax.vmap(lambda nodes, edges, globals_, action: self.q2.apply(q_params, build_graph_from_components(nodes, edges, globals_[None, :]), action))(nodes_batch, edges_batch, globals_batch, actions)

    def _soft_update(self, params, target_params, tau):
        return jax.tree_util.tree_map(lambda p, tp: tau * p + (1.0 - tau) * tp, params, target_params)

    def _update_step(self, policy_state, q1_state, q2_state, target_policy_params, target_q1_params, target_q2_params, batch, rng):
        obs = jnp.asarray(batch['obs'], dtype=jnp.float32)
        act = jnp.asarray(batch['act'], dtype=jnp.float32)
        rew = jnp.asarray(batch['rew'], dtype=jnp.float32)
        next_obs = jnp.asarray(batch['next_obs'], dtype=jnp.float32)
        done = jnp.asarray(batch['done'], dtype=jnp.float32)

        obs_nodes, obs_edges, obs_globals = flat_obs_batch_to_graph_components(obs, self.obs_mean_jnp, self.obs_std_jnp)
        next_nodes, next_edges, next_globals = flat_obs_batch_to_graph_components(next_obs, self.obs_mean_jnp, self.obs_std_jnp)

        rng, noise_key = jax.random.split(rng)
        target_actions = self._policy_apply_batch_jit(target_policy_params, next_nodes, next_edges, next_globals)
        target_noise = self.target_sigma * jnp.clip(jax.random.normal(noise_key, target_actions.shape), -self.noise_clip, self.noise_clip)
        next_action = jnp.clip(target_actions + target_noise, -1.0, 1.0)
        q1_next, _ = self._q1_apply_batch_jit(target_q1_params, next_nodes, next_edges, next_globals, next_action)
        q2_next, _ = self._q2_apply_batch_jit(target_q2_params, next_nodes, next_edges, next_globals, next_action)
        target = jax.lax.stop_gradient(rew + (1.0 - done) * self.discount * jnp.minimum(q1_next, q2_next))

        def q1_loss_fn(q_params):
            q_pred, reward_pred = self._q1_apply_batch_jit(q_params, obs_nodes, obs_edges, obs_globals, act)
            td_loss = jnp.mean((target - q_pred) ** 2)
            reward_loss = jnp.mean((rew - reward_pred) ** 2)
            return td_loss + self.reward_loss_weight * reward_loss

        def q2_loss_fn(q_params):
            q_pred, reward_pred = self._q2_apply_batch_jit(q_params, obs_nodes, obs_edges, obs_globals, act)
            td_loss = jnp.mean((target - q_pred) ** 2)
            reward_loss = jnp.mean((rew - reward_pred) ** 2)
            return td_loss + self.reward_loss_weight * reward_loss

        q1_loss, q1_grads = jax.value_and_grad(q1_loss_fn)(q1_state.params)
        q2_loss, q2_grads = jax.value_and_grad(q2_loss_fn)(q2_state.params)
        q1_state = q1_state.apply_gradients(grads=q1_grads)
        q2_state = q2_state.apply_gradients(grads=q2_grads)

        def policy_loss_fn(policy_params):
            action = self._policy_apply_batch_jit(policy_params, obs_nodes, obs_edges, obs_globals)
            q1_pi, _ = self._q1_apply_batch_jit(q1_state.params, obs_nodes, obs_edges, obs_globals, action)
            return -jnp.mean(q1_pi)

        policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_state.params)
        policy_state = policy_state.apply_gradients(grads=policy_grads)
        target_policy_params = self._soft_update(policy_state.params, target_policy_params, self.policy_tau)
        target_q1_params = self._soft_update(q1_state.params, target_q1_params, self.critic_tau)
        target_q2_params = self._soft_update(q2_state.params, target_q2_params, self.critic_tau)

        metrics = {
            'policy_loss': policy_loss,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'target_mean': jnp.mean(target),
            'target_std': jnp.std(target),
        }
        return policy_state, q1_state, q2_state, target_policy_params, target_q1_params, target_q2_params, metrics, rng

    def select_action_flat(self, flat_obs, noise_sigma=0.0):
        flat_obs = np.asarray(flat_obs, dtype=np.float32)[None, :]
        nodes_batch, edges_batch, globals_batch = flat_obs_batch_to_graph_components(flat_obs, self.obs_mean_jnp, self.obs_std_jnp)
        action = np.asarray(self._policy_apply_batch_jit(self.policy_state.params, nodes_batch, edges_batch, globals_batch)[0], dtype=np.float32)
        if noise_sigma > 0.0:
            action = np.clip(action + np.random.normal(0.0, noise_sigma, size=action.shape).astype(np.float32), -1.0, 1.0)
        return action.astype(np.float32)

    def select_action(self, obs, noise_sigma=0.0):
        return self.select_action_flat(flatten_obs(obs), noise_sigma=noise_sigma)

    def select_action_batch(self, flat_obs_batch):
        flat_obs_batch = np.asarray(flat_obs_batch, dtype=np.float32)
        nodes_batch, edges_batch, globals_batch = flat_obs_batch_to_graph_components(flat_obs_batch, self.obs_mean_jnp, self.obs_std_jnp)
        return np.asarray(self._policy_apply_batch_jit(self.policy_state.params, nodes_batch, edges_batch, globals_batch), dtype=np.float32)

    def update(self, batch, materialize_metrics=False):
        result = self._update_step_jit(
            self.policy_state,
            self.q1_state,
            self.q2_state,
            self.target_policy_params,
            self.target_q1_params,
            self.target_q2_params,
            batch,
            self.rng,
        )
        self.policy_state, self.q1_state, self.q2_state, self.target_policy_params, self.target_q1_params, self.target_q2_params, metrics, self.rng = result
        if materialize_metrics:
            try:
                metrics = jax.tree_util.tree_map(lambda value: np.asarray(jax.block_until_ready(value)).item(), metrics)
            except Exception as exc:
                raise RuntimeError(
                    f'Failed to materialize TD3 metrics after update step. '
                    f'batch_obs_shape={batch["obs"].shape}, batch_act_shape={batch["act"].shape}'
                ) from exc
            self.last_losses = {key: float(value) for key, value in metrics.items()}

    def save(
        self,
        path,
        training_step=0,
        best_success_rate=-np.inf,
        best_avg_return=-np.inf,
        best_all_targets_done_rate=-np.inf,
        best_avg_final_score=-np.inf,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            policy=serialize_params_to_uint8(self.policy_state.params),
            q1=serialize_params_to_uint8(self.q1_state.params),
            q2=serialize_params_to_uint8(self.q2_state.params),
            model_architecture=np.array(CHECKPOINT_ARCHITECTURE),
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            obs_dim=np.array(self.obs_dim, dtype=np.int32),
            act_dim=np.array(self.act_dim, dtype=np.int32),
            gnn_hidden=np.array(self.gnn_hidden, dtype=np.int32),
            mlp_hidden=np.array(self.mlp_hidden, dtype=np.int32),
            training_step=np.array(training_step, dtype=np.int32),
            best_success_rate=np.array(best_success_rate, dtype=np.float32),
            best_avg_return=np.array(best_avg_return, dtype=np.float32),
            best_all_targets_done_rate=np.array(best_all_targets_done_rate, dtype=np.float32),
            best_avg_final_score=np.array(best_avg_final_score, dtype=np.float32),
        )

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        checkpoint_architecture = str(data['model_architecture']) if 'model_architecture' in data.files else 'legacy'
        if checkpoint_architecture != CHECKPOINT_ARCHITECTURE:
            raise ValueError(
                f'Checkpoint architecture mismatch: expected {CHECKPOINT_ARCHITECTURE}, '
                f'got {checkpoint_architecture}. Train a fresh model or keep using the legacy network definition.'
            )
        if 'obs_mean' in data.files:
            self.obs_mean = data['obs_mean'].astype(np.float32)
            self.obs_mean_jnp = jnp.asarray(self.obs_mean, dtype=jnp.float32)
        if 'obs_std' in data.files:
            self.obs_std = data['obs_std'].astype(np.float32)
            self.obs_std_jnp = jnp.asarray(self.obs_std, dtype=jnp.float32)
        self.policy_state = self.policy_state.replace(params=flax.serialization.from_bytes(self.policy_state.params, npz_value_to_bytes(data['policy'])))
        self.q1_state = self.q1_state.replace(params=flax.serialization.from_bytes(self.q1_state.params, npz_value_to_bytes(data['q1'])))
        self.q2_state = self.q2_state.replace(params=flax.serialization.from_bytes(self.q2_state.params, npz_value_to_bytes(data['q2'])))
        self.target_policy_params = self.policy_state.params
        self.target_q1_params = self.q1_state.params
        self.target_q2_params = self.q2_state.params


def load_checkpoint_metadata(path):
    if not path or not os.path.exists(path):
        return {
            'training_step': 0,
            'best_success_rate': -np.inf,
            'best_avg_return': -np.inf,
            'best_all_targets_done_rate': -np.inf,
            'best_avg_final_score': -np.inf,
        }
    data = np.load(path, allow_pickle=True)
    return {
        'training_step': int(data['training_step']) if 'training_step' in data.files else 0,
        'best_success_rate': float(data['best_success_rate']) if 'best_success_rate' in data.files else -np.inf,
        'best_avg_return': float(data['best_avg_return']) if 'best_avg_return' in data.files else -np.inf,
        'best_all_targets_done_rate': float(data['best_all_targets_done_rate']) if 'best_all_targets_done_rate' in data.files else -np.inf,
        'best_avg_final_score': float(data['best_avg_final_score']) if 'best_avg_final_score' in data.files else -np.inf,
    }


def load_agent_from_checkpoint(path):
    data = np.load(path, allow_pickle=True)
    agent = TD3Agent(
        obs_dim=int(data['obs_dim']) if 'obs_dim' in data.files else OBS_DIM,
        act_dim=int(data['act_dim']) if 'act_dim' in data.files else ACT_DIM,
        gnn_hidden=int(data['gnn_hidden']) if 'gnn_hidden' in data.files else 96,
        mlp_hidden=int(data['mlp_hidden']) if 'mlp_hidden' in data.files else 256,
        obs_mean=data['obs_mean'].astype(np.float32) if 'obs_mean' in data.files else None,
        obs_std=data['obs_std'].astype(np.float32) if 'obs_std' in data.files else None,
    )
    agent.load(path)
    return agent


def _exp_decay(initial_value, decay_rate, step, steps_per_decay=100000):
    return float(initial_value * decay_rate ** (step / steps_per_decay))


def collect_normalization_stats(env, act_dim, num_steps):
    obs_samples = []
    obs = env.reset()
    rollout_horizon = min(env.max_steps, 32)
    steps_since_reset = 0
    for _ in range(num_steps):
        obs_samples.append(flatten_obs(obs))
        action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        next_obs, _, done, _ = env.step(action)
        steps_since_reset += 1
        if done or steps_since_reset >= rollout_horizon:
            obs = env.reset()
            steps_since_reset = 0
        else:
            obs = next_obs
    obs_samples = np.asarray(obs_samples, dtype=np.float32)
    obs_std = np.maximum(obs_samples.std(axis=0), 1e-2).astype(np.float32)
    return obs_samples.mean(axis=0).astype(np.float32), obs_std


def _episode_stats_from_info(env, total_reward, info, *, is_hindsight_replay=0, hindsight_metadata=None):
    return_distances = np.asarray(info.get('return_distances', np.zeros(NUM_BALLS, dtype=np.float32)), dtype=np.float32)
    num_targets_done = int(info.get('targets_done', env.num_targets_done()))
    entered_return_phase = bool(getattr(env, 'return_phase_started', False) or info.get('entered_return_phase', False))
    return EpisodeStats(
        total_reward=float(total_reward),
        final_score=float(info.get('current_score', env.current_score_value())),
        num_targets_done=num_targets_done,
        episode_length=int(env.step_count),
        total_collision_penalty=float(info.get('total_collision_penalty', env.total_collision_penalty())),
        total_collision_ball_steps=int(env.total_collision_ball_steps_value()),
        total_mean_acceleration=float(env.total_acceleration()),
        success=int(bool(info.get('success', False))),
        all_targets_done=int(bool(info.get('all_targets_done', False))),
        entered_return_phase=int(entered_return_phase),
        return_ready=int(bool(info.get('all_targets_done', False) and np.all(return_distances < env.return_threshold))),
        completion_ratio=float(num_targets_done / max(env.num_targets, 1)),
        final_mean_target_score=float(info.get('mean_target_score', 0.0)),
        final_return_to_start_score=float(info.get('return_to_start_score', 0.0)),
        final_return_distance_mean=float(np.mean(return_distances)) if return_distances.size else 0.0,
        final_return_distance_max=float(np.max(return_distances)) if return_distances.size else 0.0,
        min_pairwise_ball_distance=float(info.get('pairwise_ball_distance_min', 0.0)),
        is_hindsight_replay=int(is_hindsight_replay),
        hindsight_target_count=int(hindsight_metadata.requested_target_count if hindsight_metadata is not None else 0),
        hindsight_targets_changed=int(len(hindsight_metadata.changed_target_indices) if hindsight_metadata is not None else 0),
        hindsight_goal_delta=int(max(0, (hindsight_metadata.requested_target_count if hindsight_metadata is not None else 0) - num_targets_done)),
    )


def aggregate_episode_stats(stats_list):
    if not stats_list:
        return {}
    stats_dicts = [asdict(stats) if isinstance(stats, EpisodeStats) else dict(stats) for stats in stats_list]
    keys = stats_dicts[0].keys()
    aggregated = {f'avg_{key}': float(np.mean([float(stats[key]) for stats in stats_dicts])) for key in keys}
    aggregated['success_rate'] = aggregated['avg_success']
    aggregated['all_targets_done_rate'] = aggregated['avg_all_targets_done']
    aggregated['entered_return_phase_rate'] = aggregated['avg_entered_return_phase']
    aggregated['return_ready_rate'] = aggregated['avg_return_ready']
    aggregated['avg_targets_done'] = aggregated['avg_num_targets_done']
    aggregated['avg_length'] = aggregated['avg_episode_length']
    aggregated['avg_return'] = aggregated['avg_total_reward']
    aggregated['avg_collision_penalty'] = aggregated['avg_total_collision_penalty']
    return aggregated


def format_eval_log(step, stats):
    return (
        f"[EVAL] step={step} "
        f"phase(s/all/ret/park)={stats.get('success_rate', 0.0):.2f}/{stats.get('all_targets_done_rate', 0.0):.2f}/"
        f"{stats.get('entered_return_phase_rate', 0.0):.2f}/{stats.get('return_ready_rate', 0.0):.2f} "
        f"task(score/ret/tgt)={stats.get('avg_final_score', 0.0):.4f}/{stats.get('avg_return', 0.0):.4f}/"
        f"{stats.get('avg_targets_done', 0.0):.2f} "
        f"state(rdist/coll/len)={stats.get('avg_final_return_distance_mean', 0.0):.4f}/"
        f"{stats.get('avg_collision_penalty', 0.0):.6f}/{stats.get('avg_length', 0.0):.1f}"
    )


def summarize_hindsight_target_counts(hindsight_stats_list):
    if not hindsight_stats_list:
        return 'none'
    counts = {}
    for stats in hindsight_stats_list:
        if isinstance(stats, EpisodeStats):
            target_count = int(stats.hindsight_target_count)
        else:
            target_count = int(stats.get('hindsight_target_count', 0))
        counts[target_count] = counts.get(target_count, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda item: item[0])
    return ','.join(f'{target_count}:{count}' for target_count, count in sorted_counts)


def format_train_log(step, recent_stats, recent_hindsight, recent_hindsight_stats_list, losses, total_hindsight_episodes, total_hindsight_added):
    hindsight_target_mix = summarize_hindsight_target_counts(recent_hindsight_stats_list)
    return (
        f"[TRAIN] step={step} "
        f"phase(s/all/ret/park)={recent_stats.get('success_rate', 0.0):.2f}/{recent_stats.get('all_targets_done_rate', 0.0):.2f}/"
        f"{recent_stats.get('entered_return_phase_rate', 0.0):.2f}/{recent_stats.get('return_ready_rate', 0.0):.2f} "
        f"task(score/ret/tgt)={recent_stats.get('avg_final_score', 0.0):.4f}/{recent_stats.get('avg_return', 0.0):.4f}/"
        f"{recent_stats.get('avg_targets_done', 0.0):.2f} "
        f"state(rdist/coll)={recent_stats.get('avg_final_return_distance_mean', 0.0):.4f}/"
        f"{recent_stats.get('avg_collision_penalty', 0.0):.6f} "
        f"her(ep/tr/ret/tgt)={total_hindsight_episodes}/{total_hindsight_added}/"
        f"{recent_hindsight.get('entered_return_phase_rate', 0.0):.2f}/{recent_hindsight.get('avg_targets_done', 0.0):.2f} "
        f"her_mix(tgt:count)={hindsight_target_mix} "
        f"loss(q1/q2/pi)={losses.get('q1_loss', 0.0):.4f}/{losses.get('q2_loss', 0.0):.4f}/{losses.get('policy_loss', 0.0):.4f}"
    )


def _better_eval_candidate(candidate_stats, incumbent_stats):
    if incumbent_stats is None:
        return True
    candidate_key = (
        float(candidate_stats.get('success_rate', 0.0)),
        float(candidate_stats.get('all_targets_done_rate', 0.0)),
        float(candidate_stats.get('avg_final_score', 0.0)),
        float(candidate_stats.get('avg_targets_done', 0.0)),
        float(candidate_stats.get('avg_return', 0.0)),
        -float(candidate_stats.get('avg_final_return_distance_mean', 0.0)),
    )
    incumbent_key = (
        float(incumbent_stats.get('success_rate', 0.0)),
        float(incumbent_stats.get('all_targets_done_rate', 0.0)),
        float(incumbent_stats.get('avg_final_score', 0.0)),
        float(incumbent_stats.get('avg_targets_done', 0.0)),
        float(incumbent_stats.get('avg_return', 0.0)),
        -float(incumbent_stats.get('avg_final_return_distance_mean', 0.0)),
    )
    return candidate_key > incumbent_key


def run_episode(env, agent, episode_idx, global_step, random_exploration_episodes, policy_sigma, policy_sigma_decay_rate):
    obs = env.reset()
    start_state = env.get_start_state()

    transitions = []
    total_reward = 0.0
    while not env.terminal():
        if episode_idx < random_exploration_episodes:
            action = np.random.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
        else:
            sigma = _exp_decay(policy_sigma, policy_sigma_decay_rate, global_step)
            action = agent.select_action(obs, noise_sigma=sigma)
        next_obs, reward, done, info = env.step(action)
        transitions.append(
            {
                'obs': flatten_obs(obs),
                'act': action.astype(np.float32),
                'rew': float(reward),
                'next_obs': flatten_obs(next_obs),
                'done': float(done),
            }
        )
        total_reward += float(reward)
        obs = next_obs
        if done:
            break

    stats = _episode_stats_from_info(env, total_reward, info)
    return start_state, transitions, stats


def generate_hindsight_episode(start_state, actual_transitions, actual_stats, env_kwargs, num_targets_to_solve):
    if not actual_transitions:
        return None
    if actual_stats.num_targets_done >= num_targets_to_solve:
        return None
    observations = [flat_obs_to_dict(actual_transitions[0]['obs'])]
    observations.extend(flat_obs_to_dict(transition['next_obs']) for transition in actual_transitions)
    target_done_final = observations[-1]['target_done'] > 0.5
    unsolved_targets = [idx for idx in range(NUM_TARGETS) if not target_done_final[idx]]
    if not unsolved_targets:
        return None
    requested_target_count = min(num_targets_to_solve, NUM_TARGETS)
    targets_to_change = random_sample(unsolved_targets, requested_target_count - actual_stats.num_targets_done)
    if not targets_to_change:
        return None
    new_start_state = StartState(
        ball_pos=np.asarray(start_state.ball_pos, dtype=np.float32).copy(),
        ball_vel=np.asarray(start_state.ball_vel, dtype=np.float32).copy(),
        target_pos=np.asarray(start_state.target_pos, dtype=np.float32).copy(),
    )
    completion_event_candidates = _extract_completion_event_candidates(observations)
    source_steps = []
    source_ball_indices = []
    source_is_completion_event = []
    for target_idx in targets_to_change:
        anchor = _sample_future_pose_anchor(observations, completion_event_candidates, target_idx)
        if anchor is None:
            return None
        solution_step, ball_idx, used_completion_event = anchor
        future_obs = observations[solution_step]
        new_start_state.target_pos[target_idx] = future_obs['ball_pos'][ball_idx]
        source_steps.append(solution_step)
        source_ball_indices.append(ball_idx)
        source_is_completion_event.append(used_completion_event)

    hindsight_metadata = HindsightMetadata(
        requested_target_count=requested_target_count,
        changed_target_indices=tuple(int(idx) for idx in targets_to_change),
        source_steps=tuple(int(step) for step in source_steps),
        source_ball_indices=tuple(int(idx) for idx in source_ball_indices),
        source_is_completion_event=tuple(int(flag) for flag in source_is_completion_event),
        completion_event_candidate_count=int(len(completion_event_candidates)),
    )

    env = BallRoboBalletEnvironment(**env_kwargs)
    obs = env.reset(start_state=new_start_state)
    hindsight_transitions = []
    total_reward = 0.0
    info = {}
    for transition in actual_transitions:
        next_obs, reward, done, info = env.step(transition['act'])
        hindsight_transitions.append(
            {
                'obs': flatten_obs(obs),
                'act': transition['act'],
                'rew': float(reward),
                'next_obs': flatten_obs(next_obs),
                'done': float(done),
            }
        )
        total_reward += float(reward)
        obs = next_obs
        if done:
            break
    hindsight_stats = _episode_stats_from_info(
        env,
        total_reward,
        info,
        is_hindsight_replay=1,
        hindsight_metadata=hindsight_metadata,
    )
    return hindsight_transitions, hindsight_stats, hindsight_metadata


def _append_unique_target_count(resolved_counts, target_count):
    target_count = int(np.clip(target_count, 1, NUM_TARGETS))
    if target_count not in resolved_counts:
        resolved_counts.append(target_count)


def resolve_hindsight_target_counts(base_target_count, actual_completed_targets):
    resolved_counts = []
    curriculum_target_count = int(np.clip(base_target_count, 1, NUM_TARGETS))
    actual_completed_targets = int(np.clip(actual_completed_targets, 0, NUM_TARGETS))
    near_full_target_count = max(
        curriculum_target_count,
        actual_completed_targets + HIGH_COMPLETION_HINDSIGHT_MARGIN,
        NUM_TARGETS - HIGH_COMPLETION_HINDSIGHT_MARGIN,
    )

    _append_unique_target_count(resolved_counts, curriculum_target_count)
    if curriculum_target_count < NUM_TARGETS:
        _append_unique_target_count(resolved_counts, near_full_target_count)
        _append_unique_target_count(resolved_counts, NUM_TARGETS)
    return resolved_counts


def random_sample(values, sample_count):
    if sample_count <= 0:
        return []
    sample_count = min(sample_count, len(values))
    return list(np.random.choice(np.asarray(values), size=sample_count, replace=False))


def _extract_completion_event_candidates(observations):
    candidates = []
    previous_done = observations[0]['target_done'] > 0.5
    for step in range(1, len(observations)):
        current_obs = observations[step]
        current_done = current_obs['target_done'] > 0.5
        newly_done_targets = np.flatnonzero(current_done & (~previous_done))
        if newly_done_targets.size == 0:
            previous_done = current_done
            continue
        ball_pos = np.asarray(current_obs['ball_pos'], dtype=np.float32)
        target_pos = np.asarray(current_obs['target_pos'], dtype=np.float32)
        for target_idx in newly_done_targets:
            distances = np.linalg.norm(ball_pos - target_pos[int(target_idx)][None, :], axis=-1)
            ball_idx = int(np.argmin(distances))
            candidates.append(
                {
                    'step': int(step),
                    'ball_idx': ball_idx,
                    'target_idx': int(target_idx),
                    'distance': float(distances[ball_idx]),
                }
            )
        previous_done = current_done
    return candidates


def _sample_future_pose_anchor(observations, completion_event_candidates, target_idx):
    if completion_event_candidates and np.random.rand() < 0.8:
        candidate = completion_event_candidates[np.random.randint(0, len(completion_event_candidates))]
        return int(candidate['step']), int(candidate['ball_idx']), 1

    candidate_steps = np.arange(1, len(observations), dtype=np.int32)
    if candidate_steps.size == 0:
        return None
    step_weights = np.power(np.arange(1, candidate_steps.size + 1, dtype=np.float64), 2.0)
    step_weights = step_weights / np.sum(step_weights)
    solution_step = int(np.random.choice(candidate_steps, p=step_weights))
    future_obs = observations[solution_step]
    target_pos = np.asarray(future_obs['target_pos'][target_idx], dtype=np.float32)
    ball_pos = np.asarray(future_obs['ball_pos'], dtype=np.float32)
    ball_distances = np.linalg.norm(ball_pos - target_pos[None, :], axis=-1)
    ball_idx = int(np.argmin(ball_distances))
    return solution_step, ball_idx, 0


def evaluate_agent(agent, env_kwargs, num_episodes=8):
    eval_stats = []
    for episode_idx in range(num_episodes):
        np.random.seed(1234 + episode_idx)
        env = BallRoboBalletEnvironment(**env_kwargs)
        obs = env.reset()
        total_reward = 0.0
        info = {}
        while not env.terminal():
            action = agent.select_action(obs, noise_sigma=0.0)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            if done:
                break
        eval_stats.append(_episode_stats_from_info(env, total_reward, info))
    return aggregate_episode_stats(eval_stats)


def train(
    *,
    total_steps=30000,
    batch_size=128,
    update_after=1500,
    updates_per_step=2,
    buffer_size=250000,
    norm_steps=1500,
    random_exploration_episodes=12,
    policy_sigma=0.25,
    policy_sigma_decay_rate=0.8,
    replay_hindsight_num_targets=4,
    collision_penalty_scaling=1.0,
    resume_path=None,
    save_interval=2500,
    eval_interval=2500,
    log_interval=250,
    eval_episodes=6,
    seed=0,
    reward_loss_weight=0.0,
):
    np.random.seed(seed)
    env_kwargs = dict(
        target_score_shaping=0.0,
        collision_penalty_scaling=collision_penalty_scaling,
        acceleration_cost=0.0,
        return_to_start_required=True,
        return_to_start_weight=0.25,
    )
    env = BallRoboBalletEnvironment(**env_kwargs)
    best_success_rate = -np.inf
    best_avg_return = -np.inf
    best_all_targets_done_rate = -np.inf
    best_avg_final_score = -np.inf
    best_eval_stats = None
    start_step = 0

    if resume_path:
        metadata = load_checkpoint_metadata(resume_path)
        start_step = metadata['training_step']
        best_success_rate = metadata['best_success_rate']
        best_avg_return = metadata['best_avg_return']
        best_all_targets_done_rate = metadata['best_all_targets_done_rate']
        best_avg_final_score = metadata['best_avg_final_score']
        best_eval_stats = {
            'success_rate': best_success_rate,
            'all_targets_done_rate': best_all_targets_done_rate,
            'avg_final_score': best_avg_final_score,
            'avg_return': best_avg_return,
        }
        agent = load_agent_from_checkpoint(resume_path)
        print(f"[RESUME] loaded {resume_path} at step={start_step}")
    else:
        obs_mean, obs_std = collect_normalization_stats(env, ACT_DIM, norm_steps)
        agent = TD3Agent(OBS_DIM, ACT_DIM, seed=seed, obs_mean=obs_mean, obs_std=obs_std, reward_loss_weight=reward_loss_weight)

    buffer = ReplayBuffer(max_size=buffer_size, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    episode_idx = 0
    global_step = start_step
    last_eval = None
    completed_stats = []
    hindsight_stats = []
    total_hindsight_added = 0
    total_hindsight_episodes = 0
    while global_step < start_step + total_steps:
        start_state, episode_transitions, episode_stats = run_episode(
            env=env,
            agent=agent,
            episode_idx=episode_idx,
            global_step=global_step,
            random_exploration_episodes=random_exploration_episodes,
            policy_sigma=policy_sigma,
            policy_sigma_decay_rate=policy_sigma_decay_rate,
        )
        for transition in episode_transitions:
            buffer.add(transition['obs'], transition['act'], transition['rew'], transition['next_obs'], transition['done'])
            global_step += 1
            if buffer.size >= max(batch_size, update_after):
                should_materialize_metrics = log_interval > 0 and global_step % log_interval == 0
                for update_idx in range(updates_per_step):
                    agent.update(
                        buffer.sample(batch_size),
                        materialize_metrics=should_materialize_metrics and update_idx == updates_per_step - 1,
                    )
            if eval_interval > 0 and global_step % eval_interval == 0 and buffer.size >= max(batch_size, update_after):
                last_eval = evaluate_agent(agent, env_kwargs, num_episodes=eval_episodes)
                print(format_eval_log(global_step, last_eval))
            if save_interval > 0 and global_step % save_interval == 0:
                agent.save(
                    os.path.join(os.path.dirname(__file__), 'models', 'td3_gnn_latest.npz'),
                    training_step=global_step,
                    best_success_rate=best_success_rate,
                    best_avg_return=best_avg_return,
                    best_all_targets_done_rate=best_all_targets_done_rate,
                    best_avg_final_score=best_avg_final_score,
                )
                print(f"[SAVE] step={global_step} model saved.")
            if global_step >= start_step + total_steps:
                break

        completed_stats.append(episode_stats)

        hindsight_target_counts = resolve_hindsight_target_counts(
            replay_hindsight_num_targets,
            episode_stats.num_targets_done,
        )
        for hindsight_target_count in hindsight_target_counts:
            hindsight_result = generate_hindsight_episode(
                start_state=start_state,
                actual_transitions=episode_transitions,
                actual_stats=episode_stats,
                env_kwargs=env_kwargs,
                num_targets_to_solve=hindsight_target_count,
            )
            if hindsight_result is None:
                continue
            hindsight_transitions, imagined_episode_stats, hindsight_metadata = hindsight_result
            for transition in hindsight_transitions:
                buffer.add(transition['obs'], transition['act'], transition['rew'], transition['next_obs'], transition['done'])
                total_hindsight_added += 1
            total_hindsight_episodes += 1
            hindsight_stats.append(imagined_episode_stats)

        if log_interval > 0 and global_step % log_interval == 0 and buffer.size >= max(batch_size, update_after):
            recent_count = min(10, len(completed_stats))
            recent_stats = aggregate_episode_stats(completed_stats[-recent_count:]) if recent_count > 0 else {}
            recent_hindsight_count = min(10, len(hindsight_stats))
            recent_hindsight_stats_list = hindsight_stats[-recent_hindsight_count:] if recent_hindsight_count > 0 else []
            recent_hindsight = aggregate_episode_stats(recent_hindsight_stats_list) if recent_hindsight_count > 0 else {}
            selection_stats = last_eval if last_eval is not None else recent_stats
            if selection_stats and _better_eval_candidate(selection_stats, best_eval_stats):
                best_eval_stats = dict(selection_stats)
                best_success_rate = float(selection_stats.get('success_rate', best_success_rate))
                best_avg_return = float(selection_stats.get('avg_return', best_avg_return))
                best_all_targets_done_rate = float(selection_stats.get('all_targets_done_rate', best_all_targets_done_rate))
                best_avg_final_score = float(selection_stats.get('avg_final_score', best_avg_final_score))
                agent.save(
                    os.path.join(os.path.dirname(__file__), 'models', 'td3_gnn_best.npz'),
                    training_step=global_step,
                    best_success_rate=best_success_rate,
                    best_avg_return=best_avg_return,
                    best_all_targets_done_rate=best_all_targets_done_rate,
                    best_avg_final_score=best_avg_final_score,
                )
            losses = agent.last_losses
            print(
                format_train_log(
                    global_step,
                    recent_stats,
                    recent_hindsight,
                    recent_hindsight_stats_list,
                    losses,
                    total_hindsight_episodes,
                    total_hindsight_added,
                )
            )
        episode_idx += 1
    agent.save(
        os.path.join(os.path.dirname(__file__), 'models', 'td3_gnn_latest.npz'),
        training_step=global_step,
        best_success_rate=best_success_rate,
        best_avg_return=best_avg_return,
        best_all_targets_done_rate=best_all_targets_done_rate,
        best_avg_final_score=best_avg_final_score,
    )
    print('训练完成，模型已保存。')


def main():
    parser = argparse.ArgumentParser(description='RoboBallet-style 2D multi-ball TD3-GNN training')
    parser.add_argument('--total_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--update_after', type=int, default=1500)
    parser.add_argument('--updates_per_step', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=250000)
    parser.add_argument('--norm_steps', type=int, default=1500)
    parser.add_argument('--random_exploration_episodes', type=int, default=12)
    parser.add_argument('--policy_sigma', type=float, default=0.25)
    parser.add_argument('--policy_sigma_decay_rate', type=float, default=0.8)
    parser.add_argument('--replay_hindsight_num_targets', type=int, default=4)
    parser.add_argument('--collision_penalty_scaling', type=float, default=1.0)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=2500)
    parser.add_argument('--eval_interval', type=int, default=2500)
    parser.add_argument('--log_interval', type=int, default=250)
    parser.add_argument('--eval_episodes', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reward_loss_weight', type=float, default=0.0)
    args = parser.parse_args()
    train(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        update_after=args.update_after,
        updates_per_step=args.updates_per_step,
        buffer_size=args.buffer_size,
        norm_steps=args.norm_steps,
        random_exploration_episodes=args.random_exploration_episodes,
        policy_sigma=args.policy_sigma,
        policy_sigma_decay_rate=args.policy_sigma_decay_rate,
        replay_hindsight_num_targets=args.replay_hindsight_num_targets,
        collision_penalty_scaling=args.collision_penalty_scaling,
        resume_path=args.resume_path,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        reward_loss_weight=args.reward_loss_weight,
    )


if __name__ == '__main__':
    main()