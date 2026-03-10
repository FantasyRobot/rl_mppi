
"""
基于 Flax + Jraph 的 GNN-SAC 训练主流程
- 环境：BallGNNEnvironment
- 网络：Flax参数化GNN policy/value
- 算法：SAC
- 支持模型保存/加载
- 仅小球和目标为节点
"""
import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
import jraph
import flax
from flax import linen as nn
from flax.training import train_state
import optax
from envball_gnn import BallGNNEnvironment
import optuna


GRAPH_SENDERS = jnp.array([0, 1], dtype=jnp.int32)
GRAPH_RECEIVERS = jnp.array([1, 0], dtype=jnp.int32)
GRAPH_N_NODE = jnp.array([2], dtype=jnp.int32)
GRAPH_N_EDGE = jnp.array([2], dtype=jnp.int32)
GRAPH_GLOBALS = jnp.zeros((1, 1), dtype=jnp.float32)


def normalize_obs_array(obs_array, obs_mean, obs_std):
    return (obs_array - obs_mean) / (obs_std + 1e-8)


def normalize_graph_obs(obs, obs_mean, obs_std):
    return {
        'ball_pos': normalize_obs_array(np.asarray(obs['ball_pos'], dtype=np.float32), obs_mean[:2], obs_std[:2]),
        'ball_vel': normalize_obs_array(np.asarray(obs['ball_vel'], dtype=np.float32), obs_mean[2:4], obs_std[2:4]),
        'target_pos': normalize_obs_array(np.asarray(obs['target_pos'], dtype=np.float32), obs_mean[:2], obs_std[:2]),
    }


def flat_obs_to_dict(flat_obs, target_pos):
    goal = np.asarray(flat_obs[4:6], dtype=np.float32) if len(flat_obs) >= 6 else np.asarray(target_pos, dtype=np.float32)
    return {
        'ball_pos': np.asarray(flat_obs[:2], dtype=np.float32),
        'ball_vel': np.asarray(flat_obs[2:4], dtype=np.float32),
        'target_pos': goal,
    }


def flatten_obs(obs):
    return np.array(list(obs['ball_pos']) + list(obs['ball_vel']) + list(obs['target_pos']), dtype=np.float32)


def relabeled_reward(obs_flat, next_obs_flat, goal, reach_threshold):
    prev_distance = np.linalg.norm(obs_flat[:2] - goal)
    next_distance = np.linalg.norm(next_obs_flat[:2] - goal)
    reward = (prev_distance - next_distance) - 0.005
    success = next_distance < reach_threshold
    if success:
        reward += 5.0
    return float(reward), float(success)


def add_hindsight_replay(buffer, episode_transitions, reach_threshold, max_her_samples=64):
    if len(episode_transitions) < 2 or max_her_samples <= 0:
        return 0

    candidate_indices = np.arange(len(episode_transitions) - 1)
    sample_count = min(max_her_samples, len(candidate_indices))
    sampled_indices = np.random.choice(candidate_indices, size=sample_count, replace=False)
    added = 0

    for idx in sampled_indices:
        future_idx = np.random.randint(idx, len(episode_transitions))
        goal = episode_transitions[future_idx]['next_obs'][:2].copy()

        her_obs = episode_transitions[idx]['obs'].copy()
        her_next_obs = episode_transitions[idx]['next_obs'].copy()
        her_obs[4:6] = goal
        her_next_obs[4:6] = goal
        her_reward, her_done = relabeled_reward(her_obs, her_next_obs, goal, reach_threshold)

        buffer.add(
            her_obs,
            episode_transitions[idx]['act'],
            her_reward,
            her_next_obs,
            her_done,
        )
        added += 1

    return added


def evaluate_agent(agent, num_episodes=10, max_steps=400, base_seed=1234, reset_span=2.5, reach_threshold=0.5):
    rng_state = np.random.get_state()

    returns = np.zeros(num_episodes, dtype=np.float32)
    lengths = np.zeros(num_episodes, dtype=np.int32)
    successes = np.zeros(num_episodes, dtype=np.float32)
    final_distances = np.zeros(num_episodes, dtype=np.float32)
    active = np.ones(num_episodes, dtype=bool)
    envs = []
    observations = []

    try:
        for episode_idx in range(num_episodes):
            np.random.seed(base_seed + episode_idx)
            env = BallGNNEnvironment(
                target_pos=tuple(agent.target_pos.tolist()),
                max_steps=max_steps,
                reset_span=reset_span,
                reach_threshold=reach_threshold,
            )
            obs = env.reset(reset_span=reset_span)
            envs.append(env)
            observations.append(obs)
            final_distances[episode_idx] = float(np.linalg.norm(obs['ball_pos'] - obs['target_pos']))

        for _ in range(max_steps):
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break

            flat_obs_batch = np.stack([flatten_obs(observations[idx]) for idx in active_indices], axis=0)
            actions = agent.select_action_batch(flat_obs_batch, eval_mode=True)

            for batch_idx, episode_idx in enumerate(active_indices):
                next_obs, reward, done, info = envs[episode_idx].step(actions[batch_idx])
                returns[episode_idx] += float(reward)
                lengths[episode_idx] += 1
                observations[episode_idx] = next_obs
                final_distances[episode_idx] = float(info.get('distance', 0.0))
                if info.get('success', False):
                    successes[episode_idx] = 1.0
                if done:
                    active[episode_idx] = False
    finally:
        np.random.set_state(rng_state)

    return {
        'success_rate': float(np.mean(successes)) if num_episodes > 0 else 0.0,
        'avg_return': float(np.mean(returns)) if num_episodes > 0 else 0.0,
        'avg_length': float(np.mean(lengths)) if num_episodes > 0 else 0.0,
        'avg_final_distance': float(np.mean(final_distances)) if num_episodes > 0 else 0.0,
    }


def sample_tanh_gaussian(mu, log_std, rng):
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mu.shape)
    pre_tanh = mu + std * noise
    action = jnp.tanh(pre_tanh)
    gaussian_log_prob = -0.5 * (((pre_tanh - mu) / (std + 1e-6)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    gaussian_log_prob = jnp.sum(gaussian_log_prob, axis=-1)
    squash_correction = jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
    log_prob = gaussian_log_prob - squash_correction
    return action, log_prob


def collect_normalization_stats(env, act_dim, num_steps):
    obs_samples = []
    rew_samples = []
    obs = env.reset()
    for _ in range(num_steps):
        action = np.random.uniform(-1, 1, size=act_dim)
        next_obs, reward, done, _ = env.step(action)
        obs_samples.append(np.array(list(obs['ball_pos']) + list(obs['ball_vel']), dtype=np.float32))
        rew_samples.append(reward)
        obs = next_obs if not done else env.reset()
    obs_samples = np.asarray(obs_samples, dtype=np.float32)
    rew_samples = np.asarray(rew_samples, dtype=np.float32)
    return (
        obs_samples.mean(axis=0),
        obs_samples.std(axis=0) + 1e-8,
        float(rew_samples.mean()),
        float(rew_samples.std() + 1e-8),
    )


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
            if isinstance(scalar, (bytearray, memoryview)):
                return bytes(scalar)
            if isinstance(scalar, str):
                return scalar.encode('utf-8')
            if isinstance(scalar, np.void):
                return bytes(scalar)
        return value.tobytes()
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode('utf-8')
    return bytes(value)


def load_checkpoint_metadata(path):
    if not os.path.exists(path):
        return {
            'training_step': 0,
            'best_success_rate': -np.inf,
            'best_avg_return': -np.inf,
        }
    data = np.load(path, allow_pickle=True)
    return {
        'training_step': int(data['training_step']) if 'training_step' in data.files else 0,
        'best_success_rate': float(data['best_success_rate']) if 'best_success_rate' in data.files else -np.inf,
        'best_avg_return': float(data['best_avg_return']) if 'best_avg_return' in data.files else -np.inf,
    }


def build_graph_from_components(nodes, edges):
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=GRAPH_SENDERS,
        receivers=GRAPH_RECEIVERS,
        globals=GRAPH_GLOBALS,
        n_node=GRAPH_N_NODE,
        n_edge=GRAPH_N_EDGE,
    )


def flat_obs_batch_to_graph_components(flat_obs_batch, obs_mean, obs_std, target_pos):
    flat_obs_batch = jnp.asarray(flat_obs_batch, dtype=jnp.float32)
    obs_mean = jnp.asarray(obs_mean, dtype=jnp.float32)
    obs_std = jnp.asarray(obs_std, dtype=jnp.float32)
    if flat_obs_batch.shape[-1] >= 6:
        goal = flat_obs_batch[:, 4:6]
    else:
        goal = jnp.broadcast_to(jnp.asarray(target_pos, dtype=jnp.float32), (flat_obs_batch.shape[0], 2))

    ball_pos = (flat_obs_batch[:, :2] - obs_mean[:2]) / (obs_std[:2] + 1e-8)
    ball_vel = (flat_obs_batch[:, 2:4] - obs_mean[2:4]) / (obs_std[2:4] + 1e-8)
    relative_goal = (goal - flat_obs_batch[:, :2]) / (obs_std[:2] + 1e-8)

    ball_node = jnp.concatenate([relative_goal, ball_vel], axis=-1)
    target_node = jnp.zeros_like(ball_node)
    nodes = jnp.stack([ball_node, target_node], axis=1)

    rel = relative_goal
    dist = jnp.linalg.norm(rel, axis=-1, keepdims=True)
    edge_forward = jnp.concatenate([rel, dist], axis=-1)
    edge_backward = jnp.concatenate([-rel, dist], axis=-1)
    edges = jnp.stack([edge_forward, edge_backward], axis=1)
    return nodes, edges

# --- ReplayBuffer ---
class ReplayBuffer:
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0
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
        return dict(
            obs=self.obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )

# --- GNN建图辅助 ---
def obs_to_graph(obs):
    relative_goal = np.asarray(obs['target_pos'], dtype=np.float32) - np.asarray(obs['ball_pos'], dtype=np.float32)
    node_list = []
    node_list.append(np.concatenate([relative_goal, np.asarray(obs['ball_vel'], dtype=np.float32)]))
    node_list.append(np.zeros(4, dtype=np.float32))
    nodes = jnp.array(node_list, dtype=jnp.float32)
    n = len(nodes)
    senders, receivers, edge_features = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                senders.append(i)
                receivers.append(j)
                rel = relative_goal if (i == 0 and j == 1) else -relative_goal
                dist = np.linalg.norm(rel)
                edge_features.append(np.concatenate([rel, [dist]]))
    edges = jnp.array(edge_features, dtype=jnp.float32)
    return build_graph_from_components(nodes, edges)

# --- Flax GNN Policy ---
class FlaxGNNPolicy(nn.Module):
    action_dim: int
    gnn_hidden: int = 32
    mlp_hidden: int = 128
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        # 两层GNN
        def gnn_block():
            return jraph.GraphNetwork(
                update_edge_fn=lambda e, s, r, g: nn.relu(nn.Dense(self.gnn_hidden, kernel_init=nn.initializers.orthogonal())(jnp.concatenate([e, s, r], axis=-1))),
                update_node_fn=lambda n, s, r, g: nn.relu(nn.Dense(self.gnn_hidden, kernel_init=nn.initializers.orthogonal())(jnp.concatenate([n, s, r], axis=-1))),
                update_global_fn=lambda g, n, e: g
            )
        graph = gnn_block()(graph)
        graph = gnn_block()(graph)
        x = graph.nodes[0]
        x = nn.relu(nn.Dense(self.mlp_hidden, kernel_init=nn.initializers.orthogonal())(x))
        x = nn.relu(nn.Dense(self.mlp_hidden, kernel_init=nn.initializers.orthogonal())(x))
        mu = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal())(x)
        log_std = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros)(x)
        log_std = jnp.clip(log_std, -5, 2)
        return mu, log_std

# --- Flax GNN QNet ---
class FlaxGNNQNet(nn.Module):
    gnn_hidden: int = 32
    mlp_hidden: int = 128
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, action):
        def gnn_block():
            return jraph.GraphNetwork(
                update_edge_fn=lambda e, s, r, g: nn.relu(nn.Dense(self.gnn_hidden, kernel_init=nn.initializers.orthogonal())(jnp.concatenate([e, s, r], axis=-1))),
                update_node_fn=lambda n, s, r, g: nn.relu(nn.Dense(self.gnn_hidden, kernel_init=nn.initializers.orthogonal())(jnp.concatenate([n, s, r], axis=-1))),
                update_global_fn=lambda g, n, e: g
            )
        graph = gnn_block()(graph)
        graph = gnn_block()(graph)
        x = graph.nodes[0]
        x = jnp.concatenate([x, action], axis=-1)
        x = nn.relu(nn.Dense(self.mlp_hidden, kernel_init=nn.initializers.orthogonal())(x))
        x = nn.relu(nn.Dense(self.mlp_hidden, kernel_init=nn.initializers.orthogonal())(x))
        q = nn.Dense(1, kernel_init=nn.initializers.orthogonal())(x)
        return q.squeeze(-1)

# --- SAC Agent ---
class SACAgent:
    def __init__(self, obs_dim, act_dim, seed=0, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.01, target_entropy=None, gnn_hidden=32, mlp_hidden=128, obs_mean=None, obs_std=None, rew_mean=None, rew_std=None, reward_scale=1.0):
        self.last_losses = {}
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gnn_hidden = gnn_hidden
        self.mlp_hidden = mlp_hidden
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.fixed_alpha = float(alpha)
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim
        self.log_alpha = jnp.array(np.log(alpha), dtype=jnp.float32)
        self.alpha_tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
        self.alpha_state = train_state.TrainState.create(apply_fn=lambda x: x, params={'log_alpha': self.log_alpha}, tx=self.alpha_tx)
        self.rng = jax.random.PRNGKey(seed)
        self.rng, policy_init_key, q1_init_key, q2_init_key = jax.random.split(self.rng, 4)
        # Normalization
        self.obs_mean = np.asarray(obs_mean if obs_mean is not None else np.zeros(obs_dim), dtype=np.float32)
        self.obs_std = np.asarray(obs_std if obs_std is not None else np.ones(obs_dim), dtype=np.float32)
        self.obs_mean_jnp = jnp.asarray(self.obs_mean, dtype=jnp.float32)
        self.obs_std_jnp = jnp.asarray(self.obs_std, dtype=jnp.float32)
        self.rew_mean = rew_mean if rew_mean is not None else 0.0
        self.rew_std = rew_std if rew_std is not None else 1.0
        self.target_pos = np.array([3.0, 3.0], dtype=np.float32)
        self.target_pos_jnp = jnp.asarray(self.target_pos, dtype=jnp.float32)
        # 初始化policy
        self.policy = FlaxGNNPolicy(action_dim=act_dim, gnn_hidden=gnn_hidden, mlp_hidden=mlp_hidden)
        dummy_graph = obs_to_graph({'ball_pos':np.zeros(2), 'ball_vel':np.zeros(2), 'target_pos':np.zeros(2)})
        self.policy_params = self.policy.init(policy_init_key, dummy_graph)
        self.policy_tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
        self.policy_state = train_state.TrainState.create(apply_fn=self.policy.apply, params=self.policy_params, tx=self.policy_tx)
        # 初始化Q网络
        self.q1 = FlaxGNNQNet(gnn_hidden=gnn_hidden, mlp_hidden=mlp_hidden)
        self.q2 = FlaxGNNQNet(gnn_hidden=gnn_hidden, mlp_hidden=mlp_hidden)
        dummy_action = jnp.zeros(act_dim)
        self.q1_params = self.q1.init(q1_init_key, dummy_graph, dummy_action)
        self.q2_params = self.q2.init(q2_init_key, dummy_graph, dummy_action)
        self.q1_tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
        self.q2_tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
        self.q1_state = train_state.TrainState.create(apply_fn=self.q1.apply, params=self.q1_params, tx=self.q1_tx)
        self.q2_state = train_state.TrainState.create(apply_fn=self.q2.apply, params=self.q2_params, tx=self.q2_tx)
        # target Q网络
        self.q1_target_params = self.q1_params
        self.q2_target_params = self.q2_params
        self._policy_apply_single_jit = jax.jit(self._policy_apply_single)
        self._policy_apply_batch_jit = jax.jit(self._policy_apply_batch)
        self._select_action_eval_batch_jit = jax.jit(self._select_action_eval_batch)
        self._select_action_sample_batch_jit = jax.jit(self._select_action_sample_batch)
        self._q1_apply_batch_jit = jax.jit(self._q1_apply_batch)
        self._q2_apply_batch_jit = jax.jit(self._q2_apply_batch)
        self._update_step_jit = jax.jit(self._update_step)

    def soft_update(self, params, target_params, tau):
        return jax.tree_util.tree_map(lambda p, tp: tau * p + (1 - tau) * tp, params, target_params)

    def _policy_apply_single(self, policy_params, nodes, edges):
        graph = build_graph_from_components(nodes, edges)
        return self.policy.apply(policy_params, graph)

    def _policy_apply_batch(self, policy_params, nodes_batch, edges_batch):
        return jax.vmap(
            lambda nodes, edges: self._policy_apply_single(policy_params, nodes, edges),
            in_axes=(0, 0),
        )(nodes_batch, edges_batch)

    def _q1_apply_batch(self, q_params, nodes_batch, edges_batch, actions):
        return jax.vmap(
            lambda nodes, edges, action: self.q1.apply(q_params, build_graph_from_components(nodes, edges), action),
            in_axes=(0, 0, 0),
        )(nodes_batch, edges_batch, actions)

    def _q2_apply_batch(self, q_params, nodes_batch, edges_batch, actions):
        return jax.vmap(
            lambda nodes, edges, action: self.q2.apply(q_params, build_graph_from_components(nodes, edges), action),
            in_axes=(0, 0, 0),
        )(nodes_batch, edges_batch, actions)

    def _select_action_eval_batch(self, policy_params, flat_obs_batch):
        nodes_batch, edges_batch = flat_obs_batch_to_graph_components(flat_obs_batch, self.obs_mean_jnp, self.obs_std_jnp, self.target_pos_jnp)
        mu, _ = self._policy_apply_batch(policy_params, nodes_batch, edges_batch)
        return jnp.tanh(mu)

    def _select_action_sample_batch(self, policy_params, flat_obs_batch, rngs):
        nodes_batch, edges_batch = flat_obs_batch_to_graph_components(flat_obs_batch, self.obs_mean_jnp, self.obs_std_jnp, self.target_pos_jnp)
        mu, log_std = self._policy_apply_batch(policy_params, nodes_batch, edges_batch)
        action, _ = jax.vmap(sample_tanh_gaussian)(mu, log_std, rngs)
        return action

    def _update_step(self, policy_state, q1_state, q2_state, q1_target_params, q2_target_params, batch, rng):
        obs = jnp.asarray(batch['obs'], dtype=jnp.float32)
        act = jnp.asarray(batch['act'], dtype=jnp.float32)
        rew = self.reward_scale * (jnp.asarray(batch['rew'], dtype=jnp.float32) - self.rew_mean) / (self.rew_std + 1e-8)
        next_obs = jnp.asarray(batch['next_obs'], dtype=jnp.float32)
        done = jnp.asarray(batch['done'], dtype=jnp.float32)

        obs_nodes, obs_edges = flat_obs_batch_to_graph_components(obs, self.obs_mean_jnp, self.obs_std_jnp, self.target_pos_jnp)
        next_nodes, next_edges = flat_obs_batch_to_graph_components(next_obs, self.obs_mean_jnp, self.obs_std_jnp, self.target_pos_jnp)

        rng, target_key, policy_key = jax.random.split(rng, 3)
        target_rngs = jax.random.split(target_key, obs.shape[0])
        policy_rngs = jax.random.split(policy_key, obs.shape[0])
        alpha = jnp.asarray(self.fixed_alpha, dtype=jnp.float32)

        next_mu, next_log_std = self._policy_apply_batch_jit(policy_state.params, next_nodes, next_edges)
        next_action, next_log_prob = jax.vmap(sample_tanh_gaussian)(next_mu, next_log_std, target_rngs)
        q1_next = self._q1_apply_batch_jit(q1_target_params, next_nodes, next_edges, next_action)
        q2_next = self._q2_apply_batch_jit(q2_target_params, next_nodes, next_edges, next_action)
        min_q_next = jnp.minimum(q1_next, q2_next)
        target = rew + self.gamma * (1.0 - done) * (min_q_next - alpha * next_log_prob)
        target = jnp.clip(target, -10.0, 10.0)

        def q1_loss_fn(q_params):
            q_pred = self._q1_apply_batch_jit(q_params, obs_nodes, obs_edges, act)
            return jnp.mean((q_pred - target) ** 2)

        def q2_loss_fn(q_params):
            q_pred = self._q2_apply_batch_jit(q_params, obs_nodes, obs_edges, act)
            return jnp.mean((q_pred - target) ** 2)

        q1_loss, q1_grads = jax.value_and_grad(q1_loss_fn)(q1_state.params)
        q2_loss, q2_grads = jax.value_and_grad(q2_loss_fn)(q2_state.params)
        q1_state = q1_state.apply_gradients(grads=q1_grads)
        q2_state = q2_state.apply_gradients(grads=q2_grads)

        def policy_loss_fn(policy_params):
            mu, log_std = self._policy_apply_batch_jit(policy_params, obs_nodes, obs_edges)
            sampled_action, log_prob = jax.vmap(sample_tanh_gaussian)(mu, log_std, policy_rngs)
            q1_pi = self._q1_apply_batch_jit(q1_state.params, obs_nodes, obs_edges, sampled_action)
            q2_pi = self._q2_apply_batch_jit(q2_state.params, obs_nodes, obs_edges, sampled_action)
            min_q_pi = jnp.minimum(q1_pi, q2_pi)
            return jnp.mean(alpha * log_prob - min_q_pi)

        policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_state.params)
        policy_state = policy_state.apply_gradients(grads=policy_grads)

        q1_target_params = self.soft_update(q1_state.params, q1_target_params, self.tau)
        q2_target_params = self.soft_update(q2_state.params, q2_target_params, self.tau)

        metrics = {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'policy_loss': policy_loss,
            'alpha_loss': jnp.array(0.0, dtype=jnp.float32),
            'alpha': alpha,
            'target_mean': jnp.mean(target),
            'target_std': jnp.std(target),
            'target_min': jnp.min(target),
            'target_max': jnp.max(target),
        }
        return policy_state, q1_state, q2_state, q1_target_params, q2_target_params, metrics, rng

    def preprocess_obs(self, obs):
        return normalize_graph_obs(obs, self.obs_mean, self.obs_std)

    def preprocess_flat_obs(self, flat_obs):
        return self.preprocess_obs(flat_obs_to_dict(flat_obs, self.target_pos))

    def set_target_pos(self, target_pos):
        self.target_pos = np.asarray(target_pos, dtype=np.float32)
        self.target_pos_jnp = jnp.asarray(self.target_pos, dtype=jnp.float32)

    def select_action_flat(self, flat_obs, eval_mode=False):
        flat_obs = np.asarray(flat_obs, dtype=np.float32)
        flat_batch = flat_obs[None, :]
        if eval_mode:
            actions = self._select_action_eval_batch_jit(self.policy_state.params, flat_batch)
        else:
            self.rng, subkey = jax.random.split(self.rng)
            rngs = jax.random.split(subkey, 1)
            actions = self._select_action_sample_batch_jit(self.policy_state.params, flat_batch, rngs)
        return np.asarray(actions[0], dtype=np.float32)

    def select_action_batch(self, flat_obs_batch, eval_mode=False):
        flat_obs_batch = np.asarray(flat_obs_batch, dtype=np.float32)
        if eval_mode:
            actions = self._select_action_eval_batch_jit(self.policy_state.params, flat_obs_batch)
        else:
            self.rng, subkey = jax.random.split(self.rng)
            rngs = jax.random.split(subkey, flat_obs_batch.shape[0])
            actions = self._select_action_sample_batch_jit(self.policy_state.params, flat_obs_batch, rngs)
        return np.asarray(actions, dtype=np.float32)

    def select_action(self, obs, eval_mode=False):
        return self.select_action_flat(flatten_obs(obs), eval_mode=eval_mode)

    def update(self, batch):
        self.policy_state, self.q1_state, self.q2_state, self.q1_target_params, self.q2_target_params, metrics, self.rng = self._update_step_jit(
            self.policy_state,
            self.q1_state,
            self.q2_state,
            self.q1_target_params,
            self.q2_target_params,
            batch,
            self.rng,
        )
        metrics = jax.device_get(metrics)
        self.last_losses = {key: float(value) for key, value in metrics.items()}

    def save(self, path, training_step=0, best_success_rate=-np.inf, best_avg_return=-np.inf):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
            policy=flax.serialization.to_bytes(self.policy_state.params),
            q1=flax.serialization.to_bytes(self.q1_state.params),
            q2=flax.serialization.to_bytes(self.q2_state.params),
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            rew_mean=np.array(self.rew_mean, dtype=np.float32),
            rew_std=np.array(self.rew_std, dtype=np.float32),
            reward_scale=np.array(self.reward_scale, dtype=np.float32),
            target_pos=self.target_pos.astype(np.float32),
            obs_dim=np.array(self.obs_dim, dtype=np.int32),
            act_dim=np.array(self.act_dim, dtype=np.int32),
            gnn_hidden=np.array(self.gnn_hidden, dtype=np.int32),
            mlp_hidden=np.array(self.mlp_hidden, dtype=np.int32),
            training_step=np.array(training_step, dtype=np.int32),
            best_success_rate=np.array(best_success_rate, dtype=np.float32),
            best_avg_return=np.array(best_avg_return, dtype=np.float32),
        )
        
    def load(self, path):
        data = np.load(path, allow_pickle=True)
        if 'obs_mean' in data.files:
            self.obs_mean = data['obs_mean'].astype(np.float32)
            self.obs_mean_jnp = jnp.asarray(self.obs_mean, dtype=jnp.float32)
        if 'obs_std' in data.files:
            self.obs_std = data['obs_std'].astype(np.float32)
            self.obs_std_jnp = jnp.asarray(self.obs_std, dtype=jnp.float32)
        if 'rew_mean' in data.files:
            self.rew_mean = float(data['rew_mean'])
        if 'rew_std' in data.files:
            self.rew_std = float(data['rew_std'])
        if 'reward_scale' in data.files:
            self.reward_scale = float(data['reward_scale'])
        if 'target_pos' in data.files:
            self.target_pos = data['target_pos'].astype(np.float32)
            self.target_pos_jnp = jnp.asarray(self.target_pos, dtype=jnp.float32)
        policy_bytes = npz_value_to_bytes(data['policy'])
        q1_bytes = npz_value_to_bytes(data['q1'])
        q2_bytes = npz_value_to_bytes(data['q2'])
        self.policy_state = self.policy_state.replace(params=flax.serialization.from_bytes(self.policy_state.params, policy_bytes))
        self.q1_state = self.q1_state.replace(params=flax.serialization.from_bytes(self.q1_state.params, q1_bytes))
        self.q2_state = self.q2_state.replace(params=flax.serialization.from_bytes(self.q2_state.params, q2_bytes))


def load_agent_from_checkpoint(path):
    data = np.load(path, allow_pickle=True)
    obs_dim = int(data['obs_dim']) if 'obs_dim' in data.files else 4
    act_dim = int(data['act_dim']) if 'act_dim' in data.files else 2
    gnn_hidden = int(data['gnn_hidden']) if 'gnn_hidden' in data.files else 32
    mlp_hidden = int(data['mlp_hidden']) if 'mlp_hidden' in data.files else 128
    obs_mean = data['obs_mean'].astype(np.float32) if 'obs_mean' in data.files else None
    obs_std = data['obs_std'].astype(np.float32) if 'obs_std' in data.files else None
    rew_mean = float(data['rew_mean']) if 'rew_mean' in data.files else None
    rew_std = float(data['rew_std']) if 'rew_std' in data.files else None
    reward_scale = float(data['reward_scale']) if 'reward_scale' in data.files else 1.0
    target_pos = data['target_pos'].astype(np.float32) if 'target_pos' in data.files else np.array([3.0, 3.0], dtype=np.float32)
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        gnn_hidden=gnn_hidden,
        mlp_hidden=mlp_hidden,
        obs_mean=obs_mean,
        obs_std=obs_std,
        rew_mean=rew_mean,
        rew_std=rew_std,
        reward_scale=reward_scale,
    )
    agent.set_target_pos(target_pos)
    agent.load(path)
    return agent

# --- 训练主流程 ---
def train(
    *,
    additional_steps=20000,
    batch_size=64,
    update_after=500,
    resume_path=None,
    save_interval=500,
    eval_interval=1000,
    log_interval=100,
    eval_episodes=10,
):
    env = BallGNNEnvironment()
    obs_dim = 6  # ball_pos(2)+ball_vel(2)+target_pos(2)
    act_dim = 2

    resume_step = 0
    best_success_rate = -np.inf
    best_avg_return = -np.inf
    if resume_path:
        metadata = load_checkpoint_metadata(resume_path)
        resume_step = metadata['training_step']
        best_success_rate = metadata['best_success_rate']
        best_avg_return = metadata['best_avg_return']
        agent = load_agent_from_checkpoint(resume_path)
        env = BallGNNEnvironment(target_pos=tuple(agent.target_pos.tolist()))
        print(f"[RESUME] loaded {resume_path} at step={resume_step}")
    else:
        obs_mean, obs_std, rew_mean, rew_std = collect_normalization_stats(env, act_dim, num_steps=1000)
        agent = SACAgent(obs_dim, act_dim, obs_mean=obs_mean, obs_std=obs_std, rew_mean=rew_mean, rew_std=rew_std)
        agent.set_target_pos(env.target_pos.copy())

    buffer = ReplayBuffer(max_size=100000, obs_dim=obs_dim, act_dim=act_dim)
    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0
    completed_returns = []
    completed_lengths = []
    completed_successes = []
    recent_distances = []
    episode_transitions = []
    total_her_added = 0
    last_eval = None
    her_max_samples = 64
    for local_step in range(1, additional_steps + 1):
        step = resume_step + local_step
        flat_obs = flatten_obs(obs)
        action = agent.select_action_flat(flat_obs)
        next_obs, reward, done, info = env.step(action)
        next_flat_obs = flatten_obs(next_obs)
        ep_reward += reward
        ep_length += 1
        recent_distances.append(float(info.get('distance', 0.0)))
        buffer.add(flat_obs, action, reward, next_flat_obs, float(done))
        episode_transitions.append({
            'obs': flat_obs,
            'act': action,
            'reward': reward,
            'next_obs': next_flat_obs,
            'done': float(done),
        })
        obs = next_obs if not done else env.reset()
        if done:
            completed_returns.append(ep_reward)
            completed_lengths.append(ep_length)
            completed_successes.append(float(info.get('success', False)))
            if not info.get('success', False):
                total_her_added += add_hindsight_replay(
                    buffer,
                    episode_transitions,
                    reach_threshold=env.reach_threshold,
                    max_her_samples=her_max_samples,
                )
            episode_transitions = []
            ep_reward = 0.0
            ep_length = 0
        if local_step > update_after:
            batch = buffer.sample(batch_size)
            agent.update(batch)

        if eval_interval > 0 and step % eval_interval == 0 and local_step > update_after:
            last_eval = evaluate_agent(
                agent,
                num_episodes=eval_episodes,
                max_steps=env.max_steps,
                base_seed=1234,
                reset_span=env.reset_span,
                reach_threshold=env.reach_threshold,
            )
            print(
                f"[EVAL] step={step} eval_success_rate={last_eval['success_rate']:.2f} "
                f"eval_avg_return={last_eval['avg_return']:.4f} eval_avg_len={last_eval['avg_length']:.1f} "
                f"eval_final_dist={last_eval['avg_final_distance']:.4f}"
            )

        if step % log_interval == 0 and local_step > update_after:
            losses = agent.last_losses
            last_return = completed_returns[-1] if completed_returns else ep_reward
            recent_count = min(10, len(completed_returns))
            recent_success_rate = float(np.mean(completed_successes[-recent_count:])) if recent_count > 0 else 0.0
            recent_avg_length = float(np.mean(completed_lengths[-recent_count:])) if recent_count > 0 else float(ep_length)
            recent_avg_return = float(np.mean(completed_returns[-recent_count:])) if recent_count > 0 else float(ep_reward)
            recent_distance_mean = float(np.mean(recent_distances[-100:])) if recent_distances else 0.0
            current_distance = recent_distances[-1] if recent_distances else 0.0
            selection_success_rate = last_eval['success_rate'] if last_eval is not None else recent_success_rate
            selection_avg_return = last_eval['avg_return'] if last_eval is not None else recent_avg_return
            if selection_success_rate > best_success_rate or (selection_success_rate == best_success_rate and selection_avg_return > best_avg_return):
                best_success_rate = selection_success_rate
                best_avg_return = selection_avg_return
                best_path = os.path.join(os.path.dirname(__file__), "models", "gnn_sac_best.npz")
                agent.save(best_path, training_step=step, best_success_rate=best_success_rate, best_avg_return=best_avg_return)
            print(
                f"step={step} episode_return={last_return:.4f} recent_avg_return={recent_avg_return:.4f} "
                f"completed_eps={len(completed_returns)} success_rate={recent_success_rate:.2f} avg_ep_len={recent_avg_length:.1f} "
                f"distance={current_distance:.4f} recent_dist_mean={recent_distance_mean:.4f} "
                f"her_total={total_her_added} "
                f"q1_loss={losses.get('q1_loss', 0):.4f} q2_loss={losses.get('q2_loss', 0):.4f} "
                f"policy_loss={losses.get('policy_loss', 0):.4f} alpha={losses.get('alpha', 0):.4f} "
                f"alpha_loss={losses.get('alpha_loss', 0):.4f} target_mean={losses.get('target_mean', 0):.4f} "
                f"target_std={losses.get('target_std', 0):.4f}"
            )

        if save_interval > 0 and step % save_interval == 0:
            agent.save(
                os.path.join(os.path.dirname(__file__), "models", "gnn_sac.npz"),
                training_step=step,
                best_success_rate=best_success_rate,
                best_avg_return=best_avg_return,
            )
            print(f"[SAVE] step={step} model saved.")

    agent.save(
        os.path.join(os.path.dirname(__file__), "models", "gnn_sac.npz"),
        training_step=resume_step + additional_steps,
        best_success_rate=best_success_rate,
        best_avg_return=best_avg_return,
    )
    print("训练完成，模型已保存。")

# Optuna自动调参目标函数
def optuna_objective(trial):
    # 搜索空间
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.99)
    tau = trial.suggest_float('tau', 1e-5, 1e-2, log=True)
    gnn_hidden = trial.suggest_categorical('gnn_hidden', [32, 64, 128])
    mlp_hidden = trial.suggest_categorical('mlp_hidden', [64, 128, 256])
    total_steps = 2000
    batch_size = 64
    update_after = 500
    env = BallGNNEnvironment()
    obs_dim = 6
    act_dim = 2
    obs_mean, obs_std, rew_mean, rew_std = collect_normalization_stats(env, act_dim, num_steps=500)

    agent = SACAgent(obs_dim, act_dim, lr=lr, gamma=gamma, tau=tau, gnn_hidden=gnn_hidden, mlp_hidden=mlp_hidden, obs_mean=obs_mean, obs_std=obs_std, rew_mean=rew_mean, rew_std=rew_std)
    agent.set_target_pos(env.target_pos.copy())
    buffer = ReplayBuffer(max_size=10000, obs_dim=obs_dim, act_dim=act_dim)
    obs = env.reset()
    ep_rewards = []
    ep_return = 0.0
    ep_length = 0
    ep_successes = []
    ep_lengths = []
    recent_distances = []
    episode_transitions = []
    print(f"[Optuna Trial {trial.number}] Params: lr={lr:.2e}, gamma={gamma:.4f}, tau={tau:.2e}, gnn_hidden={gnn_hidden}, mlp_hidden={mlp_hidden}")
    for t in range(1, total_steps+1):
        flat_obs = flatten_obs(obs)
        action = agent.select_action_flat(flat_obs)
        next_obs, reward, done, info = env.step(action)
        next_flat_obs = flatten_obs(next_obs)
        ep_return += reward
        ep_length += 1
        recent_distances.append(float(info.get('distance', 0.0)))
        buffer.add(flat_obs, action, reward, next_flat_obs, float(done))
        episode_transitions.append({
            'obs': flat_obs,
            'act': action,
            'reward': reward,
            'next_obs': next_flat_obs,
            'done': float(done),
        })
        obs = next_obs if not done else env.reset()
        if done:
            ep_rewards.append(ep_return)
            ep_successes.append(float(info.get('success', False)))
            ep_lengths.append(ep_length)
            if not info.get('success', False):
                add_hindsight_replay(
                    buffer,
                    episode_transitions,
                    reach_threshold=env.reach_threshold,
                    max_her_samples=32,
                )
            episode_transitions = []
            ep_return = 0.0
            ep_length = 0
        if t > update_after:
            batch = buffer.sample(batch_size)
            agent.update(batch)
        if t % 100 == 0 and t > update_after:
            losses = agent.last_losses
            recent_count = min(10, len(ep_rewards))
            recent_success_rate = float(np.mean(ep_successes[-recent_count:])) if recent_count > 0 else 0.0
            recent_avg_length = float(np.mean(ep_lengths[-recent_count:])) if recent_count > 0 else float(ep_length)
            recent_avg_return = float(np.mean(ep_rewards[-recent_count:])) if recent_count > 0 else float(ep_return)
            recent_distance_mean = float(np.mean(recent_distances[-100:])) if recent_distances else 0.0
            current_distance = recent_distances[-1] if recent_distances else 0.0
            print(
                f"[Trial {trial.number}] step={t} episode_return={ep_rewards[-1] if ep_rewards else ep_return:.4f} "
                f"recent_avg_return={recent_avg_return:.4f} completed_eps={len(ep_rewards)} success_rate={recent_success_rate:.2f} avg_ep_len={recent_avg_length:.1f} "
                f"distance={current_distance:.4f} recent_dist_mean={recent_distance_mean:.4f} "
                f"q1_loss={losses.get('q1_loss', 0):.4f} q2_loss={losses.get('q2_loss', 0):.4f} "
                f"policy_loss={losses.get('policy_loss', 0):.4f} alpha={losses.get('alpha', 0):.4f} "
                f"alpha_loss={losses.get('alpha_loss', 0):.4f} target_mean={losses.get('target_mean', 0):.4f} "
                f"target_std={losses.get('target_std', 0):.4f}"
            )
    if len(ep_rewards) == 0:
        print(f"[Trial {trial.number}] No episode finished, return -999.")
        return -999
    avg_reward = np.mean(ep_rewards[-100:])
    print(f"[Trial {trial.number}] Final avg_reward={avg_reward:.4f}")
    return avg_reward

def run_optuna():
    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=20)
    print('Best trial:', study.best_trial.params)


def main():
    parser = argparse.ArgumentParser(description="Batched JAX GNN-SAC training for ball reaching")
    parser.add_argument('--additional_steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--update_after', type=int, default=500)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--run_optuna', action='store_true')
    args = parser.parse_args()

    if args.run_optuna:
        run_optuna()
        return

    train(
        additional_steps=args.additional_steps,
        batch_size=args.batch_size,
        update_after=args.update_after,
        resume_path=args.resume_path,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
    )

if __name__ == "__main__":
    main()
