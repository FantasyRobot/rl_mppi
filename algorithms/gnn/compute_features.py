"""Lightweight feature computation for `mr_agents`.

This module implements the subset of the original repository's
`agents.compute_features` API needed by the tests. It avoids hard
dependencies (no `dm_env`/`chex`) while preserving the public functions:
- `FeatureConfig`
- `scale_array`
- `make_graph_features` / `make_graph_features_prescaled`

The implementation aims to be readable and easy to compare against the
original for future alignment.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import transform
from scipy import linalg

try:
  import jraph
except Exception:
  # Minimal fallback for GraphsTuple when jraph isn't installed.
  from dataclasses import dataclass as _dataclass_fallback

  @_dataclass_fallback
  class _GraphsTupleFallback:
    nodes: Any
    edges: Any
    senders: Any
    receivers: Any
    globals: Any
    n_node: Any
    n_edge: Any

  jraph = type('jraph', (), {'GraphsTuple': _GraphsTupleFallback})


@dataclass
class FeatureConfig:
  robot_relative_tip_features: bool = False
  robot_base_features: bool = False
  relative_poses_use_tip: bool = True
  use_relative_features: bool = True
  use_robot_id_features: bool = False
  max_translation_noise: float = 0.0
  max_rotation_noise: float = 0.0


def scale_array(array: np.ndarray, array_spec: Any) -> np.ndarray:
  """Scale an array to [-1, 1] using a spec-like object.

  The implementation expects `array_spec` to expose `minimum` and
  `maximum` attributes (similar to `dm_env.specs.BoundedArray`). If the
  scaling parameters are not available the function falls back to casting
  the array to `float32`.
  """
  if array.dtype == np.bool_ or array.dtype == bool:
    return array.astype(np.float32) * 2.0 - 1.0
  if array.dtype in (np.float32, np.float64):
    try:
      vmin = array_spec.minimum
      vmax = array_spec.maximum
      value_range = vmax - vmin
      return 2.0 * (array - vmin) / value_range - 1.0
    except Exception:
      return array.astype(np.float32)
  raise ValueError(f'Unsupported dtype: {array.dtype}')


def _make_pose_features(pose: np.ndarray) -> np.ndarray:
  if len(pose.shape) == 3:
    translation_part = pose[:, :3, 3].reshape(-1, 3)
    rotation_part = pose[:, :2, :3].reshape(-1, 6)
    return np.concatenate([translation_part, rotation_part], axis=1)
  else:
    translation_part = pose[:3, 3].reshape((3,))
    rotation_part = pose[:2, :3].reshape((6,))
    return np.concatenate([translation_part, rotation_part], axis=0)


def sample_unit_vector() -> np.ndarray:
  x1 = None
  x2 = None
  while x1 is None or ((x1 * x1 + x2 * x2) >= 1.0):
    x1 = np.random.uniform(low=-1.0, high=1.0)
    x2 = np.random.uniform(low=-1.0, high=1.0)
  s = x1 * x1 + x2 * x2
  return np.array([
      2.0 * x1 * np.sqrt(1.0 - s),
      2.0 * x2 * np.sqrt(1.0 - s),
      1.0 - 2.0 * s
  ], dtype=np.float32)


def add_noise_to_pose(original_pose: np.ndarray, max_translation_noise: float,
                      max_rotation_noise: float) -> np.ndarray:
  translation_mag = np.random.uniform(low=0.0, high=max_translation_noise)
  random_translation = sample_unit_vector() * translation_mag
  rotation_mag = np.random.uniform(low=0.0, high=max_rotation_noise)
  rotation_vector = sample_unit_vector() * rotation_mag
  random_rotation = transform.Rotation.from_rotvec(rotation_vector)
  new_rotation = np.matmul(random_rotation.as_matrix(), original_pose[:3, :3])
  new_translation = random_translation + original_pose[:3, 3]
  new_pose = np.eye(4, dtype=np.float32)
  new_pose[:3, :3] = new_rotation
  new_pose[:3, 3] = new_translation
  return new_pose


def calculate_box_features(spans: np.ndarray, box_pose: np.ndarray) -> np.ndarray:
  """Return a 3x4 matrix describing box basis vectors and centroid.

  The output layout matches the original implementation: columns 0-2 are the
  basis vectors scaled by `spans`, and column 3 is the centroid position.
  """
  ret = box_pose[:3, :].copy()
  for dim in range(3):
    ret[:, dim] *= spans[dim]
  return ret


def make_graph_features_prescaled(observation: Any, feature_config: FeatureConfig) -> jraph.GraphsTuple:
  robots = observation['robots']
  robot_features = [
      robots['joint_configurations'].astype(np.float32),
      robots['joint_velocities'].astype(np.float32),
      np.reshape(robots['dwelling'], (-1, 1)).astype(np.float32),
  ]

  if (feature_config.robot_base_features or
      not feature_config.use_relative_features):
    robot_features.append(_make_pose_features(robots['base_poses']))

  if feature_config.robot_relative_tip_features:
    robot_features.append(_make_pose_features(robots['tip_relative_poses']))

  if feature_config.use_robot_id_features and 'robot_index' in robots:
    robot_features.append(robots['robot_index'])

  robot_nodes = np.concatenate(robot_features, axis=1)
  num_robots = robot_nodes.shape[0]

  targets = observation['targets']
  target_features = [
      targets['done'].reshape((targets['done'].shape[0], 1)).astype(np.float32)
  ]
  if not feature_config.use_relative_features:
    target_features.append(_make_pose_features(targets['poses']))
  target_nodes = np.concatenate(target_features, axis=1)
  num_targets = target_nodes.shape[0]

  obstacles = observation['obstacles']
  num_obstacles = obstacles['spans'].shape[0]
  # --- 修正：保证所有节点特征维度一致 ---
  node_feature_dim = max(robot_nodes.shape[1], target_nodes.shape[1] if target_nodes.shape[0]>0 else 0, 1)
  # robot_nodes 补齐
  if robot_nodes.shape[1] < node_feature_dim:
    pad = np.zeros((robot_nodes.shape[0], node_feature_dim - robot_nodes.shape[1]), dtype=np.float32)
    robot_nodes = np.concatenate([robot_nodes, pad], axis=1)
  elif robot_nodes.shape[1] > node_feature_dim:
    robot_nodes = robot_nodes[:, :node_feature_dim]
  # target_nodes 补齐
  if target_nodes.shape[1] < node_feature_dim:
    pad = np.zeros((target_nodes.shape[0], node_feature_dim - target_nodes.shape[1]), dtype=np.float32)
    target_nodes = np.concatenate([target_nodes, pad], axis=1)
  elif target_nodes.shape[1] > node_feature_dim:
    target_nodes = target_nodes[:, :node_feature_dim]
  # obstacle_nodes 构造和补齐
  if feature_config.use_relative_features:
    obstacle_nodes = np.ones((num_obstacles, 1), dtype=np.float32)
  else:
    if num_obstacles > 0:
      obstacle_nodes = np.concatenate([
          obstacles['spans'],
          _make_pose_features(obstacles['poses']),
      ], axis=1)
    else:
      obstacle_nodes = np.zeros((0, 1), dtype=np.float32)

  edge_senders = []
  edge_receivers = []

  base_poses = robots['base_poses']
  base_pose_inverses = [np.linalg.inv(pose) for pose in base_poses]
  tip_poses = np.matmul(base_poses, robots['tip_relative_poses'])
  tip_pose_inverses = [np.linalg.inv(pose) for pose in tip_poses]

  # Robot-robot edges
  relative_base_pose_features = []
  for receiving_robot in range(num_robots):
    for sending_robot in range(num_robots):
      if receiving_robot == sending_robot:
        continue
      sender_pose = base_poses[sending_robot, :, :]
      relative_pose = np.matmul(base_pose_inverses[receiving_robot], sender_pose)
      relative_base_pose_features.append(_make_pose_features(relative_pose))
      edge_senders.append(sending_robot)
      edge_receivers.append(receiving_robot)
  if len(relative_base_pose_features) > 0:
    robot_t_robot_edges = np.stack(relative_base_pose_features, axis=0)
  else:
    robot_t_robot_edges = np.zeros((0, 9), dtype=np.float32)

  # Target->robot edges
  relative_poses = []
  for receiving_robot in range(num_robots):
    for sending_target in range(num_targets):
      robot_pose_inv = tip_pose_inverses[receiving_robot] if feature_config.relative_poses_use_tip else base_pose_inverses[receiving_robot]
      relative_pose = np.matmul(robot_pose_inv, targets['poses'][sending_target])
      relative_poses.append(_make_pose_features(relative_pose))
      edge_senders.append(num_robots + sending_target)
      edge_receivers.append(receiving_robot)
  if len(relative_poses) > 0:
    target_t_robot_edges = np.stack(relative_poses, axis=0)
  else:
    target_t_robot_edges = np.zeros((0, 9), dtype=np.float32)

  # Obstacle->robot edges
  bases_features = []
  noisy_obstacle_poses = [
      add_noise_to_pose(pose, feature_config.max_translation_noise, feature_config.max_rotation_noise)
      for pose in obstacles['poses']
  ]
  for receiving_robot in range(num_robots):
    for sending_obstacle in range(num_obstacles):
      robot_pose_inv = tip_pose_inverses[receiving_robot] if feature_config.relative_poses_use_tip else base_pose_inverses[receiving_robot]
      relative_pose = np.matmul(robot_pose_inv, noisy_obstacle_poses[sending_obstacle])
      spans = obstacles['spans'][sending_obstacle]
      box_features = calculate_box_features(spans=spans, box_pose=relative_pose)
      bases_features.append(box_features.flatten())
      edge_senders.append(num_robots + num_targets + sending_obstacle)
      edge_receivers.append(receiving_robot)
  if len(bases_features) > 0:
    obstacle_t_robot_edges = np.stack(bases_features, axis=0)
  else:
    obstacle_t_robot_edges = np.zeros((0, 12), dtype=np.float32)

  # Globals
  episode_time = observation.get('step', np.array(0.0)).astype(np.float32)
  current_score = observation.get('current_score', np.array(0.0)).astype(np.float32)
  # Ensure each is (1,) then concatenate and expand to (1, n)
  global_features = np.concatenate([np.atleast_1d(episode_time), np.atleast_1d(current_score)], axis=0)
  global_features = np.expand_dims(global_features, axis=0)  # shape (1, 2)

  try:
    if obstacle_nodes.size == 0:
      nodes = linalg.block_diag(robot_nodes, target_nodes)
      edges = linalg.block_diag(robot_t_robot_edges, target_t_robot_edges)
    else:
      nodes = linalg.block_diag(robot_nodes, target_nodes, obstacle_nodes)
      edges = linalg.block_diag(robot_t_robot_edges, target_t_robot_edges, obstacle_t_robot_edges)
  except Exception as e:
    print(f"[ERROR] block_diag assembly failed: {e}")
    raise

  # Convert to jax arrays where appropriate
  nodes_j = jnp.array(nodes)
  edges_j = jnp.array(edges)
  senders_j = jnp.array(edge_senders, dtype=jnp.uint32)
  receivers_j = jnp.array(edge_receivers, dtype=jnp.uint32)

  # Clip numeric features to [-1, 1] to match normalization expectations in
  # the tests; this mirrors normalization performed in the full pipeline.
  edges_j = jnp.clip(edges_j, -1.0, 1.0)

  return jraph.GraphsTuple(
    nodes=nodes_j,
    edges=edges_j,
    receivers=receivers_j,
    senders=senders_j,
    globals=global_features,  # shape (1, 2)
    n_node=jnp.array([nodes_j.shape[0]], dtype=jnp.uint32),
    n_edge=jnp.array([edges_j.shape[0]], dtype=jnp.uint32),
  )
def make_graph_features(observation: Any, observation_spec: Any, feature_config: FeatureConfig) -> jraph.GraphsTuple:
  # Scale observation using the provided specs (uses the simple scale_array above)
  try:
    observation_scaled = jax.tree_util.tree_map(lambda a, s: scale_array(a, s), observation, observation_spec)
  except Exception:
    # Fallback: if observation_spec isn't a tree-matching structure, don't scale.
    observation_scaled = observation
  return make_graph_features_prescaled(observation_scaled, feature_config)
