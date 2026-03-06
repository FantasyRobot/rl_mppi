#!/usr/bin/env python3

"""Graph Neural Network (GNN) encoder for robot navigation with obstacles.

实现原理
--------
本模块使用图神经网络（GNN）将机器人状态和障碍物信息编码为统一的特征表示，
用于后续的策略网络（如 SAC）或价值函数网络的输入。

图结构
------
每个时间步将环境构建为一个异构图：

- **机器人节点**（1 个）：特征为 [x, y, vx, vy, gx - x, gy - y]，即位置、速度和相对目标方向。
- **障碍物节点**（N 个，N 可变）：特征为 [ox, oy, r]，即障碍物圆心坐标和半径。
- **有向边**：从每个障碍物节点指向机器人节点（以及可选的从机器人到障碍物），
  边特征为两节点间的相对位移 [dx, dy] 和欧氏距离 [dist]。

消息传递机制
-----------
每一层 GNNLayer 包含两个步骤：

1. **边更新（Edge Update）**：EdgeModel 利用源节点特征、目标节点特征和边特征计算新的边特征。

   .. math::

       e_{ij}^{(l+1)} = f_e\\bigl(h_i^{(l)},\\, h_j^{(l)},\\, e_{ij}^{(l)}\\bigr)

2. **节点更新（Node Update）**：NodeModel 对每个节点聚合来自其邻居的更新后边特征（求和），
   再与节点自身特征拼接后输出新的节点特征。

   .. math::

       h_i^{(l+1)} = f_n\\Bigl(h_i^{(l)},\\, \\sum_{j \\in \\mathcal{N}(i)} e_{ij}^{(l+1)}\\Bigr)

多层堆叠后，机器人节点的特征已融合了所有障碍物的空间关系信息，
可直接用作策略网络或 Q 网络的输入特征。

与 MLP 的对比优势
-----------------
- **置换不变性**：障碍物节点顺序无关，适合动态变化的障碍物数量。
- **变长输入**：支持任意数量的障碍物，而 MLP 需要固定输入维度。
- **结构归纳偏置**：显式建模机器人与每个障碍物的成对关系，利于泛化。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Small constant for numerical stability in distance computations.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Helper: MLP factory
# ---------------------------------------------------------------------------

def _make_mlp(layer_dims: list[int], act_fn: type = nn.ELU) -> nn.Sequential:
    """Build a fully-connected network with the given layer dimensions.

    Args:
        layer_dims: List of integers specifying the width of each layer,
            including input and output dimensions.  E.g. ``[64, 128, 64]``
            gives one hidden layer of size 128.
        act_fn: Activation function class (default: ELU).

    Returns:
        An ``nn.Sequential`` module.
    """
    layers: list[nn.Module] = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(nn.LayerNorm(layer_dims[i + 1]))
            layers.append(act_fn())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Edge model φ_e
# ---------------------------------------------------------------------------

class EdgeModel(nn.Module):
    """Update edge features using source/target node features and current edge features.

    .. math::

        e_{ij}^{\\prime} = \\phi_e\\bigl([h_i,\\, h_j,\\, e_{ij}]\\bigr)

    Args:
        node_dim: Dimensionality of node feature vectors.
        edge_dim: Dimensionality of edge feature vectors.
        hidden_dims: Hidden layer widths for the edge MLP.
        out_dim: Output dimensionality of updated edge features.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dims: list[int] | None = None,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        in_dim = node_dim + node_dim + edge_dim
        self.mlp = _make_mlp([in_dim] + hidden_dims + [out_dim])

    def forward(self, src: Tensor, dst: Tensor, edge_attr: Tensor) -> Tensor:
        """Compute updated edge features.

        Args:
            src: Source node features, shape ``(E, node_dim)``.
            dst: Destination node features, shape ``(E, node_dim)``.
            edge_attr: Current edge features, shape ``(E, edge_dim)``.

        Returns:
            Updated edge features, shape ``(E, out_dim)``.
        """
        x = torch.cat([src, dst, edge_attr], dim=-1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Node model φ_n
# ---------------------------------------------------------------------------

class NodeModel(nn.Module):
    """Update node features by aggregating incoming edge messages.

    Aggregation is performed by **summing** the incoming updated edge features
    for each node.  The aggregated message is then concatenated with the
    node's own features and passed through an MLP:

    .. math::

        h_i^{\\prime} = \\phi_n\\Bigl(\\bigl[h_i,\\;
            \\sum_{j \\in \\mathcal{N}(i)} e_{ij}^{\\prime}\\bigr]\\Bigr)

    Args:
        node_dim: Dimensionality of current node features.
        edge_out_dim: Dimensionality of updated edge features (output of EdgeModel).
        hidden_dims: Hidden layer widths for the node MLP.
        out_dim: Output dimensionality of updated node features.
    """

    def __init__(
        self,
        node_dim: int,
        edge_out_dim: int,
        hidden_dims: list[int] | None = None,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        in_dim = node_dim + edge_out_dim
        self.mlp = _make_mlp([in_dim] + hidden_dims + [out_dim])

    def forward(
        self,
        x: Tensor,
        aggregated_messages: Tensor,
    ) -> Tensor:
        """Compute updated node features.

        Args:
            x: Current node features, shape ``(V, node_dim)``.
            aggregated_messages: Aggregated edge messages for each node,
                shape ``(V, edge_out_dim)``.

        Returns:
            Updated node features, shape ``(V, out_dim)``.
        """
        h = torch.cat([x, aggregated_messages], dim=-1)
        return self.mlp(h)


# ---------------------------------------------------------------------------
# Single GNN layer
# ---------------------------------------------------------------------------

class GNNLayer(nn.Module):
    """One round of edge-then-node message passing.

    Given a graph with ``V`` nodes, ``E`` directed edges, node features
    ``x ∈ R^{V×d_n}`` and edge features ``e ∈ R^{E×d_e}``, this layer:

    1. Updates each edge feature using :class:`EdgeModel`.
    2. Aggregates (sums) incoming edge messages per destination node.
    3. Updates each node feature using :class:`NodeModel`.

    Args:
        node_dim: Input node feature dimensionality.
        edge_dim: Input edge feature dimensionality.
        edge_hidden: Hidden layer widths for EdgeModel.
        edge_out_dim: Output dimensionality of EdgeModel.
        node_hidden: Hidden layer widths for NodeModel.
        node_out_dim: Output dimensionality (updated node feature size).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_hidden: list[int] | None = None,
        edge_out_dim: int = 64,
        node_hidden: list[int] | None = None,
        node_out_dim: int = 64,
    ) -> None:
        super().__init__()
        if edge_hidden is None:
            edge_hidden = [64, 64]
        if node_hidden is None:
            node_hidden = [64, 64]
        self.edge_model = EdgeModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dims=edge_hidden,
            out_dim=edge_out_dim,
        )
        self.node_model = NodeModel(
            node_dim=node_dim,
            edge_out_dim=edge_out_dim,
            hidden_dims=node_hidden,
            out_dim=node_out_dim,
        )
        self.edge_out_dim = edge_out_dim
        self.node_out_dim = node_out_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run one message-passing round.

        Args:
            x: Node features, shape ``(V, node_dim)``.
            edge_index: Directed edge indices, shape ``(2, E)``.
                Row 0 contains source node indices; row 1 contains destination
                node indices.
            edge_attr: Edge features, shape ``(E, edge_dim)``.

        Returns:
            Tuple ``(x_new, edge_attr_new)`` where:

            - ``x_new``: Updated node features, shape ``(V, node_out_dim)``.
            - ``edge_attr_new``: Updated edge features, shape ``(E, edge_out_dim)``.
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # 1. Edge update.
        src_feats = x[src_idx]
        dst_feats = x[dst_idx]
        edge_attr_new = self.edge_model(src_feats, dst_feats, edge_attr)

        # 2. Aggregate messages to destination nodes (sum).
        num_nodes = x.size(0)
        agg = torch.zeros(num_nodes, self.edge_out_dim, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(edge_attr_new), edge_attr_new)

        # 3. Node update.
        x_new = self.node_model(x, agg)

        return x_new, edge_attr_new


# ---------------------------------------------------------------------------
# Full obstacle encoder
# ---------------------------------------------------------------------------

class ObstacleEncoder(nn.Module):
    """GNN-based encoder that fuses robot state and obstacle information.

    This module takes the robot's current state and a list of circular
    obstacles, builds a graph, runs ``num_layers`` rounds of message passing,
    and returns the robot node's final feature vector.  The output can be
    used directly as the state representation for a downstream policy
    (e.g. SAC) or value-function network.

    Graph topology
    ~~~~~~~~~~~~~~
    - Node 0 is the **robot node** with features::

          [x, y, vx, vy, gx - x, gy - y]    (robot_node_dim = 6)

    - Nodes 1 … N are **obstacle nodes** with features::

          [ox, oy, r]                          (obstacle_node_dim = 3)

    - **Directed edges**: for every obstacle *i* there is an edge from
      obstacle node *i* to the robot node (``i+1 → 0``), and, optionally,
      a reverse edge (``0 → i+1``).  Edge features are::

          [dx, dy, dist]                       (edge_dim = 3)

      where ``dx = ox - x``, ``dy = oy - y``, ``dist = sqrt(dx²+dy²)``.

    When there are no obstacles the robot node still passes through all
    GNN layers; its aggregated message will simply be a zero vector.

    Args:
        robot_node_dim: Dimensionality of robot node input features (default: 6).
        obstacle_node_dim: Dimensionality of obstacle node input features (default: 3).
        edge_dim: Dimensionality of edge input features (default: 3).
        hidden_dim: Width of all hidden layers in EdgeModel and NodeModel.
        node_out_dim: Node feature dimensionality after each GNN layer.
        num_layers: Number of message-passing rounds.
        add_reverse_edges: If True, also add edges from robot to each obstacle.
    """

    def __init__(
        self,
        robot_node_dim: int = 6,
        obstacle_node_dim: int = 3,
        edge_dim: int = 3,
        hidden_dim: int = 64,
        node_out_dim: int = 64,
        num_layers: int = 2,
        add_reverse_edges: bool = True,
    ) -> None:
        super().__init__()

        self.robot_node_dim = robot_node_dim
        self.obstacle_node_dim = obstacle_node_dim
        self.add_reverse_edges = add_reverse_edges

        # Project robot and obstacle node features to the same hidden_dim.
        self.robot_proj = nn.Linear(robot_node_dim, hidden_dim)
        self.obstacle_proj = nn.Linear(obstacle_node_dim, hidden_dim)

        hidden = [hidden_dim, hidden_dim]
        layers = []
        in_node_dim = hidden_dim
        in_edge_dim = edge_dim
        for _ in range(num_layers):
            layers.append(
                GNNLayer(
                    node_dim=in_node_dim,
                    edge_dim=in_edge_dim,
                    edge_hidden=hidden,
                    edge_out_dim=hidden_dim,
                    node_hidden=hidden,
                    node_out_dim=node_out_dim,
                )
            )
            in_node_dim = node_out_dim
            in_edge_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.out_dim = node_out_dim

    def forward(
        self,
        robot_state: Tensor,
        target_pos: Tensor,
        obstacle_feats: Tensor | None = None,
    ) -> Tensor:
        """Encode robot state and obstacles into a fixed-size feature vector.

        Args:
            robot_state: Tensor of shape ``(batch, 4)`` or ``(4,)`` containing
                ``[x, y, vx, vy]``.
            target_pos: Tensor of shape ``(batch, 2)`` or ``(2,)`` containing
                the goal position ``[gx, gy]``.
            obstacle_feats: Tensor of shape ``(batch, N, 3)`` or ``(N, 3)``
                containing ``[ox, oy, r]`` for each obstacle.
                Pass ``None`` or an empty tensor when there are no obstacles.

        Returns:
            Robot node feature vector of shape ``(batch, out_dim)`` or
            ``(out_dim,)`` (matching the input batch dimension).
        """
        squeeze = robot_state.dim() == 1
        if squeeze:
            robot_state = robot_state.unsqueeze(0)
            target_pos = target_pos.unsqueeze(0)
            if obstacle_feats is not None and obstacle_feats.dim() == 2:
                obstacle_feats = obstacle_feats.unsqueeze(0)

        batch_size = robot_state.size(0)
        device = robot_state.device
        dtype = robot_state.dtype

        # Build robot node feature: [x, y, vx, vy, gx-x, gy-y]
        rel_goal = target_pos - robot_state[:, :2]
        robot_node = torch.cat([robot_state, rel_goal], dim=-1)  # (B, 6)

        has_obstacles = (
            obstacle_feats is not None
            and obstacle_feats.numel() > 0
            and obstacle_feats.size(-2) > 0
        )

        outputs = []
        for b in range(batch_size):
            rn = self.robot_proj(robot_node[b : b + 1])  # (1, hidden_dim)

            if has_obstacles:
                obs = obstacle_feats[b]  # (N, 3)
                num_obs = obs.size(0)
                on = self.obstacle_proj(obs)  # (N, hidden_dim)
                x = torch.cat([rn, on], dim=0)  # (1+N, hidden_dim)

                # Build edges: obstacle_i (node i+1) → robot (node 0)
                obs_indices = torch.arange(1, 1 + num_obs, device=device)
                robot_idx = torch.zeros(num_obs, dtype=torch.long, device=device)

                src = obs_indices
                dst = robot_idx
                if self.add_reverse_edges:
                    src = torch.cat([src, robot_idx], dim=0)
                    dst = torch.cat([dst, obs_indices], dim=0)

                edge_index = torch.stack([src, dst], dim=0)  # (2, E)

                # Edge features: [dx, dy, dist]
                rx = robot_node[b, 0]
                ry = robot_node[b, 1]
                dx = obs[:, 0] - rx
                dy = obs[:, 1] - ry
                dist = torch.sqrt(dx * dx + dy * dy + _EPS)
                ea = torch.stack([dx, dy, dist], dim=-1)  # (N, 3)
                if self.add_reverse_edges:
                    ea_rev = torch.stack([-dx, -dy, dist], dim=-1)  # robot→obstacle: negate direction
                    ea = torch.cat([ea, ea_rev], dim=0)

                edge_attr = ea
            else:
                num_nodes = 1
                x = rn
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                edge_attr = torch.zeros((0, 3), dtype=dtype, device=device)

            # Message passing rounds.
            for layer in self.layers:
                x, edge_attr = layer(x, edge_index, edge_attr)

            outputs.append(x[0])  # Robot node feature (out_dim,)

        result = torch.stack(outputs, dim=0)  # (B, out_dim)
        return result.squeeze(0) if squeeze else result


# ---------------------------------------------------------------------------
# Helper: build graph tensors from numpy arrays
# ---------------------------------------------------------------------------

def build_robot_obstacle_graph(
    robot_state: "np.ndarray",
    target_pos: "np.ndarray",
    obstacles: list[tuple[float, float, float]] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Convert raw numpy environment data into graph input tensors.

    This is a convenience function for use outside the training loop
    (e.g. inside MPPI rollouts or evaluation scripts).

    Args:
        robot_state: Array ``[x, y, vx, vy]`` of shape ``(4,)``.
        target_pos: Array ``[gx, gy]`` of shape ``(2,)``.
        obstacles: List of ``(ox, oy, r)`` tuples.  Pass ``None`` or empty
            list when there are no obstacles.
        device: Torch device string or object.

    Returns:
        Tuple ``(robot_state_t, target_pos_t, obstacle_feats_t)`` where:

        - ``robot_state_t``: Float tensor, shape ``(4,)``.
        - ``target_pos_t``: Float tensor, shape ``(2,)``.
        - ``obstacle_feats_t``: Float tensor of shape ``(N, 3)``, or ``None``
          when no obstacles are provided.
    """
    import numpy as np

    rs_t = torch.tensor(np.asarray(robot_state, dtype=np.float32), device=device)
    tp_t = torch.tensor(np.asarray(target_pos, dtype=np.float32), device=device)

    if obstacles and len(obstacles) > 0:
        obs_arr = np.array([[o[0], o[1], o[2]] for o in obstacles], dtype=np.float32)
        obs_t: Tensor | None = torch.tensor(obs_arr, device=device)
    else:
        obs_t = None

    return rs_t, tp_t, obs_t
