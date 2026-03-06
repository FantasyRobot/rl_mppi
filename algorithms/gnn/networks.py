import torch
import torch.nn as nn
import numpy as np

# 参考 roboballet
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, last_layer_scale=0.005):
        super().__init__()
        # 兼容 int 或 list/tuple
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.last_layer_scale = last_layer_scale
        # Optionally scale last layer
        if hasattr(self.net[-1], 'weight'):
            nn.init.uniform_(self.net[-1].weight, -last_layer_scale, last_layer_scale)
    def forward(self, x):
        return self.net(x)

class GraphNet(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, rounds=1, use_attention=False):
        super().__init__()
        # 这里只做简单线性映射和消息传递占位
        self.node_encoder = MLP(input_node_dim, node_hidden_dims, node_dim)
        self.edge_encoder = MLP(input_edge_dim, edge_hidden_dims, edge_dim)
        # roboballet风格：嵌入层MLP输入维度严格等于原始特征维度，输出为统一embed_dim
        embed_dim = 128
        self.embed_dim = embed_dim
        self.node_embedder = MLP(input_node_dim, [embed_dim], embed_dim)
        self.edge_embedder = MLP(input_edge_dim, [embed_dim], embed_dim)
        self.global_embedder = MLP(input_global_dim, [embed_dim], embed_dim)  # input_global_dim=全局特征真实维度
        self.node_encoder = MLP(embed_dim, [embed_dim], embed_dim)
        self.edge_encoder = MLP(embed_dim, [embed_dim], embed_dim)
        self.global_encoder = MLP(embed_dim, [embed_dim], embed_dim)
        self.rounds = rounds

    def forward(self, nodes, edges, globals_, senders, receivers):
        # 1. 嵌入
        nodes = self.node_embedder(nodes)
        edges = self.edge_embedder(edges)
        globals_ = self.global_embedder(globals_)
        # 2. message passing（可多轮，这里只做一轮）
        nodes = self.node_encoder(nodes)
        edges = self.edge_encoder(edges)
        globals_ = self.global_encoder(globals_)
        return nodes, edges, globals_

class PolicyNet(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds=1, use_attention=False):
        super().__init__()
        self.gnn = GraphNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, rounds=rounds, use_attention=use_attention)
        embed_dim = 128
        self.policy_head = MLP(embed_dim, node_hidden_dims, action_dim, last_layer_scale=0.005)
    def forward(self, nodes, edges, globals_, senders, receivers, robot_node_indices):
        nodes, edges, globals_ = self.gnn(nodes, edges, globals_, senders, receivers)
        # roboballet风格：单机器人直接用所有节点均值，输出 action_dim
        node_feat = nodes.mean(dim=0, keepdim=True)
        action = self.policy_head(node_feat)
        return action.squeeze(0)

class CriticNet(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds=1, use_attention=False):
        super().__init__()
        self.gnn = GraphNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, rounds=rounds, use_attention=use_attention)
        embed_dim = 128
        self.q_head = MLP(embed_dim, node_hidden_dims, 1, last_layer_scale=0.005)
    def forward(self, nodes, edges, globals_, senders, receivers):
        nodes, edges, globals_ = self.gnn(nodes, edges, globals_, senders, receivers)
        q = self.q_head(globals_)
        return q.view(-1, 1)
