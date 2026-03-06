import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# GNN基础模块（可根据multi_robots/gnn/networks.py调整）
class GNNPolicyNet(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds=1, use_attention=False):
        super().__init__()
        from .networks import PolicyNet
        self.policy = PolicyNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim * 2, rounds=rounds, use_attention=use_attention)
        self.action_dim = action_dim
        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 2.0
        self.LOG_PROB_MIN = -20.0
        self.LOG_PROB_MAX = 2.0

    def forward(self, nodes, edges, globals_, senders, receivers, robot_node_indices):
        out = self.policy(nodes, edges, globals_, senders, receivers, robot_node_indices)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, nodes, edges, globals_, senders, receivers, robot_node_indices):
        mean, log_std = self.forward(nodes, edges, globals_, senders, receivers, robot_node_indices)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        # 计算log_prob（带tanh修正项）
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # tanh修正项
        log_prob -= (2 * (np.log(2) - x_t - nn.functional.softplus(-2 * x_t))).sum(dim=-1, keepdim=True)
        log_prob = torch.clamp(log_prob, self.LOG_PROB_MIN, self.LOG_PROB_MAX)
        return action, log_prob, mean, log_std

    def get_deterministic(self, nodes, edges, globals_, senders, receivers, robot_node_indices):
        mean, _ = self.forward(nodes, edges, globals_, senders, receivers, robot_node_indices)
        return torch.tanh(mean)

class GNNCriticNet(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds=1, use_attention=False):
        super().__init__()
        from .networks import CriticNet
        self.critic = CriticNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds=rounds, use_attention=use_attention)
    def forward(self, nodes, edges, globals_, senders, receivers):
        return self.critic(nodes, edges, globals_, senders, receivers)

class GNNReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size
    def add(self, *args):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]
    def __len__(self):
        return len(self.buffer)

class GNNRLAgent:
    def __init__(self, input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds=1, use_attention=False, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.policy_net = GNNPolicyNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds, use_attention)
        self.q_net1 = GNNCriticNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds, use_attention)
        self.q_net2 = GNNCriticNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds, use_attention)
        self.target_q_net1 = GNNCriticNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds, use_attention)
        self.target_q_net2 = GNNCriticNet(input_node_dim, input_edge_dim, input_global_dim, node_dim, edge_dim, global_dim, node_hidden_dims, edge_hidden_dims, action_dim, rounds, use_attention)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
    def select_action(self, nodes, edges, globals_, senders, receivers, robot_node_indices, evaluate=False):
        with torch.no_grad():
            if evaluate:
                action = self.policy_net.get_deterministic(nodes, edges, globals_, senders, receivers, robot_node_indices)
            else:
                action, _, _, _ = self.policy_net.sample(nodes, edges, globals_, senders, receivers, robot_node_indices)
            return action.cpu().numpy()
    def update(self, replay_buffer, batch_size=256, device='cpu'):
        # 假设replay_buffer每条为(obs, action, reward, next_obs, done)
        batch = replay_buffer.sample(batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        # 需根据实际obs结构展开
        nodes = torch.tensor(np.stack([o['nodes'] for o in obs]), dtype=torch.float32, device=device)
        edges = torch.tensor(np.stack([o['edges'] for o in obs]), dtype=torch.float32, device=device)
        globals_ = torch.tensor(np.stack([o['globals'] for o in obs]), dtype=torch.float32, device=device)
        senders = torch.tensor(obs[0]['senders'], dtype=torch.long, device=device)
        receivers = torch.tensor(obs[0]['receivers'], dtype=torch.long, device=device)
        robot_node_indices = torch.tensor(obs[0]['robot_node_indices'], dtype=torch.long, device=device)
        actions = torch.tensor(np.stack(actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        next_nodes = torch.tensor(np.stack([o['nodes'] for o in next_obs]), dtype=torch.float32, device=device)
        next_edges = torch.tensor(np.stack([o['edges'] for o in next_obs]), dtype=torch.float32, device=device)
        next_globals = torch.tensor(np.stack([o['globals'] for o in next_obs]), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)

         # 目标动作
        with torch.no_grad():
            next_actions, _, _, _ = self.policy_net.sample(next_nodes, next_edges, next_globals, senders, receivers, robot_node_indices)
            target_q1 = self.target_q_net1(next_nodes, next_edges, next_globals, senders, receivers)
            target_q2 = self.target_q_net2(next_nodes, next_edges, next_globals, senders, receivers)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1 - dones) * self.gamma * target_q
         # 当前Q
        current_q1 = self.q_net1(nodes, edges, globals_, senders, receivers)
        current_q2 = self.q_net2(nodes, edges, globals_, senders, receivers)
        q1_loss = nn.MSELoss()(current_q1, target)
        q2_loss = nn.MSELoss()(current_q2, target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        # Policy loss (SAC风格，含熵项)
        new_actions, log_prob, _, _ = self.policy_net.sample(nodes, edges, globals_, senders, receivers, robot_node_indices)
        q1_new = self.q_net1(nodes, edges, globals_, senders, receivers)
        # 保证log_prob和q1_new shape一致
        if log_prob.dim() == 2 and log_prob.shape[1] == 1:
            log_prob = log_prob.squeeze(1)
        if q1_new.dim() == 2 and q1_new.shape[1] == 1:
            q1_new = q1_new.squeeze(1)
        # 若q1_new多于1列，只取第一个Q值
        if q1_new.dim() == 2 and q1_new.shape[1] > 1:
            q1_new = q1_new[:, 0]
        # shape对齐后再计算policy_loss
        min_len = min(log_prob.shape[0], q1_new.shape[0])
        policy_loss = (self.alpha * log_prob[:min_len] - q1_new[:min_len]).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.soft_update_target_networks()
        return {'q1_loss': q1_loss.item(), 'q2_loss': q2_loss.item(), 'policy_loss': policy_loss.item()}
    def soft_update_target_networks(self):
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

# 训练主流程接口（参考algorithms/sac/sac_utils.py和roboballet/train/train.py）
def train_gnn_agent(env, agent, replay_buffer, total_steps, batch_size=256, eval_every=10000):
    # 训练主循环，采集数据、更新网络、评估
    obs = env.reset()
    for step in range(total_steps):
        # 采集数据
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()
        # 更新网络
        if len(replay_buffer) >= batch_size:
            agent.update(replay_buffer, batch_size)
        # 评估
        if (step + 1) % eval_every == 0:
            eval_reward = evaluate_gnn_agent(env, agent)
            print(f"Step {step+1}, Eval Reward: {eval_reward}")

def evaluate_gnn_agent(env, agent, episodes=5):
    total_reward = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
    return total_reward / episodes
