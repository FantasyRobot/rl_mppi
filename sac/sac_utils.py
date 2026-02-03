#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

# ========================================# Global Configuration# ========================================
DEVICE = torch.device("cpu")
EPS = 1e-8

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

# ========================================# SAC Neural Network Architectures# ========================================

# --------------------------# Policy Network (Actor) - π_φ# --------------------------
class PolicyNetwork(nn.Module):
    """
    Stochastic policy network that outputs Gaussian distribution parameters.
    Maps states to action distributions: π(a|s) = N(μ(s), σ(s))
    Uses tanh squashing for bounded action output in [-1, 1]
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ELU()])
            prev_dim = dim

        self.mlp = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, output_dim)
        self.log_std_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = x.to(DEVICE)
        features = self.mlp(x)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)

        normal = distributions.Normal(mean, std)
        z = normal.rsample()
        # Standard tanh squashing to map to [-1, 1]
        action = torch.tanh(z)

        log_prob_raw = normal.log_prob(z).sum(dim=-1, keepdim=True)
        # Jacobian correction for tanh: log(1 - tanh(z)^2)
        log_det_jacobian = torch.sum(torch.log(1 - action.pow(2) + EPS), dim=-1, keepdim=True)
        log_prob = log_prob_raw - log_det_jacobian

        return action, log_prob, mean, log_std

    def get_deterministic(self, x):
        mean, _ = self.forward(x)
        # Deterministic action: tanh(mean) maps to [-1,1]
        return torch.tanh(mean)

# --------------------------# Q-Network (Critic) - Q_θ# --------------------------
class QNetwork(nn.Module):
    """
    Q-network that outputs Q-values for state-action pairs.
    Supports both standard SAC and distributional DSAC architectures.
    """
    def __init__(self, input_dim, action_dim, hidden_dims=[256, 256], use_distributional=False):
        super(QNetwork, self).__init__()
        self.use_distributional = use_distributional

        layers = []
        prev_dim = input_dim + action_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ELU()])
            prev_dim = dim

        self.shared = nn.Sequential(*layers)
        
        if use_distributional:
            # Distributional Q-network (for DSAC models)
            self.out_mean = nn.Linear(prev_dim, 1)
            self.out_log_std = nn.Linear(prev_dim, 1)
        else:
            # Standard Q-network (for SAC models)
            self.q_value = nn.Linear(prev_dim, 1)

    def forward(self, x, u):
        x = x.to(DEVICE)
        u = u.to(DEVICE)
        xu = torch.cat([x, u], dim=-1)
        h = self.shared(xu)
        
        if self.use_distributional:
            mean = self.out_mean(h).squeeze(-1)
            log_std = self.out_log_std(h).squeeze(-1)
            return mean, torch.exp(log_std), log_std
        else:
            return self.q_value(h).squeeze(-1)

    def get_q_value(self, x, u):
        """
        Get Q-value for state-action pair.
        If distributional, returns the mean of the distribution.
        """
        if self.use_distributional:
            mean, _, _ = self.forward(x, u)
            return mean
        else:
            return self.forward(x, u)


# ========================================# SAC Agent Implementation# ========================================
class SACAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 learning_rate=3e-4,
                 alpha=0.2,
                 gamma=0.99,
                 tau=0.005,
                 auto_entropy_tuning=True):
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims=[hidden_dim, hidden_dim]).to(DEVICE)
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dims=[hidden_dim, hidden_dim]).to(DEVICE)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dims=[hidden_dim, hidden_dim]).to(DEVICE)
        
        # Target networks
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dims=[hidden_dim, hidden_dim]).to(DEVICE)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dims=[hidden_dim, hidden_dim]).to(DEVICE)
        
        # Copy weights to target networks
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        # Optimizers with weight decay for stability
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler for decay
        self.policy_lr_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=100, gamma=0.9)
        self.q1_lr_scheduler = optim.lr_scheduler.StepLR(self.q1_optimizer, step_size=100, gamma=0.9)
        self.q2_lr_scheduler = optim.lr_scheduler.StepLR(self.q2_optimizer, step_size=100, gamma=0.9)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Temperature parameter
        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(DEVICE)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha = self.log_alpha.exp().item()  # Initialize alpha
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = alpha
    
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            if evaluate:
                action = self.policy_net.get_deterministic(state)
            else:
                action, _, _, _ = self.policy_net.sample(state)
            return action.cpu().numpy()
    
    def update(self, replay_buffer, batch_size=256):
        # Sample from replay buffer
        batch = replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(batch["state"]).to(DEVICE)
        action_batch = torch.FloatTensor(batch["action"]).to(DEVICE)
        reward_batch = torch.FloatTensor(batch["reward"]).to(DEVICE)
        next_state_batch = torch.FloatTensor(batch["next_state"]).to(DEVICE)
        done_batch = torch.FloatTensor(batch["done"]).to(DEVICE)
        
        # -----------------------------
        # Update Q-networks
        # -----------------------------
        with torch.no_grad():
            next_action_batch, log_prob_next_action_batch, _, _ = self.policy_net.sample(next_state_batch)
            
            target_q1 = self.target_q_net1(next_state_batch, next_action_batch)
            target_q2 = self.target_q_net2(next_state_batch, next_action_batch)
            target_q_min = torch.min(target_q1, target_q2) - self.alpha * log_prob_next_action_batch.squeeze()
            
            # Compute target Q values
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q_min
        
        # Get current Q estimates
        current_q1 = self.q_net1(state_batch, action_batch)
        current_q2 = self.q_net2(state_batch, action_batch)
        
        # Compute Q loss
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        # Update Q networks with gradient clipping
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=1.0)
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=1.0)
        self.q2_optimizer.step()
        
        # Step learning rate schedulers
        self.q1_lr_scheduler.step()
        self.q2_lr_scheduler.step()
        
        # -----------------------------
        # Update Policy Network
        # -----------------------------
        # Freeze Q-networks so you don't waste computational effort computing gradients for them during the policy learning step.
        for param in self.q_net1.parameters():
            param.requires_grad = False
        for param in self.q_net2.parameters():
            param.requires_grad = False
        
        # Compute policy loss
        new_action_batch, log_prob_new_action_batch, _, _ = self.policy_net.sample(state_batch)
        q1_new_policy = self.q_net1(state_batch, new_action_batch)
        q2_new_policy = self.q_net2(state_batch, new_action_batch)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)
        
        policy_loss = (self.alpha * log_prob_new_action_batch.squeeze() - q_new_policy).mean()
        
        # Update policy network with gradient clipping
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # Step policy learning rate scheduler
        self.policy_lr_scheduler.step()
        
        # Unfreeze Q-networks
        for param in self.q_net1.parameters():
            param.requires_grad = True
        for param in self.q_net2.parameters():
            param.requires_grad = True
        
        # -----------------------------
        # Update Temperature Parameter
        # -----------------------------
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob_new_action_batch + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # -----------------------------
        # Update Target Q Networks
        # -----------------------------
        self.soft_update_target_networks()
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha
        }
    
    def soft_update_target_networks(self):
        """Soft update target networks: target = tau*local + (1-tau)*target"""
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# ========================================# Replay Buffer Implementation# ========================================
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.buffer = []
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return {
            "state": np.array(states),
            "action": np.array(actions),
            "reward": np.array(rewards),
            "next_state": np.array(next_states),
            "done": np.array(dones)
        }
    
    def __len__(self):
        return len(self.buffer)
    
# ========================================# Data Handling Utilities# ========================================

# ------------------------# Simple CSV-like reader (compatible with previous format)# ------------------------
def read_data_from_file(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        return np.zeros((0,))
    rows = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            clean = line.lstrip(",").strip()
            parts = [p for p in clean.split(",") if p.strip() != ""]
            if len(parts) == 0:
                continue
            try:
                vals = [float(p) for p in parts]
            except Exception:
                continue
            rows.append(vals)
    if len(rows) == 0:
        return np.zeros((0,))
    # ensure uniform width
    w = len(rows[0])
    rows = [r for r in rows if len(r) == w]
    return np.array(rows, dtype=np.float32)

# ------------------------# Min-max scaler helpers (map to [-1,1])# ------------------------
def fit_minmax_scalers(x: np.ndarray, u: np.ndarray, x_prime: np.ndarray) -> dict:
    stacked_x = np.vstack([x, x_prime]) if x_prime.size > 0 else x
    x_min = np.min(stacked_x, axis=0)
    x_max = np.max(stacked_x, axis=0)
    u_min = np.min(u, axis=0)
    u_max = np.max(u, axis=0)
    return {"x_min": x_min, "x_max": x_max, "u_min": u_min, "u_max": u_max}

def minmax_scale_to_minus1_1(arr: np.ndarray, arr_min: np.ndarray, arr_max: np.ndarray) -> np.ndarray:
    denom = (arr_max - arr_min)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    return 2.0 * (arr - arr_min) / denom_safe - 1.0

def scale_state(state, scalers):
    """
    Scale state using the provided scalers.
    
    Args:
        state (np.ndarray): Raw state
        scalers (dict): Scaling parameters
        
    Returns:
        np.ndarray: Scaled state in [-1, 1]
    """
    if not scalers:
        return state
    
    x_min = scalers['x_min']
    x_max = scalers['x_max']
    denom = (x_max - x_min)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    return 2.0 * (state - x_min) / denom_safe - 1.0

def scale_action(action, scalers):
    """
    Unscale action from [-1, 1] to the original action space.
    
    Args:
        action (np.ndarray): Scaled action in [-1, 1]
        scalers (dict): Scaling parameters
        
    Returns:
        np.ndarray: Unscaled action in original action space
    """
    if not scalers:
        return action
    
    u_min = scalers['u_min']
    u_max = scalers['u_max']
    return u_min + (action + 1.0) * (u_max - u_min) / 2.0