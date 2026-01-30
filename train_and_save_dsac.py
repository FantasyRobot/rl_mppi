#!/usr/bin/env python3
"""
Train DSAC (Distributional Soft Actor-Critic) - Paper-aligned implementation

Reference: RL-Driven Model Predictive Path Integral for Control
Paper Algorithm: Section B - RL-Driven MPPI: Offline RL Training Module

Core DSAC Algorithm Steps:
  1. Q-Network Update (Eqs. 9): Learn cost distribution via KL divergence minimization
     J_Z(θ) = -E[log p(Z(x_t,u_t)|Z_θ(·|x_t,u_t))]
     
  2. Policy Update (Eq. 10): Minimize expected cost + entropy regularization
     J_π(φ) = E[Q_θ(x_t,u_t) + α log σ_φ(u_t|x_t)]
     
  3. Alpha Update: Automatic temperature tuning for entropy regularization
     
  4. Target Networks: Soft update of target Q networks

Input Dataset:
 - x.txt:       state observations
 - u.txt:       action commands
 - x_prime.txt: next state observations
 - r.txt:       cost/reward
 - done.txt:    episode termination flags (optional)

Output Files:
 - dsac_model.pth: actor & critic state_dicts + scalers
 - sac_pi_model_deterministic.pt: TorchScript deterministic policy (tanh(μ))
 - scalers.npz: min-max normalization parameters

Usage:
    python train_and_save_dsac.py --epochs 100 --batch_size 512
    python train_and_save_dsac.py --epochs 100 --batch_size 512 --auto_alpha
"""
import os
import argparse
from typing import Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import time

# ========================================
# Global Configuration
# ========================================
DEVICE = torch.device("cpu")
EPS = 1e-8

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.set_default_dtype(torch.float32)

# ========================================
# DSAC Neural Network Architectures
# ========================================

# --------------------------
# Policy Network (Actor) - π_φ
# --------------------------
class PolicyNetwork(nn.Module):
    """
    Stochastic policy network that outputs Gaussian distribution parameters.
    Maps states to action distributions: π(a|s) = N(μ(s), σ(s))
    Uses tanh squashing for bounded action output in [-1, 1]
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], log_std_min=-20, log_std_max=2):
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
        action = torch.tanh(z)

        log_prob_raw = normal.log_prob(z).sum(dim=-1, keepdim=True)
        log_det_jacobian = torch.sum(torch.log(1 - action.pow(2) + EPS), dim=-1, keepdim=True)
        log_prob = log_prob_raw - log_det_jacobian

        return action, log_prob, mean, log_std

    def get_deterministic(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean)

# --------------------------
# Distributional Q-Network (Critic) - Z_θ
# --------------------------
class QNetwork(nn.Module):
    """
    Distributional Q-network that outputs cost distribution parameters.
    Models infinite-horizon cumulative cost as: Z^π(x,u) = E[Z(x,u)]
    Outputs both mean and standard deviation for distributional representation
    Paper Eq. (8): Q^π(x_t,v_t) := -E_{Z~Z^π(x,v)} [Z^π(x_t, v_t)]
    """
    def __init__(self, input_dim, action_dim, hidden_dims=[512, 256, 128], min_log_std=-20, max_log_std=2):
        super(QNetwork, self).__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        layers = []
        prev_dim = input_dim + action_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ELU()])
            prev_dim = dim

        self.shared = nn.Sequential(*layers)
        self.out_mean = nn.Linear(prev_dim, 1)
        self.out_log_std = nn.Linear(prev_dim, 1)

    def forward(self, x, u):
        x = x.to(DEVICE)
        u = u.to(DEVICE)
        xu = torch.cat([x, u], dim=-1)
        h = self.shared(xu)
        mean = self.out_mean(h).squeeze(-1)
        log_std = self.out_log_std(h).squeeze(-1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        return mean, std, log_std

    def sample_q(self, x, u):
        mean, std, _ = self.forward(x, u)
        normal = distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample().to(DEVICE)
        z = torch.clamp(z, -3.0, 3.0)
        q_sample = mean + z * std
        return q_sample, mean, std

# --------------------------
# DSAC Agent
# --------------------------
class DSACAgent:
    """
    Distributional Soft Actor-Critic Agent
    
    Paper: RL-Driven Model Predictive Path Integral
    Section B: Offline RL Training Module
    
    Uses distributional cost Q-networks with entropy-regularized policy improvement
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dims=[512, 256, 128],
                 lr=3e-4,
                 gamma=0.99,
                 tau=1e-3,
                 alpha=None):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dims).to(DEVICE)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(DEVICE)

        self.q1_target = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(DEVICE)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

        if alpha is None:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None
            self.alpha = torch.tensor(alpha, device=DEVICE) if not isinstance(alpha, float) else alpha

    def _compute_target_q(self, rewards, dones, next_states):
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.sample(next_states)
            q1_next_sample, _, _ = self.q1_target.sample_q(next_states, next_action)
            q2_next_sample, _, _ = self.q2_target.sample_q(next_states, next_action)
            q_next = torch.min(q1_next_sample, q2_next_sample)
            alpha = self.alpha if not isinstance(self.alpha, torch.Tensor) else self.alpha.item()
            target_q = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * self.gamma * (q_next - alpha * next_log_prob.squeeze(-1))
            return target_q

    def update_critics(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """
        Update Q-networks (critic) via distributional cost learning.
        
        Paper Eq. (9): Minimize KL divergence between target and current distributions
        J_Z(θ) = -E_{(x_t,u_t,r_t+1)~B, Z~Z_θ(-|x_t,u_t)} [log p(Z_target | Z_θ(-|x_t,u_t))]
        """
        target_q = self._compute_target_q(reward_batch, done_batch, next_state_batch).detach()

        mean1, std1, _ = self.q1.forward(state_batch, action_batch)
        normal1 = distributions.Normal(mean1, std1 + EPS)
        q1_loss = -normal1.log_prob(target_q).mean()

        mean2, std2, _ = self.q2.forward(state_batch, action_batch)
        normal2 = distributions.Normal(mean2, std2 + EPS)
        q2_loss = -normal2.log_prob(target_q).mean()

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        return q1_loss.item(), q2_loss.item()

    def update_actor(self, state_batch):
        """
        Update policy network (actor) via entropy-regularized cost minimization.
        
        Paper Eq. (10): Minimize expected cost + entropy regularization
        J_π(φ) = E_{x_t~B, u_t~π_φ(-|x_t)} [Q_θ(x_t, u_t) + α log π_φ(u_t|x_t)]
        """
        action, log_prob, _, _ = self.actor.sample(state_batch)
        q1_sample, _, _ = self.q1.sample_q(state_batch, action)
        q2_sample, _, _ = self.q2.sample_q(state_batch, action)
        q_sample = torch.min(q1_sample, q2_sample).unsqueeze(-1)
        alpha = self.alpha if not isinstance(self.alpha, torch.Tensor) else self.alpha.item()
        actor_loss = (alpha * log_prob - q_sample).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def update_alpha(self, log_prob_batch):
        """
        Automatic temperature (entropy) tuning via SAC mechanism.
        
        Adjusts α to maintain target entropy level:
        α_loss = -E[log_α * (H(π) - H_target)]
        """
        if self.alpha_optimizer is None:
            return 0.0
        alpha_loss = -(self.log_alpha * (log_prob_batch + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        return alpha_loss.item()

    def update_target_networks(self):
        """
        Soft update of target networks for TD stability.
        Targets lag behind current networks: θ_target := τ*θ + (1-τ)*θ_target
        """
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_deterministic(state)
                return action.cpu().numpy().squeeze(0)
            else:
                action, _, _, _ = self.actor.sample(state)
                return action.cpu().numpy().squeeze(0)

# ------------------------
# Simple CSV-like reader (compatible with previous format)
# ------------------------
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

# ------------------------
# Min-max scaler helpers (map to [-1,1])
# ------------------------
def fit_minmax_scalers(x: np.ndarray, u: np.ndarray, x_prime: np.ndarray) -> Dict[str, np.ndarray]:
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

# ------------------------
# Dataset wrapper
# ------------------------
class OfflineDataset(Dataset):
    def __init__(self, x, u, r, x_prime, done):
        # ensure shapes
        self.x = x.astype(np.float32)
        self.u = u.astype(np.float32)
        self.r = r.astype(np.float32).reshape(-1, 1)
        self.x_prime = x_prime.astype(np.float32)
        self.done = done.astype(np.float32).reshape(-1, 1)
        assert len(self.x) == len(self.u) == len(self.r) == len(self.x_prime) == len(self.done)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "obs": torch.from_numpy(self.x[idx]),
            "act": torch.from_numpy(self.u[idx]),
            "rew": torch.from_numpy(self.r[idx]),
            "obs2": torch.from_numpy(self.x_prime[idx]),
            "done": torch.from_numpy(self.done[idx]),
        }

# ------------------------
# Training loop
# ------------------------
def train_and_save(args):
    # load raw data
    x = read_data_from_file(os.path.join(args.data_dir, "x.txt"))
    u = read_data_from_file(os.path.join(args.data_dir, "u.txt"))
    x_prime = read_data_from_file(os.path.join(args.data_dir, "x_prime.txt"))
    r = read_data_from_file(os.path.join(args.data_dir, "r.txt"))
    done_arr = read_data_from_file(os.path.join(args.data_dir, "done.txt"))

    # build done if missing
    if done_arr.size > 0:
        done = done_arr.flatten()
        print("Using done.txt")
    else:
        if r.size == 0:
            raise RuntimeError("reward file missing or empty.")
        r_flat = r.flatten()
        done = (r_flat < 0.01).astype(np.float32)
        print("done.txt missing: generated done = (r < 0.01)")

    minlen = min(len(x), len(u), len(x_prime), len(r), len(done))
    if minlen == 0:
        raise RuntimeError("One or more data files empty or incompatible.")
    x = x[:minlen]
    u = u[:minlen]
    x_prime = x_prime[:minlen]
    r = r[:minlen].reshape(-1)
    done = done[:minlen].reshape(-1)

    # fit scalers and normalize to [-1,1]
    scalers = fit_minmax_scalers(x, u, x_prime)
    x_norm = minmax_scale_to_minus1_1(x, scalers["x_min"], scalers["x_max"])
    x_prime_norm = minmax_scale_to_minus1_1(x_prime, scalers["x_min"], scalers["x_max"])
    u_norm = minmax_scale_to_minus1_1(u, scalers["u_min"], scalers["u_max"])

    print(f"\n[STEP 1] Data preparation completed")
    print(f"  ├─ Total samples: {len(x)}")
    print(f"  ├─ State dimension: {x.shape[1]}")
    print(f"  ├─ Action dimension: {u.shape[1]}")
    print(f"  └─ Reward range: [{r.min():.4f}, {r.max():.4f}]")
    
    dataset = OfflineDataset(x_norm, u_norm, r.reshape(-1,1), x_prime_norm, done.reshape(-1,1))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    print(f"\n[STEP 2] DataLoader initialized")
    print(f"  ├─ Batch size: {args.batch_size}")
    print(f"  ├─ Num batches per epoch: {len(loader)}")
    print(f"  └─ Data normalization: [-1, 1]")

    obs_dim = x_norm.shape[1]
    act_dim = u_norm.shape[1] if u_norm.ndim > 1 else 1

    print(f"\n[STEP 3] Creating DSAC Agent")
    print(f"  ├─ Observation dimension: {obs_dim}")
    print(f"  ├─ Action dimension: {act_dim}")
    print(f"  ├─ Hidden sizes: [{args.hidden_size}, {args.hidden_size}]")
    print(f"  ├─ Learning rate: {args.lr}")
    print(f"  ├─ Discount factor (gamma): {args.gamma}")
    print(f"  ├─ Soft update rate (tau): {args.tau}")
    print(f"  └─ Automatic alpha: {args.auto_alpha}")
    
    # create agent (paper-aligned DSACAgent from DSAC_pi)
    # DSACAgent uses `alpha=None` to enable automatic alpha tuning.
    alpha_arg = None if args.auto_alpha else args.alpha
    agent = DSACAgent(obs_dim, act_dim, hidden_dims=(args.hidden_size, args.hidden_size),
                      lr=args.lr, gamma=args.gamma, tau=args.tau, alpha=alpha_arg)
    print(f"  └─ Agent initialized successfully ✓")

    # Setup logging containers if requested
    loss_log = {
        'loss_z1': [],
        'loss_z2': [],
        'loss_actor': [],
        'loss_alpha': []
    }

    # ========================================
    # DSAC Offline Training Loop
    # ========================================
    # Paper: "RL-Driven Model Predictive Path Integral"
    # Section B: RL-Driven MPPI - Offline RL Training Module
    
    print(f"\n[STEP 4] Starting training loop")
    print(f"  ├─ Total epochs: {args.epochs}")
    print(f"  ├─ Device: {DEVICE}")
    print(f"  └─ Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_steps = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f"{'='*80}")
        print(f"EPOCH [{epoch+1}/{args.epochs}]")
        print(f"{'='*80}")
        
        epoch_info = {"loss_z1": 0.0, "loss_z2": 0.0, "loss_actor": 0.0, "loss_alpha": 0.0}
        n_batches = 0
        batch_start = time.time()
        
        for batch_idx, batch in enumerate(loader):
            batch_obs = batch["obs"].float()
            batch_act = batch["act"].float()
            batch_rew = batch["rew"].float()
            batch_obs2 = batch["obs2"].float()
            batch_done = batch["done"].float()

            # ─────────────────────────────────────────────────────────────
            # Step 1: Update Q-Networks (Critic Update)
            # ─────────────────────────────────────────────────────────────
            # Paper Eq. (9): Minimize KL divergence between target and current cost distributions
            # J_Z(θ) = -E_{(x_t,u_t,r_{t+1})~B, μ~Z_θ(-|x_t,u_t)} 
            #          [log p(Z(x_t,v_t)|Z_θ(-|x_t,v_t))]
            # Where Z is the cost distribution with mean and std dev
            loss_z1, loss_z2 = agent.update_critics(batch_obs, batch_act, batch_rew, batch_obs2, batch_done)
            
            # ─────────────────────────────────────────────────────────────
            # Step 2: Update Actor (Policy Update)
            # ─────────────────────────────────────────────────────────────
            # Paper Eq. (10): Policy improvement via entropy-regularized cost minimization
            # J_π(φ) = E_{x_t~B, v_t~σ_φ(-|x_t)} [Q_θ(x_t, v_t) + α log(σ_φ(v_t|x_t))]
            # This is equivalent to maximizing entropy-weighted Q-values
            loss_actor = agent.update_actor(batch_obs)
            
            # ─────────────────────────────────────────────────────────────
            # Step 3: Update Temperature Parameter (Alpha)
            # ─────────────────────────────────────────────────────────────
            # Automatic entropy regularization tuning if enabled
            # Maintains target entropy level for robust policy learning
            with torch.no_grad():
                _, log_prob_batch, _, _ = agent.actor.sample(batch_obs)
            loss_alpha = agent.update_alpha(log_prob_batch)
            
            # ─────────────────────────────────────────────────────────────
            # Step 4: Soft Update Target Networks
            # ─────────────────────────────────────────────────────────────
            # Stabilize learning by slowly updating target cost distributions
            agent.update_target_networks()

            epoch_info["loss_z1"] += loss_z1
            epoch_info["loss_z2"] += loss_z2
            epoch_info["loss_actor"] += loss_actor
            epoch_info["loss_alpha"] += loss_alpha
            n_batches += 1
            total_steps += 1
            # record per-batch losses
            if getattr(args, 'log_dir', None) is not None:
                loss_log['loss_z1'].append(loss_z1)
                loss_log['loss_z2'].append(loss_z2)
                loss_log['loss_actor'].append(loss_actor)
                loss_log['loss_alpha'].append(loss_alpha)
            
            # Print batch progress
            if (batch_idx + 1) % max(1, len(loader) // 4) == 0 or batch_idx == 0:
                batch_elapsed = time.time() - batch_start
                print(f"  Batch [{batch_idx+1:3d}/{len(loader):3d}] | "
                      f"Loss_Q1: {loss_z1:.6f} | Loss_Q2: {loss_z2:.6f} | "
                      f"Loss_π: {loss_actor:.6f} | Loss_α: {loss_alpha:.6f} | "
                      f"Time: {batch_elapsed:.2f}s")

            # Optional: break early for quick experiments
            if getattr(args, 'max_batches_per_epoch', 0) > 0 and (batch_idx + 1) >= args.max_batches_per_epoch:
                print(f"  └─ Reached max_batches_per_epoch={args.max_batches_per_epoch}, ending epoch early.")
                break

        # Compute epoch averages and print epoch summary
        if n_batches > 0:
            for k in epoch_info:
                epoch_info[k] /= n_batches
        
        elapsed = time.time() - start_time
        print(f"  ├─ Epoch average losses:")
        print(f"  │  ├─ Loss_Q1: {epoch_info['loss_z1']:.6f}")
        print(f"  │  ├─ Loss_Q2: {epoch_info['loss_z2']:.6f}")
        print(f"  │  ├─ Loss_π:  {epoch_info['loss_actor']:.6f}")
        print(f"  │  └─ Loss_α:  {epoch_info['loss_alpha']:.6f}")
        print(f"  ├─ Total steps so far: {total_steps}")
        print(f"  ├─ Elapsed time: {elapsed:.2f}s")
        print(f"  └─ Epoch completed ✓\n")

    print(f"\n[STEP 5] Training completed")
    print(f"  ├─ Total epochs: {args.epochs}")
    print(f"  ├─ Total steps: {total_steps}")
    print(f"  ├─ Total time: {time.time() - start_time:.2f}s")
    print(f"  └─ Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========================================
    # Save Trained Model and Components
    # ========================================
    # actor: π_φ(u|x) - Policy network (deterministic sampling via tanh(μ))
    # q1, q2: Z_θ₁, Z_θ₂ - Two independent cost distribution networks
    # q1_target, q2_target: Soft-updated target networks for TD stability
    # scalers: Min-max normalization parameters for state/action spaces
    
    print(f"[STEP 6] Saving trained model")
    
    save_dict = {
        "actor_state_dict": agent.actor.state_dict(),
        "q1_state_dict": agent.q1.state_dict(),
        "q2_state_dict": agent.q2.state_dict(),
        "q1_target_state_dict": agent.q1_target.state_dict(),
        "q2_target_state_dict": agent.q2_target.state_dict(),
        "scalers": scalers
    }
    torch.save(save_dict, args.save_pth)
    print(f"  ├─ Saved main model to: {args.save_pth}")

    # Save scalers separate npz for easy loading
    np.savez(args.scalers_npz, x_min=scalers["x_min"], x_max=scalers["x_max"], u_min=scalers["u_min"], u_max=scalers["u_max"])
    print(f"  ├─ Saved scalers to: {args.scalers_npz}")

    print(f"\n[STEP 7] Exporting TorchScript deterministic policy")
    # Save TorchScript deterministic actor wrapper (returns tanh(mean) in [-1,1])
    # For deployment: Deterministic policy π_d(x) = tanh(μ_φ(x))
    # Uses only the mean output (no sampling) for fast inference
    agent.actor.eval()
    example = torch.randn(1, obs_dim).to(DEVICE)
    print(f"  ├─ Switching to eval mode")

    class ActorDetWrapper(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
        def forward(self, x):
            return self.actor.get_deterministic(x)

    wrapper = ActorDetWrapper(agent.actor).eval()
    print(f"  ├─ Created deterministic policy wrapper")
    try:
        print(f"  ├─ Tracing policy with example input...")
        traced = torch.jit.trace(wrapper, example)
        traced.save(args.script_model)
        print(f"  ├─ Saved TorchScript model to: {args.script_model}")
    except Exception as e:
        print(f"  └─ TorchScript trace failed: {e}")

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE ✓")
    print(f"{'='*80}")
    print(f"Summary:")
    print(f"  ├─ Main model: {args.save_pth}")
    print(f"  ├─ Scalers: {args.scalers_npz}")
    print(f"  └─ TorchScript policy: {args.script_model}")
    print(f"\nTotal training time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"{'='*80}\n")

    # Save per-batch loss logs if requested
    if getattr(args, 'log_dir', None) is not None:
        try:
            os.makedirs(args.log_dir, exist_ok=True)
            log_name = f"loss_log_bs{args.batch_size}_ep{args.epochs}.npz"
            log_path = os.path.join(args.log_dir, log_name)
            np.savez(log_path,
                     loss_z1=np.array(loss_log['loss_z1'], dtype=np.float32),
                     loss_z2=np.array(loss_log['loss_z2'], dtype=np.float32),
                     loss_actor=np.array(loss_log['loss_actor'], dtype=np.float32),
                     loss_alpha=np.array(loss_log['loss_alpha'], dtype=np.float32),
                     batch_size=args.batch_size,
                     epochs=args.epochs)
            print(f"Saved loss log to: {log_path}")

            # simple plot of losses across recorded batches
            try:
                import matplotlib.pyplot as plt
                arr_z1 = np.array(loss_log['loss_z1'])
                arr_z2 = np.array(loss_log['loss_z2'])
                arr_actor = np.array(loss_log['loss_actor'])
                fig, ax = plt.subplots(1,1, figsize=(10,5))
                ax.plot(arr_z1, label='Loss_Q1', alpha=0.8)
                ax.plot(arr_z2, label='Loss_Q2', alpha=0.8)
                ax.plot(arr_actor, label='Loss_actor', alpha=0.8)
                ax.set_xlabel('batch index')
                ax.set_ylabel('loss')
                ax.set_title(f'Per-batch losses (bs={args.batch_size}, ep={args.epochs})')
                ax.legend()
                plt.tight_layout()
                plot_path = os.path.join(args.log_dir, f"loss_plot_bs{args.batch_size}_ep{args.epochs}.png")
                fig.savefig(plot_path)
                plt.close(fig)
                print(f"Saved loss plot to: {plot_path}")
            except Exception as e:
                print(f"Failed to save loss plot: {e}")
        except Exception as e:
            print(f"Failed to save loss log: {e}")

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="train_data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_batches_per_epoch", type=int, default=0, help="limit number of batches per epoch for quick experiments (0 = no limit)")
    p.add_argument("--log_dir", type=str, default=None, help="directory to save per-batch loss logs and plots")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--auto_alpha", action="store_true", help="enable automatic alpha tuning")
    p.add_argument("--alpha", type=float, default=0.2, help="fixed alpha (ignored if --auto_alpha is set)")
    p.add_argument("--save_pth", type=str, default="dsac_model.pth")
    p.add_argument("--scalers_npz", type=str, default="scalers.npz")
    p.add_argument("--script_model", type=str, default="sac_pi_model_deterministic.pt")
    args = p.parse_args()
    train_and_save(args)