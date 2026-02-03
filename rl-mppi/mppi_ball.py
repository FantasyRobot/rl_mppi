#!/usr/bin/env python3

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from env.envball_utils import BallEnvironment

class MPPI:
    """
    Model Predictive Path Integral (MPPI) controller for ball environment.
    """
    
    def __init__(
        self,
        env,
        horizon=20,
        num_samples=100,
        lambda_coeff=1.0,
        noise_std=0.5,
        dt=0.01,
        *,
        vectorized_rollouts: bool = True,
    ):
        """
        Initialize MPPI controller.
        
        Args:
            env: Environment object
            horizon: Control horizon
            num_samples: Number of action sequences to sample
            lambda_coeff: Temperature parameter for weighting
            noise_std: Standard deviation of action noise
            dt: Time step
        """
        self.env = env
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_coeff = lambda_coeff
        self.noise_std = noise_std
        self.dt = dt
        self.vectorized_rollouts = bool(vectorized_rollouts)
        
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        
        # Action limits
        self.action_min = -1.0
        self.action_max = 1.0
        
        # Initialize control sequence
        self.u = np.zeros((self.horizon, self.action_dim))
    
    def compute_cost(self, state, action, target_pos):
        """
        Simple cost function focused on reaching target position.
        
        Args:
            state: Current state [x, y, vx, vy]
            action: Current action [ax, ay]
            target_pos: Target position
            
        Returns:
            cost: Scalar cost value
        """
        # Only focus on position error - extremely strong penalty
        pos_error = state[:2] - target_pos
        pos_cost = np.linalg.norm(pos_error) * 1000.0
        
        # Very small action cost - almost no penalty
        action_cost = 0.001 * np.linalg.norm(action)
        
        total_cost = pos_cost + action_cost
        return total_cost
    
    def simulate_trajectory(self, initial_state, action_sequence, target_pos):
        """
        Simulate a trajectory with given initial state and action sequence using manual dynamics.
        
        Args:
            initial_state: Initial state
            action_sequence: Sequence of actions
            target_pos: Target position
            
        Returns:
            total_cost: Total cost for the trajectory
            trajectory: List of states along the trajectory
        """
        state = initial_state.copy()
        total_cost = 0.0
        trajectory = [state.copy()]
        
        for t in range(self.horizon):
            # Get current state
            x, y, vx, vy = state
            
            # Get action (already scaled in [-1, 1] range)
            action = action_sequence[t]
            ax = action[0] * self.env.acceleration_bound
            ay = action[1] * self.env.acceleration_bound
            
            # Clip acceleration
            ax = np.clip(ax, -self.env.acceleration_bound, self.env.acceleration_bound)
            ay = np.clip(ay, -self.env.acceleration_bound, self.env.acceleration_bound)
            
            # Update dynamics manually
            new_vx = vx + ax * self.dt
            new_vy = vy + ay * self.dt
            new_x = x + new_vx * self.dt + 0.5 * ax * self.dt**2
            new_y = y + new_vy * self.dt + 0.5 * ay * self.dt**2
            
            # Clip to bounds
            new_x = np.clip(new_x, -self.env.pos_bound, self.env.pos_bound)
            new_y = np.clip(new_y, -self.env.pos_bound, self.env.pos_bound)
            new_vx = np.clip(new_vx, -self.env.vel_bound, self.env.vel_bound)
            new_vy = np.clip(new_vy, -self.env.vel_bound, self.env.vel_bound)
            
            next_state = np.array([new_x, new_y, new_vx, new_vy])
            
            # Compute cost
            cost = self.compute_cost(state, action, target_pos)
            total_cost += cost * self.dt
            
            # Update state
            state = next_state.copy()
            trajectory.append(state.copy())
        
        return total_cost, trajectory

    def simulate_trajectories_batch(self, initial_state, action_sequences, target_pos):
        """Vectorized rollout cost for many action sequences.

        Args:
            initial_state: shape (4,)
            action_sequences: shape (N, H, action_dim) in [-1,1]
            target_pos: shape (2,)

        Returns:
            total_costs: shape (N,)
        """
        action_sequences = np.asarray(action_sequences, dtype=np.float32)
        if action_sequences.ndim != 3:
            raise ValueError(f"action_sequences must be 3D (N,H,dim), got shape={action_sequences.shape}")

        n, h, ad = action_sequences.shape
        if h != self.horizon or ad != self.action_dim:
            raise ValueError(
                f"action_sequences shape mismatch: expected (N,{self.horizon},{self.action_dim}), got {action_sequences.shape}"
            )

        s0 = np.asarray(initial_state, dtype=np.float32).reshape(-1)
        if s0.shape[0] < 4:
            raise ValueError(f"initial_state must have 4 elements, got shape={s0.shape}")

        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(2)
        target_x = float(target_pos[0])
        target_y = float(target_pos[1])

        # State for all samples.
        x = np.full((n,), float(s0[0]), dtype=np.float32)
        y = np.full((n,), float(s0[1]), dtype=np.float32)
        vx = np.full((n,), float(s0[2]), dtype=np.float32)
        vy = np.full((n,), float(s0[3]), dtype=np.float32)

        dt = np.float32(self.dt)
        dt2 = np.float32(self.dt * self.dt)

        acc_bound = np.float32(self.env.acceleration_bound)
        pos_bound = np.float32(self.env.pos_bound)
        vel_bound = np.float32(self.env.vel_bound)

        pos_cost_coeff = np.float32(1000.0)
        act_cost_coeff = np.float32(0.001)

        costs = np.zeros((n,), dtype=np.float32)

        for t in range(self.horizon):
            a = action_sequences[:, t, :]

            dx = x - target_x
            dy = y - target_y
            pos_norm = np.sqrt(dx * dx + dy * dy)
            act_norm = np.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1])
            costs += (pos_norm * pos_cost_coeff + act_norm * act_cost_coeff) * dt

            ax = np.clip(a[:, 0] * acc_bound, -acc_bound, acc_bound)
            ay = np.clip(a[:, 1] * acc_bound, -acc_bound, acc_bound)

            new_vx = vx + ax * dt
            new_vy = vy + ay * dt
            new_x = x + new_vx * dt + np.float32(0.5) * ax * dt2
            new_y = y + new_vy * dt + np.float32(0.5) * ay * dt2

            x = np.clip(new_x, -pos_bound, pos_bound)
            y = np.clip(new_y, -pos_bound, pos_bound)
            vx = np.clip(new_vx, -vel_bound, vel_bound)
            vy = np.clip(new_vy, -vel_bound, vel_bound)

        return costs
    
    def get_action(self, current_state, target_pos):
        """
        Compute the optimal action using MPPI.
        
        Args:
            current_state: Current state
            target_pos: Target position
            
        Returns:
            optimal_action: Optimal action to take
        """
        # Sample noise sequences
        noise = np.random.randn(self.num_samples, self.horizon, self.action_dim) * self.noise_std
        
        # Create action sequences with noise
        action_sequences = np.tile(self.u, (self.num_samples, 1, 1)) + noise
        
        # Clip actions to valid range
        action_sequences = np.clip(action_sequences, self.action_min, self.action_max)
        
        # Evaluate all trajectories
        if self.vectorized_rollouts:
            costs = self.simulate_trajectories_batch(current_state, action_sequences, target_pos)
        else:
            costs = np.zeros(self.num_samples)
            for i in range(self.num_samples):
                # Simulate trajectory using manual dynamics (no environment state modification)
                cost, _traj = self.simulate_trajectory(current_state, action_sequences[i], target_pos)
                costs[i] = cost
        
        # Normalize costs and compute weights
        cost_min = np.min(costs)
        weights = np.exp(-(costs - cost_min) / self.lambda_coeff)
        weights /= np.sum(weights)
        
        # Compute weighted average of action sequences
        weighted_actions = np.sum(weights[:, np.newaxis, np.newaxis] * action_sequences, axis=0)
        
        # Update control sequence
        self.u[:-1] = weighted_actions[1:]
        self.u[-1] = weighted_actions[-1]  # Hold last action
        
        # Return first action
        return weighted_actions[0]
    
    def reset(self):
        """
        Reset the control sequence.
        """
        self.u = np.zeros((self.horizon, self.action_dim))

def test_mppi_ball():
    """
    Test MPPI controller on BallEnvironment.
    """
    # Create environment
    target_pos = [5.0, 5.0]
    env = BallEnvironment(target_pos=target_pos)
    
    # Initialize MPPI controller with aggressive parameters
    mppi = MPPI(
        env=env,
        horizon=15,  # Short horizon for fast computation
        num_samples=50,  # Reasonable number of samples
        lambda_coeff=0.01,  # Very small lambda for strong selection
        noise_std=2.0,  # Large noise for aggressive exploration
        dt=0.01
    )
    
    # Test parameters
    max_steps = 1000  # Increased steps for reaching target
    initial_state = env.reset(initial_state=[-3.0, -3.0, 0.0, 0.0])  # Closer initial position
    state = initial_state.copy()
    
    # Store trajectory for plotting
    trajectory = [state.copy()]
    
    print("Testing MPPI controller on BallEnvironment...")
    print(f"Target position: {target_pos}")
    print(f"Initial state: {state}")
    
    for step in range(max_steps):
        print(f"\nStep {step}:")
        print(f"  Current state: {state}")
        print(f"  Distance to target: {np.linalg.norm(state[:2] - target_pos):.4f}")
        
        # Compute action using MPPI
        action = mppi.get_action(state, target_pos)
        print(f"  Computed action: {action}")
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        print(f"  Next state: {next_state}")
        print(f"  Reward: {reward}")
        
        # Update state
        state = next_state.copy()
        trajectory.append(state.copy())
        
        # Check if close to target
        distance = np.linalg.norm(state[:2] - target_pos)
        if distance < 0.5:
            print(f"\nReached target at step {step}!")
            break
    
    print(f"\nTesting completed!")
    
    # Plot results
    trajectory = np.array(trajectory)
    
    plt.figure(figsize=(10, 10))
    
    # Plot position trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Ball trajectory')
    
    # Plot target
    plt.plot(target_pos[0], target_pos[1], 'go', markersize=10, label='Target')
    
    # Plot initial position
    plt.plot(initial_state[0], initial_state[1], 'ro', markersize=8, label='Initial position')
    
    # Add velocity vectors at intervals
    interval = 50
    for i in range(0, len(trajectory), interval):
        vx, vy = trajectory[i, 2], trajectory[i, 3]
        plt.arrow(trajectory[i, 0], trajectory[i, 1], vx*0.5, vy*0.5, 
                 head_width=0.2, head_length=0.2, fc='r', ec='r')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('MPPI Controller - Ball Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    
    # Save plot with absolute path
    import os
    plot_path = os.path.join(os.path.dirname(__file__), 'mppi_ball_trajectory.png')
    plt.savefig(plot_path)
    print(f"Trajectory plot saved as '{plot_path}'")
    
    # Close plot to avoid memory issues
    plt.close()
    
    print(f"Final state: {state}")
    print(f"Final distance to target: {np.linalg.norm(state[:2] - target_pos):.4f}")
    print(f"Total steps: {step + 1}")
    
    return trajectory

if __name__ == "__main__":
    test_mppi_ball()