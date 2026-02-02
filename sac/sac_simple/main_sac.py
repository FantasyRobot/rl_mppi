#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# Set matplotlib to non-interactive mode for headless environments
plt.switch_backend('Agg')
import sys
import os

# 添加根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sac_utils import (
    PolicyNetwork, QNetwork, DEVICE, EPS,
    fit_minmax_scalers, minmax_scale_to_minus1_1,
    scale_state, scale_action,
    SACAgent, ReplayBuffer
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ========================================# Data Generation# ========================================
def generate_sine_wave_data(n_points=1000, noise_level=0.05):
    """Generate sine wave data with noise for testing"""
    t = np.linspace(0, 10*np.pi, n_points)
    x = np.sin(t)
    y = np.cos(t)
    
    # Add noise
    x += np.random.normal(0, noise_level, n_points)
    y += np.random.normal(0, noise_level, n_points)
    
    return t, x, y

class PredictionEnvironment:
    """Environment for time series prediction"""
    def __init__(self, x, y, lookback=5):
        self.x = x
        self.y = y
        self.lookback = lookback
        self.current_step = lookback
        
        # Precompute normalization parameters
        self.x_mean = np.mean(x)
        self.x_std = np.std(x)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
    def reset(self):
        self.current_step = self.lookback
        state = np.concatenate([self.x[self.current_step-self.lookback:self.current_step], 
                               self.y[self.current_step-self.lookback:self.current_step]])
        # Normalize state
        state[:self.lookback] = (state[:self.lookback] - self.x_mean) / (self.x_std + 1e-8)
        state[self.lookback:] = (state[self.lookback:] - self.y_mean) / (self.y_std + 1e-8)
        return state
        
    def step(self, action):
        # Get the next actual point
        next_x = self.x[self.current_step]
        next_y = self.y[self.current_step]
        next_actual = np.array([next_x, next_y])
        
        # Calculate reward with improved scaling and positive reinforcement
        mse = np.mean((action - next_actual)**2)
        
        # Positive reward for good predictions, scaled reward range
        if mse < 0.01:
            reward = 1.0  # Strong positive reward for very good predictions
        elif mse < 0.1:
            reward = 0.5  # Moderate positive reward for good predictions
        elif mse < 0.2:
            reward = 0.0  # Neutral reward for acceptable predictions
        else:
            reward = -1.0  # Negative reward for bad predictions
        
        # Clip reward to stable range
        reward = np.clip(reward, -2.0, 2.0)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.x) - 1
        
        # Get next state
        if not done:
            next_state = np.concatenate([self.x[self.current_step-self.lookback:self.current_step], 
                                        self.y[self.current_step-self.lookback:self.current_step]])
            # Normalize next state
            next_state[:self.lookback] = (next_state[:self.lookback] - self.x_mean) / (self.x_std + 1e-8)
            next_state[self.lookback:] = (next_state[self.lookback:] - self.y_mean) / (self.y_std + 1e-8)
        else:
            next_state = np.zeros((self.lookback * 2,))  # Dummy state
        
        return next_state, reward, done, {"actual": next_actual}

# ========================================# Training Function# ========================================
def train_sac_agent(agent, replay_buffer, num_epochs=100, batch_size=64):
    losses = {
        "q1_loss": [],
        "q2_loss": [],
        "policy_loss": [],
        "alpha": []
    }
    
    for epoch in range(num_epochs):
        loss_info = agent.update(replay_buffer, batch_size)
        
        losses["q1_loss"].append(loss_info["q1_loss"])
        losses["q2_loss"].append(loss_info["q2_loss"])
        losses["policy_loss"].append(loss_info["policy_loss"])
        losses["alpha"].append(loss_info["alpha"])
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Q1 Loss: {loss_info['q1_loss']:.6f}, Q2 Loss: {loss_info['q2_loss']:.6f}, Policy Loss: {loss_info['policy_loss']:.6f}, Alpha: {loss_info['alpha']:.6f}")
    
    return losses

# ========================================# Testing and Prediction Function# ========================================
def test_sac_agent(agent, test_states):
    predictions = []
    
    with torch.no_grad():
        for state in test_states:
            action = agent.select_action(state, evaluate=True)
            predictions.append(action)
    
    return np.array(predictions)

# ========================================# Plotting Functions# ========================================
def plot_training_losses(losses):
    """Plot training losses"""
    plt.figure(figsize=(15, 10))
    
    # Plot Q losses
    plt.subplot(2, 3, 1)
    plt.plot(losses["q1_loss"])
    plt.title("Q1 Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    
    plt.subplot(2, 3, 2)
    plt.plot(losses["q2_loss"])
    plt.title("Q2 Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    
    # Plot policy loss
    plt.subplot(2, 3, 3)
    plt.plot(losses["policy_loss"])
    plt.title("Policy Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    
    # Plot alpha
    plt.subplot(2, 3, 4)
    plt.plot(losses["alpha"])
    plt.title("Alpha")
    plt.xlabel("Training Steps")
    plt.ylabel("Alpha")
    
    # Plot episode reward
    plt.subplot(2, 3, 5)
    plt.plot(losses["episode_reward"])
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # Plot smoothed reward
    if len(losses["episode_reward"]) > 10:
        smoothed_reward = np.convolve(losses["episode_reward"], np.ones(10)/10, mode='valid')
        plt.subplot(2, 3, 6)
        plt.plot(range(len(smoothed_reward)), smoothed_reward)
        plt.title("Smoothed Episode Reward (Window=10)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
    
    plt.tight_layout()
    plt.savefig("training_losses.png")
    plt.show()

def plot_predictions(t, x, y, predictions, lookback=5):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t[lookback+1:len(predictions)+lookback+1], x[lookback+1:len(predictions)+lookback+1], label="Actual X")
    plt.plot(t[lookback+1:len(predictions)+lookback+1], predictions[:, 0], label="Predicted X", linestyle="--")
    plt.title("X Component: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(t[lookback+1:len(predictions)+lookback+1], y[lookback+1:len(predictions)+lookback+1], label="Actual Y")
    plt.plot(t[lookback+1:len(predictions)+lookback+1], predictions[:, 1], label="Predicted Y", linestyle="--")
    plt.title("Y Component: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()

def plot_phase_space(x, y, predictions, lookback=5):
    """Plot phase space (X vs Y)"""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(x[lookback+1:len(predictions)+lookback+1], y[lookback+1:len(predictions)+lookback+1], label="Actual", alpha=0.7)
    plt.scatter(predictions[:, 0], predictions[:, 1], label="Predicted", alpha=0.7, marker="x")
    plt.title("Phase Space: Actual vs Predicted")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("phase_space.png")
    plt.show()

# ========================================# Main Function# ========================================
def main():
    print("Starting SAC Algorithm Implementation")
    
    # 1. Generate data
    print("\n1. Generating sine wave data...")
    t, x, y = generate_sine_wave_data(n_points=1000, noise_level=0.05)
    lookback = 5
    
    # 2. Initialize environment and replay buffer
    print("\n2. Initializing environment and replay buffer...")
    env = PredictionEnvironment(x, y, lookback=lookback)
    replay_buffer = ReplayBuffer(max_size=100000)
    
    # 3. Initialize SAC agent
    print("\n3. Initializing SAC agent...")
    state_dim = lookback * 2  # x and y values for lookback points
    action_dim = 2  # Predict next x and y
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=32,  # Further simplify network structure for stability
        learning_rate=1e-4,  # Lower learning rate to prevent divergence
        alpha=0.1,  # Reduce entropy for less exploration
        gamma=0.9,  # Lower gamma for more immediate rewards
        tau=0.01,  # Slower target network update
        auto_entropy_tuning=False  # Disable automatic entropy tuning for stability
    )
    
    # 4. Fill replay buffer with initial interactions
    print("\n4. Filling replay buffer with initial data...")
    for episode in range(50):
        state = env.reset()
        done = False
        
        while not done:
            # Use random actions to explore
            action = np.random.uniform(-1, 1, action_dim)
            next_state, reward, done, _ = env.step(action)
            
            # Add to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
    
    # 5. Train the agent
    print("\n5. Training SAC agent...")
    num_episodes = 50  # Reduce training episodes to prevent overfitting
    batch_size = 256  # Larger batch size for more stable updates
    update_frequency = 2  # Update more frequently for faster learning
    
    losses = {
        "q1_loss": [],
        "q2_loss": [],
        "policy_loss": [],
        "alpha": [],
        "episode_reward": []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Select action with exploration
            action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Add experience to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent periodically
            step_count += 1
            if len(replay_buffer) > batch_size and step_count % update_frequency == 0:
                loss_info = agent.update(replay_buffer, batch_size)
                
                # Record losses
                losses["q1_loss"].append(loss_info["q1_loss"])
                losses["q2_loss"].append(loss_info["q2_loss"])
                losses["policy_loss"].append(loss_info["policy_loss"])
                losses["alpha"].append(loss_info["alpha"])
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Record episode reward
        losses["episode_reward"].append(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(losses["episode_reward"][-10:])
            print(f"Episode {episode}/{num_episodes} - Avg Reward: {avg_reward:.3f}")
            if episode > 0:
                # Print latest loss info if available
                if len(losses["q1_loss"]) > 0:
                    print(f"  Latest - Q1 Loss: {losses['q1_loss'][-1]:.6f}, Q2 Loss: {losses['q2_loss'][-1]:.6f}, Policy Loss: {losses['policy_loss'][-1]:.6f}")
    
    # 5. Plot training losses
    print("\n5. Plotting training losses...")
    plot_training_losses(losses)
    
    # 6. Test the agent
    print("\n6. Testing SAC agent...")
    
    # Reset environment for testing
    state = env.reset()
    done = False
    test_predictions = []
    actual_values = []
    
    while not done:
        # Use deterministic action for testing
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        
        # Store prediction and actual value
        test_predictions.append(action)
        actual_values.append([env.x[env.current_step-1], env.y[env.current_step-1]])
        
        state = next_state
    
    # Convert to numpy arrays
    test_predictions = np.array(test_predictions)
    actual_values = np.array(actual_values)
    
    # 7. Plot prediction results
    print("\n7. Plotting prediction results...")
    
    # Update plot functions to handle the new data format
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t[lookback:lookback+len(actual_values)], actual_values[:, 0], label="Actual X")
    plt.plot(t[lookback:lookback+len(test_predictions)], test_predictions[:, 0], label="Predicted X", linestyle="--")
    plt.title("X Component: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(t[lookback:lookback+len(actual_values)], actual_values[:, 1], label="Actual Y")
    plt.plot(t[lookback:lookback+len(test_predictions)], test_predictions[:, 1], label="Predicted Y", linestyle="--")
    plt.title("Y Component: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.show()
    
    # Plot phase space
    plt.figure(figsize=(10, 8))
    plt.scatter(actual_values[:, 0], actual_values[:, 1], label="Actual", alpha=0.7)
    plt.scatter(test_predictions[:, 0], test_predictions[:, 1], label="Predicted", alpha=0.7, marker="x")
    plt.title("Phase Space: Actual vs Predicted")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("phase_space.png")
    plt.show()
    
    print("\nSAC Algorithm Implementation Complete!")

if __name__ == "__main__":
    main()