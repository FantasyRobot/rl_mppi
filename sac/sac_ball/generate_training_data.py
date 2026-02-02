#!/usr/bin/env python3

import os
import numpy as np
from env.envball_utils import BallEnvironment

def generate_training_data(env, num_steps, output_dir="train_data"):
    """
    Generate training data using the ball environment.
    
    Args:
        env: Ball environment instance
        num_steps (int): Number of steps to generate
        output_dir (str): Directory to save the generated data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store data
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    
    print(f"Generating {num_steps} steps of training data...")
    print(f"Target position: {env.target_pos}")
    
    # Add debug flag for testing
    debug_mode = False
    
    i = 0
    state_history = []  # Track states to avoid redundant positions
    
    while i < num_steps:
        # More frequent state resets with varied scenarios
        reset_interval = np.random.choice([50, 100, 150])  # Varied reset intervals
        if i % reset_interval == 0:
            # Generate completely random initial states within environment bounds
            initial_state = np.zeros(4)
            # Random position within bounds
            initial_state[0] = np.random.uniform(-env.pos_bound, env.pos_bound)
            initial_state[1] = np.random.uniform(-env.pos_bound, env.pos_bound)
            # Random velocity within bounds
            initial_state[2] = np.random.uniform(-env.vel_bound, env.vel_bound)
            initial_state[3] = np.random.uniform(-env.vel_bound, env.vel_bound)
            
            # Reset environment with new random state
            env.reset(initial_state=initial_state)
        
        # Get current state (only [x, y, vx, vy])
        state = env.state.copy()
        
        # Calculate current position and distance to target
        current_pos = state[:2]
        distance = np.linalg.norm(env.target_pos - current_pos)
        
        # Generate completely random action (acceleration) within [-1, 1] range
        action = np.random.uniform(-1.0, 1.0, size=env.action_dim)
        
        # Debug: print state and action information every 100 steps
        if debug_mode and i % 100 == 0:
            print(f"Step {i}: Current pos: {current_pos}, Distance to target: {distance:.3f}")
            print(f"  State: {state}, Action: {action}")
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Store data
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        
        i += 1
        
        # Progress update
        if i % 1000 == 0 or i == num_steps:
            print(f"  Generated {i}/{num_steps} steps")
            
        # Reset environment if done
        if done:
            # Generate new random initial state for next episode
            initial_state = np.zeros(4)
            initial_state[0] = np.random.uniform(-env.pos_bound, env.pos_bound)
            initial_state[1] = np.random.uniform(-env.pos_bound, env.pos_bound)
            initial_state[2] = np.random.uniform(-env.vel_bound, env.vel_bound)
            initial_state[3] = np.random.uniform(-env.vel_bound, env.vel_bound)
            env.reset(initial_state=initial_state)

    # Convert to numpy arrays
    states_array = np.array(states)
    actions_array = np.array(actions)
    next_states_array = np.array(next_states)
    rewards_array = np.array(rewards)
    dones_array = np.array(dones)
    
    # Save data
    np.save(os.path.join(output_dir, "states.npy"), states_array)
    np.save(os.path.join(output_dir, "actions.npy"), actions_array)
    np.save(os.path.join(output_dir, "next_states.npy"), next_states_array)
    np.save(os.path.join(output_dir, "rewards.npy"), rewards_array)
    np.save(os.path.join(output_dir, "dones.npy"), dones_array)
    
    print(f"\nTraining data generated successfully!")
    print(f"Saved to: {output_dir}")
    print(f"Number of samples: {num_steps}")
    print(f"State shape: {states_array.shape}")
    print(f"Action shape: {actions_array.shape}")
    print(f"Reward shape: {rewards_array.shape}")

def save_data_file(file_path, data):
    """
    Save data to a file in a format that matches the SAC training data format.
    
    Args:
        file_path (str): Path to save the file
        data (np.ndarray): Data to save, shape (n_samples, n_dimensions)
    """
    with open(file_path, "w") as f:
        for i in range(data.shape[0]):
            # Convert to string with comma separation and trailing comma
            line = ",".join([f"{x:.8f}" for x in data[i]]) + ",\n"
            f.write(line)

def save_reward_file(file_path, rewards):
    """
    Save rewards to a file in the format matching the training data.
    
    Args:
        file_path (str): Path to save the file
        rewards (np.ndarray): Rewards to save, shape (n_samples,)
    """
    with open(file_path, "w") as f:
        for reward in rewards:
            f.write(f"{reward:.8f}\n")

# ========================================# Main# ========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data using ball environment")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of steps to generate")
    parser.add_argument("--output_dir", type=str, default="train_data", help="Output directory for generated data")
    parser.add_argument("--target_x", type=float, default=3.0, help="Target x position")
    parser.add_argument("--target_y", type=float, default=3.0, help="Target y position")
    
    args = parser.parse_args()
    
    # Create environment
    target_pos = [args.target_x, args.target_y]
    env = BallEnvironment(target_pos=target_pos, max_steps=200)
    
    # Generate data
    generate_training_data(env, args.num_steps, args.output_dir)