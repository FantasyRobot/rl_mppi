#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from env_utils import BallEnvironment

# Import SAC components from sac_utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac_utils import SACAgent

def load_model(save_path, state_dim, action_dim):
    """Load a trained SAC agent model."""
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=True
    )
    
    import torch
    checkpoint = torch.load(save_path)
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.q_net1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q_net2.load_state_dict(checkpoint['q2_state_dict'])
    agent.target_q_net1.load_state_dict(checkpoint['target_q1_state_dict'])
    agent.target_q_net2.load_state_dict(checkpoint['target_q2_state_dict'])
    agent.alpha = checkpoint['alpha']
    if agent.auto_entropy_tuning:
        agent.log_alpha = checkpoint['log_alpha']
    
    return agent

def test_sac_ball(model_path, target_pos=None, num_tests=10, max_steps=100):
    """
    Test the trained SAC agent on the ball environment.
    
    Args:
        model_path (str): Path to the trained model
        target_pos (list): Target position for the environment
        num_tests (int): Number of test runs
        max_steps (int): Maximum steps per test run
    """
    # Create environment
    env = BallEnvironment(target_pos=target_pos, max_steps=max_steps)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    agent = load_model(model_path, state_dim, action_dim)
    
    # Test results
    test_results = []
    all_trajectories = []
    
    print(f"Testing agent for {num_tests} runs...")
    
    for test_idx in range(num_tests):
        # Reset environment
        state = env.reset()
        trajectory = [state[:2]]  # Only store position for plotting
        
        total_reward = 0
        steps_taken = 0
        success = False
        distances = []
        
        while True:
            # Get action from agent
            action = agent.select_action(state, evaluate=True)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update variables
            total_reward += reward
            steps_taken += 1
            distances.append(info["distance"])
            trajectory.append(next_state[:2])
            
            # Check if reached target
            if info["distance"] < 0.5:
                success = True
                break
                
            if done:
                break
                
            state = next_state
        
        # Store test results
        test_results.append({
            "test_idx": test_idx + 1,
            "total_reward": total_reward,
            "steps_taken": steps_taken,
            "success": success,
            "final_distance": distances[-1]
        })
        
        all_trajectories.append(np.array(trajectory))
        
        # Print test results
        print(f"Test {test_idx + 1}: Success={success}, Steps={steps_taken}, Reward={total_reward:.2f}, Final Distance={distances[-1]:.4f}")
    
    # Calculate summary statistics
    success_rate = sum(1 for r in test_results if r["success"]) / num_tests
    avg_steps = np.mean([r["steps_taken"] for r in test_results])
    avg_reward = np.mean([r["total_reward"] for r in test_results])
    avg_final_distance = np.mean([r["final_distance"] for r in test_results])
    
    print(f"\nSummary Statistics:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Final Distance: {avg_final_distance:.4f}")
    
    # Plot trajectories
    plot_trajectories(all_trajectories, env.target_pos)
    
    return test_results, all_trajectories

def plot_trajectories(trajectories, target_pos):
    """
    Plot all test trajectories.
    
    Args:
        trajectories (list): List of numpy arrays representing trajectories
        target_pos (list): Target position
    """
    plt.figure(figsize=(10, 10))
    
    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3, label=f'Test {i+1}')
        # Plot start point for each trajectory
        plt.plot(traj[0, 0], traj[0, 1], 's', markersize=8, color='green', label=f'Test {i+1} Start' if i == 0 else "")
    
    # Plot target position
    plt.plot(target_pos[0], target_pos[1], 'rx', markersize=15, label='Target')
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.title('Ball Trajectories - SAC Agent')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper right')
    
    # Save plot
    plt.savefig('ball_trajectories.png')
    print(f"Trajectory plot saved to ball_trajectories.png")
    
    plt.show()

def compare_with_baseline(model_path, target_pos=None, num_tests=5):
    """
    Compare SAC agent with baseline controller.
    
    Args:
        model_path (str): Path to the trained model
        target_pos (list): Target position for the environment
        num_tests (int): Number of comparison runs
    """
    # Create environments
    env_sac = BallEnvironment(target_pos=target_pos, max_steps=100)
    env_baseline = BallEnvironment(target_pos=target_pos, max_steps=100)
    
    # Load SAC agent
    agent = load_model(model_path, env_sac.state_dim, env_sac.action_dim)
    
    print(f"Comparing SAC agent with baseline controller for {num_tests} runs...")
    
    for test_idx in range(num_tests):
        # Reset environments
        state_sac = env_sac.reset()
        state_baseline = env_baseline.reset()
        
        # Set same initial state for both
        state_sac = state_baseline.copy()
        env_sac.state = state_sac
        
        print(f"\nTest {test_idx + 1}:")
        print(f"Initial Position: {state_sac[:2]}")
        
        # Run SAC agent
        sac_reward, sac_steps, sac_distance = run_agent(env_sac, agent, evaluate=True)
        
        # Run baseline controller
        baseline_reward, baseline_steps, baseline_distance = run_baseline_controller(env_baseline)
        
        print(f"SAC - Reward: {sac_reward:.2f}, Steps: {sac_steps}, Final Distance: {sac_distance:.4f}")
        print(f"Baseline - Reward: {baseline_reward:.2f}, Steps: {baseline_steps}, Final Distance: {baseline_distance:.4f}")

def run_agent(env, agent, evaluate=True):
    """Run an agent in the environment."""
    state = env.state.copy()
    total_reward = 0
    steps = 0
    
    while True:
        action = agent.select_action(state, evaluate=evaluate)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if done or info["distance"] < 0.5:
            break
            
        state = next_state
    
    return total_reward, steps, info["distance"]

def run_baseline_controller(env):
    """Run the baseline PD controller in the environment."""
    state = env.state.copy()
    total_reward = 0
    steps = 0
    
    while True:
        current_pos = state[:2]
        target_pos = env.target_pos
        
        # Calculate position error
        pos_error = target_pos - current_pos
        distance = np.linalg.norm(pos_error)
        
        if distance < 0.5:
            # If close to target, stop moving
            linear_vel = 0.0
            angular_vel = 0.0
        else:
            # Calculate desired direction
            desired_dir = pos_error / distance
            
            # Current direction (from theta)
            current_dir = np.array([np.cos(state[4]), np.sin(state[4])])
            
            # Angle error between current and desired direction
            angle_error = np.arctan2(desired_dir[1], desired_dir[0]) - state[4]
            # Wrap angle error to [-Ãâ‚¬, Ãâ‚¬]
            angle_error = ((angle_error + np.pi) % (2 * np.pi)) - np.pi
            
            # PD controller for linear velocity (proportional to distance)
            kp_linear = 0.5
            linear_vel = np.clip(kp_linear * distance, 0, env.linear_vel_bound)
            
            # PD controller for angular velocity (proportional to angle error)
            kp_angular = 2.0
            angular_vel = np.clip(kp_angular * angle_error, -env.angular_vel_bound, env.angular_vel_bound)
        
        action = np.array([linear_vel, angular_vel])
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if done or info["distance"] < 0.5:
            break
            
        state = next_state
    
    return total_reward, steps, info["distance"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SAC agent for ball environment")
    parser.add_argument("--model_path", type=str, default="sac_ball_model.pth", help="Path to the trained model")
    parser.add_argument("--target_x", type=float, default=3.0, help="Target x position")
    parser.add_argument("--target_y", type=float, default=3.0, help="Target y position")
    parser.add_argument("--num_tests", type=int, default=10, help="Number of test runs")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per test run")
    parser.add_argument("--compare", action="store_true", help="Compare with baseline controller")
    
    args = parser.parse_args()
    
    target_pos = [args.target_x, args.target_y]
    
    # Test the model
    test_sac_ball(
        model_path=args.model_path,
        target_pos=target_pos,
        num_tests=args.num_tests,
        max_steps=args.max_steps
    )
    
    # Compare with baseline if requested
    if args.compare:
        compare_with_baseline(
            model_path=args.model_path,
            target_pos=target_pos,
            num_tests=5
        )