#!/usr/bin/env python3

import os
import numpy as np
import casadi as ca

# Import the unified environment class
from env_utils import Environment

# ========================================# Data Generation# ========================================

def generate_controlled_actions(current_state, target_state, num_steps, action_dim):
    """
    Generate controlled actions for data generation using Model Predictive Control (MPC) with CasADi.
    
    Args:
        current_state (np.ndarray): Current state containing position and velocity [position (6), velocity (6)]
        target_state (np.ndarray): Target state containing position and velocity [position (6), velocity (6)]
        num_steps (int): Number of steps to generate
        action_dim (int): Dimension of each action
        
    Returns:
        tuple:
            - np.ndarray: Array of actions with shape (num_steps, action_dim)
            - int: Actual number of steps where MPC calculated actions before early termination
    """
    actions = []
    
    # Extract position and velocity from current_state
    current_position = current_state[:6].copy()
    current_velocity = current_state[6:].copy()
    
    # Environment parameters (matching env_utils.py)
    step_time = 0.1  # Time step
    acc_limit = 10.0  # Acceleration limit
    velocity_limit = 1.0  # Velocity limit
    
    # MPC parameters
    horizon = 3  # Shorter horizon for faster computation
    
    print("Setting up CasADi MPC optimal control problem...")
    
    # Define symbolic variables
    x = ca.SX.sym('x', 12)  # State: [position (6), velocity (6)]
    u = ca.SX.sym('u', 6)   # Control: target position (6)
    
    # Simple linear dynamics approximation for faster computation
    # Using PD controller to compute acceleration from target position
    kp = 10.0
    kd = 3.0
    pos_error = u - x[:6]
    acceleration = kp * pos_error + kd * (-x[6:])
    
    # Clip acceleration to reasonable limits
    acc_limit = 10.0
    acceleration_clipped = ca.fmin(ca.fmax(acceleration, -acc_limit), acc_limit)
    
    # State transition: x_next = x + [v, a_clipped] * step_time
    x_next = ca.vertcat(
        x[:6] + x[6:] * step_time + 0.5 * acceleration_clipped * step_time**2,
        x[6:] + acceleration_clipped * step_time
    )
    
    # Create CasADi dynamics function
    dynamics = ca.Function('dynamics', [x, u], [x_next])
    
    # Set up MPC optimization problem once
    try:
        opti = ca.Opti()
        
        # Decision variables for controls over the horizon
        U = opti.variable(6, horizon)
        
        # Initial state parameter (updated at each iteration)
        x0_param = opti.parameter(12)
        
        # Initialize cost
        total_cost = 0
        
        # Simulate dynamics over horizon and compute cost
        x_mpc = x0_param
        for k in range(horizon):
            u_k = U[:, k]
            
            # Apply dynamics
            x_mpc = dynamics(x_mpc, u_k)
            
            # Calculate cost for this step
            distance_cost = ca.sumsqr(x_mpc[:6] - target_state[:6])  # Use only position part of target
            control_cost = 0 * ca.sumsqr(u_k)  # Weight for control effort
            total_cost += distance_cost + control_cost
        
        # Define the optimization problem
        opti.minimize(total_cost)

        # Terminal constraint: zero velocity at end of horizon
       # opti.subject_to(x_mpc[6:] == 0)

        # Add bounds for target position (U)
        position_limit = 3.0
        opti.subject_to(opti.bounded(-position_limit, U, position_limit))
        opti.subject_to(opti.bounded(-velocity_limit, x_mpc[6:], velocity_limit))
        
        # Configure solver with faster settings
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 5
        }
        opti.solver('ipopt', opts)
        
        mpc_solver_ready = True
    except Exception as e:
        print(f"  MPC setup failed: {e}")
        print("  Falling back to PD control")
        mpc_solver_ready = False
    
    print("Generating optimal actions...")
    
    actual_steps = 0
    
    for i in range(num_steps):
        actual_steps += 1
        # Check if we've reached the target
        current_distance = np.linalg.norm(current_position - target_state)
        if current_distance < 0.1:  # Target tolerance
            print(f"  Reached target at step {i+1}, filling remaining trajectory with last action")
            # Fill remaining trajectory with last action
            if actions:
                last_action = np.zeros(action_dim)
                remaining_actions = np.tile(last_action, (num_steps - i, 1))
                actions = np.concatenate([actions, remaining_actions])
            break
        
        # Try MPC control first
        if mpc_solver_ready:
            try:
                # Set current state as parameter
                current_state_combined = np.concatenate([current_position, current_velocity])
                opti.set_value(x0_param, current_state_combined)
                
                # Solve optimization problem
                sol = opti.solve()
                optimal_controls = sol.value(U)
                target_position = optimal_controls[:, 0]  # Take first control action (target position)
            except Exception as e:
                # Fallback to target position calculation if MPC fails
                # Simple target position: move towards env target with some proportional distance
                delta = target_state - current_position
                target_position = current_position + delta * 0.5  # Move 50% towards target
                print(f"  MPC failed at step {i+1}, using simple target position: {str(e)[:50]}...")
        else:
            # Use simple target position if MPC setup failed
            delta = target_state - current_position
            target_position = current_position + delta * 0.5  # Move 50% towards target
        
        # Add exploration noise for better data diversity
        exploration_noise = np.random.normal(0, 0.1, size=action_dim)
        target_position += exploration_noise
        
        # Normalize target position to [-1, 1] range for action
        position_limit = 3.0
        action = target_position / position_limit
        
        # Clip action to valid range for the environment
        action = np.clip(action, -1.0, 1.0)
        
        # Store the action
        actions.append(action)
        
        # Update current position and velocity using the actual environment dynamics
        # First scale back action from [-1, 1] to environment's position range
        position_limit = 3.0
        target_position = action * position_limit
        
        # Clip to ensure within bounds
        target_position = np.clip(target_position, -position_limit, position_limit)
        
        # Use PD controller to compute acceleration
        kp = 10.0
        kd = 3.0
        pos_error = target_position - current_position
        acceleration = kp * pos_error + kd * (-current_velocity)
        
        # Clip accelerations
        acc_clipped = np.clip(acceleration, -acc_limit, acc_limit)
        
        # Update velocity and position
        new_velocity = current_velocity + acc_clipped * step_time
        new_velocity = np.clip(new_velocity, -velocity_limit, velocity_limit)
        new_position = current_position + new_velocity * step_time + 0.5 * acc_clipped * step_time ** 2
        
        # Update state variables for next iteration
        current_position = new_position
        current_velocity = new_velocity
        
        # Print progress
        if (i + 1) % 10 == 0 or i == num_steps - 1:
            distance = np.linalg.norm(current_position - target_state)
            print(f"  Step {i+1}/{num_steps}, distance to target: {distance:.4f}")
    
    # Ensure we have exactly num_steps actions
    if len(actions) < num_steps:
        # If no actions were generated yet, use zeros
        if not actions:
            actions = np.zeros((num_steps, action_dim))
        else:
            # Fill with last action
            #last_action = actions[-1]
            last_action = np.zeros(action_dim)
            remaining = num_steps - len(actions)
            actions = np.concatenate([actions, np.tile(last_action, (remaining, 1))])
    
    print("Optimal action generation completed!")
    print(f"  Actual MPC-calculated steps: {actual_steps}")
    return np.array(actions), actual_steps


def test_controlled_actions():
    """
    Test function for generate_controlled_actions that visualizes the generated trajectory.
    """
    import matplotlib.pyplot as plt
    
    # Parameters for the test
    num_steps = 200  # Reduced for faster testing
    action_dim = 6
    target_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    print("Testing generate_controlled_actions function...")
    print(f"Target position: {target_state}")
    print(f"Number of steps: {num_steps}")
    
    # Generate actions using CasADi-based optimal control
    # Create initial state [position (6), velocity (6)]
    initial_position = np.zeros(6)  # Start closer to target
    initial_velocity = np.zeros(6)
    initial_state = np.concatenate([initial_position, initial_velocity])
    
    actions, actual_steps = generate_controlled_actions(initial_state, target_state, num_steps, action_dim)
    
    print("Simulating environment dynamics to generate trajectory...")
    
    # Simulate the environment dynamics to get the full trajectory
    # Start with the same initial state as in generate_controlled_actions
    current_position = initial_position  # Recreate random start position
    current_velocity =  initial_velocity  # Same as in the function
    
    # Environment parameters (exactly matching the function)
    step_time = 0.1
    acc_limit = 10.0
    velocity_limit = 1.0
    
    # Store trajectory data
    positions = [current_position.copy()]
    velocities = [current_velocity.copy()]
    
    # Simulate each step
    for i in range(num_steps):
        action = actions[i]
        
        # Apply the same dynamics as in the function
        # Scale back action from [-1, 1] to environment's position range
        position_limit = 3.0
        target_position = action * position_limit
        
        # Clip to ensure within bounds
        target_position = np.clip(target_position, -position_limit, position_limit)
        
        # Use PD controller to compute acceleration
        kp = 10.0
        kd = 3.0
        pos_error = target_position - current_position
        acceleration = kp * pos_error + kd * (-current_velocity)
        
        # Clip accelerations
        acc_clipped = np.clip(acceleration, -acc_limit, acc_limit)
        
        # Update velocity and position
        new_velocity = current_velocity + acc_clipped * step_time
        new_velocity = np.clip(new_velocity, -velocity_limit, velocity_limit)
        new_position = current_position + new_velocity * step_time + 0.5 * acc_clipped * step_time ** 2
        
        # Update state
        current_position = new_position
        current_velocity = new_velocity
        
        # Store data
        positions.append(new_position.copy())
        velocities.append(new_velocity.copy())
    
    # Convert to numpy arrays for easier plotting
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    # Calculate distances to target over time
    distances = np.linalg.norm(positions - target_state, axis=1)
    
    print("Creating visualization...")
    
    # Create a comprehensive plot for position dimensions
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Trajectory Generated by CasADi Optimal Control', fontsize=16)
    
    # Plot all 6 position dimensions in the 3x2 grid
    for i in range(6):
        row = i // 2
        col = i % 2
        axes[row, col].plot(positions[:, i], label=f'Position x{i+1}')
        axes[row, col].axhline(y=target_state[i], color='r', linestyle='--', label=f'Target x{i+1}')
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel(f'Position x{i+1}')
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    # Plot distance to target
    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.xlabel('Time Step')
    plt.ylabel('Distance to Target')
    plt.title('Distance to Target Over Time')
    plt.grid(True)
    
    # Save plots
    plt.tight_layout()
    plt.savefig('trajectory_test_results.png')
    
    # Show plots
    plt.show()
    
    # Print summary statistics
    print(f"\nTrajectory Summary:")
    print(f"Initial position: {positions[0]}")
    print(f"Final position: {positions[-1]}")
    print(f"Initial distance to target: {distances[0]:.4f}")
    print(f"Final distance to target: {distances[-1]:.4f}")
    print(f"Average distance: {np.mean(distances):.4f}")
    print(f"Maximum velocity: {np.max(np.abs(velocities)):.4f}")
    print(f"Maximum acceleration: {np.max(np.abs(actions)):.4f}")
    
    print("\nTest completed successfully!")
    print("Plot saved as 'trajectory_test_results.png'")


def generate_training_data(env, num_steps, output_dir="train_data"):
    """
    Generate training data using the environment model.
    
    Args:
        env: Data generation environment
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
    
    step_time = env.step_time
    # Ensure we have a valid target state
    target_state = env.target_state
    if target_state is None:
        target_state = np.zeros(6)  # Default target position is origin
        env.target_state = target_state
    
    print(f"Generating {num_steps} steps of training data...")
    print(f"Target state: {target_state}")
    
    i = 0
    while i < num_steps:
        if i % 200 == 0:
            # Generate random initial state for diversity
            state_type = np.random.randint(0, 3)
            
            if state_type == 0:
                # Far from target
                random_offset = np.random.randn(6) * 6.0
                initial_position = target_state + random_offset
            elif state_type == 1:
                # Medium distance from target
                random_offset = np.random.randn(6) * 2.0
                initial_position = target_state + random_offset
            else:
                # Close to target
                random_offset = np.random.randn(6) * 0.5
                initial_position = target_state + random_offset
            
            # Generate random initial velocity
            initial_velocity = np.random.randn(6) * 0.5
            
            initial_state = np.zeros(12)
            initial_state[:6] = initial_position
            initial_state[6:] = initial_velocity
            
            state = env.reset(initial_state=initial_state)
            current_position = state[:6]
            current_velocity = state[6:]
            
            # Generate a batch of actions using MPC
            batch_size = min(50, num_steps - i)
            mpc_actions, actual_steps = generate_controlled_actions(
                state,
                target_state,
                batch_size,
                env.action_dim
            )
            batch_idx = 0
        
        # Use MPC-generated action from the batch
        # Regenerate batch if needed
        if batch_idx >= len(mpc_actions):
            batch_size = min(50, num_steps - i)
            mpc_actions, actual_steps = generate_controlled_actions(
                state,
                target_state,
                batch_size,
                env.action_dim
            )
            batch_idx = 0
        
        action = mpc_actions[batch_idx]
        batch_idx += 1
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Store data
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        
        # Update state variables
        state = next_state
        current_position = state[:6]
        current_velocity = state[6:]
        
        i += 1
        
        # Progress update
        if i % 1000 == 0 or i == num_steps:
            print(f"  Generated {i}/{num_steps} steps")
            
        # Check if close to target
        distance = np.linalg.norm(current_position - target_state)
        if distance < 0.5 and i % 200 == 50:
            # Generate new initial state to continue exploration
            random_offset = np.random.randn(6) * 4.0
            initial_position = target_state + random_offset
            initial_velocity = np.random.randn(6) * 0.5
            
            initial_state = np.zeros(12)
            initial_state[:6] = initial_position
            initial_state[6:] = initial_velocity
            
            state = env.reset(initial_state=initial_state)
            current_position = state[:6]
            current_velocity = state[6:]
            
            # Generate new batch of MPC actions
            batch_size = min(50, num_steps - i)
            mpc_actions, actual_steps = generate_controlled_actions(
                state,
                target_state,
                batch_size,
                env.action_dim
            )
            batch_idx = 0
    
    # Convert to numpy arrays
    states_array = np.array(states)
    actions_array = np.array(actions)
    next_states_array = np.array(next_states)
    rewards_array = np.array(rewards)
    
    # Save data in the format matching train_data
    save_data_file(os.path.join(output_dir, "x.txt"), states_array)
    save_data_file(os.path.join(output_dir, "u.txt"), actions_array)
    save_data_file(os.path.join(output_dir, "x_prime.txt"), next_states_array)
    save_reward_file(os.path.join(output_dir, "r.txt"), rewards_array)
    
    print(f"\nTraining data generated successfully!")
    print(f"Saved to: {output_dir}")
    print(f"Number of samples: {num_steps}")
    print(f"State shape: {states_array.shape}")
    print(f"Action shape: {actions_array.shape}")
    print(f"Reward shape: {rewards_array.shape}")

def save_data_file(file_path, data):
    """
    Save data to a file in the format matching the training data.
    
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
    test_controlled_actions()

    parser = argparse.ArgumentParser(description="Generate training data using environment dynamics")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of steps to generate")
    parser.add_argument("--output_dir", type=str, default="train_data_generated", help="Output directory for generated data")
    parser.add_argument("--initial_state", type=float, nargs='*', default=None, help="Initial state for data generation")
    parser.add_argument("--target_state", type=float, nargs='*', default=None, help="Target state for reward calculation")
    
    args = parser.parse_args()
    
    # Process initial state if provided
    initial_state = None
    if args.initial_state is not None:
        if len(args.initial_state) != 12:
            print(f"Warning: Expected 12-dimensional initial state, got {len(args.initial_state)} dimensions")
            initial_state = np.array(args.initial_state[:12])
            if len(initial_state) < 12:
                pad = np.zeros(12 - len(initial_state))
                initial_state = np.concatenate([initial_state, pad])
        else:
            initial_state = np.array(args.initial_state)
        print(f"Using custom initial state: {initial_state[:6]}...")
    
    # Process target state if provided
    target_state = None
    if args.target_state is not None:
        if len(args.target_state) != 6:
            print(f"Warning: Expected 6-dimensional target state, got {len(args.target_state)} dimensions")
            target_state = np.array(args.target_state[:6])
            if len(target_state) < 6:
                pad = np.zeros(6 - len(target_state))
                target_state = np.concatenate([target_state, pad])
        else:
            target_state = np.array(args.target_state)
        print(f"Using custom target state: {target_state}")
    
    # Create environment
    env = Environment(target_state=target_state, initial_state=initial_state)
    
    # Generate data
    generate_training_data(env, args.num_steps, args.output_dir)