#!/usr/bin/env python3

import sys
import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.envball_utils import BallEnvironment

class BallMPC:
    """
    Model Predictive Control (MPC) controller for 2D ball environment using CasADi.
    This version focuses on numerical stability.
    """
    
    def __init__(self, env, horizon=10, dt=0.01, solver_opts=None):
        """
        Initialize MPC controller.
        
        Args:
            env: BallEnvironment object
            horizon: Control horizon
            dt: Time step
            solver_opts: Solver options dictionary
        """
        self.env = env
        self.horizon = horizon
        self.dt = dt
        
        self.state_dim = env.state_dim  # [x, y, vx, vy]
        self.action_dim = env.action_dim  # [ax, ay]
        
        # State and control bounds
        self.state_bounds = {
            'x_min': -env.pos_bound,
            'x_max': env.pos_bound,
            'y_min': -env.pos_bound,
            'y_max': env.pos_bound,
            'vx_min': -env.vel_bound,
            'vx_max': env.vel_bound,
            'vy_min': -env.vel_bound,
            'vy_max': env.vel_bound
        }
        
        self.control_bounds = {
            'ax_min': -env.acceleration_bound,
            'ax_max': env.acceleration_bound,
            'ay_min': -env.acceleration_bound,
            'ay_max': env.acceleration_bound
        }
        
        # Target position (will be updated during control)
        self.target_pos = env.target_pos
        
        # Initialize optimization problem
        self.setup_mpc_problem()
        
        # Set solver options for stability
        if solver_opts is None:
            solver_opts = {
                'ipopt.max_iter': 50,
                'ipopt.print_level': 0,
                'ipopt.tol': 1e-3,
                'ipopt.acceptable_tol': 1e-2,
                'ipopt.acceptable_iter': 5,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-9,
                'ipopt.warm_start_bound_frac': 1e-9,
                'print_time': 0
            }
        
        # Create solver instance
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp, solver_opts)
        
        # Store previous solution for warm start
        self.prev_solution = None
        
        # PID fallback controller parameters
        self.kp_pos = 2.0
        self.kd_pos = 0.5
        self.prev_pos_error = None
    
    def setup_mpc_problem(self):
        """
        Setup MPC optimization problem using CasADi with stability in mind.
        """
        # Define variables with proper dimensions
        x = ca.MX.sym('x', self.state_dim, self.horizon + 1)  # States [x, y, vx, vy]
        u = ca.MX.sym('u', self.action_dim, self.horizon)      # Controls [ax, ay]
        
        # Define parameters
        x0 = ca.MX.sym('x0', self.state_dim)    # Initial state
        target = ca.MX.sym('target', 2)         # Target position
        
        # Define cost weights (lower values for better stability)
        w_pos = 10.0     # Position tracking weight
        w_vel = 5.0      # Velocity regularization weight
        w_control = 1.0  # Control effort weight
        w_terminal = 50.0 # Terminal weight (reduced from 1000)
        
        # Initialize cost
        cost = 0.0
        
        # Dynamics constraints: Simplified Euler integration for stability
        constraints = []
        
        # Initial condition - add each state as separate constraint
        constraints.append(x[0, 0] - x0[0])  # x
        constraints.append(x[1, 0] - x0[1])  # y
        constraints.append(x[2, 0] - x0[2])  # vx
        constraints.append(x[3, 0] - x0[3])  # vy
        
        # Recursive constraints
        for k in range(self.horizon):
            # Current state
            x_k = x[0, k]
            y_k = x[1, k]
            vx_k = x[2, k]
            vy_k = x[3, k]
            
            # Current control
            ax_k = u[0, k]
            ay_k = u[1, k]
            
            # Next state prediction - simplified dynamics (stable)
            x_kp1 = x_k + vx_k * self.dt
            y_kp1 = y_k + vy_k * self.dt
            vx_kp1 = vx_k + ax_k * self.dt
            vy_kp1 = vy_k + ay_k * self.dt
            
            # Add dynamics constraints
            constraints.append(x[0, k+1] - x_kp1)
            constraints.append(x[1, k+1] - y_kp1)
            constraints.append(x[2, k+1] - vx_kp1)
            constraints.append(x[3, k+1] - vy_kp1)
            
            # Add cost for current step
            pos_error = ca.vertcat(x_k - target[0], y_k - target[1])
            control_effort = ca.vertcat(ax_k, ay_k)
            
            cost += w_pos * ca.norm_2(pos_error)**2 + w_control * ca.norm_2(control_effort)**2
        
        # Terminal cost (reduced weight for stability)
        terminal_pos_error = ca.vertcat(x[0, -1] - target[0], x[1, -1] - target[1])
        terminal_vel = ca.vertcat(x[2, -1], x[3, -1])
        
        cost += w_terminal * ca.norm_2(terminal_pos_error)**2 + w_vel * ca.norm_2(terminal_vel)**2
        
        # Define constraint bounds (all equality constraints)
        lbg = [0.0] * len(constraints)
        ubg = [0.0] * len(constraints)
        
        # Define variable bounds
        lbx = []
        ubx = []
        
        # State bounds
        for k in range(self.horizon + 1):
            lbx.extend([
                self.state_bounds['x_min'],  # x
                self.state_bounds['y_min'],  # y
                self.state_bounds['vx_min'], # vx
                self.state_bounds['vy_min']  # vy
            ])
            ubx.extend([
                self.state_bounds['x_max'],  # x
                self.state_bounds['y_max'],  # y
                self.state_bounds['vx_max'], # vx
                self.state_bounds['vy_max']  # vy
            ])
        
        # Control bounds
        for k in range(self.horizon):
            lbx.extend([
                self.control_bounds['ax_min'],  # ax
                self.control_bounds['ay_min']   # ay
            ])
            ubx.extend([
                self.control_bounds['ax_max'],  # ax
                self.control_bounds['ay_max']   # ay
            ])
        
        # Define NLP structure
        self.nlp = {
            'x': ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1)),
            'p': ca.vertcat(x0, target),
            'f': cost,
            'g': ca.vertcat(*constraints)
        }
        
        # Store bounds for later use
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg
    
    def pid_fallback(self, state, target_pos):
        """
        PID fallback controller for when MPC fails.
        
        Args:
            state: Current state [x, y, vx, vy]
            target_pos: Target position [x, y]
            
        Returns:
            control_action: Control action [ax, ay]
        """
        # Extract current position and velocity
        current_pos = state[:2]
        current_vel = state[2:]
        
        # Calculate position error
        pos_error = target_pos - current_pos
        
        # Calculate velocity error (we want to reach target with zero velocity)
        vel_error = -current_vel
        
        # Calculate derivative of position error
        if self.prev_pos_error is not None:
            pos_error_dot = (pos_error - self.prev_pos_error) / self.dt
        else:
            pos_error_dot = np.zeros(2)
        
        # Store previous error
        self.prev_pos_error = pos_error.copy()
        
        # PD control action
        control_action = (
            self.kp_pos * pos_error +  # Position error correction
            self.kd_pos * vel_error    # Velocity error correction
        )
        
        # Clip control action to bounds
        control_action = np.clip(control_action,
                                [-self.control_bounds['ax_min'], -self.control_bounds['ay_min']],
                                [self.control_bounds['ax_max'], self.control_bounds['ay_max']])
        
        return control_action
    
    def generate_initial_guess(self, current_state, target_pos):
        """
        Generate a good initial guess for the optimization problem.
        
        Args:
            current_state: Current state [x, y, vx, vy]
            target_pos: Target position [x, y]
            
        Returns:
            initial_guess: Initial guess for the solver
        """
        if self.prev_solution is not None:
            # Use previous solution as warm start
            return self.prev_solution
        
        # Generate initial guess for states and controls
        x_guess = []
        u_guess = []
        
        # Simple linear trajectory to target
        for k in range(self.horizon + 1):
            t = k * self.dt
            # Smoothly interpolate towards target using exponential decay
            alpha = 1.0 - np.exp(-t / 1.0)  # More stable than linear interpolation
            
            # Position guess
            x_k = current_state[0] + alpha * (target_pos[0] - current_state[0])
            y_k = current_state[1] + alpha * (target_pos[1] - current_state[1])
            
            # Velocity guess (smooth decay to final velocity)
            # Final velocity should be zero at target
            vx_k = current_state[2] * (1.0 - alpha)
            vy_k = current_state[3] * (1.0 - alpha)
            
            x_guess.extend([x_k, y_k, vx_k, vy_k])
        
        # Initial guess for controls: use PID controller
        for k in range(self.horizon):
            # Generate control based on current guess
            t = k * self.dt
            alpha = 1.0 - np.exp(-t / 1.0)
            
            # Current state guess
            x_k = current_state[0] + alpha * (target_pos[0] - current_state[0])
            y_k = current_state[1] + alpha * (target_pos[1] - current_state[1])
            vx_k = current_state[2] * (1.0 - alpha)
            vy_k = current_state[3] * (1.0 - alpha)
            
            # PID control action for this state
            state_guess = [x_k, y_k, vx_k, vy_k]
            control = self.pid_fallback(np.array(state_guess), target_pos)
            u_guess.extend(control.tolist())
        
        # Combine state and control guesses
        initial_guess = np.array(x_guess + u_guess)
        
        # Replace any NaN or inf values with reasonable defaults
        initial_guess = np.nan_to_num(initial_guess, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return initial_guess
    
    def solve_mpc(self, current_state, target_pos):
        """
        Solve MPC optimization problem with improved stability.
        
        Args:
            current_state: Current state [x, y, vx, vy]
            target_pos: Target position [x, y]
            
        Returns:
            optimal_action: Optimal control action [ax, ay]
            predicted_trajectory: Predicted state trajectory
        """
        # Prepare parameters
        params = np.concatenate([current_state, target_pos])
        
        # Generate initial guess
        initial_guess = self.generate_initial_guess(current_state, target_pos)
        
        try:
            # Solve optimization problem
            sol = self.solver(
                x0=initial_guess,
                p=params,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg
            )
            
            # Extract solution
            state_dim_total = self.state_dim * (self.horizon + 1)
            x_sol = sol['x'][:state_dim_total]
            u_sol = sol['x'][state_dim_total:]
            
            # Reshape to proper dimensions
            x_sol_reshaped = ca.reshape(x_sol, self.state_dim, self.horizon + 1)
            u_sol_reshaped = ca.reshape(u_sol, self.action_dim, self.horizon)
            
            # Convert to numpy arrays
            x_sol_np = np.array(x_sol_reshaped.full())
            u_sol_np = np.array(u_sol_reshaped.full())
            
            # Check for NaN or inf values in solution
            if np.isnan(x_sol_np).any() or np.isinf(x_sol_np).any() or \
               np.isnan(u_sol_np).any() or np.isinf(u_sol_np).any():
                raise ValueError("Solution contains NaN or inf values")
            
            # Store solution for warm start only if it's valid
            full_sol = sol['x'].full().flatten()
            if not (np.isnan(full_sol).any() or np.isinf(full_sol).any()):
                self.prev_solution = full_sol
            
            # Return first control action and predicted trajectory
            return u_sol_np[:, 0], x_sol_np.T
            
        except Exception as e:
            #print(f"Warning: MPC solver failed: {e}")
            # Fallback to PID controller
            pid_action = self.pid_fallback(current_state, target_pos)
            return pid_action, None
    
    def reset(self):
        """
        Reset MPC controller.
        """
        self.prev_solution = None
        self.prev_pos_error = None

def test_mpc_ball():
    """
    Test MPC controller on BallEnvironment.
    """
    # Create environment with a reasonable target
    target_pos = [3.0, 3.0]  # Closer target for faster convergence
    env = BallEnvironment(target_pos=target_pos, max_steps=1000)
    
    # Initialize MPC controller with stable parameters
    mpc = BallMPC(
        env=env,
        horizon=5,   # Short horizon for stability
        dt=env.dt
    )
    
    # Initial state
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # Start from origin
    state = env.reset(initial_state=initial_state)
    
    # Store trajectory data
    trajectory = [state.copy()]
    control_history = []
    predicted_trajectories = []
    
    print("Testing Stable MPC controller on BallEnvironment...")
    print(f"Target position: {target_pos}")
    print(f"Initial state: {state}")
    
    max_steps = 500
    target_threshold = 0.2  # Distance threshold to consider target reached
    
    for step in range(max_steps):
        # Solve MPC problem
        control_action, predicted_traj = mpc.solve_mpc(state, target_pos)
        
        # Store control action and predicted trajectory
        control_history.append(control_action)
        if predicted_traj is not None:
            predicted_trajectories.append(predicted_traj)
        
        # Scale control action to [-1, 1] range for environment
        scaled_action = control_action / env.acceleration_bound
        scaled_action = np.clip(scaled_action, -1.0, 1.0)
        
        # Take step in environment
        next_state, reward, done, info = env.step(scaled_action)
        
        # Store trajectory
        trajectory.append(next_state.copy())
        
        # Update state
        state = next_state.copy()
        
        # Calculate distance to target
        distance = np.linalg.norm(state[:2] - target_pos)
        
        # Print progress
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"  Position: ({state[0]:.4f}, {state[1]:.4f})")
            print(f"  Velocity: ({state[2]:.4f}, {state[3]:.4f})")
            print(f"  Control: ({control_action[0]:.4f}, {control_action[1]:.4f})")
            print(f"  Distance to target: {distance:.4f}")
        
        # Check if target reached
        if distance < target_threshold:
            print(f"\nReached target at step {step}!")
            print(f"  Final position: ({state[0]:.4f}, {state[1]:.4f})")
            print(f"  Final velocity: ({state[2]:.4f}, {state[3]:.4f})")
            print(f"  Final distance: {distance:.4f}")
            break
    
    # Convert trajectory to numpy array
    trajectory = np.array(trajectory)
    control_history = np.array(control_history)
    
    print(f"\nTesting completed!")
    print(f"Total steps: {step + 1}")
    
    # Print final results
    final_distance = np.linalg.norm(state[:2] - target_pos)
    print(f"Final state: {state}")
    print(f"Final distance to target: {final_distance:.4f}")
    
    if final_distance < target_threshold:
        print("Success: Target reached!")
    else:
        print("Info: Target not reached within maximum steps.")
    
    # Try to plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot position trajectory
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Actual trajectory')
        ax1.plot(target_pos[0], target_pos[1], 'go', markersize=12, label='Target')
        ax1.plot(initial_state[0], initial_state[1], 'ro', markersize=8, label='Initial position')
        
        # Plot predicted trajectories if available
        if predicted_trajectories:
            num_pred_to_plot = min(5, len(predicted_trajectories))
            indices = np.linspace(0, len(predicted_trajectories)-1, num_pred_to_plot, dtype=int)
            for i in indices:
                pred_traj = predicted_trajectories[i]
                ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'g--', alpha=0.3)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Ball Trajectory (MPC Control)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(-6, 6)
        
        # Plot control inputs over time
        ax2.plot(control_history[:, 0], 'r-', label='Control X (ax)')
        ax2.plot(control_history[:, 1], 'b-', label='Control Y (ay)')
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(env.acceleration_bound, color='gray', linestyle='--', alpha=0.3)
        ax2.axhline(-env.acceleration_bound, color='gray', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Control Input')
        ax2.set_title('Control Inputs Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(__file__), 'mpc_ball_trajectory.png')
        plt.savefig(plot_path)
        print(f"Trajectory plot saved as '{plot_path}'")
        
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    return trajectory

if __name__ == "__main__":
    test_mpc_ball()