#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BallEnvironment:
    """
    Ball environment where the agent controls the ball's acceleration in x and y directions
    directly to reach a target position.
    
    State space: [x, y, vx, vy]  (position, linear velocity)
    Action space: [ax, ay]  (desired acceleration components)
    """
    
    def __init__(self, target_pos=None, max_steps=100):
        # Environment parameters
        self.max_steps = max_steps
        self.dt = 0.01  # Time step
        
        # State bounds
        self.pos_bound = 10.0  # Position bounds [-10, 10]
        self.vel_bound = 2.0   # Linear velocity bounds [-5, 5]
        self.acceleration_bound = 1.0  # Acceleration bounds
        
        # Target position
        if target_pos is None:
            self.target_pos = np.array([0.0, 0.0])  # Default target at origin
        else:
            self.target_pos = np.array(target_pos)
            
        # Initialize state
        self.reset()
    
    def reset(self, initial_state=None):
        """Reset the environment to initial state."""
        self.current_step = 0
        
        if initial_state is None:
            # Random initial position in [-5, 5], zero velocity
            self.state = np.zeros(4)  # [x, y, vx, vy]
            self.state[0] = np.random.uniform(-5.0, 5.0)
            self.state[1] = np.random.uniform(-5.0, 5.0)
            # No orientation or angular velocity anymore
        else:
            self.state = np.array(initial_state)[:4]  # Only keep [x, y, vx, vy]
            
        # Initialize previous distance for delta reward calculation
        current_pos = self.state[:2]
        self.prev_distance = np.linalg.norm(current_pos - self.target_pos)
        
        return self.state
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        
        Args:
            action (np.ndarray): Action [ax, ay] - expected in range [-1, 1]
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Store last action (in [-1,1] range) for reward calculation
        self.last_action = action.copy()
        
        # Scale action from [-1, 1] to environment's action space (acceleration range)
        ax = action[0] * self.acceleration_bound  # Scale to [-acceleration_bound, acceleration_bound]
        ay = action[1] * self.acceleration_bound
        
        # Clip to ensure within bounds
        ax = np.clip(ax, -self.acceleration_bound, self.acceleration_bound)
        ay = np.clip(ay, -self.acceleration_bound, self.acceleration_bound)
        
        # Get current state (only x, y, vx, vy)
        x, y, vx, vy = self.state
        
        # Update velocity using acceleration: v = v0 + a*dt
        new_vx = vx + ax * self.dt
        new_vy = vy + ay * self.dt
        
        # Update position using velocity: x = x0 + v*dt
        new_x = x + new_vx * self.dt+0.5*ax*self.dt**2
        new_y = y + new_vy * self.dt+0.5*ay*self.dt**2
        
        # Clip to bounds
        new_x = np.clip(new_x, -self.pos_bound, self.pos_bound)
        new_y = np.clip(new_y, -self.pos_bound, self.pos_bound)
        new_vx = np.clip(new_vx, -self.vel_bound, self.vel_bound)
        new_vy = np.clip(new_vy, -self.vel_bound, self.vel_bound)
        
        # Update state
        self.state = np.array([new_x, new_y, new_vx, new_vy])
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        info = {
            "distance": np.linalg.norm(self.state[:2] - self.target_pos),
            "applied_acceleration": np.linalg.norm([ax, ay]),
            "step": self.current_step
        }
        
        return self.state, reward, done, info
    
    def _calculate_reward(self):
        """Enhanced reward function for direct position control."""
        # Calculate current state
        current_pos = self.state[:2]
        velocity = self.state[2:4]
        vel_magnitude = np.linalg.norm(velocity)
        
        # Calculate distance to environment target
        distance_to_env_target = np.linalg.norm(current_pos - self.target_pos)
        
        # Reward 1: Distance reduction to environment target (using exponential function)
        if hasattr(self, 'prev_distance'):
            distance_delta = self.prev_distance - distance_to_env_target  # Positive if getting closer
            # Exponential reward for distance reduction
            distance_reward = 50.0 * (1.0 - np.exp(-5.0 * distance_delta))  # Stronger reward for larger improvements
        else:
            # First step - exponential reward based on absolute distance
            distance_reward = 100.0 * np.exp(-0.5 * distance_to_env_target) - 20.0
        
        # Store current distance for next step's delta calculation
        self.prev_distance = distance_to_env_target
        
        # Combine all rewards and penalties with clear weights
        total_reward = (
            distance_reward             # Main: get closer to environment target
        )
        
        return total_reward
    
    def render(self):
        """Render the environment."""
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # Set plot limits
        ax.set_xlim(-self.pos_bound, self.pos_bound)
        ax.set_ylim(-self.pos_bound, self.pos_bound)
        ax.set_aspect('equal')
        
        # Draw target
        target = patches.Circle(self.target_pos, radius=0.5, color='green', alpha=0.5)
        ax.add_patch(target)
        
        # Draw ball
        ball = patches.Circle(self.state[:2], radius=0.3, color='blue')
        ax.add_patch(ball)
        
        # Draw ball direction based on velocity
        velocity = self.state[2:4]
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude > 0.1:  # Only draw direction if velocity is significant
            direction = velocity / vel_magnitude
            ax.arrow(self.state[0], self.state[1], direction[0] * 0.5, direction[1] * 0.5, 
                    head_width=0.1, head_length=0.1, color='red')
        
        plt.grid()
        plt.title(f"Ball Environment - Step {self.current_step}")
        plt.show()

    @property
    def state_dim(self):
        return 4  # Only [x, y, vx, vy] now
        
    @property  
    def action_dim(self):
        return 2

# Test the environment if this file is run directly
if __name__ == "__main__":
    # Create environment
    env = BallEnvironment(target_pos=[5.0, 5.0])
    
    # Test reset
    initial_state = env.reset()
    print(f"Initial state: {initial_state}")
    
    # Test step
    action = np.array([1.0, 0.5])  # Move forward with some rotation
    next_state, reward, done, info = env.step(action)
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Test rendering
    print("Rendering environment...")
    env.render()