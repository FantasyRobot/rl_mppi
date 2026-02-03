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

    def __init__(
        self,
        target_pos=None,
        max_steps=100,
        reward_scale: float = 1.0,
        reset_span: float = 5.0,
        reach_threshold: float = 0.5,
    ):
        # Environment parameters
        self.max_steps = max_steps
        self.dt = 0.01  # Time step

        # State bounds
        self.pos_bound = 10.0  # Position bounds [-10, 10]
        self.vel_bound = 2.0   # Linear velocity bounds
        self.acceleration_bound = 1.0  # Acceleration bounds

        # Reward scale (multiplies the distance-delta reward; does not change sign)
        self.reward_scale = float(reward_scale)

        # Reset sampling span around target (symmetric)
        self.reset_span = float(reset_span)

        # Episode termination distance threshold
        self.reach_threshold = float(reach_threshold)

        # Target position
        if target_pos is None:
            self.target_pos = np.array([0.0, 0.0])
        else:
            self.target_pos = np.array(target_pos)

        # Initialize state via reset
        self.reset()

    def reset(self, initial_state=None, *, reset_span: float | None = None):
        """Reset the environment to initial state."""
        self.current_step = 0

        if initial_state is None:
            # Random initial position sampled *around the target* (symmetric), zero velocity
            # This avoids a biased dataset where most states are on one side of the target.
            self.state = np.zeros(4)
            span = float(self.reset_span if reset_span is None else reset_span)
            self.state[0] = np.random.uniform(self.target_pos[0] - span, self.target_pos[0] + span)
            self.state[1] = np.random.uniform(self.target_pos[1] - span, self.target_pos[1] + span)
            self.state[0] = np.clip(self.state[0], -self.pos_bound, self.pos_bound)
            self.state[1] = np.clip(self.state[1], -self.pos_bound, self.pos_bound)
            self.state[2] = 0.0
            self.state[3] = 0.0
        else:
            self.state = np.array(initial_state, dtype=float)[:4]
            if self.state.shape[0] < 4:
                # pad with zeros if incomplete
                s = np.zeros(4)
                s[: self.state.shape[0]] = self.state
                self.state = s

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
        action = np.array(action, dtype=float)
        # Store last action (in [-1,1] range) for reward calculation
        self.last_action = action.copy()

        # Scale action from [-1, 1] to environment's action space (acceleration range)
        ax = action[0] * self.acceleration_bound
        ay = action[1] * self.acceleration_bound

        # Clip to ensure within bounds
        ax = np.clip(ax, -self.acceleration_bound, self.acceleration_bound)
        ay = np.clip(ay, -self.acceleration_bound, self.acceleration_bound)

        # Get current state (only x, y, vx, vy)
        x, y, vx, vy = self.state

        # Update velocity using acceleration: v = v0 + a*dt
        new_vx = vx + ax * self.dt
        new_vy = vy + ay * self.dt

        # Update position using velocity and acceleration (kinematic)
        new_x = x + new_vx * self.dt + 0.5 * ax * self.dt ** 2
        new_y = y + new_vy * self.dt + 0.5 * ay * self.dt ** 2

        # Clip to bounds
        new_x = np.clip(new_x, -self.pos_bound, self.pos_bound)
        new_y = np.clip(new_y, -self.pos_bound, self.pos_bound)
        new_vx = np.clip(new_vx, -self.vel_bound, self.vel_bound)
        new_vy = np.clip(new_vy, -self.vel_bound, self.vel_bound)

        # Detect boundary hit before updating state (if clipping occurred)
        hit_boundary_x = (new_x <= -self.pos_bound + 1e-9) or (new_x >= self.pos_bound - 1e-9)
        hit_boundary_y = (new_y <= -self.pos_bound + 1e-9) or (new_y >= self.pos_bound - 1e-9)
        hit_boundary = bool(hit_boundary_x or hit_boundary_y)

        # Update state
        self.state = np.array([new_x, new_y, new_vx, new_vy])
        self.current_step += 1

        # Calculate reward (only encourages moving toward target)
        reward = self._calculate_reward()

        # Do NOT terminate on boundary hit.
        # Terminating on boundary creates an incentive to "crash" to end the episode early
        # (especially when not sure how to reach the goal), which looks like boundary attraction.
        done = False

        # Check done: reached target or exceeded max steps
        distance = np.linalg.norm(self.state[:2] - self.target_pos)
        time_limit = False
        if distance < self.reach_threshold:
            done = True
        elif self.current_step >= self.max_steps:
            done = True
            time_limit = True

        info = {
            "distance": distance,
            "applied_acceleration": np.linalg.norm([ax, ay]),
            "step": self.current_step,
            "hit_boundary": hit_boundary,
            "time_limit": time_limit,
        }

        return self.state, reward, done, info

    def _calculate_reward(self):
        """Reward only for moving toward the target.

        Returns positive when distance to target decreases, negative when it increases.
        No reach bonus, no speed/action penalties, no boundary penalties.
        """
        current_pos = self.state[:2]
        distance = np.linalg.norm(current_pos - self.target_pos)

        # Distance delta reward: + if getting closer
        prev = getattr(self, "prev_distance", distance)
        reward = (prev - distance) * self.reward_scale
        self.prev_distance = distance
        return float(reward)

    def render(self):
        """Render the environment."""
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Set plot limits
        ax.set_xlim(-self.pos_bound, self.pos_bound)
        ax.set_ylim(-self.pos_bound, self.pos_bound)
        ax.set_aspect("equal")

        # Draw target
        target = patches.Circle(self.target_pos, radius=0.5, color="green", alpha=0.5)
        ax.add_patch(target)

        # Draw ball
        ball = patches.Circle(self.state[:2], radius=0.3, color="blue")
        ax.add_patch(ball)

        # Draw ball direction based on velocity
        velocity = self.state[2:4]
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude > 0.1:  # Only draw direction if velocity is significant
            direction = velocity / vel_magnitude
            ax.arrow(
                self.state[0], self.state[1], direction[0] * 0.5, direction[1] * 0.5, head_width=0.1, head_length=0.1, color="red"
            )

        plt.grid()
        plt.title(f"Ball Environment - Step {self.current_step}")
        plt.show()

    @property
    def state_dim(self):
        return 4

    @property
    def action_dim(self):
        return 2


# Test the environment if this file is run directly
if __name__ == "__main__":
    env = BallEnvironment(target_pos=[5.0, 5.0])
    initial_state = env.reset()
    print(f"Initial state: {initial_state}")
    action = np.array([1.0, 0.5])
    next_state, reward, done, info = env.step(action)
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    print("Rendering environment...")
    env.render()