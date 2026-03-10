import numpy as np

class BallGNNEnvironment:
    """
    小球-障碍物-目标点环境，适用于GNN建图。
    状态包括小球位置、速度、障碍物、目标点。
    """
    def __init__(self, target_pos=(3.0, 3.0), max_steps=400, obstacle_list=None, reset_span=2.5, reach_threshold=0.5):
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.max_steps = max_steps
        self.reset_span = reset_span
        self.reach_threshold = reach_threshold
        self.obstacle_list = obstacle_list if obstacle_list is not None else [((1.5, 1.5), 0.5)]
        self.state_dim = 4
        self.action_dim = 2
        self.pos_bound = 5.0
        self.vel_bound = 5.0
        self.reset()

    def reset(self, initial_state=None, *, reset_span: float = None):
        self.step_count = 0
        span = float(self.reset_span if reset_span is None else reset_span)
        if initial_state is None:
            self.state = np.zeros(4)
            self.state[0] = np.random.uniform(self.target_pos[0] - span, self.target_pos[0] + span)
            self.state[1] = np.random.uniform(self.target_pos[1] - span, self.target_pos[1] + span)
            self.state[0] = np.clip(self.state[0], -self.pos_bound, self.pos_bound)
            self.state[1] = np.clip(self.state[1], -self.pos_bound, self.pos_bound)
            self.state[2] = 0.0
            self.state[3] = 0.0
        else:
            self.state = np.array(initial_state, dtype=float)[:4]
            if self.state.shape[0] < 4:
                s = np.zeros(4)
                s[: self.state.shape[0]] = self.state
                self.state = s
        current_pos = self.state[:2]
        self.prev_distance = np.linalg.norm(current_pos - self.target_pos)
        return self._get_obs()

    def step(self, action):
        action = np.array(action, dtype=float)
        self.last_action = action.copy()
        # 动作缩放
        ax = np.clip(action[0], -1, 1) * 1.0
        ay = np.clip(action[1], -1, 1) * 1.0
        x, y, vx, vy = self.state
        dt = 0.01
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_x = x + new_vx * dt + 0.5 * ax * dt ** 2
        new_y = y + new_vy * dt + 0.5 * ay * dt ** 2
        new_x = np.clip(new_x, -self.pos_bound, self.pos_bound)
        new_y = np.clip(new_y, -self.pos_bound, self.pos_bound)
        new_vx = np.clip(new_vx, -self.vel_bound, self.vel_bound)
        new_vy = np.clip(new_vy, -self.vel_bound, self.vel_bound)
        hit_boundary_x = (new_x <= -self.pos_bound + 1e-9) or (new_x >= self.pos_bound - 1e-9)
        hit_boundary_y = (new_y <= -self.pos_bound + 1e-9) or (new_y >= self.pos_bound - 1e-9)
        hit_boundary = bool(hit_boundary_x or hit_boundary_y)
        self.state = np.array([new_x, new_y, new_vx, new_vy])
        self.step_count += 1
        done = False
        distance = np.linalg.norm(self.state[:2] - self.target_pos)
        time_limit = False
        success = distance < self.reach_threshold
        if success:
            done = True
        elif self.step_count >= self.max_steps:
            done = True
            time_limit = True
        reward = self._calculate_reward(distance=distance, success=success, time_limit=time_limit)
        info = {
            "distance": distance,
            "applied_acceleration": np.linalg.norm([ax, ay]),
            "step": self.step_count,
            "hit_boundary": hit_boundary,
            "time_limit": time_limit,
            "success": success,
        }
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return {
            'ball_pos': self.state[:2].copy(),
            'ball_vel': self.state[2:4].copy(),
            'target_pos': self.target_pos.copy(),
            'obstacles': [(np.array(center), radius) for center, radius in self.obstacle_list]
        }

    def _calculate_reward(self, distance, success, time_limit):
        prev = getattr(self, "prev_distance", distance)
        reward = (prev - distance) * 1.0
        reward -= 0.005
        if success:
            reward += 5.0
        elif time_limit:
            reward -= 1.0
        self.prev_distance = distance
        return float(reward)
