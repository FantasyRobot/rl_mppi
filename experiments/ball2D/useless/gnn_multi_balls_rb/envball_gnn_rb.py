from dataclasses import dataclass

import numpy as np


@dataclass
class StartState:
    """Reproducible start state for rerunning a trajectory, including HER relabels."""

    ball_pos: np.ndarray
    ball_vel: np.ndarray
    target_pos: np.ndarray


class BallRoboBalletEnvironment:
    """RoboBallet-style 2D multi-ball environment with score-difference rewards."""

    def __init__(
        self,
        num_balls=4,
        targets_per_ball=2,
        max_steps=320,
        reset_span=0.65,
        target_span=1.1,
        reach_threshold=0.35,
        return_threshold=0.35,
        target_score_shaping=0.0,
        target_score_coeff=0.5,
        return_score_shaping=0.5,
        return_score_coeff=0.5,
        return_progress_horizon=4.0,
        return_to_start_weight=0.45,
        return_to_start_required=True,
        return_phase_bonus=0.15,
        return_takeover_threshold=None,
        collision_margin=0.30,
        collision_penalty_scaling=4.0,
        acceleration_cost=0.0,
        pos_bound=5.0,
        vel_bound=3.0,
        dt=0.05,
    ):
        self.num_balls = int(num_balls)
        self.targets_per_ball = int(targets_per_ball)
        self.num_targets = self.num_balls * self.targets_per_ball
        self.max_steps = int(max_steps)
        self.reset_span = float(reset_span)
        self.target_span = float(target_span)
        self.reach_threshold = float(reach_threshold)
        self.return_threshold = float(return_threshold)
        self.target_score_shaping = float(target_score_shaping)
        self.target_score_coeff = float(target_score_coeff)
        self.return_score_shaping = float(return_score_shaping)
        self.return_score_coeff = float(return_score_coeff)
        self.return_progress_horizon = float(return_progress_horizon)
        self.return_to_start_weight = float(return_to_start_weight)
        self.return_to_start_required = bool(return_to_start_required)
        self.return_phase_bonus = float(return_phase_bonus)
        if return_takeover_threshold is None:
            return_takeover_threshold = return_threshold
        self.return_takeover_threshold = float(return_takeover_threshold)
        self.collision_margin = float(collision_margin)
        self.collision_penalty_scaling = float(collision_penalty_scaling)
        self.acceleration_cost = float(acceleration_cost)
        self.pos_bound = float(pos_bound)
        self.vel_bound = float(vel_bound)
        self.dt = float(dt)
        self.state_dim = self.num_balls * 4
        self.action_dim = self.num_balls * 2
        self.cluster_centers = np.array(
            [
                [-2.5, -2.5],
                [2.5, -2.5],
                [-2.5, 2.5],
                [2.5, 2.5],
            ],
            dtype=np.float32,
        )[: self.num_balls]
        self.target_cluster = np.repeat(np.arange(self.num_balls, dtype=np.int32), self.targets_per_ball)
        self.target_slot = np.tile(np.arange(self.targets_per_ball, dtype=np.int32), self.num_balls)
        self.reset()

    def _sample_target_positions(self):
        target_pos = np.zeros((self.num_targets, 2), dtype=np.float32)
        for target_idx in range(self.num_targets):
            center = self.cluster_centers[self.target_cluster[target_idx]]
            offset = np.random.uniform(-self.target_span, self.target_span, size=2).astype(np.float32)
            target_pos[target_idx] = np.clip(center + offset, -self.pos_bound, self.pos_bound)
        return target_pos

    def get_start_state(self):
        """Return a serializable start state for trajectory replay and HER reruns."""

        return StartState(
            ball_pos=self.ball_start_pos.copy(),
            ball_vel=np.zeros_like(self.ball_vel),
            target_pos=self.target_pos.copy(),
        )

    def reset(self, start_state=None):
        """Reset the environment either randomly or from an explicit start state."""

        self.step_count = 0
        if start_state is None:
            ball_pos = self.cluster_centers + np.random.uniform(
                -self.reset_span,
                self.reset_span,
                size=(self.num_balls, 2),
            ).astype(np.float32)
            ball_vel = np.zeros((self.num_balls, 2), dtype=np.float32)
            target_pos = self._sample_target_positions()
        else:
            ball_pos = np.asarray(start_state.ball_pos, dtype=np.float32).copy()
            ball_vel = np.asarray(start_state.ball_vel, dtype=np.float32).copy()
            target_pos = np.asarray(start_state.target_pos, dtype=np.float32).copy()

        self.ball_pos = np.clip(ball_pos, -self.pos_bound, self.pos_bound).astype(np.float32)
        self.ball_vel = np.clip(ball_vel, -self.vel_bound, self.vel_bound).astype(np.float32)
        self.ball_start_pos = self.ball_pos.copy()
        self.target_pos = np.clip(target_pos, -self.pos_bound, self.pos_bound).astype(np.float32)
        self.target_done = np.zeros(self.num_targets, dtype=np.float32)
        self.last_action = np.zeros((self.num_balls, 2), dtype=np.float32)
        self.last_colliding_balls = np.zeros(self.num_balls, dtype=bool)
        self.current_score = 0.0
        self.total_collision_penalty_value = 0.0
        self.total_collision_ball_steps = 0
        self.total_mean_acceleration = 0.0
        self.total_acceleration_cost_value = 0.0
        self.return_phase_started = False
        return self._get_obs()

    def _distance_score(self, distance, threshold, max_shaped_score, coeff):
        if distance < threshold:
            return 1.0
        if max_shaped_score <= 0.0:
            return 0.0
        return max_shaped_score * coeff / (coeff + distance)

    def _target_distances(self):
        return np.linalg.norm(self.ball_pos[:, None, :] - self.target_pos[None, :, :], axis=-1)

    def _pairwise_ball_distances(self):
        delta = self.ball_pos[:, None, :] - self.ball_pos[None, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(distances, np.inf)
        return distances

    def _colliding_ball_mask(self):
        distances = self._pairwise_ball_distances()
        return np.any(distances < self.collision_margin, axis=1)

    def _resolve_collisions_with_rollbacks(self, previous_pos, previous_vel):
        """Rollback colliding balls one by one, matching RoboBallet's safety behavior."""

        rolled_back_balls = np.zeros(self.num_balls, dtype=bool)
        iterations = 0
        while True:
            colliding_mask = self._colliding_ball_mask()
            colliding_indices = np.flatnonzero(colliding_mask & (~rolled_back_balls))
            if colliding_indices.size == 0:
                break
            rollback_idx = int(np.random.choice(colliding_indices))
            self.ball_pos[rollback_idx] = previous_pos[rollback_idx]
            self.ball_vel[rollback_idx] = 0.0
            rolled_back_balls[rollback_idx] = True
            iterations += 1
            if iterations > self.num_balls:
                break
        self.last_colliding_balls = rolled_back_balls
        return rolled_back_balls

    def _collision_penalty(self):
        step_max_penalty_per_ball = 1.0 / self.max_steps / self.num_balls
        penalty = self.collision_penalty_scaling * step_max_penalty_per_ball * int(np.sum(self.last_colliding_balls))
        return float(penalty)

    def _return_distances(self):
        return np.linalg.norm(self.ball_pos - self.ball_start_pos, axis=1).astype(np.float32)

    def _all_balls_are_parked(self):
        return bool(np.all(self._return_distances() < self.return_threshold))

    def _return_to_start_score(self):
        distances = self._return_distances()
        smooth_scores = [
            self._distance_score(distance, self.return_threshold, self.return_score_shaping, self.return_score_coeff)
            for distance in distances
        ]
        smooth_scores = np.asarray(smooth_scores, dtype=np.float32)
        parked_mask = (distances < self.return_threshold).astype(np.float32)
        threshold_progress = np.clip(
            1.0 - distances / (self.return_progress_horizon * self.return_threshold),
            0.0,
            1.0,
        )
        inverse_distance_progress = np.clip(
            self.return_threshold / np.maximum(distances, self.return_threshold),
            0.0,
            1.0,
        ).astype(np.float32)
        dense_progress = np.maximum(threshold_progress, inverse_distance_progress)
        scores = np.maximum(parked_mask, 0.35 * smooth_scores + 0.65 * dense_progress)
        return float(np.mean(scores)), float(np.mean(parked_mask))

    def _apply_return_takeover(self, action):
        """Directly drive already-near balls back to start during the return phase."""

        if not np.all(self.target_done > 0.5):
            return action, 0

        pos_diff = self.ball_start_pos - self.ball_pos
        takeover_mask = np.all(np.abs(pos_diff) < self.return_takeover_threshold, axis=1)
        takeover_count = int(np.sum(takeover_mask))
        if takeover_count == 0:
            return action, 0

        desired_velocity = pos_diff / self.dt
        desired_acceleration = (desired_velocity - self.ball_vel) / self.dt
        action = action.copy()
        action[takeover_mask] = desired_acceleration[takeover_mask]
        return action, takeover_count

    def _score_and_update_all_targets(self):
        total = 0.0
        all_done = True
        target_distances = self._target_distances()
        for target_idx in range(self.num_targets):
            if self.target_done[target_idx] > 0.5:
                total += 1.0
                continue
            highest_score = 0.0
            for ball_idx in range(self.num_balls):
                distance = float(target_distances[ball_idx, target_idx])
                highest_score = max(
                    highest_score,
                    self._distance_score(distance, self.reach_threshold, self.target_score_shaping, self.target_score_coeff),
                )
                if distance < self.reach_threshold:
                    self.target_done[target_idx] = 1.0
                    break
            if self.target_done[target_idx] > 0.5:
                total += 1.0
            else:
                all_done = False
                total += highest_score

        mean_target_score = total / max(self.num_targets, 1)
        total_score = (1.0 - self.return_to_start_weight) * mean_target_score
        return_to_start_score = 0.0
        parked_ratio = 0.0
        if all_done:
            return_to_start_score, parked_ratio = self._return_to_start_score()
            if self.return_to_start_required:
                return_phase_score = 0.5 * return_to_start_score + 0.5 * parked_ratio
                total_score += self.return_to_start_weight * return_phase_score
        return float(total_score), float(mean_target_score), float(return_to_start_score), float(parked_ratio)

    def step(self, action):
        """Advance one step with score-difference reward and RoboBallet-style penalties."""

        action = np.asarray(action, dtype=np.float32).reshape(self.num_balls, 2)
        action, return_takeover_count = self._apply_return_takeover(action)
        action = np.clip(action, -1.0, 1.0)
        self.last_action = action.copy()
        velocities_before = self.ball_vel.copy()
        positions_before = self.ball_pos.copy()

        self.ball_vel = np.clip(self.ball_vel + action * self.dt, -self.vel_bound, self.vel_bound)
        self.ball_pos = np.clip(
            self.ball_pos + self.ball_vel * self.dt + 0.5 * action * (self.dt ** 2),
            -self.pos_bound,
            self.pos_bound,
        )
        self.step_count += 1

        self._resolve_collisions_with_rollbacks(positions_before, velocities_before)
        new_score, mean_target_score, return_to_start_score, parked_ratio = self._score_and_update_all_targets()
        collision_penalty = self._collision_penalty()
        acceleration = (self.ball_vel - velocities_before) / self.dt
        mean_acc2 = float(np.mean(np.square(acceleration)))
        acceleration_cost = (self.acceleration_cost / max(self.num_targets, 1)) * mean_acc2 * self.dt
        all_targets_done_now = bool(np.all(self.target_done > 0.5))
        entered_return_phase = all_targets_done_now and (not self.return_phase_started)
        phase_switch_bonus = self.return_phase_bonus if entered_return_phase else 0.0
        reward = float(new_score - self.current_score - collision_penalty - acceleration_cost + phase_switch_bonus)

        self.total_collision_penalty_value += collision_penalty
        self.total_collision_ball_steps += int(np.sum(self.last_colliding_balls))
        self.total_mean_acceleration += float(np.mean(np.abs(acceleration))) * self.dt
        self.total_acceleration_cost_value += acceleration_cost
        self.current_score = new_score
        if entered_return_phase:
            self.return_phase_started = True

        all_targets_done = all_targets_done_now
        success = all_targets_done and (not self.return_to_start_required or self._all_balls_are_parked())
        time_limit = self.step_count >= self.max_steps
        done = bool(success or time_limit)

        info = {
            'step': self.step_count,
            'success': success,
            'time_limit': bool(time_limit and not success),
            'targets_done': int(np.sum(self.target_done > 0.5)),
            'targets_remaining': int(np.sum(self.target_done < 0.5)),
            'current_score': float(new_score),
            'mean_target_score': float(mean_target_score),
            'return_to_start_score': float(return_to_start_score),
            'return_parked_ratio': float(parked_ratio),
            'return_takeover_count': int(return_takeover_count),
            'phase_switch_bonus': float(phase_switch_bonus),
            'entered_return_phase': bool(entered_return_phase),
            'collision_penalty': float(collision_penalty),
            'collision_ball_count': int(np.sum(self.last_colliding_balls)),
            'total_collision_penalty': float(self.total_collision_penalty_value),
            'return_distances': self._return_distances().copy(),
            'all_targets_done': all_targets_done,
            'pairwise_ball_distance_min': float(np.min(self._pairwise_ball_distances())),
        }
        return self._get_obs(), reward, done, info

    def observation_spec_summary(self):
        """Return a lightweight summary of observation shapes for debugging."""

        obs = self._get_obs()
        return {key: tuple(np.asarray(value).shape) for key, value in obs.items()}

    def current_score_value(self):
        return float(self.current_score)

    def num_targets_done(self):
        return int(np.sum(self.target_done > 0.5))

    def total_collision_penalty(self):
        return float(self.total_collision_penalty_value)

    def total_collision_ball_steps_value(self):
        return int(self.total_collision_ball_steps)

    def total_acceleration(self):
        return float(self.total_mean_acceleration)

    def terminal(self):
        all_targets_done = bool(np.all(self.target_done > 0.5))
        return bool(
            self.step_count >= self.max_steps
            or (all_targets_done and (not self.return_to_start_required or self._all_balls_are_parked()))
        )

    def _get_obs(self):
        time_fraction = float(self.step_count / max(self.max_steps, 1))
        return {
            'ball_pos': self.ball_pos.copy(),
            'ball_vel': self.ball_vel.copy(),
            'ball_start_pos': self.ball_start_pos.copy(),
            'target_pos': self.target_pos.copy(),
            'target_done': self.target_done.copy(),
            'target_cluster': self.target_cluster.copy(),
            'target_slot': self.target_slot.copy(),
            'current_score': np.asarray(self.current_score, dtype=np.float32),
            'time_fraction': np.asarray(time_fraction, dtype=np.float32),
        }