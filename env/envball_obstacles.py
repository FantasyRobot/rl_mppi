#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.envball_utils import BallEnvironment


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    r: float

    @property
    def center(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float32)


class BallEnvironmentObstacles(BallEnvironment):
    """BallEnvironment with circular obstacles.

    Obstacles are treated as hard collisions for termination (optional) and also
    contribute an additional reward penalty when the ball is inside (or near) an obstacle.

    This environment is intended for *evaluation/comparison* of controllers.
    """

    def __init__(
        self,
        *args,
        obstacles: list[CircleObstacle] | None = None,
        obstacle_margin: float = 0.2,
        obstacle_penalty: float = 200.0,
        terminate_on_collision: bool = True,
        **kwargs,
    ):
        self.obstacles = list(obstacles) if obstacles is not None else []
        self.obstacle_margin = float(obstacle_margin)
        self.obstacle_penalty = float(obstacle_penalty)
        self.terminate_on_collision = bool(terminate_on_collision)
        super().__init__(*args, **kwargs)

    def _obstacle_clearance(self, pos_xy: np.ndarray) -> tuple[float, bool]:
        if not self.obstacles:
            return float("inf"), False
        p = np.asarray(pos_xy, dtype=np.float32).reshape(2)
        min_clearance = float("inf")
        collided = False
        for obs in self.obstacles:
            dx = float(p[0] - obs.x)
            dy = float(p[1] - obs.y)
            dist = float(np.sqrt(dx * dx + dy * dy))
            clearance = dist - float(obs.r)
            if clearance < min_clearance:
                min_clearance = clearance
            if dist <= float(obs.r):
                collided = True
        return min_clearance, collided

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        clearance, collided = self._obstacle_clearance(next_state[:2])

        # Soft penalty if inside inflated obstacle.
        if self.obstacles:
            # margin region counts as penalty too
            # penalty magnitude grows as we go deeper into margin.
            # clearance < margin => penalty
            margin = float(self.obstacle_margin)
            if clearance < margin:
                depth = float(margin - clearance)
                reward -= self.obstacle_penalty * depth

        if collided and self.terminate_on_collision:
            done = True

        info = dict(info)
        info["min_obstacle_clearance"] = float(clearance)
        info["hit_obstacle"] = bool(collided)

        return next_state, float(reward), bool(done), info
