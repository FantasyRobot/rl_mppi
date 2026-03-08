#!/usr/bin/env python3

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class MultiRobotGNNEnv:
    """
    兼容roboballet风格和SAC风格的多机器人GNN环境。
    支持GNN训练（返回roboballet结构obs），也兼容SAC风格reset/step接口。
    """
    def __init__(
        self,
        xml_path,
        num_robots=1,
        num_obstacles=0,
        max_steps=2500,
        seed=0,
        goal_site: str = "goal",
        eef_site: str = "end_effector",
        reach_tol: float = 0.03,
        action_repeat: int = 5,
        qvel_scale: float = 5.0,
        reward_dist_coeff: float = 1.0,
        reward_ctrl_coeff: float = 0.01,
        success_bonus: float = 10.0,
        reset_noise: float = 0.05,
        **kwargs
    ):
        import mujoco
        self.xml_path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(xml_path))))
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML not found: {self.xml_path}")
        self.goal_site = str(goal_site)
        self.eef_site = str(eef_site)
        self.max_steps = int(max_steps)
        self.reach_tol = float(reach_tol)
        self.action_repeat = int(max(1, action_repeat))
        self.qvel_scale = float(qvel_scale)
        self.reward_dist_coeff = float(reward_dist_coeff)
        self.reward_ctrl_coeff = float(reward_ctrl_coeff)
        self.success_bonus = float(success_bonus)
        self.reset_noise = float(reset_noise)
        self.num_robots = int(num_robots)
        self.num_obstacles = int(num_obstacles)
        self._step_count = 0
        self._rng = np.random.default_rng(int(seed))
        # 加载mujoco模型
        xml_dir = os.path.dirname(self.xml_path)
        with _pushd(xml_dir):
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        if self.nu <= 0:
            raise ValueError("Model has no actuators (model.nu == 0).")
        # ctrlrange for position actuators
        self.ctrl_min = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=np.float64)
        self.ctrl_max = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=np.float64)
        self.ctrl_center = 0.5 * (self.ctrl_min + self.ctrl_max)
        self.ctrl_half = 0.5 * (self.ctrl_max - self.ctrl_min)
        self.ctrl_half = np.where(self.ctrl_half == 0.0, 1.0, self.ctrl_half)
        # site id
        self.goal_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.goal_site)
        self.eef_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.eef_site)
        if self.goal_sid < 0:
            raise ValueError(f"Goal site not found: {self.goal_site}")
        if self.eef_sid < 0:
            raise ValueError(f"EEF site not found: {self.eef_site}")
        self.goal_pos = np.zeros((3,), dtype=np.float64)
        self._mujoco.mj_forward(self.model, self.data)

        self.action_dim = int(self.nu)
        self.obs_dim = int(self.nu + self.nv + 3)

    def reset(self, init_qpos=None):
        self._step_count = 0
        self.data.qvel[:] = 0.0
        # 多机器人初始位姿
        if init_qpos is not None:
            self.data.qpos[:len(init_qpos)] = np.array(init_qpos, dtype=np.float64)
        else:
            self.data.qpos[:] = self._rng.normal(0.0, 0.05, size=self.data.qpos.shape)
        # 随机采样goal_xyz，覆盖工作空间
        goal_low = np.array([0.3, 0.0, 0.2], dtype=np.float64)
        goal_high = np.array([0.8, 0.4, 0.5], dtype=np.float64)
        self.goal_pos = self._rng.uniform(goal_low, goal_high)
        self._mujoco.mj_forward(self.model, self.data)

        return self.get_obs()

    def step(self, actions):
        # actions: shape=(num_robots, action_dim)
        # 这里假设每个机器人独立控制
        ctrl = self.action_to_ctrl(actions)
        self.data.ctrl[:ctrl.size] = np.asarray(ctrl).reshape(-1)
        self._mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self.get_obs()
        # 打印action和tip末端xyz坐标及分布诊断
        # tip = obs['robots']['tip_relative_poses'][0, :3, 3]
        # a = np.asarray(actions).reshape(-1)
        # c = np.asarray(ctrl).reshape(-1)
        # print(f"[DEBUG] action: {a} | ctrl: {c} | tip_xyz: {tip} | goal_xyz: {self.goal_pos}")
        # print(f"[STAT] action mean/std: {a.mean():.4f}/{a.std():.4f} | ctrl mean/std: {c.mean():.4f}/{c.std():.4f} | tip_xyz mean/std: {tip.mean():.4f}/{tip.std():.4f}")
        reward = self.compute_reward(obs, actions)
        done = self._step_count >= self.max_steps
        info = {}
        return obs, reward, done, info

    def get_obs(self):
        # roboballet风格obs，增加归一化
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        # 归一化qpos到[-1,1]
        qpos_norm = 2 * (qpos - self.ctrl_min) / (self.ctrl_max - self.ctrl_min + 1e-8) - 1
        # 归一化qvel到[-1,1]（假设qvel_scale为最大速度）
        qvel_norm = np.clip(qvel / (self.qvel_scale + 1e-8), -1, 1)
        # 获取end_effector site的4x4位姿
        eef_pos = self._eef_pos()  # (3,)
        eef_mat = np.asarray(self.data.site_xmat[self.eef_sid], dtype=np.float32).reshape(3, 3)
        tip_pose = np.eye(4, dtype=np.float32)
        tip_pose[:3, :3] = eef_mat
        tip_pose[:3, 3] = eef_pos
        robots = {
            'joint_configurations': qpos_norm[None, :],
            'joint_velocities': qvel_norm[None, :],
            'dwelling': np.zeros((1,), dtype=np.float32),
            'base_poses': np.tile(np.eye(4, dtype=np.float32), (1, 1, 1)),
            'tip_relative_poses': tip_pose[None, :, :],
        }

        self._update_goal()
        goal_pose = np.eye(4, dtype=np.float32)
        goal_pose[:3, 3] = self.goal_pos.astype(np.float32)
        targets = {
            'done': np.zeros((1,), dtype=np.float32),
            'poses': goal_pose[None, :, :],
        }
        obstacles = {
            'spans': np.zeros((self.num_obstacles, 3), dtype=np.float32),
            'poses': np.tile(np.eye(4, dtype=np.float32), (self.num_obstacles, 1, 1)) if self.num_obstacles > 0 else np.zeros((0, 4, 4), dtype=np.float32),
        }
        if 'poses' not in obstacles:
            num_obstacles = obstacles.get('positions', np.zeros((0, 2))).shape[0]
            obstacles['poses'] = np.tile(np.eye(4, dtype=np.float32), (num_obstacles, 1, 1))
        obs = {
            'robots': robots,
            'targets': targets,
            'obstacles': obstacles,
        }
        return obs

    def compute_reward(self, obs, actions):
        # roboballet风格reward：距离惩罚+控制代价+成功奖励
        tip = self._eef_pos().astype(np.float32)
        goal = self.goal_pos.astype(np.float32)
        dist = np.linalg.norm(tip - goal)
        ctrl_cost = float(self.reward_ctrl_coeff) * np.linalg.norm(actions)
        reward = -float(self.reward_dist_coeff) * dist - ctrl_cost
        # 成功奖励
        if dist < float(self.reach_tol):
            reward += float(self.success_bonus)
        # 碰撞惩罚（如有info['collision']可加）
        # if info.get('collision', False):
        #     reward -= 15.0
        return reward

    def export_graph_features(self):
        # 可调用gnn/compute_features.py生成graph结构
        from .compute_features import make_graph_features_prescaled, FeatureConfig
        obs = self.get_obs()
        feature_config = FeatureConfig(robot_relative_tip_features=True, robot_base_features=True)
        g = make_graph_features_prescaled(obs, feature_config)
        return g

    def _update_goal(self) -> None:
        #self._mujoco.mj_forward(self.model, self.data)
        self.goal_pos = np.asarray(self.data.site_xpos[self.goal_sid], dtype=np.float64).copy()
        return self.goal_pos

    def _eef_pos(self) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.eef_sid], dtype=np.float64).copy()

    def action_to_ctrl(self, action_norm: np.ndarray) -> np.ndarray:
        a = np.asarray(action_norm, dtype=np.float64).reshape(self.nu)
        a = np.clip(a, -1.0, 1.0)
        return np.clip(self.ctrl_center + a * self.ctrl_half, self.ctrl_min, self.ctrl_max)

    def ctrl_to_action(self, ctrl: np.ndarray) -> np.ndarray:
        c = np.asarray(ctrl, dtype=np.float64).reshape(self.nu)
        a = (c - self.ctrl_center) / self.ctrl_half
        return np.clip(a, -1.0, 1.0)