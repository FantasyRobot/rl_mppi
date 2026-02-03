import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加cdf_2d目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../cdf_2d'))
from cdf import CDF2D
from primitives2D_torch import Circle
from robot_plot2D import plot_2d_manipulators

# 设置设备为CPU
device = torch.device("cpu")

print("正在初始化CDF2D类...")
# 创建CDF2D实例
cdf = CDF2D(device)

print("初始化成功！")
print(f"配置空间维度: {cdf.num_joints}")
print(f"关节角度范围: [{cdf.q_min.tolist()}] 到 [{cdf.q_max.tolist()}]")

# 创建简单障碍物
print("\n创建障碍物...")
scene = [Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device)]

# 使用shooting算法生成防碰撞路径
print("\n使用shooting算法生成防碰撞路径...")
# 定义初始关节角度配置
q0 = torch.tensor([[0.0, 0.0]], device=device)  # 初始配置

# 调用shooting算法生成路径
trajectories = cdf.shooting(q0, scene, method='CDField', timestep=500)
print(f"生成的轨迹形状: {trajectories.shape}")

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制障碍物
print("绘制障碍物...")
for obstacle in scene:
    obstacle_patch = obstacle.create_patch(color='red')
    ax.add_patch(obstacle_patch)

# 获取shooting算法生成的轨迹数据
trajectory = trajectories[0]  # 获取第一条轨迹
print(f"轨迹初始点: {trajectory[0]}")
print(f"轨迹结束点: {trajectory[-1]}")

# 绘制机器人防碰撞路径
plot_2d_manipulators(
    link1_length=2,
    link2_length=2,
    joint_angles_batch=trajectory,
    ax=ax,
    color='green',
    alpha=0.5,
    show_eef_traj=True  # 显示末端执行器轨迹
)

# 设置合适的坐标轴范围以同时显示机器人和障碍物
ax.set_xlim([-5, 5.0])
ax.set_ylim([-5, 5.0])
ax.set_aspect("equal", "box")
plt.title("2 link Robot Collision-Free Path using CDF-MPPI")
plt.grid(True)
plt.show()