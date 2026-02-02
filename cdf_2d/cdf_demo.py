import torch
import matplotlib.pyplot as plt
from cdf import CDF2D
from primitives2D_torch import Circle

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化CDF
cdf = CDF2D(device)

# 创建障碍物场景
scene = [
    Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device)
]

# 初始配置
q_start = torch.tensor([[-0.2, 0.0]], device=device)

# 使用CDF进行运动规划
trajectory = cdf.shooting(q_start, scene, dt=1e-2, timestep=500, method='CDField')

# 可视化结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 绘制配置空间距离场
cdf.plot_cdf(ax1, scene)
ax1.plot(trajectory[0, :, 0], trajectory[0, :, 1], 'r-', linewidth=2, label='Trajectory')
ax1.plot(q_start[0, 0], q_start[0, 1], 'go', markersize=10, label='Start')

# 绘制机器人在任务空间的运动
from robot_plot2D import plot_2d_manipulators

# 转换配置到机器人末端轨迹
robot = cdf.robot
robot.init_states = torch.tensor(trajectory[0], device=device)
robot.forward_kinematics()
eef_traj = robot.ee_positions.cpu().numpy()

# 绘制任务空间
ax2.set_xlim(-3, 5)
ax2.set_ylim(-3, 5)
ax2.set_aspect('equal')
ax2.set_title('Task Space')

# 绘制障碍物
cdf.plot_objects(ax2, scene)

# 绘制末端轨迹
ax2.plot(eef_traj[:, 0], eef_traj[:, 1], 'r-', linewidth=2, label='End-effector trajectory')

plt.legend()
plt.show()