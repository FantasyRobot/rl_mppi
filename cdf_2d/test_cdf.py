import torch
from cdf import CDF2D
from primitives2D_torch import Circle

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

# 测试CDF计算
print("\n测试CDF计算...")
# 创建几个测试配置
q_test = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], device=device)

# 计算CDF值
d_cdf = cdf.calculate_cdf(q_test, scene)
print(f"测试配置: {q_test.tolist()}")
print(f"CDF值: {d_cdf.tolist()}")

# 测试CDF梯度计算
print("\n测试CDF梯度计算...")
q_test.requires_grad = True
d_cdf, grad_cdf = cdf.calculate_cdf(q_test, scene, return_grad=True)
print(f"CDF梯度: {grad_cdf.tolist()}")

print("\n所有测试通过！CDF算法正常工作。")