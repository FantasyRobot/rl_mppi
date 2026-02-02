#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CasADi优化问题Demo
展示CasADi在无约束优化和有约束优化中的基本用法
"""

import casadi as ca
import numpy as np

def unconstrained_optimization():
    """
    无约束优化示例：最小化 Rosenbrock 函数
    Rosenbrock函数: f(x, y) = (a - x)^2 + b*(y - x^2)^2
    全局最小值在 (a, a^2) 处，函数值为 0
    """
    print("=== 无约束优化示例：Rosenbrock 函数 ===")
    
    # 创建优化变量
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    
    # 定义目标函数 (Rosenbrock函数, a=1, b=100)
    a = 1.0
    b = 100.0
    f = (a - x)**2 + b*(y - x**2)**2
    
    # 创建优化问题
    opti = ca.Opti()
    
    # 设置优化变量
    opti_x = opti.variable()
    opti_y = opti.variable()
    
    # 设置目标函数（最小化）
    opti.minimize((a - opti_x)**2 + b*(opti_y - opti_x**2)**2)
    
    # 设置初始猜测
    opti.set_initial(opti_x, 0.0)
    opti.set_initial(opti_y, 0.0)
    
    # 选择求解器（这里使用IPOPT）
    opti.solver('ipopt')
    
    try:
        # 求解优化问题
        sol = opti.solve()
        
        # 输出结果
        print(f"优化成功！")
        print(f"最优解: x = {sol.value(opti_x):.6f}, y = {sol.value(opti_y):.6f}")
        print(f"目标函数值: {sol.value(f):.6f}")
        print(f"理论最小值: x = {a:.6f}, y = {a**2:.6f}, f = 0.0")
        
    except RuntimeError as e:
        print(f"优化失败: {e}")

def constrained_optimization():
    """
    有约束优化示例：最小化二次函数，带线性约束
    问题：minimize (x-3)^2 + (y-2)^2
    约束：x + y ≤ 4
          x ≥ 0, y ≥ 0
    """
    print("\n=== 有约束优化示例：带线性约束的二次函数 ===")
    
    # 创建优化问题
    opti = ca.Opti()
    
    # 设置优化变量
    x = opti.variable()
    y = opti.variable()
    
    # 设置目标函数（最小化）
    opti.minimize((x - 3)**2 + (y - 2)**2)
    
    # 添加约束条件
    opti.subject_to(x + y <= 4)  # 不等式约束
    opti.subject_to(x >= 0)       # 变量下界
    opti.subject_to(y >= 0)       # 变量下界
    
    # 设置初始猜测
    opti.set_initial(x, 1.0)
    opti.set_initial(y, 1.0)
    
    # 选择求解器
    opti.solver('ipopt')
    
    try:
        # 求解优化问题
        sol = opti.solve()
        
        # 输出结果
        print(f"优化成功！")
        print(f"最优解: x = {sol.value(x):.6f}, y = {sol.value(y):.6f}")
        print(f"目标函数值: {sol.value((x-3)**2 + (y-2)**2):.6f}")
        print(f"约束检查: x + y = {sol.value(x + y):.6f} ≤ 4.0")
        print(f"x ≥ 0: {sol.value(x) >= 0}, y ≥ 0: {sol.value(y) >= 0}")
        
    except RuntimeError as e:
        print(f"优化失败: {e}")

def optimal_control_example():
    """
    最优控制示例：简单的一维运动控制
    问题：控制一个物体从位置0移动到位置10，在时间T内停止
    最小化控制能量和时间
    """
    print("\n=== 最优控制示例：一维运动控制 ===")
    
    # 创建优化问题
    opti = ca.Opti()
    
    # 定义时间参数
    T = opti.variable()  # 总时间 (作为优化变量)
    N = 20              # 离散化步数
    dt = T/N            # 时间步长
    
    # 定义状态变量和控制变量
    x = opti.variable(N+1)  # 位置
    v = opti.variable(N+1)  # 速度
    u = opti.variable(N)    # 控制输入（加速度）
    
    # 初始条件
    opti.subject_to(x[0] == 0)
    opti.subject_to(v[0] == 0)
    
    # 终端条件
    opti.subject_to(x[-1] == 10)
    opti.subject_to(v[-1] == 0)
    
    # 动力学约束（欧拉积分）
    for k in range(N):
        opti.subject_to(x[k+1] == x[k] + v[k]*dt)
        opti.subject_to(v[k+1] == v[k] + u[k]*dt)
    
    # 控制约束
    opti.subject_to(opti.bounded(-2, u, 2))  # 控制输入范围
    
    # 目标函数：最小化控制能量和时间
    J = 0.1*T + 0.01*ca.sum1(u**2)
    opti.minimize(J)
    
    # 添加时间约束（确保T为正数且有合理上限）
    opti.subject_to(T >= 0.1)  # 最小时间约束
    opti.subject_to(T <= 20)   # 最大时间约束
    
    # 设置初始猜测
    opti.set_initial(T, 5.0)
    opti.set_initial(x, np.linspace(0, 10, N+1))
    
    # 选择求解器
    opti.solver('ipopt')
    
    try:
        # 求解优化问题
        sol = opti.solve()
        
        # 输出结果
        print(f"优化成功！")
        print(f"最优时间: T = {sol.value(T):.6f}秒")
        print(f"初始位置: x[0] = {sol.value(x[0]):.6f}")
        print(f"终端位置: x[-1] = {sol.value(x[-1]):.6f}")
        print(f"终端速度: v[-1] = {sol.value(v[-1]):.6f}")
        print(f"最大控制输入: {np.max(np.abs(sol.value(u))):.6f}")
        
    except RuntimeError as e:
        print(f"优化失败: {e}")

if __name__ == '__main__':
    print("CasADi优化问题Demo")
    print("=" * 40)
    
    # 运行无约束优化示例
    unconstrained_optimization()
    
    # 运行有约束优化示例
    constrained_optimization()
    
    # 运行最优控制示例
    optimal_control_example()
    
    print("\nDemo完成！")