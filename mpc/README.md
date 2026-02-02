# CasADi优化问题Demo

这是一个基于CasADi库的Python优化问题示例集合，展示了无约束优化、有约束优化和最优控制的基本用法。

## 安装CasADi

在运行这个demo之前，需要先安装CasADi库。可以通过以下方式安装：

### 使用pip安装（推荐）

```bash
pip install casadi
```

### 从源码编译安装

如果需要最新版本的CasADi，可以从源码编译安装。详细步骤请参考[CasADi官方文档](https://web.casadi.org/get/)。

## 运行Demo

1. 确保已经安装了Python 3和CasADi
2. 运行demo脚本：

```bash
python casadi_optimization_demo.py
```

## Demo内容

这个demo包含三个优化问题示例：

### 1. 无约束优化：Rosenbrock函数

最小化Rosenbrock函数：
```
f(x, y) = (a - x)^2 + b*(y - x^2)^2
```
其中a=1，b=100。该函数的全局最小值在(x, y)=(1, 1)处，函数值为0。

### 2. 有约束优化：带线性约束的二次函数

问题：
```
minimize (x-3)^2 + (y-2)^2
subject to:
    x + y ≤ 4
    x ≥ 0
    y ≥ 0
```

### 3. 最优控制：一维运动控制

问题：控制一个物体从位置0移动到位置10，在时间T内停止，同时最小化控制能量和总时间。

动力学方程：
```
x_dot = v
v_dot = u
```

约束：
- 初始条件：x(0)=0, v(0)=0
- 终端条件：x(T)=10, v(T)=0
- 控制约束：-2 ≤ u ≤ 2

## 求解器

demo中使用了IPOPT求解器，这是CasADi默认的非线性规划求解器。IPOPT是一个开源的内点法求解器，适用于大规模的连续优化问题。

## 注意事项

1. 确保系统中已经安装了IPOPT求解器。如果没有，CasADi可能会使用其他可用的求解器。
2. 对于最优控制示例，离散化步数N=20，可以根据需要调整以获得更高的精度。
3. 如果遇到求解器错误，可以尝试调整初始猜测或求解器参数。

## CasADi官方资源

- [CasADi官方网站](https://web.casadi.org/)
- [CasADi文档](https://web.casadi.org/docs/)
- [CasADi Python API](https://web.casadi.org/python-api/)