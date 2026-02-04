下面是一份可直接作为“多关节机械臂避障轨迹规划（MPPI + CDF）”论文基础资料/预备知识的整理稿，重点补全符号、公式与常用实现细节。内容尽量保持“可粘贴进论文”的写法；你可以在此基础上再加入具体机械臂参数、碰撞模型与实验设定。

---

# MPPI + CDF：多关节机械臂避障轨迹规划基础资料

## 1. 问题定义与符号（Manipulator Planning Setup）

本文设定与你的任务一致：**6 自由度机械臂**（$n=6$），目标为**末端到达指定位置**（只关心位置，不强制姿态），运动过程中避开**球形障碍物**。

关节变量 $q\in\mathbb{R}^6$，关节速度 $\dot q\in\mathbb{R}^6$，控制输入 $u\in\mathbb{R}^6$。

常见的两类控制建模：

- **关节速度控制**（最简单、常用于轨迹规划器外环）：
  $$q_{t+1}=q_t + \Delta t\,u_t,\qquad u_t\approx \dot q_t.$$

- **关节加速度控制**（更接近动力学、但实现更复杂）：
  $$\begin{aligned}
  q_{t+1} &= q_t + \Delta t\,\dot q_t + \tfrac12\Delta t^2\,u_t,\\
  \dot q_{t+1} &= \dot q_t + \Delta t\,u_t.
  \end{aligned}$$

统一写成离散系统
$$x_{t+1}=f(x_t,u_t),\qquad x_t=\begin{bmatrix}q_t\\ \dot q_t\end{bmatrix} \in\mathbb{R}^{2n}.$$

目标是在滚动时域 $H$ 上最小化累计代价（或最大化回报）：
$$S(\mathbf U;x_t)=\sum_{i=0}^{H-1}\ell(x_{t+i},u_{t+i})+\ell_f(x_{t+H}),\quad \mathbf U=\{u_t,\ldots,u_{t+H-1}\}.$$

并满足关节/速度/控制边界（硬约束或软惩罚）：
$$q_{\min}\le q_t\le q_{\max},\quad \dot q_{\min}\le \dot q_t\le \dot q_{\max},\quad u_{\min}\le u_t\le u_{\max}.$$

---

## 2. 机械臂几何：正向运动学、雅可比与离散连杆点

### 2.1 正向运动学（Forward Kinematics）

用 $\mathrm{FK}_j(q)\in\mathbb{R}^3$ 表示第 $j$ 个几何点（例如连杆上一点、关节、末端、或胶囊体采样点）在任务空间中的位置。

- 末端位姿：$p_{ee}(q)=\mathrm{FK}_{ee}(q)$
- 连杆采样点：$p_j(q)=\mathrm{FK}_j(q)$（$j=1,\dots,M$）

通常避障用 **离散点集** 覆盖整条机械臂几何：
$$\mathcal P(q)=\{p_1(q),\ldots,p_M(q)\}.$$

### 2.2 雅可比（Jacobian）

对任意点 $p_j(q)$，其雅可比矩阵
$$J_j(q)=\frac{\partial p_j(q)}{\partial q}\in\mathbb{R}^{3\times n}.$$

雅可比用于把任务空间距离场梯度映射到关节空间（CDF 梯度推导会用到）。

---

## 3. 任务空间有符号距离场（SDF）与碰撞距离（球形障碍物特化）

给定环境障碍集合 $\mathcal O\subset\mathbb{R}^3$，定义任务空间 SDF：
$$\mathrm{SDF}(x)=\begin{cases}
+d(x,\partial\mathcal O), & x\notin\mathcal O,\\
0, & x\in \partial\mathcal O,\\
-d(x,\partial\mathcal O), & x\in\mathcal O.
\end{cases}$$
其中 $d(\cdot,\cdot)$ 为欧氏距离。

### 3.1 球形障碍物的 SDF

若障碍物由 $N_o$ 个球组成，第 $m$ 个球的中心与半径分别为 $c_m\in\mathbb{R}^3$、$r_m>0$，则单球 SDF 为：
$$\mathrm{SDF}_m(x)=\|x-c_m\|_2-r_m.$$
多球障碍物的 SDF 可写为：
$$\mathrm{SDF}(x)=\min_{m\in\{1,\dots,N_o\}}\mathrm{SDF}_m(x).$$

### 3.2 机械臂-障碍物距离（点集/球串/胶囊近似）

对机械臂配置 $q$，常用“最小间隙”定义机械臂到障碍物的距离。若用离散点集（或球串）覆盖机械臂几何：
$$\mathcal P(q)=\{p_1(q),\ldots,p_M(q)\},$$
并给每个点一个等效半径 $r_j\ge 0$（球串/连杆厚度近似；若只是点采样可取 $r_j=0$），则与第 $m$ 个球障碍的**有符号间隙**：
$$d_{j,m}(q)=\|p_j(q)-c_m\|_2-(r_m+r_j).$$
机械臂对多球障碍的最小间隙：
$$d_{\text{task}}(q)=\min_{j\in\{1,\dots,M\},\,m\in\{1,\dots,N_o\}} d_{j,m}(q).$$

- 若 $d_{\text{task}}(q)>0$：整机臂在安全区外。
- 若 $d_{\text{task}}(q)=0$：刚好接触。
- 若 $d_{\text{task}}(q)<0$：发生穿透/碰撞。

> 工程上常加入额外安全裕度（膨胀） $r_{\text{safety}}>0$：
> $$d_{\text{task}}^{\text{safe}}(q)=d_{\text{task}}(q)-r_{\text{safety}}.$$

### 3.3 梯度（用于斥力 nominal / 投影；MPPI 本身不强制需要）

若 $d_{\text{task}}(q)$ 的最小值在某个 $(j^*,m^*)$ 上取得（实际实现可用 soft-min 平滑），记
$$x^*(q)=p_{j^*}(q),\quad d(q)=d_{j^*,m^*}(q)=\|x^*(q)-c_{m^*}\|_2-(r_{m^*}+r_{j^*}),$$
则
$$\nabla_x d=\frac{x^*(q)-c_{m^*}}{\|x^*(q)-c_{m^*}\|_2},\qquad \nabla_q d=J_{j^*}(q)^\top\nabla_x d.$$

---

## 4. 配置空间距离场（CDF）的定义与性质

### 4.1 C-obstacle 与 CDF

配置空间 $\mathcal C\subset\mathbb{R}^n$ 中，使机械臂发生碰撞的配置集合定义为 C-obstacle：
$$\mathcal C_{\text{obs}}=\{q\in\mathcal C\mid \exists p\in\mathcal P(q),\ \mathrm{SDF}(p)\le 0\}.$$

CDF 是定义在配置空间的有符号距离场：
$$\mathrm{CDF}(q)=\text{sign}(q)\,\min_{q^*\in\partial\mathcal C_{\text{obs}}}\|q-q^*\|_2.$$

其中 $\text{sign}(q)$ 满足：外部为 $+1$，内部为 $-1$，边界为 $0$。

### 4.2 CDF 与 SDF 的关系（常用近似）

严格意义上 CDF 是“关节空间到碰撞边界的距离”，而 $d_{\text{task}}(q)$ 是“任务空间点到障碍物表面的距离”。两者不同：

- **SDF**：在 $\mathbb{R}^3$ 上；直接描述几何间隙。
- **CDF**：在 $\mathbb{R}^n$ 上；直接描述“关节变化多少能脱离碰撞/保持安全”。

在许多规划器中，常用 $d_{\text{task}}(q)$ 或其安全版本 $d_{\text{task}}^{\text{safe}}(q)$ 作为避障距离（因为更容易算），并把它当作“近似 CDF”参与代价。

如果你希望严格使用 CDF（与仓库内的 CDF2D 思路一致），可以使用“边界配置集” $\{q^*\}$ 做最近邻距离来获得 $|\mathrm{CDF}(q)|$，符号由任务空间 SDF 决定（见下一节）。

---

## 5. CDF 的计算：在线边界集 / 离线网格 / 神经网络拟合

参考你仓库的 CDF2D 文档，常见三种路线：

### 5.1 在线计算（Online computation）

核心思想：先得到障碍物对应的“零水平集边界配置集合” $\mathcal Q_0\approx\{q\mid \mathrm{SDF}(\mathcal P(q))=0\}$，然后对任意 $q$ 计算到该集合的最近距离。

- 绝对值：
  $$|\mathrm{CDF}(q)|\approx \min_{q^*\in\mathcal Q_0}\|q-q^*\|_2.$$
- 符号：用任务空间最小 SDF 判断是否碰撞：
  $$\mathrm{CDF}(q)=\begin{cases}
  +|\mathrm{CDF}(q)|, & d_{\text{task}}(q)>0,\\
  -|\mathrm{CDF}(q)|, & d_{\text{task}}(q)\le 0.
  \end{cases}$$

优点：不需要预生成大表；缺点：首次构建 $\mathcal Q_0$ 可能较慢。

### 5.2 离线网格/模板（Offline grid / template）

将任务空间障碍物表面点采样映射到对应的“边界配置集合”，做成模板表（例如用网格索引 / KD-tree），查询时只在局部候选集上做最近邻。

优点：查询快；缺点：预处理成本和内存开销大，维度高时更明显（$n$ 越大越难做全空间网格）。

### 5.3 神经网络拟合（Neural CDF / Differentiable Approximation）

训练一个网络 $g_\phi(q)\approx \mathrm{CDF}(q)$ 或 $g_\phi(q)\approx d_{\text{task}}^{\text{safe}}(q)$。

优点：推理快、可自动微分得到梯度；缺点：需要数据，且需要控制泛化误差（尤其碰撞边界附近）。

---

## 6. CDF/距离的梯度：用于名义控制、投影或优化（可选）

MPPI 本身不要求梯度，但在构造 nominal（例如 PD + 斥力）或做投影/约束处理时，梯度非常有用。

### 6.1 任务空间距离梯度到关节空间梯度

令 $p^*(q)$ 是达到最小距离的“最近点”（某个连杆采样点），则
$$d_{\text{task}}(q)=\mathrm{SDF}(p^*(q)).$$

链式法则：
$$\nabla_q d_{\text{task}}(q)=J_{p^*}(q)^\top\,\nabla_x \mathrm{SDF}(x)\big|_{x=p^*(q)}.$$

其中 $\nabla_x\mathrm{SDF}(x)$ 是任务空间 SDF 的空间梯度（方向指向“远离障碍物”的法向）。

### 6.2 基于 CDF 的投影（Projection to boundary）

若你有可微的 $\mathrm{CDF}(q)$，可用投影公式把任意 $q$ 拉回边界：
$$q_{\text{proj}} = q - \mathrm{CDF}(q)\,\frac{\nabla_q \mathrm{CDF}(q)}{\|\nabla_q \mathrm{CDF}(q)\|_2}.$$

---

## 7. MPPI：用于机械臂关节空间避障轨迹规划（末端位置目标 + 球形障碍物）

### 7.1 采样、权重与更新

在时刻 $t$ 维护名义控制序列 $\bar{\mathbf U}=\{\bar u_0,\ldots,\bar u_{H-1}\}$。

对 $k=1,\ldots,K$ 采样噪声：
$$\epsilon_i^k\sim\mathcal N(0,\Sigma),\quad i=0,\ldots,H-1,$$
并形成候选控制：
$$u_i^k = \bar u_i + \epsilon_i^k.$$

对每条样本 roll out 得到轨迹 $\{x_{t+i}^k\}$ 并计算代价 $S_k$。

使用路径积分权重（带 baseline 稳定数值）：
$$w_k=\exp\left(-\frac{1}{\lambda}(S_k-S_{\min})\right),\quad \tilde w_k=\frac{w_k}{\sum_j w_j},\quad S_{\min}=\min_j S_j.$$

更新名义控制：
$$\bar u_i\leftarrow \bar u_i+\sum_{k=1}^{K}\tilde w_k\,\epsilon_i^k,\quad i=0,\ldots,H-1.$$

执行 $u_t=\bar u_0$ 并 shift 序列进入下一时刻。

> 说明：$\epsilon$ 就是采样扰动噪声（探索/搜索半径）；$\Sigma$ 决定每个关节扰动幅度及关节间相关性；$\lambda$ 控制“只相信最优样本”还是“平均一批好样本”。

### 7.2 代价函数（到达 + 平滑 + 约束 + 避障）

对机械臂常见结构：
$$\ell(x,u)=\ell_{\text{goal}}(q)+\ell_{\text{smooth}}(u)+\ell_{\text{limits}}(x,u)+\ell_{\text{obs}}(q),\qquad \ell_f(x)=\beta\,\ell_{\text{goal}}(q).$$

- **目标项（末端位置）**：给定目标点 $p_g\in\mathbb{R}^3$，
  $$\ell_{\text{goal}}(q)=\|p_{ee}(q)-p_g\|_2^2.$$
  终端代价常用更大权重：
  $$\ell_f(x)=\beta\,\|p_{ee}(q_{t+H})-p_g\|_2^2,\quad \beta\gg 1.$$

- **控制/平滑项**（避免关节抖动）：
  $$\ell_{\text{smooth}}(u)=\rho\|u\|_2^2\quad\text{或}\quad \rho\|u-u_{\text{prev}}\|_2^2.$$

- **边界软惩罚**（关节限位、速度限位、控制限位）：
  $$\ell_{\text{limits}}=w_q\|[q-q_{\max}]_+\|_2^2+w_q\|[q_{\min}-q]_+\|_2^2+\cdots$$
  其中 $[z]_+=\max(z,0)$ 按元素作用。

- **避障项（CDF/SDF）**：令 $d(q)$ 为距离（论文与实现里最常用的是 $d_{\text{task}}^{\text{safe}}(q)$；若你有 $\mathrm{CDF}(q)$ 也可直接替换）。在“球形障碍物 + 球串/点集机械臂近似”下，可以直接使用第 3 节的
  $$d(q)=d_{\text{task}}^{\text{safe}}(q)=\min_{j,m}\big(\|p_j(q)-c_m\|_2-(r_m+r_j)\big)-r_{\text{safety}}.$$
  给出两种常用形式：

  1) **安全距离 hinge**（推荐，稳定易调）：
  $$\ell_{\text{obs}}(q)=w_{\text{obs}}\,\big[ -d(q) \big]_+^2 + w_{\text{col}}\,\mathbb{I}[d(q)<0].$$

  2) **光滑 barrier**（更“连续”，但可能数值偏硬）：
  $$\ell_{\text{obs}}(q)=w_{\text{obs}}\,\exp\big(-\kappa\,d(q)\big),\quad d(q)\to -\infty\Rightarrow\text{代价爆炸}.$$

  其中 $w_{\text{col}}$ 是碰撞硬惩罚（通常远大于其它项）。注意这里把安全裕度写进了 $d(q)$（通过 $r_{\text{safety}}$）；若你更习惯显式 $d_{\text{safe}}$，可把 hinge 写成 $[d_{\text{safe}}-d_{\text{task}}(q)]_+^2$。

---

## 8. MPPI + CDF 的一个论文级算法描述（可直接引用）

**Algorithm: MPPI-CDF for Manipulator Obstacle Avoidance**

**Input:** 当前状态 $x_t$，时域 $H$，样本数 $K$，温度 $\lambda$，噪声协方差 $\Sigma$，离散动力学 $f$，代价项（含避障距离 $d(q)$），约束（关节/速度/控制边界）。

1. 初始化/继承名义控制序列 $\bar{\mathbf U}$。
2. 对 $k=1,\dots,K$：采样 $\epsilon_i^k\sim\mathcal N(0,\Sigma)$，令 $u_i^k=\bar u_i+\epsilon_i^k$，并对 $u_i^k$ 做饱和/投影以满足 $u$ 边界。
3. 对每条样本控制序列从 $x_t$ rollout：$x_{t+i+1}^k=f(x_{t+i}^k,u_i^k)$，累计代价
   $$S_k=\sum_{i=0}^{H-1}\ell(x_{t+i}^k,u_i^k)+\ell_f(x_{t+H}^k),\quad \ell\text{中含}\ell_{\text{obs}}(q)\text{基于 }d(q).$$
4. 计算权重 $w_k=\exp(-(S_k-S_{\min})/\lambda)$ 并归一化。
5. 更新名义控制：$\bar u_i\leftarrow\bar u_i+\sum_k \tilde w_k\epsilon_i^k$。
6. 执行 $u_t=\bar u_0$；shift $\bar{\mathbf U}$ 并进入下一时刻。

---

## 9. 实现建议（写论文时可放到 Implementation Details）

- **碰撞几何建模（球形障碍物最省事版本）**：
  - 机械臂：用“球串/点集 + 半径”近似（每条连杆采样若干点，给定 $r_j$），或进一步用 capsule；
  - 障碍物：球体 SDF/距离有闭式表达式（第 3 节），计算非常快，适合放进 MPPI 的 $K\times H$ 批量 rollout。
- **距离计算的“最小值”不可微**：若你需要梯度，可用 soft-min：
  $$\min_j a_j \approx -\tau\log\sum_j \exp(-a_j/\tau).$$
- **高维 CDF 的离线网格难做**：$n\ge 6$ 时更推荐：任务空间 SDF + 连杆采样点最小间隙作为避障距离；或训练 NN 近似 $d(q)$。
- **MPPI 约束处理**：最常用的是“采样后 clip / squash”，再加 soft penalty；硬约束（投影到可行集）也可行但实现更复杂。
- **数值稳定**：使用 $S_{\min}$ baseline；对过大的碰撞代价使用上限裁剪，避免权重下溢导致“全 0”。

---

如果你希望我把这份资料进一步“论文化”，我可以按你的机械臂设定（例如 6-DOF，DH 参数、末端目标是位姿还是位置）把 $\ell_{\text{goal}}$、$d(q)$（连杆胶囊体到三角网格障碍物的距离）写成更具体、可直接复现的形式，并补一段和你上传的《基于 MPPI 的机械臂轨迹规划》PDF 对齐的符号表与章节结构。