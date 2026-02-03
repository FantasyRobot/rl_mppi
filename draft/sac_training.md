下面给你一段可以直接放进论文“方法/训练原理”部分的文字（偏正式、可复用），包含：SAC 的优化目标与更新公式、为什么必须采用在线交互数据收集（避免离线数据集下 OOD/外推误差导致的不收敛）、以及两个在你这个任务里决定性稳定的工程细节（time-limit 截断处理、相对目标归一化）。

---

**方法：基于在线交互数据的 Soft Actor-Critic（SAC）训练**

我们在连续控制任务中采用 Soft Actor-Critic（SAC）学习随机策略 $\pi_\phi(a|s)$。SAC 在最大熵强化学习框架下，同时最大化期望回报与策略熵，从而在学习过程中保持探索性并提高训练稳定性。其目标函数为
$$
\max_{\pi}\ \mathbb{E}_{\tau\sim\pi}\Big[\sum_{t=0}^{\infty}\gamma^t\big(r(s_t,a_t)+\alpha\,\mathcal{H}(\pi(\cdot|s_t))\big)\Big],
$$
其中 $\gamma$ 为折扣因子，$\mathcal{H}(\pi(\cdot|s))$ 表示策略熵，$\alpha$ 为温度参数（可通过自动熵调节自适应学习）。

**Critic 更新**  
SAC 采用双 Q 网络 $Q_{\theta_1}(s,a),Q_{\theta_2}(s,a)$ 以缓解 Q 值过估计。对于经验回放池 $\mathcal{D}$ 中的样本 $(s,a,r,s',d)$，soft Bellman 备份目标为
$$
y=r+\gamma(1-d)\Big(\min_{i\in\{1,2\}}Q_{\bar\theta_i}(s',a')-\alpha\log\pi_\phi(a'|s')\Big),\quad a'\sim\pi_\phi(\cdot|s'),
$$
并最小化 TD 误差
$$
\min_{\theta_i}\ \mathbb{E}_{(s,a,r,s',d)\sim\mathcal{D}}\big[(Q_{\theta_i}(s,a)-y)^2\big].
$$
其中 $\bar\theta_i$ 为 target critic 参数，通过软更新维持训练稳定性。

**Actor 更新**  
策略通过最大化“高 Q + 高熵”目标来更新：
$$
\min_{\phi}\ \mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_\phi}\Big[\alpha\log\pi_\phi(a|s)-\min_{i\in\{1,2\}}Q_{\theta_i}(s,a)\Big].
$$
在实现上，我们采用高斯策略并通过 $\tanh$ 将动作限制到 $[-1,1]$，同时在对数概率中加入 $\tanh$ 的雅可比修正以保证梯度正确。

---

**为什么需要“在线交互数据收集”（否则离线 OOD 严重）**

尽管 SAC 算法本身属于 off-policy 方法，但在本任务中，训练必须采用**在线交互数据收集（online / interactive）**：即训练数据持续由当前策略与环境交互产生并写入回放池，而不是固定使用离线数据集。

原因在于：当仅使用固定离线数据集 $\mathcal{D}_\mu$（由某行为策略 $\mu$ 收集）时，critic 的目标 $y$ 需要评估 $Q(s',a')$，其中 $a'\sim\pi_\phi(\cdot|s')$。随着训练推进，$\pi_\phi$ 往往会偏离 $\mu$，导致 $(s',a')$ 落入数据集覆盖之外的区域（out-of-distribution, OOD）。此时 critic 只能对 OOD 区域进行函数外推，容易产生显著的外推误差（extrapolation error）或虚高的 Q 值，从而形成不稳定的闭环：

1) critic 在 OOD 动作上错误高估 $Q(s,a)$；  
2) actor 被“虚高 Q”吸引，输出更 OOD 的动作；  
3) OOD 程度进一步增加，critic 更不可靠，最终导致策略发散、塌缩或产生投机行为。

因此，采用在线交互训练使回放池分布 $\mathcal{D}$ 随策略同步更新，更接近当前策略的访问分布，从而显著缓解 OOD 外推误差，提高 critic 目标的可靠性与策略学习的稳定性。

论文中推荐的严谨表述是：
> SAC 虽为 off-policy 算法，但在该任务中需要在线交互采样以维持数据分布与当前策略的一致性；固定离线数据集会导致严重的 OOD 动作与 critic 外推误差，进而使策略更新被错误的 Q 值引导而难以收敛。

---

**与本任务收敛性强相关的两个实现细节**

1) **Time-limit 截断不应视为真正终止（truncation $\neq$ terminal）**  
当回合因达到最大步数 $T_{\max}$ 结束时，这通常是“时间截断”，并不意味着进入吸收终止状态。若将其作为 terminal（即令 $d=1$），会使目标 $y$ 在 $T_{\max}$ 处停止自举（bootstrap），从而系统性低估价值函数，诱发策略向“小动作/不动”的方向塌缩。标准做法是在 time-limit 截断时令 $d=0$ 继续自举。

2) **相对目标的状态表示与归一化（提升可学习性与泛化）**  
将状态从绝对坐标改为相对目标坐标（例如使用 $(x-x_g, y-y_g)$），并按环境边界对位置/速度缩放，使输入量纲一致且具有平移对称性，有助于减轻函数逼近难度并提升训练稳定性。进一步地，可采用“先近后远”的初始化 curriculum（例如初始位置采样半径从小逐步增大），使策略先掌握末端收敛行为再学习远距离情形的泛化控制。

---

**可选参考文献（你可按论文格式改 BibTeX）**
- Haarnoja, Tuomas, et al. “Soft Actor-Critic Algorithms and Applications.” (SAC)
- Kumar, Aviral, et al. “Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction.” (BEAR，讨论离线 RL 的 OOD/外推问题)
- Kumar, Aviral, et al. “Conservative Q-Learning for Offline Reinforcement Learning.” (CQL，离线 RL 中抑制 OOD 过估计)
- Fujimoto, Scott, et al. “Off-Policy Deep Reinforcement Learning without Exploration.” (BCQ，离线 RL 中 OOD 动作问题的经典讨论)

---

如果你告诉我：论文是中文还是英文、以及你“方法章节”的小标题结构（比如 3.1 算法、3.2 训练细节、3.3 稳定性处理），我可以把这段进一步改成完全贴合你论文格式的一版（含小节编号、符号表一致、参考文献占位符如 [1][2][3]）。