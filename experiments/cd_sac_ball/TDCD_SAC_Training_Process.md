# TD-CD-SAC 训练流程（论文可直接引用）

我们采用 TD-CD（Temporal-Difference Constraint-Discounting）的思想将约束信息以“软终止”形式注入到 SAC 的时序差分备份中，实现对约束的折扣化处理。训练过程为在线交互式 off-policy 学习：智能体在受约束动力学环境中滚动采样转移 $(s_t,u_t,r_t,s_{t+1})$，并将每一步的约束违反程度映射为一个软终止强度 $\delta_t\in[0,1]$，进而得到时变折扣因子 $\gamma_t=\gamma(1-\delta_t)$。与传统做法（对违反约束直接终止或额外奖励惩罚）不同，TD-CD 不改变即时奖励 $r_t$ 的定义，而是通过降低未来回报的传播强度来抑制“风险轨迹”对价值函数的贡献，从而诱导策略趋向安全区域。

具体而言，我们首先构造约束信号 $c_t$。在二值模式下，若任一约束被违反则 $c_t=1$，否则 $c_t=0$；在连续模式下，$c_t$ 取为约束超限量（例如速度分量越界的溢出量之和），从而能够区分轻微与严重违规。随后根据归一化的约束强度计算软终止强度：

$$
\delta_t = p^{\max}\cdot \mathrm{clip}\!\left(\frac{|c_t|}{c^{\max}},0,1\right),
$$

其中 $p^{\max}\in[0,1]$ 控制软终止的最大强度，$c^{\max}$ 为约束信号的归一化尺度。为避免 $c^{\max}$ 由单次极端样本造成剧烈波动，我们在固定窗口上统计当前窗口内观测到的最大违规量 $\bar c^{\max}$，并使用指数滑动平均（EMA）更新尺度：

$$
 c^{\max} \leftarrow \tau_c\,c^{\max} + (1-\tau_c)\,\bar c^{\max},
$$

其中 $\tau_c\in[0,1]$ 为平滑系数。最终，每一步用于 TD 学习的有效折扣为

$$
\gamma_t=\gamma(1-\delta_t).
$$

当状态为真正终止（到达目标）时，我们令 $\gamma_t=0$；对于时间上限（time-limit）截断，我们将其视为非终止样本以避免对 bootstrap 引入偏差。

在学习更新方面，我们使用标准 SAC 的双 Q 网络与最大熵策略更新框架，但在 critic 的 TD 目标中使用时变折扣 $\gamma_t$。对从回放池采样的 minibatch $\{(s_i,u_i,r_i,s'_i,\gamma_i)\}$，先构造软值函数目标

$$
V(s'_i)=\mathbb{E}_{a\sim\pi}\Big[\min(Q_{\bar\theta_1}(s'_i,a),Q_{\bar\theta_2}(s'_i,a)) - \alpha\log\pi(a|s'_i)\Big],
$$

并令

$$
y_i = r_i + \gamma_i\,V(s'_i).
$$

随后通过最小化均方误差更新两个 critic：$\mathcal{L}_{Q_j}=\mathbb{E}[(Q_{\theta_j}(s_i,u_i)-y_i)^2]$。actor 与温度参数 $\alpha$ 的更新保持 SAC 标准形式不变，约束信息仅通过 critic 的时变折扣在价值传播中体现。最后对目标 Q 网络执行软更新 $\bar\theta_j\leftarrow\tau\theta_j+(1-\tau)\bar\theta_j$。这一流程使得约束影响以连续、可解释的方式进入 TD 学习，通常比“硬终止/硬惩罚”更平滑、更稳定，也更适合与 off-policy 回放结合。
