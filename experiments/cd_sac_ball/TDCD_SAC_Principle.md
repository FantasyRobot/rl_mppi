# TD‑CD‑SAC：原理、公式与优势（对应本项目实现）

本文总结 **TD‑CD‑SAC（Temporal‑Difference Constraint‑Discounted SAC）** 的核心思想：

- 用 TD‑CD‑MPPI Eq.(6)–(9) 的“**随机终止（stochastic termination）/软终止**”机制，将约束影响注入到 TD 备份中；
- 在 SAC 的 critic 更新里把固定折扣 $\gamma$ 替换成**每步时变折扣** $\gamma_t$；
- actor/entropy 更新仍是标准 SAC，仅通过 Q 传播间接感知约束风险。

本项目实现落地为：**每条 transition 存一个 `discount_t`，critic target 用**

$$
 y_t = r_t + \texttt{discount}_t\,V(s_{t+1})
$$

其中 $\texttt{discount}_t = \gamma_t$。

---

## 1. 问题设置

状态与动作（以本项目 ball 环境为例）：

- 状态：$s_t=[x_t, y_t, v_{x,t}, v_{y,t}]$
- SAC 策略输出：$u_t\in[-1,1]^2$
- 物理加速度（分量缩放）：

$$
 a_{x,t}=a_{\max}u_{x,t},\qquad a_{y,t}=a_{\max}u_{y,t}
$$

约束（分量约束示例）：

$$
|v_{x,t}|\le v_{\max},\quad |v_{y,t}|\le v_{\max}
$$

环境可以构造一个约束信号 $c_t\ge 0$（连续）或 $c_t\in\{0,1\}$（二值）。例如连续版本：

$$
 c_t = \max(0,|v^{raw}_{x,t+1}|-v_{\max}) + \max(0,|v^{raw}_{y,t+1}|-v_{\max})
$$

---

## 2. TD‑CD 的关键：从约束到“软终止强度” $\delta_t$

TD‑CD‑MPPI Eq.(6)–(9) 的本质是：把约束违规解释为“终止的概率/强度”。

### 2.1 约束集合（Eq.6）

一般形式：

$$
 c_i(x,u)\le 0,\quad \forall i\in\mathcal{I}
$$

### 2.2 软终止信号（Eq.7）

将归一化违规程度映射到 $\delta_t\in[0,1]$：

$$
\delta_t = p^{\max}\cdot\mathrm{clip}\Big(\frac{c_t}{c^{\max}}, 0, 1\Big)
$$

- $p^{\max}\in[0,1]$ 控制“对违规的敏感度”（本项目参数 `tdcd_p_max`）
- $c^{\max}$ 是用于归一化的尺度

### 2.3 归一化尺度更新（Eq.8）

论文中使用指数平滑：

$$
 c^{\max} \leftarrow \tau_c\,c^{\max} + (1-\tau_c)\,\bar c^{\max}
$$

- $\bar c^{\max}$：当前迭代/window 内观测到的最大违规量
- $\tau_c\in[0,1]$：平滑系数（本项目参数 `tdcd_tau_c`）

在本项目实现中，`eval_every` 作为一个 window：每个 window 收集到 `c_max_seen`，在评估点用 EMA 更新 `c_max_ema`。

---

## 3. 从 Eq.(9) 得到“每步时变折扣” $\gamma_t$

论文的 constraint‑discounted return（Eq.9）为：

$$
R=\sum_{t=0}^{H}\Big(\prod_{k=0}^{t}\gamma_0(1-\delta_k)\Big)\,r_t
$$

注意到每一步都出现乘子 $\gamma_0(1-\delta_t)$，因此可以定义：

$$
\gamma_t \triangleq \gamma_0(1-\delta_t)
$$

这就是 TD‑CD 要注入 TD 学习的“有效折扣”。

---

## 4. TD‑CD‑SAC：把 $\gamma$ 换成 $\gamma_t$

SAC 的 critic 目标通常写成：

$$
 y_t = r_t + (1-d_t)\gamma\,V(s_{t+1})
$$

其中

$$
V(s_{t+1}) = \mathbb{E}_{a\sim\pi_\phi}\big[\min(Q_{\bar\theta_1},Q_{\bar\theta_2})(s_{t+1},a) - \alpha\log\pi_\phi(a|s_{t+1})\big]
$$

TD‑CD‑SAC 的做法是把固定折扣替换为时变折扣：

$$
 y_t = r_t + \gamma_t\,V(s_{t+1}),\qquad \gamma_t=\gamma(1-\delta_t)
$$

对应的双 Q 损失：

$$
\mathcal{L}_{Q_i}(\theta_i)=\mathbb{E}\big[(Q_{\theta_i}(s_t,a_t)-y_t)^2\big],\quad i\in\{1,2\}
$$

**actor 更新（标准 SAC）不变**：

$$
\mathcal{L}_\pi(\phi)=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_\phi}\big[\alpha\log\pi_\phi(a|s)-\min(Q_{\theta_1},Q_{\theta_2})(s,a)\big]
$$

entropy 温度 $\alpha$ 的自动调节也保持不变。

> 直观理解：约束不通过“终止/惩罚”硬写入奖励，而是通过降低 $\gamma_t$ 抑制未来价值传播，使违反/接近违反约束的状态‑动作对长期价值贡献变小，actor 会被间接推向更安全的行为。

---

## 5. 本项目的实现对应（从公式到代码）

- 环境提供约束信号：
  - `constraint_violation`（二值）与 `vel_violation_amount`（连续）
  - 位置：`env/envball_constraints.py`
- 训练时根据 Eq.(7)(8) 计算 $\delta_t$ 并写入 replay 的 `discount_t=\gamma(1-\delta_t)`：
  - 位置：`experiments/cd_sac_ball/train_cd_sac_ball_online.py`
- SAC 更新时若 batch 含 `discount` 字段，则用 TD‑CD 目标：
  - 位置：`algorithms/sac/sac_utils.py`（`SACAgent.update()` 中 `reward + discount * target`）

---

## 6. 优势总结（为什么用 TD‑CD‑SAC）

1) **学习信号更平滑**
- 与把违规当作硬终止相比，TD‑CD 不会频繁切断 bootstrap；约束影响以连续的 $\gamma_t$ 形式进入 TD 目标，训练更稳定。

2) **样本利用率更高**
- 违规轨迹不会被“整段作废”，而是按违规强度降低未来回报传播，仍能提供有用的梯度信息。

3) **与 off‑policy replay 自然兼容**
- 只需把 $\gamma_t$ 作为每条 transition 的字段存入 replay；训练时无需重放整条轨迹。

4) **约束影响直观、可解释**
- $\delta_t$ 可解释为“软终止强度/概率”，$\gamma_t$ 是“考虑约束后的有效折扣”。

5) **可扩展：二值/连续约束信号都支持**
- 用二值 $c_t$ 时实现简单；用连续 $c_t$ 时可让“轻微违规/严重违规”在价值传播中产生不同影响。
