# RLHF（基于PPO）完整流程详解

RLHF通常分为三个阶段，每个阶段都有明确的目标和挑战。

## 第一阶段：监督微调（Supervised Fine-Tuning, SFT）

**目标**：让预训练语言模型学会遵循指令，生成符合人类期望的初步回答。

**数据**：收集大量**人类撰写的高质量对话/指令数据**。例如：  
> 用户：解释什么是机器学习。  
> 人类：机器学习是人工智能的一个分支，它让计算机通过数据学习规律，而不需要显式编程……

**过程**：在预训练模型基础上，用上述数据进行标准的**最大似然估计（MLE）** 微调。  
$$
\max_{\pi_{\text{SFT}}} \mathbb{E}_{(x,y)\sim \mathcal{D}_{\text{SFT}}} \left[ \log \pi_{\text{SFT}}(y \mid x) \right]
$$  
得到的模型记为 $\pi^{\text{SFT}}$。

**作用**：这一步让模型初步具备“对话”和“指令跟随”的能力，但它回答的质量、无害性等未必符合人类偏好。

---

## 第二阶段：奖励建模（Reward Modeling）

**目标**：训练一个能够对任意 $(x,y)$ 打分的奖励模型 $r_\phi(x,y)$，该分数反映人类对该回答的偏好程度。

**数据**：基于 $\pi^{\text{SFT}}$ 对每个提示 $x$ 生成多个回答（通常2～4个），由人类标注**偏好排序**（例如 $y_w \succ y_l$，$w$ 表示“赢”，$l$ 表示“输”）。形成数据集 $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$。

**偏好模型**：假设人类偏好遵循 **Bradley-Terry（BT）模型**：
$$
P^*(y_w \succ y_l \mid x) = \frac{\exp(r^*(x,y_w))}{\exp(r^*(x,y_w)) + \exp(r^*(x,y_l))}
$$
其中 $r^*$ 是隐含的真实奖励函数。

**训练**：用参数化的 $r_\phi$ 拟合 $r^*$，采用负对数似然（等价于二分类交叉熵）：
$$
\mathcal{L}_R(r_\phi) = - \mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}} \left[ \log \sigma\big( r_\phi(x,y_w) - r_\phi(x,y_l) \big) \right]
$$
$\sigma$ 是 sigmoid 函数。该损失迫使模型给 $y_w$ 的分数高于 $y_l$。这个损失函数用于训练奖励模型（Reward Model），目的是让模型学会根据人类偏好对不同的回答进行评分。

## 公式解释

$$
\mathcal{L}_R(r_\phi) = - \mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}} \left[ \log \sigma\big( r_\phi(x,y_w) - r_\phi(x,y_l) \big) \right]
$$

**各符号含义：**
- $r_\phi$：待训练的奖励模型，参数为 $\phi$
- $x$：提示（用户输入）
- $y_w$：被人类标注为“更好”的回答（win）
- $y_l$：被人类标注为“更差”的回答（loss）
- $\mathcal{D}$：由 $(x, y_w, y_l)$ 三元组组成的偏好数据集
- $\sigma$：sigmoid函数，$\sigma(z) = \frac{1}{1+e^{-z}}$
- $\mathbb{E}$：期望值，即对数据集中所有样本取平均

**损失函数的工作原理：**
1. 对于每个 $(x, y_w, y_l)$ 三元组，计算 $r_\phi(x,y_w) - r_\phi(x,y_l)$
2. 通过sigmoid函数将其映射到(0,1)区间
3. 取对数后加负号作为损失

**直观理解：**
- 如果模型给 $y_w$ 的分数显著高于 $y_l$，则 $r_\phi(x,y_w) - r_\phi(x,y_l)$ 是一个较大的正数
- 经过sigmoid后接近1，$\log(接近1)$ 接近0，损失很小
- 如果模型给分错误（$y_w$ 分数低于 $y_l$），则差值为负，sigmoid输出接近0，$\log(接近0)$ 为负大数，加上负号后损失很大

## 具体计算示例

假设：
- 提示 $x$："解释什么是机器学习"
- 好回答 $y_w$："机器学习是人工智能的一个分支，让计算机从数据中学习规律"
- 差回答 $y_l$："机器学习就是编程让电脑做事"

**步骤1：模型给出初始分数**
- 奖励模型初始参数下：
  - $r_\phi(x, y_w) = 2.1$
  - $r_\phi(x, y_l) = 3.5$（错误：差回答分数更高）

**步骤2：计算分数差**
$$
r_\phi(x,y_w) - r_\phi(x,y_l) = 2.1 - 3.5 = -1.4
$$

**步骤3：通过sigmoid函数**
$$
\sigma(-1.4) = \frac{1}{1 + e^{1.4}} \approx \frac{1}{1 + 4.055} \approx 0.198
$$

**步骤4：计算损失**
$$
-\log(0.198) \approx -(-1.62) = 1.62
$$

**步骤5：反向传播与参数更新**
损失值1.62较大，梯度会：
1. 增加 $r_\phi(x, y_w)$ 的分数
2. 降低 $r_\phi(x, y_l)$ 的分数

**步骤6：更新后的分数（假设学习后）**
- $r_\phi(x, y_w) = 3.8$
- $r_\phi(x, y_l) = 2.2$
- 差值：$3.8 - 2.2 = 1.6$
- $\sigma(1.6) \approx 0.832$
- 损失：$-\log(0.832) \approx 0.184$

此时损失显著减小，模型学会了正确评分。

## 与二分类交叉熵的等价性

该损失函数等价于一个二分类问题：
- 将"$y_w$ 优于 $y_l$"视为正类（标签为1）
- 模型预测的概率为 $\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))$
- 使用二分类交叉熵损失：$-\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]$

因为 $y=1$（总是认为 $y_w$ 更好），所以简化为：
$$
-\log(\sigma(r_\phi(x,y_w) - r_\phi(x,y_l)))
$$

这正是原公式的形式。这种设计确保了模型学习的是相对偏好（哪个更好），而不是绝对分数。

**实现细节**：
- 通常将 $r_\phi$ 初始化为 $\pi^{\text{SFT}}$ 的权重，并在最后一层替换为一个线性层，输出一个标量分数。
- 训练时对奖励进行**归一化**（如减均值），使分数稳定。

**结果**：得到一个奖励模型 $r_\phi$，它能对任意回答打分。但此时模型还不能直接用于生成，因为最大化该奖励会导致语言退化（如重复、混乱）。

---

## 第三阶段：强化学习微调（RL Fine-Tuning）

**目标**：在最大化奖励的同时，约束新策略 $\pi_\theta$ 不要偏离 $\pi^{\text{SFT}}$ 太远，避免“奖励黑客”问题。

**优化目标**（KL约束的奖励最大化）：
$$
\max_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D}, y\sim \pi_\theta(\cdot\mid x)} \left[ r_\phi(x,y) \right] - \beta \,\mathbb{D}_{\text{KL}}\left[ \pi_\theta(y\mid x) \,\|\, \pi_{\text{ref}}(y\mid x) \right]
$$
其中：
- $\pi_{\text{ref}}$ 通常是 $\pi^{\text{SFT}}$，固定不变。
- $\beta$ 控制 KL 惩罚强度。

**为何不能直接梯度上升**：语言模型输出是离散的，无法直接对 $\pi_\theta$ 求导。所以采用强化学习（PPO）。

### 3.1 将目标转化为 PPO 可用的奖励

将 KL 惩罚项融合进即时奖励，定义**修正奖励**：
$$
\hat{r}(x,y) = r_\phi(x,y) - \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$
这样，原目标等价于最大化 $\mathbb{E}_{y\sim\pi_\theta} [\hat{r}(x,y)]$。

### 3.2 PPO 算法关键要素

PPO 是一种 on-policy 的 actor-critic 方法，包含：

- **Actor（策略）** $\pi_\theta$：生成回答。
- **Critic（价值网络）** $V_\psi(x)$：估计状态价值（即给定 $x$ 下能获得的期望累积奖励），用于计算优势函数。

**优势函数** $A(x,y)$：衡量当前回答相对于平均水平的好坏。常用 **GAE（Generalized Advantage Estimation）** ：
$$
A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \dots
$$
其中 $\delta_t = \hat{r}_t + \gamma V(x_{t+1}) - V(x_t)$。对于文本生成，通常将整个回答视为一个“轨迹”，计算序列级优势。

**PPO 的 Clip 目标**：
$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t(\theta) A_t, \; \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$
其中 $\rho_t(\theta) = \frac{\pi_\theta(y_t\mid x)}{\pi_{\text{old}}(y_t\mid x)}$，$\epsilon$ 为裁剪阈值（通常 0.1～0.2）。该目标限制了策略更新幅度，防止震荡。

### 3.3 训练流程

1. **采样**：用当前策略 $\pi_\theta$ 对一批提示 $x$ 生成回答 $y$。
2. **计算奖励**：用 $r_\phi$ 打分，并加上 KL 惩罚项得到 $\hat{r}$。
3. **计算优势**：通过 Critic $V_\psi$ 计算优势 $A$。
4. **更新 Actor**：用 PPO clip 目标最大化，通常使用 Adam 优化器。
5. **更新 Critic**：最小化 $ \left(V_\psi(x) - \hat{R}(x,y)\right)^2 $，其中 $\hat{R}$ 为实际累积奖励。
6. **重复**直到收敛。

### 3.4 RLHF 的复杂性与不稳定来源

- **四个模型**：Actor, Reference, Reward Model, Critic。显存占用巨大，需精心调度。
- **超参数众多**：$\beta$（KL 系数）、$\epsilon$（PPO clip）、GAE 的 $\gamma, \lambda$、学习率、batch size、rollout 数量等。调参困难。
- **奖励模型可能被“攻击”**：由于奖励模型是近似学习的，策略可能找到它未覆盖的漏洞（如输出无意义但得分高的文本）。
- **on-policy 采样**：每轮更新都要重新采样，计算成本高，且样本利用率低。
- **Critic 训练不稳定**：价值估计误差会影响优势计算，进而影响策略更新。
- **KL 约束的权衡**：$\beta$ 太小，模型可能忘记原始能力；太大，则奖励提升有限。

这些复杂性正是 DPO 试图解决的核心问题——**去掉奖励模型和 RL 循环，直接用偏好数据以监督方式训练**。

---
# RLHF-PPO 单样本损失计算数值示例

## 1. 生成与概率记录

当前策略 $\pi_\theta$ 对提示 $x$ 生成回答 $y = [w_1, w_2, w_3]$（三个词）。记录每个 token 的 log 概率（当前策略和参考模型）。

| token | $\log \pi_\theta$ | $\log \pi_{\text{ref}}$ |
|-------|------------------|-------------------------|
| w1    | -0.5             | -0.6                    |
| w2    | -0.8             | -0.7                    |
| w3    | -1.0             | -0.9                    |

序列总 log 概率：
$$
\log \pi_\theta(y|x) = -0.5 -0.8 -1.0 = -2.3
$$
$$
\log \pi_{\text{ref}}(y|x) = -0.6 -0.7 -0.9 = -2.2
$$

同时，为了计算 PPO 的比率 $\rho(\theta)$，需要知道**旧策略** $\pi_{\text{old}}$（即采样时的策略）在同样回答上的概率。假设旧策略下各 token log 概率分别为 -0.6, -0.7, -0.8，则：
$$
\log \pi_{\text{old}}(y|x) = -0.6 -0.7 -0.8 = -2.1
$$

## 2. 奖励与修正奖励

奖励模型 $r_\phi$ 对 $(x,y)$ 输出分数 $r_\phi(x,y) = 1.2$。

KL 惩罚系数 $\beta = 0.1$。计算 KL 项：
$$
\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} = 0.1 \times (-2.3 - (-2.2)) = 0.1 \times (-0.1) = -0.01
$$

修正奖励（即 PPO 中实际用于优势计算的奖励）为：
$$
\hat{r} = r_\phi(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} = 1.2 - (-0.01) = 1.21
$$
（注意：减去 KL 项，所以减负数得加法。）

由于这是一个完整的回答，通常将其视为一个“步骤”，最终累积回报 $R = \hat{r} = 1.21$。

## 3. 价值估计与优势

批评家网络 $V_\psi$ 对提示 $x$ 估计价值 $V_\psi(x) = 0.5$。

优势 $A = R - V_\psi(x) = 1.21 - 0.5 = 0.71$。

## 4. PPO Actor 损失

计算比率 $\rho(\theta) = \frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}$：
$$
\log \rho(\theta) = \log \pi_\theta - \log \pi_{\text{old}} = -2.3 - (-2.1) = -0.2
$$
$$
\rho(\theta) = \exp(-0.2) \approx 0.8187
$$

设定 PPO 裁剪参数 $\epsilon = 0.2$，则 clip 范围为 $[0.8, 1.2]$。

- 未裁剪项：$\rho(\theta) \cdot A = 0.8187 \times 0.71 \approx 0.5813$
- 裁剪项：$\text{clip}(\rho(\theta), 1-\epsilon, 1+\epsilon) = 0.8187$（在范围内），乘以 $A$ 得 $0.5813$

取最小值 $\min = 0.5813$。PPO 的 actor 目标是最大化这个值，但实际损失通常取负号，所以：
$$
\mathcal{L}_{\text{actor}} = - \min(...) = -0.5813
$$

## 5. Critic 损失

Critic 损失为价值估计与实际回报的均方误差：
$$
\mathcal{L}_{\text{critic}} = \left( V_\psi(x) - R \right)^2 = (0.5 - 1.21)^2 = (-0.71)^2 = 0.5041
$$

## 6. 总损失（简化）

忽略熵奖励，总损失：
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} + \mathcal{L}_{\text{critic}} = -0.5813 + 0.5041 = -0.0772
$$
实际优化器会使用梯度下降（最小化损失），因此这个负值表示更新会增加该样本的优势贡献。

## 表1：RLHF（PPO）阶段数值计算示例
| 步骤                 | 变量/表达式          | 符号/公式                                                                                   | 数值                                           | 说明               |
| ------------------ | --------------- | --------------------------------------------------------------------------------------- | -------------------------------------------- | ---------------- |
| **1. 生成与概率**       | 提示              | $x$                                                                                     | “写一句猫的诗”                                     | 固定输入             |
|                    | 生成回答（3个token）   | $y = [w_1, w_2, w_3]$                                                                   | `["喵", "轻", "步"]`                            | 示例               |
|                    | 当前策略token级log概率 | $\log \pi_\theta(w_i \mid x, w_{<i})$                                                   | [-0.5, -0.8, -1.0]                           | 求和得到序列log概率      |
|                    | 序列总log概率        | $\log \pi_\theta(y \mid x)$                                                             | -2.3                                         | -0.5-0.8-1.0     |
|                    | 参考模型token级log概率 | $\log \pi_{\text{ref}}(w_i \mid \dots)$                                                 | [-0.6, -0.7, -0.9]                           |                  |
|                    | 序列总log概率（参考）    | $\log \pi_{\text{ref}}(y \mid x)$                                                       | -2.2                                         | -0.6-0.7-0.9     |
|                    | 旧策略token级log概率  | $\log \pi_{\text{old}}(w_i \mid \dots)$                                                 | [-0.6, -0.7, -0.8]                           | 采样时策略            |
|                    | 序列总log概率（旧）     | $\log \pi_{\text{old}}(y \mid x)$                                                       | -2.1                                         | -0.6-0.7-0.8     |
| **2. 奖励与修正奖励**     | 奖励模型输出          | $r_\phi(x, y)$                                                                          | 1.2                                          | 任意标量             |
|                    | KL惩罚系数          | $\beta$                                                                                 | 0.1                                          | 超参数              |
|                    | KL项             | $\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$                    | $0.1 \times (-2.3 + 2.2) = -0.01$            | 注意log ratio负值    |
|                    | 修正奖励            | $\hat{r} = r_\phi - \beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$                     | $1.2 - (-0.01) = 1.21$                       | 作为单步回报           |
|                    | 累积回报            | $R = \hat{r}$                                                                           | 1.21                                         | 本例中仅一步           |
| **3. 价值估计与优势**     | 批评家价值           | $V_\psi(x)$                                                                             | 0.5                                          | 预测该提示下期望回报       |
|                    | 优势              | $A = R - V_\psi(x)$                                                                     | $1.21 - 0.5 = 0.71$                          | 正优势表示回答优于平均水平    |
| **4. PPO Actor损失** | 概率比率            | $\rho = \frac{\pi_\theta(y \mid x)}{\pi_{\text{old}}(y \mid x)}$                        | $\exp(-2.3 + 2.1) = e^{-0.2} \approx 0.8187$ | 旧策略采样，当前策略更新     |
|                    | PPO裁剪范围         | $[1-\epsilon, 1+\epsilon]$                                                              | [0.8, 1.2]                                   | 设 $\epsilon=0.2$ |
|                    | 裁剪后的比率          | $\text{clip}(\rho, 0.8, 1.2)$                                                           | 0.8187                                       | 在范围内，不变          |
|                    | 未裁剪项            | $\rho \cdot A$                                                                          | $0.8187 \times 0.71 \approx 0.5813$          |                  |
|                    | 裁剪项             | $\text{clip}(\rho) \cdot A$                                                             | $0.8187 \times 0.71 = 0.5813$                | 相同               |
|                    | 取最小值            | $\min(\dots)$                                                                           | 0.5813                                       |                  |
|                    | Actor损失         | $\mathcal{L}_{\text{actor}} = -\min(\dots)$                                             | -0.5813                                      | 梯度下降时最小化         |
| **5. Critic损失**    | 均方误差            | $\mathcal{L}_{\text{critic}} = (V_\psi(x) - R)^2$                                       | $(0.5 - 1.21)^2 = 0.5041$                    |                  |
| **6. 总损失**         | 简单相加            | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} + \mathcal{L}_{\text{critic}}$ | $-0.5813 + 0.5041 = -0.0772$                 |                  |

---
### 总结：RLHF 的“重”与 DPO 的“轻”

| 方面   | RLHF-PPO        | DPO            |
| ---- | --------------- | -------------- |
| 模型数量 | 4               | 2              |
| 训练方式 | 强化学习（on-policy） | 监督学习（offline）  |
| 超参数  | 多，敏感            | 少（主要是 $\beta$） |
| 稳定性  | 容易不稳定           | 相对稳定           |
| 计算成本 | 高               | 低              |

理解了 RLHF 的繁琐流程，就能明白 DPO 的贡献：它从数学上证明了**可以不训练独立奖励模型，直接优化策略**，并且将目标函数简化为一个与 SFT 类似的交叉熵损失。

# DPO关键原理
DPO核心主张：一步到位
- DPO的核心主张可以概括为一句话：  
	**绕过显式的奖励模型，将偏好对齐问题直接转化为一个带有隐式奖励函数的监督学习问题。****
## 步骤1：先明确 RLHF-PPO 的目标是什么
同学们，我们先看看传统 RLHF-PPO 在第三阶段究竟想优化什么。它的目标函数（也就是我们想让模型达到的理想状态）是这样的：
$$
\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} [r_\phi(x, y)] - \beta \mathbb{D}_{KL}[\pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)]
$$
这个公式很关键，它包含了两部分，我们拆开看：
- **第一部分：$\mathbb{E}[r_\phi(x, y)]$**。这部分的目标是**最大化奖励模型的打分**。我们希望模型生成的回答 $y$，能被奖励模型 $r_\phi$ 打高分。
- **第二部分：$-\beta \cdot \mathbb{D}_{KL}$**。这部分是**一个约束条件**。它要求我们正在训练的模型 $\pi_\theta$ 不要和一开始的参考模型 $\pi_{\text{ref}}$（通常是 SFT 模型）偏离得太远。这就像告诫运动员，在改进投篮姿势的同时，别把基本的运球、上篮技巧全给忘了。
## 步骤2：再发现一个数学上的“等价关系”
这里有一个关键的数学洞察。我们可以证明，上面那个复杂的优化问题，它的最优解 $\pi^*$ 有一个显式的表达式。也就是说，如果我们知道了一个理想的奖励函数 $r(x, y)$，那么**最优的策略 $\pi^*$ 长什么样是可以算出来的**：
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$
这个公式，我们可以用一句话来理解：**最优的策略，就是在参考策略的基础上，给那些“奖励高”的回答指数级地增加概率。**
  
**我们就能从这个表达式中，反过来把奖励函数 $r$ 表示成策略 $\pi$ 的函数**。把上面的公式做个变换，把 $r$ 单独拎出来，就能得到：
$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$
这个公式的含义是：**奖励模型 $r$ 的值，可以由当前策略 $\pi$ 和参考策略 $\pi_{\text{ref}}$ 的“概率比值”来决定**。$Z(x)$ 只是一个和 $y$ 无关的归一化项。
具体的推导在[[#DPO 数学推导：从优化目标到最优策略的闭式解]]
## 步骤3：最后将“偏好”代入，实现“一步到位”
现在，我们手上有什么？我们有最原始的、**直接由人类提供的偏好数据**，比如 $\langle \text{prompt } x, \text{chosen } y_w, \text{reject } y_l \rangle$。我们该如何利用这些数据？我们可以用 Bradley-Terry 模型来刻画人类选择偏好 $y_w$ 而不是 $y_l$ 的概率：
$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$
这个公式的意思是，**人类觉得 $y_w$ 比 $y_l$ 好的概率，取决于它们“奖励分数”的差值**。
好了，关键的一步来了。我们**把“步骤2”里用策略表示奖励的公式，代入到“步骤3”这个偏好概率的公式中**。注意看，那个只和 $x$ 相关的归一化项 $\beta \log Z(x)$，在作差的时候神奇地**抵消了**！
我们得到：
$$
P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$
看，这个最终表达式里，**奖励模型 $r$ 完全消失了**！只剩下我们想要的目标策略 $\pi^*$ 和参考策略 $\pi_{\text{ref}}$。这告诉我们，**人类偏好的概率，可以直接由当前策略和参考策略的“对数概率比值之差”来表达**。
因此，我们的训练目标就变得无比直接了：**既然 $P(y_w \succ y_l | x)$ 的概率应该越大越好，那我们就直接最大化这个概率**。把它转换成损失函数（负对数似然），就得到了 DPO 的最终形式：
$$
\mathcal{L}_{\text{DPO}}(\pi_\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$
---

# DPO 知识测验

## 一、选择题（每题只有一个正确选项）

1. **关于 DPO 相比 RLHF-PPO 的主要优势，下列说法错误的是？**  
   A. DPO 不需要单独训练一个奖励模型  
   B. DPO 可以完全避免使用参考模型（$\pi_{\text{ref}}$）  
   C. DPO 使用监督学习式的损失函数，训练更稳定  
   D. DPO 通常需要更少的显存和计算资源  

2. **在 DPO 的推导中，从 RLHF 优化目标得到最优策略的显式解时，引入了配分函数 $Z(x)$。关于 $Z(x)$，下列说法正确的是？**  
   A. $Z(x)$ 依赖于当前待优化的策略 $\pi_\theta$  
   B. $Z(x)$ 的作用是让 $\pi^*$ 成为一个合法的概率分布  
   C. $Z(x)$ 与奖励函数 $r$ 无关  
   D. $Z(x)$ 在后续代入 Bradley-Terry 模型时无法被消去  

3. **DPO 损失函数中，超参数 $\beta$ 的作用是？**  
   A. 控制偏好数据的置信度  
   B. 控制模型偏离参考模型的惩罚强度  
   C. 控制学习率的大小  
   D. 控制奖励模型的更新步长  

4. **假设我们有一个偏好数据样本：prompt $x$，chosen $y_w$，rejected $y_l$。在 DPO 训练中，我们希望模型的什么行为？**  
   A. 最大化 $\pi_\theta(y_w|x)$ 并最小化 $\pi_\theta(y_l|x)$  
   B. 最大化 $\pi_\theta(y_w|x) - \pi_\theta(y_l|x)$  
   C. 最大化 $\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$  
   D. 最小化 $\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$  

---

## 二、填空题

1. RLHF-PPO 的优化目标可以写成：  
   $\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} [r(x,y)] - \beta \mathbb{D}_{\text{KL}}[\pi(\cdot|x) \parallel \_\_\_\_\_\_\_\_\_\_\_\_\_\_]$。

2. 在 DPO 推导中，最优策略的显式解为：  
   $\pi^*(y|x) = \frac{1}{\_\_\_\_\_\_\_\_\_\_\_\_} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$。

3. 将奖励函数用最优策略和参考策略表示，得到：  
   $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log \_\_\_\_\_\_\_\_\_\_\_\_$。

4. 在 Bradley-Terry 模型中，人类偏好概率为 $P(y_w \succ y_l | x) = \sigma(\_\_\_\_\_\_\_\_\_\_\_\_ - r(x,y_l))$。

---

## 三、案例分析题

**场景**：你是一家 AI 公司的算法工程师，公司正在开发一款面向客服场景的对话模型。你已经完成了预训练和 SFT，得到了一个基础模型 $\pi_{\text{SFT}}$。现在你收集了 10 万条客服对话数据，每条数据包含：
- 用户提问（prompt）
- 两个候选回答（response A, response B）
- 一个标注，标明哪个回答更好（由客服专家判断）

你需要在两周内完成偏好对齐训练，并确保模型在真实客服场景中表现稳定。公司算力有限，只有一台 8 卡 A100 服务器，且希望尽快上线。

**请回答以下问题**：
1. 你会选择 RLHF-PPO 还是 DPO 进行对齐？请给出两个理由支持你的选择。
2. 基于你的选择，简要描述你的训练流程，包括需要用到的模型、数据格式、关键超参数。
3. 如果训练过程中发现模型生成的回答开始变得非常冗长，甚至偏离了客服应有的简洁专业风格，你认为可能是哪个超参数设置不当？应该增大还是减小它？

---

## 四、简答题

1. **请用自己的话解释**：为什么 DPO 不需要显式训练一个奖励模型？在推导中，奖励模型是如何被“吸收”到策略中去的？

2. **配分函数 $Z(x)$ 的消去**：在将奖励函数代入 Bradley-Terry 模型时，$Z(x)$ 为什么可以消去？这个消去对于 DPO 的目标函数有什么重要意义？




# DPO 数学推导：从优化目标到最优策略的闭式解
## 第一步：明确我们要解的问题
在 RLHF 中，我们想找一个策略（也就是我们最终的语言模型）$\pi$，让它能最大化奖励，同时又不能和参考策略 $\pi_{\text{ref}}$（通常是 SFT 模型）离得太远。这个目标写成数学公式就是：
$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot \mathbb{KL}\left[ \pi(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x) \right]
$$
这里的 $r(x,y)$ 是一个已知的奖励函数（比如训练好的奖励模型），$\beta$ 是控制 KL 惩罚强度的超参数。$\mathbb{KL}[ \pi \| \pi_{\text{ref}} ]$ 是 KL 散度，它衡量两个分布的差异。
**问题**：在给定 $r$ 和 $\pi_{\text{ref}}$ 的情况下，这个最大值对应的最优策略 $\pi^*$ 究竟是什么？
---
## 第二步：将目标函数改写成熟悉的形式
KL 散度是有明确表达式的：$\mathbb{KL}[ \pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x) ] = \mathbb{E}_{y \sim \pi(\cdot|x)} \left[ \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$。
所以目标函数可以写成：
$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$
注意，期望里是 $r(x,y) - \beta \log \frac{\pi}{\pi_{\text{ref}}}$。为了方便，我们提取出 $\beta$，并把最大化问题变成最小化问题（负号转换）：
$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
= \min_{\pi} \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} \left[ \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - r(x,y) \right]
$$

这个形式让我们想到，如果能把它写成一个 KL 散度加上一个与 $\pi$ 无关的项，那么最小值就可以直接读出来。
---
## 第三步：引入一个“配分函数”来凑出完美形式
我们想构造一个分布 $\pi^*(y|x)$，使得
$$
\beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - r(x,y) = \beta \log \frac{\pi(y|x)}{\pi^*(y|x)} - \beta \log Z(x)
$$
其中 $Z(x)$ 是一个只依赖于 $x$ 的常数。如果这样的 $\pi^*$ 存在，那么目标函数就变成了
$$
\min_{\pi} \mathbb{E}_{x, y \sim \pi} \left[ \beta \log \frac{\pi(y|x)}{\pi^*(y|x)} - \beta \log Z(x) \right] = \beta \min_{\pi} \mathbb{E}_{x \sim D} \left[ \mathbb{KL}\left( \pi(\cdot|x) \| \pi^*(\cdot|x) \right) - \log Z(x) \right]
$$
因为 $\beta$ 和 $Z(x)$ 与 $\pi$ 无关，最小化问题就变成了 **最小化 KL 散度**。根据吉布斯不等式，当 $\pi = \pi^*$ 时 KL 散度为 0，这就是最小值。所以最优策略 $\pi^*$ 就是我们要找的那个。
那么，$\pi^*$ 究竟应该长什么样？我们从等式
$$
\beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - r(x,y) = \beta \log \frac{\pi(y|x)}{\pi^*(y|x)} - \beta \log Z(x)
$$
消去 $\beta \log \pi(y|x)$，得到：
$$
-\beta \log \pi_{\text{ref}}(y|x) - r(x,y) = -\beta \log \pi^*(y|x) - \beta \log Z(x)
$$
移项可得：
$$
\beta \log \pi^*(y|x) = \beta \log \pi_{\text{ref}}(y|x) + r(x,y) - \beta \log Z(x)
$$
两边除以 $\beta$ 再取指数：
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)
$$
这里 $Z(x)$ 必须使得 $\pi^*$ 对所有 $y$ 求和为 1，因此它自然定义为：
$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)
$$
这个 $Z(x)$ 就是**配分函数**，它只依赖于 $x$ 和参考策略，与我们要优化的 $\pi$ 无关。于是，我们得到了最优策略的闭式解。
---
## 第四步：反解出奖励函数
现在我们已经知道，对于给定的奖励 $r$，最优策略 $\pi^*$ 必须满足上面那个等式。反过来，我们也可以把奖励 $r$ 用 $\pi^*$ 和 $\pi_{\text{ref}}$ 表示出来。
从
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x,y) \right)
$$
两边取对数，并整理：
$$
\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x,y) - \log Z(x)
$$
$$
\frac{1}{\beta} r(x,y) = \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \log Z(x)
$$
$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$
这就是我们想要的关系：**奖励函数可以表示为当前最优策略与参考策略的对数概率比值，再加上一个只与 $x$ 有关的常数项**。
---
## 第五步：用一个离散的例子来帮助理解
让我们用一个极简的离散例子来具象化这个过程。假设对于某个 $x$，只有两个可能的回答 $y_1, y_2$。参考策略 $\pi_{\text{ref}}$ 对它们的概率是 $[0.4, 0.6]$。奖励函数给它们打分为 $[1.0, 0.5]$。设 $\beta = 1$。
先计算配分函数：
$$
Z(x) = 0.4 \cdot e^{1} + 0.6 \cdot e^{0.5} \approx 0.4 \cdot 2.718 + 0.6 \cdot 1.648 \approx 1.087 + 0.989 = 2.076
$$
然后最优策略 $\pi^*$ 为：
$$
\pi^*(y_1|x) = \frac{1}{2.076} \cdot 0.4 \cdot 2.718 \approx 0.523
$$
$$
\pi^*(y_2|x) = \frac{1}{2.076} \cdot 0.6 \cdot 1.648 \approx 0.477
$$
现在，我们用这个 $\pi^*$ 来反解奖励函数（忽略常数 $Z(x)$）：
$$
\beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} = 1 \cdot \log \frac{0.523}{0.4} = \log 1.3075 \approx 0.268
$$
$$
\beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} = \log \frac{0.477}{0.6} = \log 0.795 \approx -0.229
$$
这两个数分别加上 $\beta \log Z(x) = \log 2.076 \approx 0.731$，就得到了原来的奖励：
$$
0.268 + 0.731 = 0.999 \approx 1.0
$$
$$
-0.229 + 0.731 = 0.502 \approx 0.5
$$
完美吻合。这个例子展示了“等价关系”是如何运作的：奖励函数和最优策略通过配分函数牢牢绑定在一起。