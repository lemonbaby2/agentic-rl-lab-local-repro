# 多模态与决策类实验笔记（Spec-o3 / Tournament-GRPO / PyTRIO-Jev / 基础篇）

> 来源：`agentic-rl-lab` 仓库对应章节的 readme.md、start.md 与同目录 py 代码。所有数字均取自原文，未作换算。

---

## Spec-o3：工具增强视觉语言智能体的光谱候选审核

- **论文**：*Spec-o3: A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Vetting via Automated Spectral Inspection*，arXiv **2601.06498**。定位：让视觉语言模型在光谱分类中主动指定波段、调用工具重绘局部光谱，形成"思考—工具—新图片"交错的多模态思维链（论文称 iMCoT）。
- **解决什么痛点**：大规模光谱巡天的候选审核要在大量易混淆光谱上逐一复核，单看整图细节不足；固定流水线无法根据当前判断决定"下一张该看哪一段"。Spec-o3 把审核拆成五个独立的 YES/NO 判断，让模型自己决定放大哪个波长区间，再用新证据继续推断。
- **核心公式**：
  - 奖励（readme §5.2，`train_rl.py:trajectory_reward`）：

    $$r(\tau)=\mathbb{1}[\text{答案正确}]-0.2\,\mathbb{1}[\text{格式错误}]$$

    $\tau$ 为一条完整轨迹；$\mathbb{1}[\cdot]$ 为指示函数；正确性由最后一轮 `</think>` 之后的正文开头 YES/NO 与该题 `label` 比较决定；格式错误扣 0.2。四象限取值：正确+合格 1.0，正确+不合格 0.8，错误+合格 0.0，错误+不合格 −0.2。不奖励工具调用次数。
  - 组内 advantage（readme §5.3，`train_rl.py:build_training_samples`）：

    $$A_i=\frac{r_i-\mathrm{mean}(r_1,\ldots,r_8)}{\mathrm{std}(r_1,\ldots,r_8)+10^{-6}}$$

    $r_i$ 为第 $i$ 条轨迹奖励；$\mathrm{mean}$ 为组内均值；$\mathrm{std}$ 为**样本标准差**（`ddof=1`）；$10^{-6}$ 为防止除零的 epsilon。$A_i$ 为标量，同一条轨迹内所有 assistant token 共享这一 advantage。
  - 策略梯度目标（readme 只给裁剪区间，未写公式本体；`train_rl.py` 传 `loss_fn="ppo"`、`clip_low_threshold=1-clip_ratio`、`clip_high_threshold=1+clip_ratio`，即 $[0.8,1.2]$，下式为该裁剪目标的通用形式）：

    $$\mathcal{L}(\theta)=-\frac{1}{\sum_i T_i}\sum_{i}\sum_{t=1}^{T_i}\min\!\Big(\rho_{i,t}A_i,\ \mathrm{clip}(\rho_{i,t},\,0.8,\,1.2)\,A_i\Big),\qquad \rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid s_{i,t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid s_{i,t})}$$

    $\rho_{i,t}$ 为新旧策略在同一 token 上的概率比；$T_i$ 为该轨迹参与训练的 assistant token 数。代码实际做法是先把每条 Datum 的 advantage 除以**整批** assistant token 总数（`advantages / tokens`），因此等价于对整个 batch 的 token 取平均。
- **reward 设计**：判分只看两件事——最终答案是 YES 还是 NO 与标签是否一致，以及轨迹是否以符合严格动作协议的最终答案收尾。训练侧答案提取只读最后一轮思考结束后正文开头的 YES/NO，允许"答案对但缺标签"拿 0.8；评测侧则从最后一轮最后一个 `</think>` 之后提取最后一个完整 `<answer>` 块，两套规则分别实现。没有把工具调用次数、图片数量计入奖励。
- **关键实现细节**：
  - **SFT 冷启动**：942 条示范轨迹（848 训练 / 94 验证），由诊断指南与标签辅助 GPT-5 生成初稿、天文学家修订；Qwen3.5-4B、LoRA rank 32、batch 4、lr `2e-5`、2 个 epoch，共 **424 个训练 step**，`cross_entropy` 只监督 assistant 生成的思考、工具调用、最终答案与结束 token，system/user/角色头/工具返回全部不参与 loss。
  - **图片如何进入上下文**：`protocol.py` 先渲染消息，再把每个 `<|image_pad|>` 替换为 `trio.ImageChunk(data=PNG 字节, format="png", expected_tokens=...)`，保持"文本—图片—文本"原序；`expected_tokens` 由匹配基模的 image processor 按图片尺寸算出，决定图片占多少视觉位置。图片占位全部进上下文但**训练权重为零**；工具返回第二、第三张图按同样方式追加，数组统一右移保证预测下一个 token。
  - **工具协议**：Qwen3.5 原生模板上改两件事——始终保留各轮 `reasoning_content`（含跨新 user 问题的早期 reasoning），用 `{% generation %}` 标出 assistant 学习区间以返回监督 mask。动作格式为 XML：`<think>…</think>` + `<tool_call><function=spectral_visualization_tool><parameter=wavelength_range>[6400, 6700]</parameter><parameter=label>Hα region</parameter></function></tool_call>`；工具结果放入 `<tool_response>` 消息，下一轮从 assistant 的 `<think>` 前缀继续。
  - **"放大光谱"实现**：`tools.py` 先做红移校正 `wavelength = wavelength / (1 + redshift)`，再按模型给的 `[low, high]` 选点重绘 PNG；窗口内少于 2 个点则返回纯文本说明 `No spectrum data in wavelength range (low, high) Å.`，模型下一轮继续判断。画布尺寸随窗口宽度分档（<50 / <200 / <1000 / 其余）。
  - **数据规模**：SFT 848 / 94；RL 训练 **3,108 题**；开发集 **256 条**；测试集 6,754 条（尚未评测）。每批 8 题 × 每题 8 条轨迹 = 64 条 rollout，一个 epoch 共 **24,864 条轨迹、389 个 rollout batch**，尾批 4 题 32 条。
  - **训练步数与预算**：RL 1 个 epoch、lr `1e-6`、`--max-turns 8`（含最终答案那一轮，实际最多执行前 7 轮工具调用）、单轮生成上限 2,048、总上下文上限 16,384；三组 RL 评测均 temperature 0.6、seed 42、并发 16。整批 rollout 共用一个固定版本 sampler，全部完成后才更新，因此组内 reward 与 old logprob 对应同一版策略；整组 reward 完全相同则跳过，且不补采样。
- **原文给出的结果数字**（同一批 256 条开发集）：

  | 模型 / 阶段 | Macro F1 | Accuracy | Format | 答对题数 |
  | --- | ---: | ---: | ---: | ---: |
  | Qwen3.5-4B Base | 53.99% | 38.67% | 0.39% | 99 / 256 |
  | Qwen3.5-4B SFT epoch 2 | 63.05% | 53.91% | 85.94% | 138 / 256 |
  | Qwen3.5-4B RL epoch 1 | 71.68% | 67.97% | 96.09% | 174 / 256 |
  | Qwen3.5-4B RL epoch 3 | 70.93% | 68.36% | 98.05% | 175 / 256 |
  | Qwen3.5-9B RL epoch 3 | 75.94% | 74.61% | 99.61% | 191 / 256 |

  分任务 F1（括号为题目数）：CC 碳星（56）4B 52.31% / 9B 67.80%；CV 激变变星（50）4B 61.11% / 9B 71.79%；GM M 型巨星（51）4B **92.06%** / 9B 87.72%；SS S 型星（50）4B 79.17% / 9B **84.62%**；WD 白矮星（49）4B **70.00%** / 9B 67.80%。
  其他数字：4B 从 Base 到 SFT 多答对 39 题，首轮 RL 又多 36 题；epoch 3 比 epoch 1 多答对 1 题，格式合规率升到 98.05%，Macro F1 从 71.68% 降到 70.93%；9B 比 4B RL epoch 3 多 16 题、Accuracy 高 6.25 个百分点、Macro F1 高 5.01 个百分点。9B 格式合规 255/256（99.61%），1,011 次工具调用全部成功；4B 251/256（98.05%），1,087 次工具调用全部成功。SFT 训练 loss 从约 0.9 降到约 0.5。费用：SFT 会话 6.11M train tokens、¥27.72；4B RL epoch 1 会话 prefilling 239.39M / train 79.13M / sample 26.38M、¥671.96；两者合计 **¥699.68**；从看论文到完成不到一天。原论文使用 8 张 NVIDIA H100。
- **坑与边界**：公开文本含损坏科学符号，当前仍有 **273 条轨迹**带可见标记并被保留；SFT 与 RL 的差值仍可能包含生成预算差异（Base 6,132 / SFT 2,048 / RL 6,144 token），4B 与 9B 基模与权重不同，结果只用于比较当前 checkpoint；epoch 3 出现"格式更好但 Macro F1 下滑"，多 epoch 未带来全指标同步提升；训练侧与评测侧答案提取规则不同；完整测试集还有 2 个 SFT/test 光谱对象重叠未处理；未逐项复刻上游 loss，不能与论文榜单对齐；`train/assistant_tokens` 受跳组与尾批影响，不能用来判断推理是否变短。
- **一句话记住它**：边看图边放大光谱再答题。

---

## Tournament-GRPO：组内淘汰赛奖励训练长文本研究报告

- **论文**：*Tournament-GRPO: Group-Wise Tournament Rewards for Reinforcement Learning in Open-Ended Long-Form Generation*，arXiv **2605.26958v2**（EMNLP 2026 主会）。定位：把 GRPO 的奖励从"裁判逐份打绝对分"换成"同组回答多轮两两淘汰、按累计胜场给奖励"，只改 Judge → Reward 一段，其余仍是标准 GRPO + PPO。
- **解决什么痛点**：Deep research 类任务没有标准答案，只能靠 rubric 加 LLM 裁判。逐条 rubric 打分与整体 1–10 打分都有三个毛病：分数尺度难校准、组内区分度弱（容易项人人满足、难题人人不满足）、训练久了容易饱和。GRPO 只需要"同题内谁比谁好"，因此直接让裁判做比较更匹配优化目标。
- **核心公式**（readme §4 与 `reward.py`）：
  - 淘汰赛累计胜场 → 奖励（组内 min-max 归一化）：

    $$r_i^{\text{tour}}=\frac{s_i-\min_j s_j}{\max_j s_j-\min_j s_j}$$

    $s_i$ 为第 $i$ 份回答在 $M$ 次淘汰赛中的累计胜场（本章取值 0–12）；$K=8$ 为每题回答数，$G=2$ 每场人数，$W=1$ 每场晋级人数，$F=1$ 剩几人时结束，$M=4$ 重复次数。所有 $s_i$ 相同时 $r_i^{\text{tour}}=0.5$；没有最终答案的回答在归一化之后置 0。
  - 总奖励：

    $$r_i=r_i^{\text{tour}}+r_i^{\text{fmt}}$$

    $r_i^{\text{fmt}}\in\{0,1\}$ 为格式奖励，故总奖励范围 0–2。
  - 组内标准化 advantage（`train.py:assign_rewards`）：

    $$A_i=\frac{r_i-\bar r}{s_r+10^{-6}}$$

    $\bar r$ 为组内 reward 均值；$s_r$ 为以 $K-1$ 为分母的样本标准差（`ddof=1` / `correction=1`）；$10^{-6}$ 防除零。
  - 自定义 PPO loss（`train.py:ppo_objective`，逐 token 计算后对全 batch assistant token 取均值）：

    $$\rho=e^{\mathrm{clip}(l_\theta-l_{\text{old}},\,-20,\,20)},\quad
    \mathcal{L}=\frac{1}{N}\sum\Big[\mathbb{1}[A\ge 0]\cdot\max(-A\rho,\,-A\,\mathrm{clip}(\rho,0.8,1.2))+\mathbb{1}[A<0]\cdot\min(-3A,\ \max(-A\rho,\,-A\,\mathrm{clip}(\rho,0.8,1.2)))\Big]$$

    $l_\theta$、$l_{\text{old}}$ 为当前与旧策略的 token logprob，差值先截断到 $[-20,20]$ 再取指数得 ratio $\rho$；$A$ 为该轨迹的 advantage；`clip` 区间 $[0.8,1.2]$（$\epsilon=0.2$）；dual clip 常数 3.0 **只作用于负 advantage**；$N$ 为 action token 总数。没有 KL 项、没有熵项。每批只更新一次，故 old 直接取训练端前向的 `current.detach()`，此时 ratio 恒为 1、clip 不触发，但梯度仍是 $-A\nabla\log\pi$。
- **reward 设计**：
  - **淘汰赛如何构造奖励**：同一题的 8 份回答参赛。每次淘汰赛随机洗牌，按轮转方式两两分组（第 $i$ 个候选进第 $i \bmod \text{num\_groups}$ 组，人数为奇数时会出现 3 人组），每组由裁判选 1 个胜者、胜者 +1 分，胜者晋级，直到剩 1 人。一次淘汰赛 4+2+1=7 场，重复 4 次共 **28 场**。裁判只看问题、完整 rubric 和每份回答的最终 `<answer>`，看不到推理与搜索过程；提示词要求"恰好选 1 个胜者，优先准确、完整、有证据、直接回答问题"。
  - **格式奖励**：0/1，检查整条轨迹——开头非空 think、至少一次搜索、每次工具调用后必须接 `tool_output` 和 think、最后恰好一个非空 answer、不能有标签外杂文或损坏 XML、工具名只能是 `google_search` / `browse_webpage`。
  - **评测怎么判分（RACE）**：裁判按每题 criteria 把待评报告与作者参考报告逐条比较、各打 0–10 分，先按细则权重、再按维度权重聚合；单维度得分是"待评 ÷ (待评 + 参考)"，50 分相当于与参考报告打平。
- **关键实现细节**：
  - **数据规模**：作者 Parquet 原始 8,380 训练 / 441 验证（DR-Tulu 4,627 + RaR-Science 2,793 + RaR-Medicine 960），按 verl 配置与 Qwen3.5-4B tokenizer 去掉超过 4096 token 的 prompt 后剩 **8,378 / 440**；评测为 Deep Research Bench 全部 100 题。rubric 训练前用 GPT-4o 生成后保持静态。
  - **工具与协议**：两个工具 `google_search`（返回搜索摘要）与 `browse_webpage`（读网页正文），XML 协议 `<think>` / `<call_tool name="...">` / `<tool_output>` / `<answer>`；observation 与作者新增的"预算用尽"提示都 mask 为 0，模型只为自己的 think、工具调用与 answer 负责。
  - **交互预算**：为降成本把搜索从 Top 10 改成 Top 3，轮数从 8 轮 / 7 次工具改成 **6 轮 / 4 次工具**；response 预算 18,000 token（含 observation）、prompt 上限 4,096、模型 token 上限 25,000、工具返回按字符中间截断到 3,000（首尾各 1,500）。第 4 次工具结果末尾追加环境提示，让模型收尾。
  - **训练规模**：Qwen3.5-4B、LoRA rank 32、lr `4e-5`、每步 8 题 × 8 条轨迹 = 64 条轨迹 / 224 场裁判比较，共 **20 step**，每批只更新一次，`--save-every 10`。裁判失败重试 3 次仍失败则停止训练（作者是随机选胜者）。
- **原文给出的结果数字**：
  - 训练总量：20 步 = **1,280 条轨迹、4,480 场裁判比较**，答案完成率 64% → 97%。

    | 指标 | Step 1 | Step 10 | Step 20 |
    | --- | ---: | ---: | ---: |
    | 答案完成率 | 64.1% | 93.8% | 96.9% |
    | 格式合规率 | 20.3% | 65.6% | 45.3% |
    | 总 reward | 0.489 | 1.022 | 0.794 |
    | 淘汰赛 reward | 0.286 | 0.365 | 0.340 |
    | 每条轨迹搜索次数 | 2.58 | 3.17 | 2.47 |
    | 每条轨迹浏览次数 | 0.73 | 0.70 | 0.47 |
    | 每条轨迹 response tokens | 2,778 | 3,556 | 2,725 |

    前 3 步答案完成率均值 61.5%，后 3 步 95.3%；格式合规前 3 步均值 19.3% → 后 3 步 52.6%；`reward/degenerate_group_rate` 全程为 0；`rollout/finish_no_tool_call` 从 0.78 升到接近 1，前几步约 22% 轨迹在第 5 次工具请求时被截断，第 8 步后基本消失。
  - 评测（100 题，贪心 1 条轨迹，seed 42）：RACE 全面性 19.21 → 32.33（+13.12）、洞察力 17.11 → 29.67（+12.56）、指令遵循 24.95 → 40.77（+15.82）、可读性 23.91 → 40.01（+16.10）、**Overall 20.65 → 34.70（+14.05）**。生成指标：答案完成 60/100 → 98/100，格式合规 42/100 → 72/100，每题工具调用 3.98 → 3.94，每题模型生成 tokens 826 → 1,641。Base 有 40 题白卷（记 0 分），Step 20 只有 2 题；两边都写出报告的 59 题上 Overall 仅 34.49 → 35.25。Base 白卷平均只有 220 token。
  - 成本与耗时：训练 **78.92 元（4 小时 3 分钟）** + 评测 **13.12 元** = **92.04 元**（明细 PyTRIO 训练 60.96 / DeepSeek Search 16.96 / Jev 裁判 1.00 / PyTRIO 采样 3.20 / DeepSeek 9.92）。平均每步 12.2 分钟、约 3.95 元；4,480 场比较累计约 2 小时 22 分钟，占 58%，费用仅 1 元；裁判共读入约 758 万 token、输出约 13.9 万 token；模型共搜索 3,218 次、浏览网页 722 次；评测生成 100 份报告 Base 用 4.5 分钟、Step 20 用 13.2 分钟。
- **坑与边界**：policy（4B LoRA vs 7B 全参）、裁判（Jev `jev-1.14.0` vs Qwen2.5-72B）、搜索后端与训练规模都不同，绝对分数不能与论文对比，只能验证链路与早期行为；涨幅主要来自"白卷变少"，单份报告质量几乎没变（59 题上 34.49 → 35.25）；早期总 reward 增量约 92% 来自格式奖励（0.193 → 0.526），淘汰赛奖励仅 0.293 → 0.323；K=8 时 M=4 的 28 场与两两全比 C(8,2) 等量，**并不省裁判调用**（论文里省调用的是 M=2–3 的配置）；一组 8 条全空答案时 reward 与 advantage 全 0，该步无梯度；裁判接口连续失败会让 reward 变成随机数（曾出现 224 场中 150 场退回随机胜者），故本章改为直接停训。
- **一句话记住它**：裁判打淘汰赛，胜场当奖励。

---

## PyTRIO-Jev：单 token 受限 softmax 的决策模型

- **论文**：未标注（readme 与代码均未给出论文标题或 arXiv 编号；方法参考 Intern-Decision、Jev 架构逆向文章与 Datawhale AgentJev 教程，不是论文复现）。
- **解决什么痛点**：意图理解、工单路由、风险分级这类任务选项固定、高频发生，用通用大模型逐字生成答案既慢又贵，还拿不到可用的置信度。Jev 式决策模型不生成文本，只在固定候选符号上输出归一化概率，可直接做阈值判断与灰度放行。
- **核心公式**（`inference.py`、`calibrate.py`、`data.py`）：
  - 候选符号上的受限 softmax（带校准温度）：

    $$p_T(s\mid f)=\frac{\exp\!\big(z_{f,s}/T\big)}{\sum_{s'\in\mathcal{C}_f}\exp\!\big(z_{f,s'}/T\big)}$$

    $f$ 为某一问题字段；$\mathcal{C}_f$ 为该字段的候选符号集（choice 取 A/B/C…，noul 固定 `[no, yes]`，score 取 1/2/3/4/5）；$z_{f,s}$ 是模型在 marker 位置对符号 $s$ 的 logprob（由 `topk_prompt_logprobs` 读回，未进 top-k 的候选按 $\min(\text{topk})-10$ 近似为概率 0）；$T$ 为标量校准温度，$T=1$ 即未校准。代码实现为 `z = logprobs / T; z -= z.max(); p = exp(z); p /= p.sum()`。
  - 温度拟合目标（最小化校准集平均 NLL，`calibrate.py:mean_nll`）：

    $$\mathrm{NLL}(T)=\frac{1}{N}\sum_{n=1}^{N}\Big[\log\sum_{s\in\mathcal{C}_{f_n}}\exp\!\big(z_{n,s}/T\big)-z_{n,g_n}/T\Big]$$

    $n$ 索引一个"字段级"实例；$g_n$ 为该字段的 gold 符号下标；$N$ 为字段实例总数（本章 120 案例 / 600 题）。在 $\log T\in[\log 0.01,\log 100]$ 上做 80 轮黄金分割搜索。
  - score 题型的期望分：$\hat{y}=\sum_{v} v\cdot p_T(v)$，$v$ 为等级值（1–5）。
  - 训练目标：`cross_entropy`，但 `weights` 只在 **marker 前一个位置**为 1，其余全 0，因此
    $$\mathcal{L}=-\sum_{f}\log \pi_\theta\big(y_f \mid \text{prefix}\cdot\texttt{marker}_f\big)$$
    $y_f$ 为字段 $f$ 的答案符号 token；一条记录的所有字段在同一次前向里被同时监督。
- **reward 设计**：**没有 reward**，本章是纯监督（masked SFT），不做 RL。评测按字段准确率计分：choice 取 argmax 是否等于标签；noul 比较 $p(\text{yes})$ 与 $p(\text{no})$；score 输出期望分与分布。校准另用 ECE（按置信度分桶，比较桶内平均置信度与实际准确率的差距，越小越好）与 Brier（预测分布与真实标签的均方误差，越小越好）两个指标，二者在原文中只有文字定义，未给公式。
- **关键实现细节**：
  - **不让模型"生成"答案**：assistant 侧输入是一个含占位符的 JSON 骨架，每个字段值的位置放 marker；占位符就是**一个普通词表 token** `decision`（Qwen3.5 tokenizer id **61781**，代码断言必须编码为单 token），不新增 special token（服务端词表不可扩展）。所有字段共享同一段 state、同一次前向；字段答案不出现在输入里，字段之间互相看不到彼此答案，问题排列顺序不影响输出概率。
  - **clean chat template**：不沿用模型自带模板，专门写 `templates/qwen3_5_jev.jinja`，除对话结构外不多塞一个字，以去除系统提示词等额外文本对决策的干扰。
  - **marker 定位**：marker 索引靠"prefix 之外、骨架之内"定位而非扫全文，因此 state/instructions 里出现单词 `decision` 不会误判；骨架里若字段名本身就叫 `decision`（`{"decision": "decision"}`），按字段顺序先跳过 key 命中再取 value。监督位置 = marker 索引 − 1（右移后预测 answer 符号）。
  - **推理不采样文本**：`sample(max_tokens=1, topk_prompt_logprobs=K)` 一次前向读回各 marker 位置的下一个 token 分布，过滤到该字段候选符号后归一化；K 取所有字段候选数的最大值。
  - **数据规模**：typed-decisions train split 960 案例 / 4,800 题（4 个 workflow × 300 案例，案例级切 240/30/30 为 train/dev/calibration，覆盖 choice/score/noul）+ ToolACE 1,200 条（**先剔除 310 条评测行**再采样）= **2,160 条记录、6,000 个决策题**。评测集与 Intern-Decision 完全一致：Jevbench Easy 48 / Original 72 / Hard 111、Typed Decision test 400 案例 2,000 题、ToolACE 310。防污染：Jevbench 全部 231 条与 ToolACE 310 条不进训练，跨划分做 state 文本哈希查重，温度只在校准划分拟合、测试集在 checkpoint 与 T 冻结后才读。
  - **训练配置**：Qwen3.5-4B、LoRA rank 32（mlp/attn/unembed）、lr `1e-4`、batch 16（每步约 45 个监督 token）、2 个 epoch、**270 步**；全异步提交 `forward_backward_async` 与 `optim_step_async`，日志由独立 task 消费，epoch 末必存权重。
  - noul 题型的训练信号只来自 typed-decisions（ToolACE 是纯 choice）。
- **原文给出的结果数字**（准确率 %，评测集与 Intern-Decision 官方榜单一致）：

  | 模型 | Easy | Original | Hard | Typed | ToolACE | 平均 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Qwen3.5-4B（base） | 100.00 | 87.50 | 55.86 | 50.40 | 87.74 | 76.30 |
  | PyTrio-Jev（本文） | 100.00 | 95.83 | 47.75 | 79.20 | 94.52 | 83.46 |
  | PyTrio-Jev（+温度校准） | 100.00 | 95.83 | 49.55 | 79.10 | 94.19 | **83.74** |
  | Intern-Decision-0.8B | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 80.52 |
  | Intern-Decision-2B | 100.00 | 84.72 | 63.96 | 79.35 | 96.45 | 84.90 |
  | Intern-Decision-4B | 100.00 | 98.61 | 73.87 | 80.55 | 96.45 | 89.90 |
  | Jev | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 87.06 |
  | Laya | 95.83 | 72.22 | 28.83 | 35.95 | 63.87 | 59.34 |

  其他数字：训练第一步 loss **3.78**，十几步内压到 1 以下；`supervised_tokens` 在 20–70 间波动（每条记录字段数不同，不是异常）。温度拟合得 **T = 1.13**，NLL 从 **0.4990** 降到 **0.4959**；Hard 档 ECE 从 **0.2311** 降到 **0.1884**，Brier 从 **0.6510** 降到 **0.6331**。训练会话消耗 **2.72M train tokens、¥12.36**；两个 ¥0.99 会话是五项基准评测（各 0.66M prefilling tokens）；训练 + 两次评测合计 **14.34 元**（原文亦称"不到 15 块钱"）。数据分布：训练 state 中位约 **727 字符**，Hard 档 state 中位约 **1,611 字符**。
- **坑与边界**：Jevbench-Hard 从 base 的 55.86% 一路降到 47.75%，"训练在帮倒忙"——训练数据以短 case 为主、ToolACE 是"请求含关键词、工具名里就有答案"的表面匹配，而 Hard 全是长政策文档、多跳（multi_hop）、多规则权衡（judge_hard），模型学到"抓关键词快速下结论"的风格后迁移到长文档就是灾难，Hard 的 ECE 也印证了系统性过度自信；这不是方法问题而是数据问题。口径差异：官方在 Typed Decision 上是**零样本**，本章是 specialist 模式（train/test 同 workflow）；官方全参数微调，本章只 LoRA + 2,400 条通用数据，故不能直接对标。温度不改变 argmax，accuracy 的 0.1–2 个点微波动来自服务端采样数值不确定性；noul 训练信号偏科。
- **一句话记住它**：一次前向只吐一个 token。

---

## 基础篇：IS / PPO / CISPO 与 SFT、rollout、on-policy、advantage、LoRA、OPD、observation mask

> 合并 `experiments/loss-functions/readme.md`（第 0 篇：损失函数篇）与 `experiments/getting-started/readme.md`（环境配置与基础概念）。

- **论文**：未标注。两篇都是原创教学章节，不是论文复现；loss-functions 是后续所有 RL 章节的公式预备，getting-started 对应 Datawhale 组队学习 Task01 的概念词汇表。
- **解决什么痛点**：初读 RL loss 的人容易卡在"模型自己生成一段话、事后才拿到 reward，loss 到底在优化什么"。这一篇把它压缩成一句话：被证明有帮助的 token 下次更容易生成，拖后腿的 token 下次少生成；三个量——`advantage`（这一步是好是坏）、`sampling_logprobs`（采样时旧策略多喜欢这个 token）、`target_logprobs`（训练时当前策略多喜欢它）——就能区分 IS / PPO / CISPO。
- **核心公式**：
  - **Importance Sampling**：

    $$\mathcal{L}_{\text{IS}}(\theta)=-\mathbb{E}_{x\sim q}\Big[\frac{p_\theta(x)}{q(x)}A(x)\Big]$$

    $q$ 为采样策略（旧策略），$p_\theta$ 为当前策略，$A(x)$ 为该 token 的 advantage。实现写成 `prob_ratio = exp(target_logprobs - sampling_logprobs)`，再 `loss = -(prob_ratio * advantages).sum()`；负号只因优化器默认最小化，真正最大化的是 $\text{ratio}\cdot A$。$A>0$ 推高概率，$A<0$ 压低概率，ratio 负责修正采样偏差。
  - **PPO**（clip objective）：

    $$\mathcal{L}_{\text{PPO}}(\theta)=-\mathbb{E}_{x\sim q}\Big[\min\Big(\frac{p_\theta(x)}{q(x)}A(x),\ \mathrm{clip}\big(\frac{p_\theta(x)}{q(x)},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}}\big)A(x)\Big)\Big]$$

    $\epsilon_{\text{low}},\epsilon_{\text{high}}$ 为裁剪幅度（各章实现多为 0.2，即 $[0.8,1.2]$）。直觉：ratio 已经到 1.8 说明当前模型比采样时更喜欢该 token 很多，PPO 不按 1.8 的力度推，而限制到约 1.2——"可以学习，但别一步冲太远"。`min` 不是随便取小，而是在做保守更新。
  - **CISPO**（clip ratio 但 detach）：

    $$\mathcal{L}_{\text{CISPO}}(\theta)=-\mathbb{E}_{x\sim q}\Big[\mathbf{sg}\Big(\mathrm{clip}\big(\frac{p_\theta(x)}{q(x)},1-\epsilon_{\text{low}},1+\epsilon_{\text{high}}\big)\Big)\cdot\log p_\theta(x)\cdot A(x)\Big]$$

    $\mathbf{sg}(\cdot)$ 即 `detach`/stop-gradient：**clipped ratio 不参与梯度回传，只作固定系数**；真正被优化的是 $\log p_\theta(x)$。所以 CISPO 是"给每个 token 的学习力度加一个上限，但不直接把某些 token 的梯度剪没"；PPO 是 clip objective，CISPO 是 clip ratio 再当砝码。
  - 三者一句话对照：

    | Loss | 人话比喻 | 核心动作 | 最易误解处 |
    | --- | --- | --- | --- |
    | Importance Sampling | 按功劳发奖金 | `ratio * advantage` | loss 可能为负，因为代码在最小化负目标 |
    | PPO | 奖金有预算上限 | `min(unclipped, clipped)` | `min` 不是随便取小，而是保守更新 |
    | CISPO | 给学习力度挂固定砝码 | `detach(clipped_ratio) * logprob * advantage` | detach 后 ratio 不走梯度，主要优化 logprob |
  - **基础概念的精确定义**（getting-started §4 原义）：
    - **SFT 与 RL**：SFT 给标准答案，模型按 `cross_entropy` 学；RL 让模型自己作答，用 reward 打分，把好的回答强化、差的压下去。GRPO 往后各章都是 RL 或蒸馏，SFT 只在 Spec-o3、Medical OPD 等章节作为热启动出现。
    - **policy / rollout / on-policy**：policy 就是当前正在训练的权重；从 prompt 出发生成一次回答叫一条 rollout。on-policy 指训练数据由**当前权重**采样而来，所以每个 step 先存权重拿 sampler、采完再更新；用旧权重采的数据是 off-policy，两者之间的偏差靠重要性比率补偿。
    - **reward**：给一条回答打的分。GSM8K 是 0/1（boxed 答案是否匹配）；Search-R1 会加上格式与搜索行为；ALFWorld 看任务是否完成。reward 设计往往比算法本身更影响结果。
    - **advantage 与组**：GRPO 对同一道题采 `group_size` 个回答，advantage = reward − 组内平均 reward：比平均好的被鼓励、比平均差的被压低，因此**不需要额外训 critic**，这是 GRPO 相对 PPO 最大的简化。整组 reward 全相同（全对或全错）时 advantage 全 0，这一组白采，即日志里的 degenerate group。（注：本篇给的是"减均值"的定义；各章代码实际进一步除以组内标准差，形如 $(r_i-\bar r)/(s_r+10^{-6})$。）
    - **old logprobs 与重要性比率**：logprob 是每个 token 的对数概率；同一 token 在新旧 policy 下的概率之比即重要性采样比率，RL loss 基本都在对比率乘 advantage 做文章——`ppo` 给比率加 clip，`cispo` 换了 clip 的位置。
    - **LoRA 与权重**：不动 base model，只训一个低秩适配器（demo 里 rank 32）。训练结束保存的就是这份适配器：`save_weights_for_sampler` 存给采样与评测用，`save_state` 存完整 checkpoint 用于续训；权重地址是 `trio://` 形式。
    - **OPD（On-Policy Distillation）**：同样用 student 自己采样的回答，但监督信号不是 reward——teacher 对 student 的回答**逐 token** 算 logprobs，用 student 与 teacher 的差（reverse KL）当 advantage，相当于老师逐 token 批改。OPSD 再进一步：teacher 就是 step 0 的自己，只是 prompt 里多给了参考解答。
    - **多轮与 observation mask**：Agentic RL 让模型在回答中途调工具（搜索、代码解释器、TextWorld 环境）。工具返回的内容（observation）拼进上下文给模型看，但**不参与 loss**——环境说的话不是模型的动作，不该被强化。代码里体现为这些 token 的 logprobs 与 advantages 填 0。
  - **分工边界**（getting-started §0，属于概念而非安装）：PyTRIO 在云端负责前向、反向、优化器更新、采样与权重保存；SwanLab 负责指标曲线、超参数与终端输出；本地代码只负责数据集、tokenizer、reward 与训练循环。本地 torch 是 CPU 版，仅自定义 loss 走 `forward_backward_custom`（grpo 里的 cispo、DAPO、GSPO 等）时用得上；`cross_entropy`、`importance_sampling`、`ppo` 这些内置 loss 全程在云端算。
- **reward 设计**：本篇不定义具体 reward，只给分类：0/1 答案匹配（GSM8K）、叠加格式与搜索行为（Search-R1）、任务完成度（ALFWorld）；并强调 reward 设计常比算法本身更决定结果，故各章单独解释自己的 reward。
- **关键实现细节**：本篇给的是读代码所需的最小词汇表，不涉及图片进入上下文、工具协议、数据规模与训练步数；唯一与实现绑定的细节是"on-policy 要求每个 step 先存权重拿 sampler、采完再更新"，以及"observation 的 logprobs 与 advantages 填 0"这两条代码级约定。
- **原文给出的结果数字**：该章未给定量结果（仅给出示例超参与环境版本：LoRA rank 32、`pytrio==0.2.9`、`swanlab==0.10.4`、Python ≥ 3.13、本地 torch 为 CPU 版；这些是配置事实而非实验结果）。
- **坑与边界**：本篇明确只讲到"能读代码的程度"，推导与对比留给各章正文；`--swanlab-mode` 的可选值各章略有差别（多数四档 `online/local/offline/disabled`，grpo 三个脚本只有 `online` 与 `disabled`），以 `--help` 为准；advantage 的"减均值"简写与各章"再除标准差"的实现存在表述落差，读代码时以具体章节为准；CISPO 的 `detach` 是最易误解处——ratio 被 clip 后只是常数权重，梯度全部来自 `log p_\theta`。
- **一句话记住它**：ratio 决定谁被强化。
