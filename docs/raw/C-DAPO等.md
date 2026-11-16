# DAPO / GSPO / TEMPO / ALFWorld 章节精读笔记

## DAPO

- **论文**：《DAPO: An Open-Source LLM Reinforcement Learning System at Scale》，arXiv:2503.14476（Qiying Yu et al., 2025）；定位：面向 long CoT 的大规模 LLM 强化学习开源系统，以一套「改进 recipe」在 GRPO 骨架上做四处改动，而非重写算法。

- **解决什么痛点**：GRPO 在 long CoT 场景有四类毛病——正 advantage 的低概率 token 提升空间被对称 clip 压住导致 entropy collapse；组内全对/全错的 prompt 提供不了相对信号；sample mean 让 4,000 token 的长回答和 100 token 的短回答在 batch 最外层等权；硬截断让 reward 突然从「可能对」变成「错」。DAPO 对应给出 Clip-Higher、Dynamic Sampling、Token-level Loss、Soft Overlong Punishment 四个开关，与 GRPO 唯一的代码差异集中在 `train.py` 的 preset 表。

- **核心公式**：

逐 token 的 PPO 概率比：

$$\rho_{i,t}(\theta)=\frac{\pi_\theta\left(y_{i,t}\mid x,y_{i,\lt t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(y_{i,t}\mid x,y_{i,\lt t}\right)}$$

DAPO 的目标（token mean，代码 `train.py:280` 的 `-torch.cat(token_objectives).mean()`）：

$$\mathcal{L}_{\mathrm{DAPO}}(\theta)=-\frac{1}{\sum_{i=1}^{G}\lvert y_i\rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert y_i\rvert}\min\left(\rho_{i,t}\hat A_i,\ \mathrm{clip}\left(\rho_{i,t},\,0.8,\,1.28\right)\hat A_i\right)$$

GRPO preset 的对照形式（sample mean，代码 `train.py:277` 的 `torch.stack(sequence_losses).mean()`）：

$$\mathcal{L}_{\mathrm{GRPO}}(\theta)=-\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert y_i\rvert}\sum_{t=1}^{\lvert y_i\rvert}\min\left(\rho_{i,t}\hat A_i,\ \mathrm{clip}\left(\rho_{i,t},\,0.8,\,1.2\right)\hat A_i\right)$$

组内标准化 advantage（`rollout.py:154-178`，`std` 用 N−1 分母、`epsilon=1e-8`）：

$$\hat A_i=\frac{r_i^{\mathrm{shaped}}-\mathrm{mean}\left(r^{\mathrm{shaped}}\right)}{\mathrm{std}\left(r^{\mathrm{shaped}}\right)+\epsilon}$$

符号解释：$G$ 是同题 completion 数（默认 8）；$\lvert y_i\rvert$ 是第 $i$ 条 completion 的 token 数；$\rho_{i,t}$ 是第 $t$ 个 token 的新旧策略概率比，代码为 `exp(current_logprob - old_logprob)`；$\hat A_i$ 为整条回答共享的标量 advantage，代码把它广播到该条所有 completion token（prompt token 的 advantage 与 mask 均为 0）；$r_i^{\mathrm{shaped}}$ 是 shaped reward。两种 reduction 只差归一化粒度：token mean 让每个 completion token 等权，长 reasoning 自然获得更大梯度权重。

- **reward 设计**：单条 completion 取最后 300 字符里最后一个 `\boxed{}`（支持嵌套括号），用 `math_verify` 判数学等价；正确 +1，错误或格式非法 −1，即 $r_i^{\mathrm{base}}\in\{+1,-1\}$。DAPO preset 再叠加分段线性长度惩罚：

$$p_i=\begin{cases}0, & L\le L_{\max}-C\\[2pt] -\dfrac{L-\left(L_{\max}-C\right)}{C}, & L_{\max}-C<L\le L_{\max}\\[2pt] -1, & L>L_{\max}\end{cases}\qquad r_i^{\mathrm{shaped}}=r_i^{\mathrm{base}}+p_i$$

$L$ 为 completion token 数，$L_{\max}$=`max_tokens`，$C$=`overlong_cache`；惩罚起点 `penalty_start = max_tokens - overlong_cache`。最初配置 8192/2048，惩罚从 6,144 token 起、8,192 处到 −1；现默认 4096/1024，对应「0～3,072 不惩罚、3,584 为 −0.5、4,096 为 −1.0」。评分粒度是 **trajectory-level**（一条 completion 一个标量 reward），长度惩罚按 completion token 数计算、不是 per-turn；prompt token 全 mask，只有模型生成的 completion token 进 loss。

- **关键实现细节**：
  - clip 区间：GRPO `[0.8, 1.2]`，DAPO `[0.8, 1.28]`（`clip_low` 两侧都是 0.8，只放宽上界）。
  - reduction：GRPO sample（先每条回答 token 均值、再回答间均值）；DAPO token（拼接全部 completion token 后直接均值）。
  - 序列级/ token 级：ratio 与 clip 都是 **token 级**，只有 reduction 粒度不同；advantage 是序列级标量、广播到 token。
  - Dynamic Sampling：先按**原始正确性**筛组，条件为 $0<\sum_j \mathbb{1}\left[r_j^{\mathrm{base}}=+1\right]<G$（`is_effective_group`），advantage 仍用 shaped reward 计算；最多采样 `groups_per_step × max_candidate_multiplier`（默认 2×）个候选组，候选用尽后用已有有效组训练部分 batch，一个有效组都没有则 `train/update_skipped=1` 跳过更新——这个 2× 上限与 partial batch 是本文实现改动，不属于原论文 full-batch 语义。
  - 环境存档读档：不涉及（纯数学推理，无环境状态）。
  - critic 共享参数：无 critic，group-relative advantage 代替 value model。
  - 数据规模：DAPO-Math-17k，剥掉原始 `Answer:` 模板、规范化去重、GT 冲突整组删除、固定 seed 切 50 条 dev，得到 `train.jsonl` 17,126 题 / `dev.jsonl` 50 题，dataset revision 写死。
  - 训练规模与耗时：最初配置 Qwen3.5-4B + LoRA rank 32，每步 4 个目标题组 × 8 条 completion = 32 条，prompt 上限 1,024、completion 上限 8,192、overlong 最后 2,048、rollout concurrency 16、Adam lr `4e-5` β `(0.9, 0.95)`、seed 42；**训练到第 35 step 已约 4.17 小时，作者手动关闭**。现默认缩为 prompt 512 / completion 4,096 / overlong 1,024，groups 4 与 group_size 8 不变；代码硬上限 `groups_per_step × group_size ≤ 128`。

- **原文给出的结果数字**：该章**未给定量效果结果**，明确写「本文不展示训练曲线，也不据此比较两个算法的优劣」，且不把未完成的 DAPO run 当效果对比。唯一可引用数字是工程侧：35 step 约耗时 4.17 小时；候选倍率公式「最多采样 group 数 = M × B」「候选 token 计算量上限约为正常 batch 的 M 倍」「首轮只差 1 个有效组时最多经历 rollout wave 数 = 1 + (M−1)×B」，故 B=4、M=2 最坏为「1 轮初始 batch + 4 轮单 group 补采 = 5 个等待 wave」，M=3 最坏为 `1 + 2×4 = 9` 个 wave。

- **坑与边界**：
  - Dynamic Sampling 的被丢弃 completion 已经付过 prefill → autoregressive sampling → reward parsing → 正确性校验的完整成本，只是不进 PPO loss；因此必须同时看 `rollout/effective_fill_ratio`、`rollout/oversample_ratio`、`train_completion_tokens / completion_tokens`、`time/step_seconds` 四类指标。
  - 补采存在串行 refill barrier：虽然**每轮内部**用 `asyncio.gather` + semaphore 并发，但下一轮采几组必须等上一轮全部返回并判完 reward 才知道；越接近填满并发宽度越小，最后一个有效 group 成为拖慢整批更新的尾部任务，与异步 RL「先完成先流转」的目标结构性冲突。
  - wall-clock 放大不严格受 2×/3× 限制：每个 wave 都可能撞上接近 8,192 token 的长尾 completion。
  - 单步 token 与 wall-clock 波动难预估：每步都出现到达上限的长回答，候选组数、有效组数、train token 数随补采结果变化。
  - 章节定位是「讨论 Dynamic Sampling 的代码行为与工程问题」，规模小于论文实验，只能观察机制。

- **一句话记住它**：四处改动，卡在补采。

## GSPO

- **论文**：《Group Sequence Policy Optimization》，arXiv:2507.18071（Qwen 团队，2025）；定位：沿用 GRPO 的 group rollout / reward / group-relative advantage，只把重要性比率与裁剪从 token 级提升到 sequence 级。

- **解决什么痛点**：GRPO 对一条含 $T$ 个 token 的回答会产生 $T$ 个 ratio、做 $T$ 次 clipping 判断，长序列里逐 token ratio 带来高方差噪声；而 reward 本身是按完整回答给的，粒度和优化目标不一致。GSPO 把整条回答压成 **1 个 ratio、1 次 clipping**，让 reward、重要性采样与优化目标处在同一粒度上。相对 DAPO 它更集中：不加 Dynamic Sampling、不补采、不做长度惩罚。

- **核心公式**：

GRPO 的 token 级比率（作为对照）：

$$w_{i,t}(\theta)=\frac{\pi_\theta\left(y_{i,t}\mid x,y_{i,\lt t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(y_{i,t}\mid x,y_{i,\lt t}\right)}$$

GSPO 的序列级比率（逐 token log-ratio 求均值再取指数，即几何平均，代码 `loss.py:133-134`）：

$$s_i(\theta)=\exp\left(\frac{1}{\lvert y_i\rvert}\sum_{t=1}^{\lvert y_i\rvert}\log\frac{\pi_\theta\left(y_{i,t}\mid x,y_{i,\lt t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(y_{i,t}\mid x,y_{i,\lt t}\right)}\right)$$

目标函数（论文式 (5)/(7)）：

$$\mathcal{J}_{\mathrm{GSPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i\hat A_i,\ \mathrm{clip}\left(s_i,\,1-\varepsilon_{\mathrm{low}},\,1+\varepsilon_{\mathrm{high}}\right)\hat A_i\right)$$

实现中的实际归一化（`loss.py:155-157`）：

$$\mathcal{L}_{\mathrm{GSPO}}=-\frac{1}{N_{\mathrm{orig}}}\sum_{i\in\mathcal{A}} \min\left(s_i\hat A_i,\ \mathrm{clip}\left(s_i,1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\right)\hat A_i\right)$$

符号解释：$\lvert y_i\rvert$ 是第 $i$ 条 completion 的 token 数，$1/\lvert y_i\rvert$ 是长度归一化，使不同长度回答的 ratio 落在相近数值范围；$s_i$ 是整条回答共享的唯一 ratio，因此也共享唯一一次裁剪结果；$\hat A_i$ 是组内标准化 advantage（每条回答一个标量，整条序列共享）；$\varepsilon_{\mathrm{low}}=3\times10^{-4}$、$\varepsilon_{\mathrm{high}}=4\times10^{-4}$（`GSPOConfig` 默认值，取论文原文）；$\mathcal{A}$ 是 advantage 非零的序列集合，$N_{\mathrm{orig}}$ 是**过滤前的原始 rollout 序列数**（每步固定 64），退化组的零目标仍留在分母里。裁剪区间看起来比 GRPO 小几个数量级，因为约束对象不同：GRPO 约束单 token 概率比，GSPO 约束长度归一化后的整条 sequence ratio。

advantage：

$$\hat A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+10^{-8}}$$

其中 $r_i\in\{0,1\}$，$\mathrm{std}$ 用 N−1 分母；组内全对或全错时 std=0，整组 advantage 全为 0。

- **reward 设计**：提取回答中最后一个完整 `\boxed{}`（支持嵌套括号，**不限于末尾 300 字符**，与 DAPO 实现不同），用 `math_verify` 判等价；正确得 `1`、错误或格式缺失得 `0`。**没有长度惩罚、没有额外 format reward**，`shaped_reward` 恒等于 `base_reward`（字段保留但未用）。评分粒度是 **trajectory-level**（一条回答一个 0/1），不是 turn-level。

- **关键实现细节**：
  - clip 区间：`1 - 3e-4` 到 `1 + 4e-4`，即约 `[0.9997, 1.0004]`；整条回答只判断一次。
  - reduction：每条 sequence objective 等权求和后除以原始序列数 `normalization_sequences`；`normalization_tokens`（原始 token 总数）只用于算 `gspo/token_clip_fraction` 指标，不参与 loss 归一化。所以「过滤只节省计算，不会放大剩余样本的 loss」。
  - 序列级 / token 级：ratio 与 clipping 都是 **sequence 级**，loss 是 sequence 级聚合；token 只用于算 log-ratio 均值。
  - Datum 构造：`model_input = prompt_tokens + completion_tokens[:-1]`，`target_tokens = [0] * (len(prompt_tokens) - 1) + completion_tokens`；prompt 段 target 仅占位，loss 只取末尾 completion 区间（`current[-completion_tokens:]`），token 与 sampling logprob 长度被严格校验。
  - 环境存档读档：不涉及。
  - critic 共享参数：无 critic、无 value model。
  - 数据规模：DAPO-Math 清洗后 17,126 道训练题（固定 revision），外加 30 道 AIME25 评测集。
  - 训练规模与耗时：Qwen3.5-4B + LoRA rank 32，100 steps，每步 8 个 prompt group × 8 条 completion = 单步 64 条；prompt/completion 上限 2,048 / 4,096；temperature 1.0 / top-p 1.0 / top-k −1；Adam lr `4e-5` β `(0.9, 0.95)`；每 20 步存 checkpoint。平均每步约 **85 秒**，总时长约 **2 小时 22 分钟**，PyTRIO 会话账单 **¥155.80**。一个 rollout batch 只调一次 `forward_backward` 和一次 `optim_step`，应用侧不拆 mini-batch。

- **原文给出的结果数字**：AIME25 固定 30 题、每题采样 12 次共 360 条、temperature 1.0、max_tokens 8192：

| Checkpoint | Average@12 | Pass@12 | Format | 正确回答 | 通过题目 | 平均 completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base (Run 1) | 41.11% | 56.67% | 46.67% | 148/360 | 17/30 | 6489.8 |
| Base (Run 2) | 40.56% | 66.67% | 47.50% | 146/360 | 20/30 | 6442.5 |
| Step 20 | 41.11% | 73.33% | 62.22% | 148/360 | 22/30 | 5775.4 |
| Step 40 | 37.78% | 60.00% | 87.22% | 136/360 | 18/30 | 4785.8 |
| Step 60 | 35.00% | 63.33% | 92.50% | 126/360 | 19/30 | 4438.7 |
| Step 80 | 34.72% | 73.33% | 97.78% | 125/360 | 22/30 | 3886.9 |
| Step 100 | 33.89% | 66.67% | 99.44% | 122/360 | 20/30 | 3358.1 |

封面「AIME25 提高 6 个点」的具体口径：Step 20 的 Pass@12 相比 Base Run 2 从 66.67% → 73.33%，**+6.66 个百分点**。训练侧：100 steps 平均 sequence clip fraction **13.11%**，最后一步 **6.25%**；Format 从 47.50% → 99.44%；平均 completion tokens 从 6442.5 → 3358.1，**缩短约 48%**。

- **坑与边界**：
  - Step 20 的 Average@12 为 41.11%，仍在两次 Base 评测的波动范围内，正确回答数与 Base Run 1 相同（148/360）；Step 20 之后 Average@12 逐步下降。
  - 两次独立 Base 的 Pass@12 相差 **10 个百分点**（56.67% vs 66.67%）；AIME25 只有 30 题，单次 Pass@12 极易受采样波动影响，+6.66 pp 需更多重复评测与随机种子确认。
  - 缺一组同模型、同数据、同训练预算的 matched GRPO 对照，因此不能据此说 GSPO 优于 GRPO。
  - 与论文规模不等价：原论文用 `Qwen3-30B-A3B-Base` 微调的冷启动模型，把一个 rollout batch 切成 4 个 optimizer mini-batch，报告 AIME 2024 / LiveCodeBench / CodeForces；本文是 Qwen3.5-4B LoRA、一 batch 一次更新、AIME25，**未验证论文中的 MoE 稳定性结论**。
  - 不同指标的最优 checkpoint 不同：Step 20 Average@12 最好，Step 20 与 Step 80 Pass@12 最高，Step 100 格式率最高。

- **一句话记住它**：一条回答，一个 ratio。

## TEMPO

- **论文**：《TEMPO: Test-Time-Scaled Value Estimation with Macro-Step Policy Optimization》——**论文尚未发布，无 arXiv 编号**，唯一来源是小红书 Dots 团队技术博客（studio.dots.ai/dots/tempo-blog.html），超参、训练曲线、消融实验全部未公开；定位：把长轨迹切成 macro-step、用会推理的生成式 critic 做价值估计的算法级复现。

- **解决什么痛点**：GRPO 在几百轮的长程任务上有两个硬问题——结果导向奖励要等轨迹彻底结束才能观测（rollout 跑八小时，信号周期下限就是八小时）；开局第三个动作与最终成败隔几百步，只按终局奖励算组内相对信号无法判断中间哪些决定关键。PPO 用 scalar value head 在中间状态预测未来回报，但一次前向计算量固定，而长程状态估值往往需要检查 actor 的假设、搜索方向和未解决障碍，本身需要推理。TEMPO 因此把优化单位从「整局」降到「一段」，并把估值变成生成任务。

- **核心公式**（readme 第 2 节，与 `tempo.py` 逐行对应）：

分支 return（截断尾部由 critic 补齐）：

$$R_n=r_n+\hat V\left(s_n^{\mathrm{end}}\right)$$

TD target：

$$G=\frac{1}{N}\sum_{n=1}^{N}R_n$$

actor advantage（`tempo.py:71`，只减均值、**不除标准差**）：

$$A_n=R_n-\mathrm{mean}\left(R\right)$$

critic 的误差奖励：

$$r_k=-\frac{\left\lvert \hat V_k-G\right\rvert}{R_{\max}},\qquad R_{\max}=1$$

符号解释：$n$ 是同一个起点状态的第 $n$ 条分支，$N$=`branches`（默认 4）；$r_n$ 是该 macro-step 段内环境奖励（纯 0/1，几乎全 0，只有成功那一轮为 1）；$\hat V(s_n^{\mathrm{end}})$ 是 critic 对终点状态的估值，**终局分支（`stop_reason != "macro_boundary"`）估值恒为 0**，只有停在 macro 边界的分支参与 bootstrap；$G$ 既是 TD target 也是「这个起点值多少」，warm-up 阶段改用完整轨迹的 Monte Carlo return(0/1)；$k$ 是同一状态第 $k$ 次独立估值，$K$=`critic_samples`（默认 4）；$R_{\max}$ 是该状态剩余回报跨度，纯 0/1 奖励下恒为 1，因此误差就是 $|\hat V_k-G|$。两侧都做组内中心化（标准 GRPO），advantage 再广播到各自 completion token；环境 observation token 的 logprob / advantage 一律填 0 占位。

- **reward 设计**：环境奖励是**纯 0/1 won 信号**，成功 = 1；刻意**不用**第 8 篇的非法动作惩罚，为的是保住 $R_{\max}=1$ 的严格性。critic 输出形式为结尾 `<value>0.42</value>`，严格正则解析 `[0-9]*\.?[0-9]+`、取最后一个匹配、收敛到 [0,1]；解析失败样本在 critic 训练侧按最差 reward `-R_max` 处理（误差记为 $R_{\max}$），在终点估值侧则被剔除、全失败时悲观取 0.0。评分粒度是 **macro-step（segment）级**：段内累加奖励 + 终点估值，而不是整条轨迹一个标量，也不是逐 turn 打分。

- **关键实现细节**：
  - clip 区间：**没有自定义 clip**，服务端用 pytrio 内置 `loss_fn="importance_sampling"`，actor 与 critic 共用同一 loss，服务端无任何自定义 loss。
  - reduction：actor 与 critic 的 Datum 合并进同一个 `forward_backward` batch（超过平台上限时按 `MAX_TRAIN_BATCH_SIZE` 分块做梯度累积，再统一 `optim_step`）；advantage 为 0 的分支/样本被跳过。
  - 序列级 / token 级：advantage 是段级标量、广播到该段所有 assistant token；`build_segment_datum` 把「保存的 token_prefix + 段内各轮」拼成**完整序列**，逐 token 右移对齐（`input=full[:-1]`、`target=full[1:]`），环境 token 的 old logprob/advantage 为 0。
  - 环境存档读档（本篇最硬的工程点）：一个 `MacroState` 同时存两份快照——环境侧 `action_history`（顺序重放即可恢复游戏内部状态）+ token 侧 `token_prefix`（保证下一轮 prompt 仍是历史真实 token 的严格前缀扩展），外加 `messages`、`round_index`、`macro_index`、`environment_seed`、当前 observation 与 admissible actions。读档 = 新开一组环境 → reset → 逐条重放动作历史，再用**保存时的 observation 做逐字校验**，对不上直接抛异常拒绝训练。`StateStore`（容量 256、FIFO、均匀不重复抽样）每轮存入所有停在边界的分支终点，下一轮从仓库抽样消费；终局分支、轮数用尽、token 前缀无续跑空间的分支不入库。压测：3 个游戏各走 5~6 步 walkthrough，用全新环境重放 observation 逐字一致，恢复后再走一步两环境行为一致。
  - critic 共享参数：**只维护一份参数**，交互时是 actor、到边界切 critic prompt 估值，共用同一采样器和同一次权重更新，甚至可以共享同一个 LoRA；角色只靠 prompt、上下文、奖励区分。critic 可见**特权信息**——`game.tw-pddl` 自带的 `walkthrough` 专家动作序列，actor prompt 里只有任务与交互历史；`condensed_history` 还会裁掉 observation 里 `Available actions` 之后的文本以控长。
  - 数据规模：ALFWorld train split，**3,553 局**（readme 表格口径），T=50 轮、H=10、每局 M=5 段。
  - 训练规模与耗时：Qwen3.5-4B + LoRA rank 32，warm-up 4 次 update + TD 4 次 update 的冒烟规模（正式算法验证把 `--td-updates` 拉到 40），每批 4 个起点、`states_per_batch=4`，每步 Datum 数 = states × (N + K) = 4×(4+4)；`endpoint_samples=2`（拼 return 时每个终点估几次取均值，不是训练 group）；单轮 assistant 上限 512、critic 上限 512。**本次冒烟总开销 15.02 元**。

- **原文给出的结果数字**：
  - 冒烟第 1 个 warm-up update：`warmup=1 critic/abs_error_mean=0.606 warmup/boundary_states=22 warmup/steps_mean=19.688 warmup/success_rate=0.750 loss=0.0513`——16 条轨迹平均 19.7 轮终局、收集 22 个边界状态（平均长度不足两段属正常）。
  - actor 侧 8 次 update（4 warm-up + 4 TD）：`actor/segment_reward_mean` 从 **0.62 升到 0.85**；`actor/degenerate_group_rate` 从 **0.19 降到 0**（初期约 1/5 起点组 return 完全相同只能跳过）。
  - critic 侧：估值误差从 **0.674 降到 0.315**（warm-up 三步下降约四成）；`critic/value_corr` 在 warm-up 阶段从 **0.60 升到 0.72**；解析失败率从 **7.3% 降到约 1%**。
  - 成本：本次冒烟 **15.02 元**（对照第 8 篇完整训练的四位数账单）。
  - 网络文档未给出任何排行榜/成功率对比数字。

- **坑与边界**：
  - 论文未发布，H、N、K 与实验数字在博客里全部脱敏，所以只做算法级复现，不追榜单。
  - **前缀分布偏移的重要性采样修正被省略**：严格等价性要求按前缀概率比给 advantage 加权；附录证明前缀 on-policy 时无需修正，且终点状态下一轮立即被消费、偏差有限。已想好的实现路径是保存状态时累积 assistant token 的旧 logprob 之和、每轮对每个起点调一次 `compute_logprobs` 算当前值、把比值乘进 advantage。
  - 环境从 ARC-AGI-3（付费 API、服务端状态无法恢复）换成 ALFWorld，T=50 只是机制验证；拉到 100+ 会遇到上下文长度这个真正硬约束。
  - 初期约 1/5 起点组 return 完全相同（退化组）拿不到训练信号；`abs_error_mean=0.606` 说明未训练 critic 基本没在预测成败。
  - 每轮要为 N 条分支各新开环境重放前缀，game file 加载开销需预估。
  - 冒烟规模很小（各 4 次 update），观察是「同步出现的」而非因果结论。

- **一句话记住它**：切段估值，一参两用。

## ALFWorld

- **论文**：环境侧是《ALFWorld: Aligning Text and Embodied Environments for Interactive Learning》，arXiv:2010.03768（Mohit Shridhar et al., ICLR 2021），定位：把 TextWorld 文字环境与 ALFRED/AI2-THOR 具身环境对齐的交互式环境；**算法侧不是该论文的方法**——本文复现的是一套「基于 ALFWorld 的 LLM Agentic RL recipe」：GRPO-style group rollout + 组相对 advantage，用 PPO 在线更新 LoRA，ALFWorld 原论文的 DAgger 训练与视觉 BUTLER 系统不在范围内。

- **解决什么痛点**：相对 GRPO 的数学题设定，这里没有「同一道题的 8 个答案」，改为**同一个游戏、同一个初始状态下的 8 条独立环境轨迹**做分组；同时把 group-relative advantage 从「短工具调用」推到「最多 50 次真实改变 PDDL 状态的多轮交互、单序列 12K token」的长轨迹场景，并保留从任务开始到当前的**完整工具历史**（不丢早期历史，避免重复搜索、拿错物体、忘记已完成步骤）。相对标准 GRPO 它还做了两处简化：advantage 只减均值不除标准差；reward 里显式扣非法动作。

- **核心公式**：

轨迹 reward（`rollout.py:572-575`）：

$$r_i=\mathbb{1}\left[\mathrm{won}_i\right]-0.1\cdot n_i^{\mathrm{invalid}}$$

同游戏组内 advantage（`advantages.py:18-41`，只减均值、**不除标准差**）：

$$A_i=r_i-\frac{1}{K}\sum_{j=1}^{K}r_j$$

PPO 逐 token 目标（`train.py` 调内置 loss，`clip_low_threshold=0.8`、`clip_high_threshold=1.2`）：

$$\rho_{i,t}(\theta)=\frac{\pi_\theta\left(a_{i,t}\mid s_{i,t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(a_{i,t}\mid s_{i,t}\right)},\qquad \mathcal{L}=-\frac{1}{N}\sum_{i}\sum_{t\in\mathcal{T}_i}\min\left(\rho_{i,t}A_i,\ \mathrm{clip}\left(\rho_{i,t},0.8,1.2\right)A_i\right)$$

符号解释：$i$ 是一条独立轨迹；$\mathrm{won}_i$ 是环境给出的终局成功标志；$n_i^{\mathrm{invalid}}$ 是非法动作计数（**工具格式非法或环境不可执行**任一即计一次，`valid_action = valid_format and admissible`）；$K$=`group_size`（默认 8），是同一游戏的分支数；$A_i$ 是整条轨迹共享的标量，广播到该轨迹所有 assistant completion token，system/user/tool observation token 的 advantage 与 old logprob 均为 0；$\mathcal{T}_i$ 是轨迹 $i$ 的 assistant token 集合；$N$ 为参与 loss 的 token 总数（由内置 PPO loss 决定归一化）。轨迹打断口径：`won=True` 成功、`done=True, won=False` 环境结束、达到 `max-episode-steps`、或 prompt 超过 12K token 提前截断（`trajectory_too_long`）。

- **reward 设计**：成功基础分 1、失败 0，每出现一次非法工具格式或环境不可执行 action 扣 0.1：成功 + 2 次非法 → 0.8；失败 + 3 次非法 → −0.3。**不计算 step reward、step return、step advantage 或 action advantage**，环境原生 `score` 只写进评测轨迹、不进训练 reward，也没有长度惩罚。评分粒度是 **trajectory-level**（一条完整 episode 一个标量），**turn-level 信号完全不用**——这与 TEMPO 的段级信号形成直接对照。

- **关键实现细节**：
  - clip 区间：`0.8 / 1.2`（`--ppo-clip-low` / `--ppo-clip-high` 默认值，校验 `0 < low ≤ 1 ≤ high`）。
  - reduction：直接用 PyTRIO 内置 `loss_fn="ppo"`，每条完整轨迹构成**一个** `Datum`（整条多轮轨迹一次性右移对齐，不是每轮一个 Datum），一次 `forward_backward` + 一次 `optim_step`；advantage 全 0 的轨迹被过滤，不提交无效 backward。
  - 序列级 / token 级：advantage 轨迹级、广播到 token；clip 与 ratio 是 token 级；`model_input = full_tokens[:-1]`、`target_tokens = full_tokens[1:]`，环境 token 的 old logprob/advantage 填 0。
  - 环境存档读档：训练**不做存档读档**（对比 TEMPO 的关键差异）——同一游戏开 K 个独立 TextWorld 实例，`reset()` 断言 K 条分支初始 observation、task、game file 完全一致，`step()` 一次推进整组 K 个 action；文件里含 TextWorld 1.7.0 在 CPython 3.13+ 的动态变量作用域兼容处理与异步 worker 安全关闭，可在 Python 3.14 跑。训练轨迹只在进程内存，不落盘。
  - critic 共享参数：无 critic；优势全部来自同游戏组内相对 reward。
  - 数据规模：从本地 `datasets/alfworld/json_2.1.1` 发现 `game.tw-pddl`，只保留 `solvable=true` 的六类任务（pick_and_place_simple / look_at_obj_in_light / pick_clean_then_place_in_recep / pick_heat_then_place_in_recep / pick_cool_then_place_in_recep / pick_two_obj_and_place）：`train` **3,553** 局、`valid_seen` **140**、`valid_unseen` **134**（评测共 274 局）；训练按 seed 打乱并 `take_batch()` 循环取样。
  - 训练规模与耗时：Qwen3.5-4B + LoRA rank 32，**80 updates**，每 update 8 个游戏 × 8 条轨迹 = 单步 64 条长轨迹，总 rollout **5,120 条轨迹**；单轨迹最多 50 次交互、12,000 tokens（平台单序列硬上限），单轮 assistant 上限 2,048；temperature 1.0 / top-p 1.0；Adam lr `1e-6` β `(0.9, 0.95)`、grad clip 1.0；每 20 updates 同时存 `state` 与 `sampler weights`。**PyTRIO 账单 ¥1613.71**；token 构成 `prefilling: 843.51M`、`train: 62.24M`、`sample: 14.28M`。
  - 首轮采样优化：同游戏 K 条轨迹首轮 prompt 完全相同，只提交一个 `num_samples=K` 的请求；第一步之后状态分叉，每条未结束轨迹各提交一个候选，轨迹间并发、轨迹内严格串行；环境推进用 `ThreadPoolExecutor(max_workers=min(len(calls), 32))`。下一轮 prompt 由 `build_next_prompt()` 在上一轮真实 prompt token 与 completion token 后直接追加 assistant 闭合符和 tool observation，不重新 tokenize 历史文本，保证严格前缀扩展与 old logprob 逐 token 对齐。

- **原文给出的结果数字**：评测口径 Base / Step 40 / Step 80 完全一致（temperature 0.01、top_p 1.0、seed 42、max 50 环境步、274 局）：

| 模型 | Valid Seen | Valid Unseen | 总成功率 | 成功游戏 | 平均 Reward | 平均非法动作数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 52.86% | 52.99% | 52.92% | 145 / 274 | -0.0117 | 5.41 |
| Checkpoint 40 | **57.14%** | 52.24% | 54.74% | 150 / 274 | 0.0237 | 5.24 |
| Checkpoint 80 | 55.00% | **58.96%** | **56.93%** | **156 / 274** | **0.0880** | **4.81** |

Checkpoint 80 相对 Base：总体成功率 52.92% → 56.93%（**+4.01 pp**）、Valid Unseen 52.99% → 58.96%（**+5.97 pp**）、成功游戏 145 → 156（**+11**）、平均非法动作 5.41 → 4.81（**−0.60**）。任务级：`valid_unseen` 中 Heat then place **+26.1 pp**、Clean then place **+9.7 pp**、Pick two & place **−11.8 pp**；`valid_seen` 中 Pick two & place **+16.7 pp**，Look under light 与 Heat then place 回退。成本：¥1613.71；prefilling 843.51M / train 62.24M / sample 14.28M tokens；评测平均交互约 22 步（正文另称「21～23 步」）。

- **坑与边界**：
  - 结果来自**一次训练 + 单个评测 seed**，只能说明流程跑通并给出正向信号，稳定结论还需要多 seed 训练、重复评测和更长 checkpoint 曲线。
  - 收益**不沿所有维度单调**：Checkpoint 40 的 `valid_seen` 最好（57.14%），Checkpoint 80 的 `valid_unseen` 与总成功率最好，选 checkpoint 要看目标 split。
  - 任务级提升**很不均匀**：`valid_unseen` 的 Pick two & place 下降 11.8 pp，`valid_seen` 的 Look under light 与 Heat then place 回退。
  - 成本结构失衡：prefilling 远高于 train 与 sample，因为每执行一次工具，下一轮都要带上更长的任务描述、历史 tool call 与环境 observation，同一段早期上下文反复参与 prefix 计算；训练 token 只是账单的一部分。
  - 12K 是平台单序列硬上限（`MAX_SEQUENCE_TOKENS`），`--max-trajectory-tokens` 不得超过它；batch 上限同样受限（超限抛错）。
  - 在线 rollout 的任务类型与难度随 batch 变化，训练 success rate 与 reward 波动明显，不能直接当能力曲线；训练阶段不保存轨迹，只有评测阶段写 JSONL。
  - TextWorld 的 batch_size=1 会退化成进程内 `SyncBatchEnv`，多个 TatSu parser 是模块级单例、不能跨线程调用，因此 `group_size <= 1` 或 `asynchronous=False` 时需要全局锁。

- **一句话记住它**：同局八条，胜负减均值。
