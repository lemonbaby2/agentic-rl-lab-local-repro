# B · OPD 蒸馏系列（4 章）

> 提炼自 `agentic-rl-lab` 仓库的 4 个实验章节，仅记录原文与代码中出现的事实。

---

## 通用 OPD 教程（general-opd）
- **论文**：未标注。该章是 OPD 的入门教程而非单篇论文复现；其方法论来源在文末明确标为 GKD《On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes》（arXiv:2306.13649）与 MiniLLM《Knowledge Distillation of Large Language Models》（arXiv:2306.08543）。定位：用一份可读的最小闭环，把"学生采样、老师打分、反向 KL 更新"跑通，并给出同步/异步两版实现。
- **解决什么痛点**：传统离线蒸馏让学生学固定数据或 teacher 预生成的答案，学生真正推理时进入自己的生成分布，训练与推理之间出现偏移。OPD 把训练 completion 换成"当前 student 实时采样"，teacher 不再写答案，只给这条真实轨迹逐 token 打 logprob，用反向 KL 把学生拉向老师，从而缓解分布偏移。相对 SFT，OPD 的训练信号是"沿学生自己犯错的路径做逐 token 纠正"，原文称其比 SFT"更柔和"。
- **核心公式**：设 $x$ 为 prompt，$y$ 为 student 采样出的 completion，$\pi_\theta$ 为正在训练的 student，$\pi_T$ 为冻结的 teacher。目标是

$$
D_{\mathrm{KL}}\left(\pi_\theta \parallel \pi_T\right)
=
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}
\left[
\log \pi_\theta(y \mid x)-\log \pi_T(y \mid x)
\right]
$$

拆到 token 后，第 $t$ 个 token 的采样估计为

$$
r_t^{\mathrm{KL}}
=
\log \pi_\theta(y_t \mid x,y_{\lt t})
-
\log \pi_T(y_t \mid x,y_{\lt t})
$$

再转成 advantage：

$$
A_t=-\beta\, r_t^{\mathrm{KL}}
$$

符号：$x$ 是 prompt；$y_t$ 是 student 实际采样出的第 $t$ 个 token；$y_{\lt t}$ 是其前缀；$\pi_\theta$ 是带可训练 LoRA 的 student；$\pi_T$ 是冻结 teacher；$\beta$ 即代码里的 `kl_penalty_coef`，默认 `1.0`，控制向 teacher 靠拢的力度。直觉：`student_logprob > teacher_logprob` 时 advantage 为负，压低该 token；反之抬高。原文强调单个 token 的 $r_t^{\mathrm{KL}}$ 可以为负，只有对完整分布取期望后 KL 才保证非负。

- **reward 或 teacher 信号设计**：信号是 **teacher 对 student 已生成 token 的逐 token logprob 差**，即 token 级 reverse KL，不是 turn 级 credit，也没有 verifier / reward model。teacher 用 `compute_logprobs(prompt + student completion)` 一次性前向打分，切片 `all_logprobs[len(prompt_ids):]` 取 completion 区间，并强制校验长度一致且无 `None`。代码里只有两行：

```python
reverse_kl = np.asarray(student_lps) - np.asarray(teacher_lps)
advantages = -args.kl_penalty_coef * reverse_kl
```

- **关键实现细节**：
  - **teacher 与 student 关系**：student 是 `Qwen/Qwen3.5-4B` + rank 32 LoRA（`training_client`，可训练）；teacher 是 `Qwen/Qwen3.6-27B`（`teacher_client`，只算 logprob，不参与优化）。若传 `teacher_model_path` 指向已训练 LoRA sampler，可蒸馏领域 teacher；不传则直接用 `teacher_base_model`。两者**必须共享兼容 tokenizer**，因为信号是相同 token ID 下的 logprob 差。
  - **数据来源与规模**：ModelScope `AI-ModelScope/DeepMath-103K`，共 10 个 parquet 分片，默认只下载并读取第 1 个分片（`--num-shards` 控制）。这是 **prompt-only** 用法：只读 `question` 字段，不用标准答案，也不计算数学正确率。
  - **prompt 是否 mask**：是。构造 Datum 时 `prompt_loss_len = len(prompt_ids) - 1`，prompt 区间的 `target_tokens`、`logprobs`、`advantages` 全部用 0 占位，只为对齐长度，不产生训练信号；`input_ids = prompt_ids + completion_ids[:-1]`、`target_ids = [0]*prompt_loss_len + completion_ids` 做标准自回归右移。
  - **训练配置**：`loss_fn="importance_sampling"`；Adam lr `4e-5`，beta1 `0.9`、beta2 `0.95`；`max_tokens` 默认 512；`temperature=1.0`、`top_p=1.0`、`top_k=-1`；`sampler_refresh_steps=1`（每次更新后刷新 sampler，最严格 on-policy）；日志用 SwanLab。
  - **`group_size` 的语义**：仅让同一 prompt 多产生几条 on-policy 轨迹，**不做 GRPO 式组内相对 reward**，每条 completion 独立接受 teacher 逐 token 监督。
  - **同步/异步差异**：`02-demo-async.py` 不改算法，只把 `sample()` / `compute_logprobs()` 换成 `sample_async()` / `compute_logprobs_async()`，用 `asyncio.gather` 并发 batch 内各 prompt 的 rollout，并并发提交同一 prompt 下多条 completion 的 teacher logprob 请求；`forward_backward_async` / `optim_step_async` 也异步提交。同步版适合阅读和调试，异步版提高远程采样与打分阶段的并发利用率。
- **原文给出的结果数字**：该章未给定量结果。只给出指标定义（`data/datums`、`data/completion_tokens_mean/total/per_second`、`opd/reverse_kl_mean/std`、`trainer/*`），并说明 `completion_tokens_per_second` 是含 sampler 刷新、采样、teacher 打分、前后向与 optimizer 的**整步吞吐**，不等于纯推理 tokens/s。
- **坑与边界**：
  - student 与 teacher tokenizer 不兼容时无法比较同一 token ID 的 logprob，必须重新设计文本级对齐或换兼容模型。
  - token 级 `reverse_kl` 可正可负，日志里偶尔出现负的 token 均值不等于公式失效，需结合更多 step、序列长度和方差判断。
  - **KL 下降只说明 student 更接近 teacher，不自动等于任务正确率提高**；原文建议同时看 KL 是否多步趋稳、completion 长度是否突然坍缩、loss 是否 NaN/尖峰、训练前后独立评测是否改善。
  - `sampler_refresh_steps` 调大会减少权重同步开销，但轨迹变成 stale policy 数据。
  - 采样过于保守 → 轨迹多样性不足、teacher 能纠正的错误类型有限；过于随机 → 低质量 token 多、方差与 teacher 成本升高；`max_tokens` 直接决定单步成本。
  - `kl_penalty_coef` 太小收敛慢，太大则单 token advantage 过强、更新不稳定；建议先保持 `1.0`。
- **一句话记住它**：学生采样，老师只打分。

---

## 医疗 OPD：SAR-OPD 与 IDT-OPD（medical-opd）
- **论文**：未标注。该章复现的是 OPD 的核心训练机制（来源同为 GKD arXiv:2306.13649 与 MiniLLM arXiv:2306.08543），并在其上提出两套自命名方案 **SAR-OPD（Staged Anchor-Restoration OPD）** 与 **IDT-OPD（Interleaved Dual-Teacher OPD）**；原文明确说明这两个名字"并不是已有论文里的标准算法名称"。参考文献另列 HuatuoGPT-o1（arXiv:2412.18925）作为医疗推理数据来源。
- **解决什么痛点**：先做 Medical SFT 能把 MedQA-zh 从 `72.17%` 提到 `79.00%`，但 C-Eval 通用能力从 `81.67%` 掉到 `69.33%`——领域能力学进去了，通用能力被明显破坏。单 teacher 的普通 OPD 只能往一个方向拉，因此该章要回答"能否让 Medical Teacher 教医疗、同时让原始 Base Model 当通用锚点，最后两项都比 Base 更好"。相对 SFT，OPD 让学生不背 `complex_cot`/`response` 固定答案，而是在自己生成的医疗回答上接受逐 token 监督；相对普通 OPD，这里把"蒸馏目标"拆成医疗与通用两个 teacher。
- **核心公式**：token 级 reverse KL 与 advantage 与通用 OPD 完全一致：

$$
r_t^{\mathrm{KL}}
=
\log \pi_\theta(y_t \mid x,y_{\lt t})
-
\log \pi_T(y_t \mid x,y_{\lt t}),
\qquad
A_t=-\beta\, r_t^{\mathrm{KL}}
$$

符号：$\pi_\theta$ 为 fresh LoRA Student；$\pi_T$ 为当前 step 指定的冻结 Teacher（Medical SFT epoch 3，或原始 `Qwen/Qwen3.5-4B`）；$y_t$ 为 Student 采样 token；$\beta$ 为 `kl_penalty_coef`，默认 `1.0`。Student 比 Teacher 更偏爱某 token 时 $r_t^{\mathrm{KL}}>0$，advantage 为负，压低该 token；反之抬高。

前置的 Medical SFT 用带权重的交叉熵做 prompt mask：

$$
\mathcal{L}_{\mathrm{SFT}}
=
-\sum_{t \in \text{assistant}}
\log \pi_\theta(a_t \mid \text{system},\text{user},a_{\lt t})
$$

对应代码 `weights = [0.0]*len(prompt_tokens) + [1.0]*len(assistant_tokens)`，即 system / user prompt 只作上下文不参与 loss，assistant completion 与 EOS 才有非零权重。

- **reward 或 teacher 信号设计**：信号是 **teacher 对 student 同一条 `prompt + completion` 的逐 token logprob 差**（token 级 reverse KL），无 verifier、无 reward model、不使用数据集答案。Teacher 只调用 `compute_logprobs_async`，不重新生成答案；原文强调"如果两条 completion 不一样，就无法进行这种 token 级别的对齐"。所有方案都是 **prompt-only distillation**：Medical OPD 只读 `question`，丢弃 `complex_cot` 与 `response`；C-Eval 阶段只读题目和选项，**明确忽略 `answer_idx`**。更新 loss 用 PyTRIO 内置 **PPO**（`loss_fn="ppo"`）。
- **关键实现细节**：
  - **teacher 与 student 关系**：两套方案共用同一个 Medical Teacher——在中文 `medical-o1-reasoning-SFT` 上把 `Qwen/Qwen3.5-4B` SFT 3 个 epoch 后的 **sampler_weights 冻结**。Student 在 Stage 2 与 IDT-OPD 中都是 **fresh `Qwen/Qwen3.5-4B` LoRA**，只有 Stage 3 是从 Medical OPD 的 Train state 恢复，不从 SFT 权重续训。
  - **Stage 划分**：SAR-OPD = Stage 1 Medical SFT → Stage 2 fresh Student 做 Medical OPD（teacher = Medical SFT）→ Stage 3 从 Medical OPD 的 **Train state** 恢复 Student，用原始 Base 当 teacher 做 Base-anchor C-Eval OPD；IDT-OPD = 一个 fresh Student、一个 TrainingClient、一份 Adam state、同一个全局 step，Medical step 与 C-Eval step 按 `1:1` 交替。
  - **调度实现**：`schedule = ("medical",) * medical_steps_per_cycle + ("ceval",) * ceval_steps_per_cycle`，`task = schedule[step % len(schedule)]`；两种任务各维护独立 `task_steps`，避免交替时跳过一半数据。
  - **数据来源与规模**：`FreedomIntelligence/medical-o1-reasoning-SFT` 中文数据（SFT 用 `question + complex_cot + response`，OPD 只用 `question`）；`bigbio/med_qa` 的 MedQA-zh 4 options 作 600 题 held-out 医疗评测；`ceval/ceval-exam` 精选 8 个 non-med subset（`computer_network`、`college_programming`、`advanced_mathematics`、`discrete_mathematics`、`college_physics`、`logic`、`chinese_language_and_literature`、`college_economics`），先按 80%/20% 划 OPD train pool 与 held-out test pool，再从 test pool 固定抽 300 题；排除了 `basic_medicine`、`clinical_medicine`、`physician`、`veterinary_medicine` 以及高中/初中生物。
  - **prompt 是否 mask**：是。与通用 OPD 相同，`prompt_loss_len = len(prompt_ids) - 1`，prompt 区间的 `target_tokens`/`logprobs`/`advantages` 全 0 占位，仅 completion 区间进 loss。Medical OPD 额外加与 SFT 一致的 system message："你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。"，并默认 `enable_thinking=True`。
  - **训练配置**：代码默认 LoRA rank 16（限制 4–64）；Medical OPD 300 steps、batch 4 prompts、group 4、max_tokens 2048、lr `4e-5`、PPO、每 step 刷新 sampler；Base-anchor 阶段 300 steps、同 batch/group/max_tokens、lr 降到 `5e-6`、每 50 step 存 checkpoint；IDT-OPD 600 全局 steps（= M300 + C300）、batch 4、group 4、max_tokens 2048、lr `1e-5`、每全局 step 刷新 sampler。评测口径固定：MedQA-zh 600 题、输出预算 1024 tokens；C-Eval non-med 300 题、输出预算 8192 tokens。
  - **同步/异步差异**：`03-medical-opd-sync.py` 用 `sample()` / `compute_logprobs()` / `forward_backward()` 顺序执行；异步版把三者换成 `_async` 并 `await asyncio.gather` 并发同一 step 内各 prompt 的 rollout，算法、Datum 构造与 mask 完全一致。仓库另有 `04-ceval-opd-async.py`（Base-anchor 阶段）与 `05-interleaved-multi-teacher-opd.py`（IDT）。原文提醒 **Train state 与 sampler_weights 不能混用**：Teacher 与推理用 `sampler_weights`，SAR 第三阶段续训必须用 `training_state`。
- **原文给出的结果数字**：

| 阶段 / Checkpoint | MedQA-zh（600 题） | C-Eval non-med（300 题） |
| --- | ---: | ---: |
| 4B Base | 72.17% `433/600` | 81.67% `245/300` |
| Medical SFT epoch 1 | 77.67% `466/600` | 71.67% `215/300` |
| Medical SFT epoch 2 | 79.17% `475/600` | 69.33% `208/300` |
| Medical SFT epoch 3（Medical Teacher） | 79.00% `474/600` | 69.33% `208/300` |
| Medical OPD step 300（SAR 第三阶段 step 0） | 80.00% `480/600` | 73.00% `219/300` |
| Base-anchor OPD step 50 | 84.17% `505/600` | 76.67% `230/300` |
| Base-anchor OPD step 100 | 82.50% `495/600` | 79.33% `238/300` |
| Base-anchor OPD step 150 | 83.00% `498/600` | 80.33% `241/300` |
| Base-anchor OPD step 200 | 82.17% `493/600` | 83.67% `251/300` |
| Base-anchor OPD step 250 | 83.33% `500/600` | 82.67% `248/300` |
| **Base-anchor OPD step 300（SAR-OPD 最终）** | **84.33% `506/600`** | **84.67% `254/300`** |

IDT-OPD：

| Checkpoint | 任务进度 | MedQA-zh（600 题） | C-Eval non-med（300 题） |
| --- | --- | ---: | ---: |
| 4B Base | - | 72.17% `433/600` | 81.67% `245/300` |
| Medical SFT Teacher | - | 79.00% `474/600` | 69.33% `208/300` |
| **IDT-OPD step 100** | `M50 + C50` | **79.17% `475/600`** | 81.67% `245/300` |
| **IDT-OPD step 200** | `M100 + C100` | 76.50% `459/600` | **86.00% `258/300`** |
| IDT-OPD step 300 | `M150 + C150` | 76.33% `458/600` | 84.33% `253/300` |
| IDT-OPD step 400 | `M200 + C200` | 75.17% `451/600` | 81.33% `244/300` |
| IDT-OPD step 500 | `M250 + C250` | 75.33% `452/600` | 83.67% `251/300` |
| IDT-OPD step 600 | `M300 + C300` | 75.00% `450/600` | 82.00% `246/300` |

原文给出的提升幅度：Medical SFT 使 C-Eval 掉了 `12.34` 个百分点；SAR-OPD step 300 相对 Base，MedQA-zh `+12.17` 个百分点、C-Eval `+3.00` 个百分点；IDT-OPD step 200 相对 Base 两项指标都提高 `4.33` 个百分点。另给出实验花费的量级描述："大概 500 块钱左右"，复现约可在 200 元以内，只做 DeepMath OPD 约 50 元。

- **坑与边界**：
  - 每套方案只有**一条主要训练轨迹，没有多随机种子均值和方差**；`84.33%` 这类结果可能含 checkpoint 波动，不能把小幅差异当作稳定提升。
  - SAR-OPD 与 IDT-OPD 学习率不同（`4e-5`、`5e-6`、`1e-5`），因此两套方案差异**不能全部归因于"顺序训练"或"交替训练"**。
  - 未系统消融 `group_size`、`kl_penalty_coef`、PPO 与 importance sampling、sampler refresh 间隔、Medical/C-Eval 调度比例；`1:1` 只是跑完的一条配置，不代表最优。
  - IDT-OPD **没有"所有指标都持续上涨"的终点**：step 100 偏医疗、step 200 偏通用，继续到 step 600 反而回落；"固定 1:1 调度并不会自动收敛到一个越来越好的平衡点"，最后一步不等于最好结果。
  - SAR 第三阶段医疗准确率反升被原文明确限定："只能把它当作这次训练结果，不能直接推断成普遍规律"。
  - MedQA-zh 与 C-Eval 都是选择题 benchmark，**不能代表真实临床问答安全性**，不能把准确率解释为可用于医疗诊断。
  - 该章不是对 GKD / MiniLLM 全部数据、模型和 benchmark 的严格复刻；reverse KL 下降只表示 Student 更接近当前 Teacher，不代表 benchmark 提高，尤其双 Teacher 时单看 loss/KL 无法判断能力平衡。
  - 保存间隔应与调度周期对齐：`1:1` 周期长度为 2、`2:1` 为 3，checkpoint 间隔应取其倍数。
- **一句话记住它**：先学医，再用锚点补通用。

---

## OPSD：同模型开卷自蒸馏（opsd）
- **论文**：*Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models*，arXiv:2601.18734（2026，Siyan Zhao 等）。一句话定位：不找更大的 teacher，让同一个初始模型"闭卷答题 + 拿参考答案开卷判卷"，在 student 自己的轨迹上做逐 token 自蒸馏。
- **解决什么痛点**：SFT 只让模型看专家写好的 `problem → expert solution`，推理时却要面对自己已经写错的 token 前缀，属于典型 off-policy distribution mismatch；GRPO 一类 RLVR 虽在自身 rollout 上学习，但整条上千 token 的轨迹只拿一个 0/1 结果奖励，信号稀疏，且同题 8 条轨迹全对或全错时 group-relative advantage 整体变成 0。OPSD 同时保留三件事：on-policy（轨迹来自当前 student）、dense feedback（每个 completion token 都有 teacher 信号）、self-distillation（不需要再部署更大 teacher）。
- **核心公式**：论文给两种实现——full-vocabulary logit distillation（逐位置取整个词表的分布算 JSD 或 KL，论文主实验用 `JSD β=0.5`，官方代码另加 point-wise KL clipping 防止少数风格 token 支配信号）与 sampled-token distillation。**该章使用第二种**：

$$
r_t^{\mathrm{KL}}
=
\log p_S(\hat y_t \mid x, \hat y_{\lt t})
-
\log p_T(\hat y_t \mid x, y^{*}, \hat y_{\lt t})
$$

$$
A_t = -\beta\, r_t^{\mathrm{KL}}
= \beta\left(\log p_T - \log p_S\right)
$$

符号：$x$ 是题目 prompt；$y^{*}$ 是数据集里的参考解答，**只有 teacher 能看到**（privileged information）；$\hat y_t$ 是 Student 在位置 $t$ 实际采样出的 token（$S$ 与 $T$ 评的是同一个 token）；$\hat y_{\lt t}$ 是 Student 自己写出的前缀；$p_S$ 是 Student 策略；$p_T$ 是带参考解答条件的 Teacher 策略；$\beta$ 为 `kl_penalty_coef`，复现用 `1.0`。Teacher 比 Student 更认可该 token 时 $\log p_T - \log p_S>0$，该 token 被鼓励，反之被压低。

- **reward 或 teacher 信号设计**：信号是 **privileged teacher 与 student 在同一 token 上的 logprob 差**，token 级、稠密、无需 verifier。Teacher 收到 `teacher_prompt_ids + student_completion_ids`，调用 `compute_logprobs_async()`（**不是** `sample_async()`）做 teacher forcing 前向，再切 `all_logprobs[len(teacher_prompt_ids):]` 取 completion 区间。Teacher 不生成另一条轨迹——整条 completion 只由 Student 生成一次。代码强制校验 `completion tokens`、`student rollout logprobs`、`teacher completion logprobs` **三者严格等长且无 `None`**，"只要有一个 token 对不上，reverse KL 就没有意义"，且不做静默截断。
- **关键实现细节**：
  - **student 与 teacher 的关系**：两者都从 `Qwen/Qwen3.5-4B` 开始。Student 挂可训练 LoRA（rank 64，`train_attn=True`、`train_mlp=True`、`train_unembed=False`）；Teacher 创建时不传 `model_path`，**永远是 step-0 base policy**，没有 optimizer，不随 Student 漂移。原文强调固定 teacher 很重要，否则监督目标不断移动、训练更易不稳定。
  - **"同模型两个角色"靠 prompt 区分**：Student prompt 只含 `Problem: {problem}` 加 `Please reason step by step, and put your final answer within \boxed{}.`；Teacher prompt 额外插入 `=== Reference Solution Begin === {solution} === Reference Solution End ===` 与一段"理解但不要复制、用自己的方式推导同一答案"的过渡语。参考解答不是新答案，而是改变 teacher 对"下一个 token"判断的特权上下文；能力来自更多上下文，而非更多参数。
  - **数据来源与规模**：`siyanzhao/Openthoughts_math_30k_opsd`，固定 revision，共 **29,434** 条 `problem + solution`（代码里 `EXPECTED_DATASET_ROWS = 29_434` 会校验行数）。`solution` 不是 SFT label，不拼进 Student target，只用于改变 Teacher 的条件分布。评测用 `yentinglin/aime_2025` 的 30 道题。
  - **prompt 是否 mask**：是。`prompt_loss_len = len(student_prompt_ids) - 1`，prompt 区间的 `target_tokens`、`old_logprobs`、`advantages` 全填 0，只作上下文；`input_ids = student_prompt_ids + completion_ids[:-1]`、`target_tokens = [0]*prompt_loss_len + completion_ids` 完成自回归右移，保证位置 $t$ 的输入预测位置 $t+1$ 的目标。**真正进入 loss 的只有 Student 自己生成的 completion token。**
  - **训练配置**：100 steps；每 step 32 道题；每题 1 条 completion（`group_size=1`）；max_tokens 1024；`temperature 1.1 / top_p 0.95 / top_k 20`；KL coefficient `1.0`；lr `5e-6`；`loss_fn="importance_sampling"`；`sampler_refresh_steps=1`（每 step 刷新，严格 on-policy）；每 25 step 保存 state + sampler weights；最大远程并发 32；Student/Teacher thinking 均关闭。训练 run 最初把上限写成 200 step，故 SwanLab 名称仍带 `steps200`，本文只分析实际完成并持久化的前 100 step。
  - **并发实现**：单题内部先完成 Student→Teacher 的依赖，再在题目之间 `asyncio.gather` 并发，并用 `asyncio.Semaphore(32)` 统一限制 sampling 与 teacher scoring 的总并发。原文强调这种并发不破坏 on-policy：当前 batch 的所有 rollout 基于同一个 Student checkpoint，全部完成后才做一次 optimizer update。
  - **同步/异步差异**：同步脚本 `01-opsd-sync.py` 顺序调用 `sample()` / `compute_logprobs()`；异步版 `01-opsd-async.py` 全部换 `_async` 且 batch 内 gather。算法、Datum 构造与 mask 完全一致，差别只在并发编排。本地只需 CPU，模型采样与 LoRA 训练由 PyTRIO 远端执行（本次 run 的 SwanLab 元数据记录的是一台 `Apple M4` MacBook Air）。
- **原文给出的结果数字**：固定 30 题 × 每题 12 次采样，评测配置 `temperature=1.0, top_p=1.0, top_k=-1, max_tokens=38,912, thinking=off`：

| 模型 / Checkpoint | Average@12 | Pass@12 | 正确 generations | 至少答对一次的题目 | Format Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B Base | 51.67% | 80.00% | 186 / 360 | 24 / 30 | 100% |
| Step 25 | 51.11% | 73.33% | 184 / 360 | 22 / 30 | 100% |
| Step 50 | 50.28% | 86.67% | 181 / 360 | 26 / 30 | 100% |
| Step 75 | 48.61% | 76.67% | 175 / 360 | 23 / 30 | 100% |
| **Step 100** | **52.78%** | **86.67%** | **190 / 360** | **26 / 30** | **100%** |

Step 100 相对 Base：Average@12 `51.67% → 52.78%`（`+1.11` 个百分点），Pass@12 `80.00% → 86.67%`（`+6.67` 个百分点）；正确 generation 从 `186` 增到 `190`，至少答对一次的题目从 `24` 增到 `26`。训练曲线（前 100 step）：`trainer/loss_mean 0.0577 → 0.0425`，`reverse_kl_mean 0.0644 → 0.0471`，`reverse_kl_std 0.4378 → 0.3687`。耗时：100 step 的 `time/step_elapsed_time` 累计约 `7,540.6` 秒，约 `2 小时 6 分钟`。成本：正式训练 PyTRIO 花销 `41.81` 元；5 次完整评测实际 sample tokens 合计 `18.05M`、花销 `¥83.61`（Base 3.49M/¥16.15、Step 25 3.40M/¥15.79、Step 50 3.53M/¥16.37、Step 75 3.77M/¥17.44、Step 100 3.86M/¥17.86）；可归属合计 `125.42` 元。单次评测理论上限 `30 × 12 × 38,912 = 14,008,320` tokens（`14.01M`），实际因提前 EOS 为 3.40M～3.86M。
- **坑与边界**：
  - 提升**是正的但不大**；中间 checkpoint 不单调，**Step 75 的 Average@12 甚至低于 Base**（48.61% vs 51.67%）。
  - 30 道题、单次训练、没有多随机种子，"还不足以宣称稳定复现了论文收益"。
  - **loss 和 reverse KL 下降只能说明 Student 正在靠近 Teacher，不能直接证明数学能力一定提高**；AIME25 曲线证明训练目标稳定下降时 checkpoint 准确率仍会波动。
  - `opd/advantage_mean` 为负是正常的：此处 advantage 定义为 $-(s\log p - t\log p)$，当 Student 在自己的采样分布下平均比 Teacher 更偏好这些 token 时，sampled reverse KL 为正、平均 advantage 为负。
  - **Teacher 是否真理解参考解答至关重要**："模型太弱、题目太难时，privileged context 并不会自动变成可靠监督。"
  - 训练目标不需要最终答案正确：哪怕 Student 最终答错，只要 Teacher 对中间 token 偏好不同，该轨迹仍能产生信号——这也意味着信号质量完全依赖 teacher 的条件分布质量。
- **一句话记住它**：同模型带着答案自评。

---

## AgentOPSD：Agent 长轨迹的 turn 级信用（agentopsd）
- **论文**：*Fine-Grained Agentic Reinforcement Learning via On-Policy Self-Distillation*，arXiv:2608.05987。一句话定位：保留 GRPO 的终局 0/1 方向，用同一参数快照带 Skill 复评每个 turn，把序列级 advantage 细化为 turn 级强度。环境为 ALFWorld（arXiv:2010.03768，text-only）。
- **解决什么痛点**：普通 GRPO 只看整条轨迹——成功的轨迹整条拿正 advantage，失败的整条拿负 advantage，一条轨迹里所有 action token 继承同一个序列级信号。放进 ALFWorld 就很直观：Agent 可能先开错两个柜门、绕十几步，最后才找到苹果放进冰箱，GRPO 只知道"这局最终成功"，于是前面的绕路和最后的关键动作一起被鼓励，轨迹越长粗粒度 credit assignment 越明显。相对第 4 篇的普通 OPSD，AgentOPSD 把粒度从"推理 token / 参考解答"改成"Agent turn / 任务技能"，并且不训练额外 Critic、不让 Teacher 重写标准轨迹。
- **核心公式**：设某组 8 条轨迹的终局 reward 为 $R_i \in \{0,1\}$，先算标准 GRPO 序列 advantage 与初始 belief：

$$
A^{\mathrm{seq}}_i=\frac{R_i-\bar R}{\sigma_R+\epsilon},
\qquad
B_0=\bar R
$$

对每个 turn $t$，把该 turn 内所有 action token 的 logprob 差求和得到 evidence：

$$
e_t=\sum_{j\in \text{turn }t}\left(\log \pi_T(a_j)-\log \pi_S(a_j)\right)
$$

evidence 按真实时间顺序递推成一条"当前轨迹会成功"的相对 belief（$\gamma=0.95$）：

$$
c_t=\gamma\,c_{t-1}+e_t,
\qquad
B_t=\sigma\!\left(\mathrm{logit}(B_0)+c_t\right),
\qquad
\Delta B_t=B_t-B_{t-1}
$$

再把相邻 turn 的 belief 变化按终局方向定向、组内标准化、裁剪成有界权重：

$$
o_t=\mathrm{sign}\!\left(A^{\mathrm{seq}}\right)\cdot \Delta B_t,
\qquad
z_t=\frac{o_t-\bar o}{\mathrm{std}(o)+\epsilon}
$$

$$
w_t=\mathrm{clip}\!\left(1+\delta\, z_t,\;1-\delta,\;1+\delta\right),
\qquad
A_t=A^{\mathrm{seq}}\cdot\left[(1-\lambda)+\lambda\, w_t\right]
$$

符号：$a_j$ 是 Student 实际生成的 action token；$\pi_S$ 是无 Skill 的 Student，$\pi_T$ 是同一快照但带 Skill 的 Self-Teacher；$e_t$ 是第 $t$ 个 turn 的 evidence；$c_t$ 是衰减累计证据；$B_t$ 是 turn $t$ 结束后的 belief；$\Delta B_t$ 是该 turn 的局部关键性（即图中的 `ΔB`）；$o_t$ 是按终局方向定向后的 outcome credit；$z_t$ 是组内标准化；$\delta$ 为 `weight_bound=0.2`；$\lambda$ 为 `reshape_lambda=0.5`；$A_t$ 是该 turn 所有 action token 最终使用的 advantage。在 $\delta=0.2,\lambda=0.5$ 下 $w_t\in[0.8,1.2]$、乘子 $\in[0.9,1.1]$，即"Self-Teacher 最多只把某个 turn 多强调或少强调 10%，绝不会把成功轨迹改成负优势"。原文明确 $B$ 只是相对支持度、未经过成功概率校准，训练真正使用的是相邻 turn 的变化，绝对数值不参与关键性判断。
- **reward 或 teacher 信号设计**：**两层信号叠加**。方向来自环境 Verifier 的纯 0/1 终局 reward（训练 reward 也是纯 0/1，没有额外无效动作惩罚），经 GRPO 组内标准化得到 $A^{\mathrm{seq}}$；强度来自 Self-Teacher 的 turn 级证据——同一参数快照带 Skill 后对 Student 完整轨迹调一次 `compute_logprobs`，按 action span 把 `teacher - student` 的 token logprob 差相加得到每个 turn 的 evidence，再经上述 belief 递推变成有界乘子。Teacher **不改动作、不重跑环境、不生成更好轨迹**，只对 Student 已生成的 `assistant action token` 复评。
- **关键实现细节**：
  - **Student 与 Self-Teacher 的关系**：两者都由**当前模型**（`Qwen/Qwen3.5-4B`）承担，来自同一个冻结的参数快照。Student prompt 只有任务、历史 observation、当前允许动作和 `alfworld_step` 工具协议，**从头到尾看不到 SkillBank**，保证训练访问的状态分布与最终无 Skill 评测一致；Teacher 版本在初始前缀里额外加入一份所有任务共享的 general skill 与一份任务专属 skill。
  - **Skill 映射**：`skills.py` 从上游固定 commit `0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b` 的 SkillBank 读取 `skill_mapping.json`，按 ALFWorld 的 `task_type` **确定性映射**到固定技能文件（不做模糊语义检索），并把 `pick_and_place_simple` 别名归一到 `pick_and_place`；最终注入文本为 `general_skills + 任务技能`。原文强调这样"更容易审计，也不会把 reward 或未来 observation 泄漏给 Teacher"。**Skill 只在训练期进入 Teacher prompt**，Student rollout、Base 评测和所有 checkpoint 评测都不注入。
  - **一次请求复评整条轨迹**：因为 ALFWorld 后续 turn 始终是前一段真实 token 的严格扩展，只需一次完整 Teacher 请求；teacher 用 `prefix_offset = teacher_initial_length - student_initial_length` 把 Student 的 `action_start/action_end` 平移到 Teacher 序列，再校验 token 数量与位置。若中途重新渲染历史或静默截断上下文，Teacher 就不再是复评 Student 当时见过的同一条轨迹。
  - **prompt / observation 是否 mask**：是，且是 **action-only mask**。`loss.py` 里 `old_logprobs_by_token` 与 `advantages_by_token` 全零初始化，只把每个 turn 的 `step.action_start` 起的 action token 位置填上 `old_logprob` 与该 turn 的 `turn_advantage`；system、user、环境 observation 和聊天模板 token 的 advantage 全为 0，只作上下文。
  - **数据与规模**：text-only ALFWorld，训练每 update 抽 16 个游戏、每游戏 8 条彼此独立的 TextWorld 环境与轨迹（共 128 条轨迹/update），最多 50 turns；80 updates 共 10,240 条 rollout。Student 轨迹上限 14,336 tokens，另为 Teacher Skill 前缀预留 2,048。固定评测 274 个游戏（`valid_seen` 140 + `valid_unseen` 134），`temperature 0.01`、`seed 42`、全程无 Skill；`analysis.py` 会校验三个 JSONL 的 `(split, game_id)` 一致。
  - **训练配置**：LoRA rank 32、lr `4e-6`；turn credit `gamma=0.95`、`lambda=0.5`、`weight_bound=0.2`；PPO clip `[0.8, 1.24]`；一次 rollout 只做一个 PPO epoch 以守住 on-policy 边界；checkpoint 在 Step 40、Step 80；`teacher-concurrency=16`；Student 采样 `temperature 1.0 / top_p 1.0`。
  - **同步/异步差异**：该章本身是异步实现（`asyncio.gather` + `asyncio.Semaphore(max_concurrency)` 并发 teacher scoring）。相对普通 GRPO 的增量不是"同步/异步"，而是同快照的第二次带 Skill 前向。若整组 8 条全成功或全失败，`sequence_advantage` 全为 0，`build_training_datums` 会直接过滤该轨迹、跳过 backward（`degenerate_groups` 计数）。
- **原文给出的结果数字**：

| 模型 / Checkpoint | Valid Seen | Valid Unseen | 整体成功率 | 平均无效动作 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B Base | 60.0% | 57.5% | 58.8% | 7.07 |
| AgentOPSD Step 40 | 61.4% | 62.7% | 62.0% | 6.74 |
| **AgentOPSD Step 80** | **64.3%** | **67.2%** | **65.7%** | **5.68** |

Step 80 相对 Base：整体成功率 `+6.9` 个百分点，成功游戏从 `161 / 274` 增到 `180 / 274`（多完成 19 局）；`valid_unseen` `+9.7` 个百分点，比 seen split 提升更明显；每局平均无效动作 `7.07 → 5.68`，少了 `1.39` 次。成本（80-step 正式训练 PyTRIO 用量）：Prefilling `762.21M` tokens（其中 Cache `549.26M`）、Train `30.52M` tokens、Sample `9.98M` tokens，花销 **¥675.69**。锁定版本 PyTRIO `0.2.9`、ALFWorld `0.4.2`、SwanLab `0.10.4`。
- **坑与边界**：
  - `reward/mean` 与 `reward/success_rate` 在 80 步里**持续波动，没有单调上升曲线**：每个 update 抽不同游戏、难度会变，且同 batch 里可能出现全成功/全失败组（无 group-relative gradient）。这张图只适合检查 rollout 是否健康，**不能回答模型是否变强**。
  - 诊断指标 `evidence_abs_mean`、`delta_b_abs_mean` 持续非零只证明 turn credit 链路在工作（`weight_min` 多次触到 0.8 下界说明确有 turn 被降权），但"`evidence` 大、`ΔB` 大，不代表游戏一定做得更好"。
  - **最大未解问题**：这些提升中 turn-level credit 与同预算 PPO/GRPO 本身各贡献多少仍未知；需要在相同游戏顺序、seed、rollout 数和 token budget 下补一条 `reshape_lambda=0` 的 GRPO 对照，并对 Base / GRPO / AgentOPSD 做多 seed 固定评测。
  - 与论文有明确差异：论文使用的模型与全参数训练栈不同，本文用 Qwen3.5-4B LoRA + PyTRIO 内置 PPO，full-vocabulary entropy、部分 paper-close reduction 未原样复刻——定位是**算法级复现，不追求复刻论文榜单数字**。
  - Smoke 阶段单个游戏 4 条轨迹可能全成功或全失败，此时 sequence advantage 全为 0，脚本会保留 rollout 指标但跳过 backward。
  - 成本结构：`762.21M` prefilling 是账单主体，Agentic RL 的钱主要花在模型为走到这些 action 而反复读取的历史上，最终进 loss 的 action token 只占较小部分。
- **一句话记住它**：终局定方向，turn 定强度。
