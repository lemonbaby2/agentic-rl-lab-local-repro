# A · GRPO 系列四章精读笔记

> 覆盖 `experiments/grpo`、`experiments/vision-grpo`、`experiments/search-r1`、`experiments/retool` 四章。所有数字均取自各章 readme.md / start.md / py 源码，未作外推。

## GRPO（GSM8K 最小闭环）

- **论文**：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models，arXiv 2402.03300。定位：GRPO 首次被系统提出的论文，本章用 GSM8K 复现其"去 critic"的组内相对策略优化。

- **解决什么痛点**：PPO 是 actor-critic，除 policy 外还要训一个 value model 估 baseline，对 LLM 而言这个 value model 往往也很大，显存与计算都要翻倍。而数学题常常只有最终答案能打分，让 value model 逐 token 估准并不自然。GRPO 换掉了 baseline 的来源：同题采样一组回答，用组内平均 reward 当 baseline，不再训练 critic，只保留 policy。

- **核心公式**：

组内 advantage（demo 实际使用的形式）：

$$
A_i = r_i - \mathrm{mean}(r_1, r_2, \ldots, r_G)
$$

符号：$r_i$ 为第 $i$ 条 completion 的 reward（本章为 0/1）；$G$ 为同一 prompt 的采样条数（`group_size`）；$\mathrm{mean}(\cdot)$ 为本组平均 reward。该式即"比组内平均好则正的 advantage，差则负的 advantage"。

论文版还会除以组内标准差做标准化（readme 明确说 demo 未采用）：

$$
A_i = \frac{r_i - \mathrm{mean}(r_1, r_2, \ldots, r_G)}{\mathrm{std}(r_1, r_2, \ldots, r_G)}
$$

三个可切换 loss（ratio 定义来自 `03-grpo.py` 文件头注释与 readme）：

$$
\rho = \exp\big(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{old}}(a_t \mid s_t)\big)
$$

其中 $\pi_\theta$ 为当前策略、$\pi_{\text{old}}$ 为本 step 采样时保存的 sampler（old policy），$\rho$ 是重要性采样修正比。

- `importance_sampling`：$\mathrm{loss} = -\rho \cdot A$，不做截断，新旧策略偏差大时容易不稳定。
- `ppo`：$\mathrm{loss} = -\min\big(\rho A,\ \mathrm{clip}(\rho, 1-\epsilon, 1+\epsilon)\,A\big)$，$\epsilon$ 由 PyTRIO 内置 ppo loss 管理。
- `cispo`（readme 与 `01-demo-sync.py` 中的 custom loss；`03-grpo.py` 的 `LOSS_FNS` 只有前两个）：

$$
\text{objective} = \mathrm{detach}\big(\mathrm{clamp}(\rho,\ c_{\text{low}},\ c_{\text{high}})\big)\cdot \log \pi_\theta(a_t\mid s_t)\cdot A,\qquad \mathrm{loss} = -\sum_t \text{objective}
$$

即 ratio 被 clip 后 `.detach()` 成固定权重，真正被求导的是 `target_logprobs`。

- **reward 设计**：纯 rule-based 0/1，无 reward model、无格式奖励、无长度惩罚。流程是取回答中**最后一个** `\boxed{...}`，与 GSM8K `####` 后的标准答案比较；仅做轻量归一化（去逗号、去首尾空白、去结尾句点），一致给 `1.0`，否则 `0.0`；解析不到 boxed 也是 `0.0`。

- **关键实现细节**：
  - **observation mask**：`observation_len = len(prompt_tokens) - 1`，`model_input = prompt + completion[:-1]`，`target_tokens = [0]*observation_len + completion`，`logprobs` 与 `advantages` 前段同样填 `0` / `0.0`，只在 completion token 上非零。prompt 只作上下文，不参与策略梯度。
  - **group 分叉**：每个 step 先用 `training_client.save_weights_and_get_sampling_client()` 从当前 LoRA 权重导出 sampler，再对同一 prompt 一次 `num_samples=group_size` 采样，旧 logprob 必须来自这个 sampler，不能在更新后重算。
  - **退化组**：组内 reward 全相同（全 0 或全 1）→ 全部 advantage 为 0 → 该组 `continue` 跳过，并计入 `frac_degenerate`。
  - **loss 入口**：`BUILTIN_LOSS_FNS = {"importance_sampling", "ppo"}` 走 `forward_backward`；`cispo` 走 `forward_backward_custom` + 本地 `loss_fn`（远端只负责 forward 出逐 token logprob）。
  - **LoRA / 优化器**：`03-grpo.py` 默认 `Qwen/Qwen3.5-4B`、LoRA rank 32、Adam lr `4e-5`、$\beta=(0.9, 0.95)$。
  - **数据**：`openai/gsm8k` 的 train split，首次运行自动下载到 HF 缓存；prompt 带一条 few-shot（"strawberry 里有几个 r"）。
  - **训练步数**：脚本 `--steps` 默认 10；readme 演示 `--steps 10 --batch-size 4 --group-size 8 --max-tokens 512`，对比三种 loss 时用 `--steps 30`；另有 `--all-data` 走完整个 train split 一次并覆盖 `--steps`。
  - **采样参数**：temperature 1.0、top_p 1.0、stop 为 eos 与 `<|im_end|>`；`enable_thinking=False`。

- **原文给出的结果数字**：
  - 调试这套代码"一共花了 55 块钱"，"如果只跑 10 步，成本可以低到 5 块钱"。
  - 小规模跑通配置：`--steps 1 --batch-size 2 --group-size 2 --max-tokens 64`；低成本版本 `--steps 10 --batch-size 4 --group-size 8 --max-tokens 512`。
  - advantage 示例：reward `[1, 0, 1, 0]` → 均值 `0.5` → advantage `[+0.5, -0.5, +0.5, -0.5]`。
  - 该章未给准确率、提升幅度或训练耗时等定量结果（readme 未报告 reward 曲线数值）。

- **坑与边界**：
  - 三处显式简化：① reward 只用 rule-based 0/1，未用 reward model / process supervision；② advantage 未除以组内 std；③ 没有显式 reference KL 约束。
  - `importance_sampling` 没有 clip，"策略可能更新得比较猛"，小实验方便、正式训练要小心。
  - 组全对或全错都拿不到梯度：模型太弱（全错）和太强（全对）都不利于 GRPO，所以要盯 `frac_degenerate`。
  - 只看 `reward` 会误判：reward 不动可能是没起效，也可能是退化组太多，或模型转向更长推理。`clip_fraction` 偏高则提示 lr 过大或采样策略偏离当前策略。
  - `train_tokens` 统计的是 advantage 非零的 token 数，不是总 token 数。

- **一句话记住它**：组内均值当 baseline

## Vision GRPO（GeoQA 多模态）

- **论文**：readme 只标注 GRPO 来源为 DeepSeekMath（arXiv 2402.03300），本章本身不是独立论文，而是把 GRPO 推到多模态几何选择题的复现；独立 arXiv 号未标注。

- **解决什么痛点**：前八篇的 prompt 都以文本为主，RL 链路没有覆盖图像输入。本章把"看图 → 读题与四选项 → 同题并发采样一组回答 → 可验证选项奖励 → 在线更新"整条链路跑通，等于给 GRPO 补上视觉 observation 的入口。算法骨架不变，变的只是 prompt 由 `EncodedTextChunk + ImageChunk + EncodedTextChunk` 三段拼成。

- **核心公式**：

单条 completion 的 rule reward：

$$
r_i =
\begin{cases}
1, & \text{最后一个 boxed 选项与 label 一致} \\
0, & \text{其他情况}
\end{cases}
$$

组内相对 advantage：

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j
$$

符号：$G$ 为同题采样条数（正式配置为 8）；$r_j$ 是同组第 $j$ 条的 0/1 reward；$A_i$ 直接广播到该条 completion 的每一个 token 上。readme 在本章只展开这两条公式，loss 侧写的是"group-relative advantage + `importance_sampling`"（`train.py` 中 `loss_fn="importance_sampling"`），未再重复 ratio 表达式。

readme 给的算例：8 条里 6 对 2 错 → 均值 0.75 → 对的 $A=1-0.75=0.25$，错的 $A=0-0.75=-0.75$。

- **reward 设计**：正则 `\\boxed\{\s*([A-D])\s*\}` 取**最后一个** boxed 选项，与 `CHOICE_LETTERS[int(label)]` 严格相等给 1，否则 0。可以容忍推理正文里出现临时候选项。无格式奖励、无长度惩罚，但 `format_rate`（可解析出 boxed 的比例）被单独监控——所以 0 分同时混合了"答错"和"格式不合规"两种失败。

- **关键实现细节**：
  - **observation mask**：`observation_length = group.prompt_length - 1`，`model_input` = 图文三段 chunks + `EncodedTextChunk(sample.tokens[:-1])`；`target_tokens` / `logprobs` / `advantages` 前 `observation_length` 位填 `0` / `0.0` / `0.0`，只有 completion token 带 advantage。图片与 prompt 文本一律不作为训练目标。
  - **视觉 token 对齐**：图片先合成到白底转 RGB（避免透明区变黑），再用 image processor 的 `get_number_of_image_patches` 算 patch 数，`expected_tokens = patches // merge_size ** 2` 写入 `ImageChunk`。采样后强制校验 `response.input_tokens == len(prompt)`，不一致立即报错——因为 target、old logprob、advantage 共用同一套位置坐标。
  - **group 分叉**：同一 step 内不同题目用**同一版 sampler** 并发 `tqdm_asyncio.gather` 提交；每题一个 `sample_async(num_samples=group_size)`。
  - **退化组**：`len(set(rewards)) == 1` 即跳过该组，计入 `degenerate_fraction`；只把 `sample.tokens` 非空的样本转成 Datum。
  - **loss**：PyTRIO 内置 `importance_sampling`，`forward_backward_async` 之后还要 `await` 返回的 future，再 `optim_step_async`。
  - **LoRA / 优化器**：`Qwen/Qwen3.5-4B`、rank 32、lr `4e-5`、temperature 1.0、top_p 1.0、max_tokens 1024、`enable_thinking=False`。
  - **数据来源与规模**：`hz2475/geoQA`，源数据 5,010 条 —— train 3,503（全部训练）、test 759（`seed=42` 打乱后取最后 100 条）、dev 748（不使用）。
  - **训练步数**：正式 100 steps，`batch-size 8`、`group-size 8`，每 25 step 存 sampler weights 与 training state；另有 20-step 小规模试跑。

- **原文给出的结果数字**：
  - 固定 100 条 GeoQA test 上：Base `71.0%` Accuracy / `75.0%` Format；GRPO step 100 `87.0%` / `91.0%`；提升 `+16.0 pp` / `+16.0 pp`。
  - 评测开销：Base `2.20 sample/s`、总耗时 `48.65s`；step 100 `2.35 sample/s`、总耗时 `49.45s`（temperature 0.0、max tokens 1024、并发 `sample_async()`）。
  - 条件准确率：Base `71/75 = 94.7%`；step 100 `87/91 = 95.6%`；即已成功输出 boxed 的样本只提高 `0.9` 个百分点，+16pp 主要来自格式更稳。
  - 仍有 `9` 条无法解析，全部打满 `1,024` token 被截断；其余 91 条格式正确回答平均长度 `278.8` tokens。
  - 20-step 小规模会话：`33.79K` prefill tokens、`0.40M` train tokens、`0.62M` sample tokens，页面显示花销 `¥5.06`。

- **坑与边界**：
  - 结果是"从原始 test 中固定抽的 100 条"，不能当完整 759 条 test 的 benchmark；仓库未提交逐样本 JSON 和远端 checkpoint。
  - 单次 Base-vs-checkpoint 对比，未覆盖完整 test、多 seed、其他视觉模型。
  - `expected_tokens` 必须与远端视觉 token 计数一致，否则直接报错。
  - 无法解析 boxed 得 0 分，所以指标同时含答题能力和格式遵循能力；训练后要把 reward、format rate、completion length 一起看，防止模型把"写更长"当捷径。
  - 单题 4 条回答可能全部同分，此时该 step 正常完成 rollout 但 `train_datums` 为 0、没有 backward。
  - `sample_async()` 一次 `await` 即返回结果；`forward_backward_async()` / `optim_step_async()` 返回的远端 future 还要继续 `await`。
  - 在线 batch 的 reward 曲线受题目难度影响，结论只能来自固定测试集；100-step 成本不能由 ¥5.06 直接倍乘。

- **一句话记住它**：给 GRPO 装上眼睛

## Search-R1（多轮搜索）

- **论文**：Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning，arXiv 2503.09516。定位：把"搜索"从回答前的固定步骤，变成推理轨迹中模型可反复选择、并靠 RL 学进参数的动作。

- **解决什么痛点**：传统 RAG 是"问题 → 检索一次 → 塞进 prompt → 回答"，遇到多跳问题（如"《小王子》作者出生在哪个国家"）就不够用。Search-R1 相对单轮 GRPO 的改动有两层：轨迹从单条 completion 变成"assistant 生成 → 工具调用 → 等环境 → 继续生成"的多轮序列；训练只对模型自己生成的 assistant token 做策略梯度，工具返回的 observation 只作上下文。搜索引擎和 embedding 模型都是固定环境，被训练的只有 LLM 的 LoRA 权重。

- **核心公式**：

组内中心化 advantage（本章明写 $G=8$）：

$$
A_i = r_i - \frac{1}{8}\sum_{j=1}^{8} r_j
$$

符号：$r_i$ 是该轨迹的最终 reward（三档：$1.0$ / $0.0$ / $-0.1$）；求和是本问 8 条轨迹的平均。readme 算例：$[1.0, 0.0, 0.0, -0.1, 1.0, 0.0, 0.0, -0.1]$ 均值 $0.225$，则 $1.0 \to +0.775$、$0.0 \to -0.225$、$-0.1 \to -0.325$。

逐 token 的训练信号（`train.py::build_datum`）：

$$
a_t =
\begin{cases}
0, & t \in \text{system/user/tool observation} \\
A_i, & t \in \text{assistant tool call / final answer}
\end{cases}
$$

序列对齐为一次统一的右移：`input_tokens = full_tokens[:-1]`、`target_tokens = full_tokens[1:]`、`old_logprobs = old_logprobs_by_token[1:]`、`advantages = advantages_by_token[1:]`，四者长度严格一致。

loss 为 PyTRIO 内置 `importance_sampling`（ratio 即当前策略与 rollout 时 Student sampler 的 logprob 之比，计算方式同 GRPO 章）。

- **reward 设计**：只看最终答案，三档 —— 格式合法且 EM 正确 `1.0`；格式合法但答案错 `0.0`；格式非法或没有最终答案 `-0.1`。判定口径：必须**恰好一行**非空 `Answer: <short answer>`（`len(matches) != 1` 直接判非法）；比对时答案小写、按 Unicode 标点类去标点、去英文冠词 a/an/the、合并多余空格后与任意参考答案完全相等才算 EM，不给部分分。没有搜索次数奖励、没有关键词命中奖励、没有长度惩罚 —— readme 明确说中间 reward 容易诱发 reward hacking。

- **关键实现细节**：
  - **observation mask**：不额外发 `mask` 字段，直接给 observation token 填 `advantage = 0` 与 `old_logprob = 0`。构造时 `full_tokens += delta_observation + turn.completion_tokens`，`old_logprobs_by_token += [0.0]*len(delta_observation) + turn.logprobs`，`advantages_by_token += [0.0]*len(delta_observation) + [trajectory.advantage]*len(turn.completion_tokens)`。observation 保留在上下文里影响后续决策，但不进 policy loss。
  - **group 分叉（本章最关键的工程点）**：首轮 8 条轨迹 prompt 完全相同，直接 `sample_async(prompt=shared_prompt, num_samples=8)` 一次分叉；首次搜索后 query/observation 各不相同，下轮 prompt 已分叉，必须改成"每条未结束轨迹一个独立 prompt × `num_samples=1`"，用 `asyncio.gather` 并发。轨迹内部严格串行（assistant → search → tool observation → 下一轮），轨迹之间才并发。
  - **token-in token-out**：下一轮 prompt 由真实采样 token 连续拼接得到（`build_next_prompt`），不做 text↔token 重编码；`build_datum` 会校验 `turn.prompt_tokens[:len(full_tokens)] == full_tokens`，不是前缀扩展就报错。
  - **loss / 梯度**：每个有效 step 一次 `forward_backward` + 一次 `optim_step`；PyTRIO 0.2.9 自动梯度累加，SDK 按服务端限制拆物理请求、按完整 logical batch 归一化并返回聚合指标。advantage 为 0 的轨迹仍保留在 batch 里，让归一化分母覆盖完整 rollout；整批都没信号则跳过更新。
  - **LoRA / 数据 / 规模**：`Qwen/Qwen3.5-4B`、LoRA rank 32；训练数据 `PeterJinGo/nq_hotpotqa_train`，NQ 79,168 + HotpotQA 90,447 = 169,615 题；固定 dev 70 题（7 benchmark × 10 题）。
  - **轨迹预算**：每题 8 条、每 step 8 题（=64 条轨迹），最大搜索 4 次、最大 assistant turns 6、单条轨迹 8,192 tokens、单轮 1,024、单次工具回包 1,024。
  - **搜索后端**：DeepSeek 官方 Search API（`POST https://api.deepseek.com/v1/tools/search`，Top 3，`need_content=false`，优先 `snippet`、缺失回退 `summary`，并发 16、超时 60s）与 Wikipedia Action API（并发 3、超时 15s、约 200 RPM、免费免 key），共用 `search(query) → observations`。工具结果按完整条目装预算，首条过长只缩短正文、保留标题与来源；query 超 128 字符或空会返回可观察的工具错误。

- **原文给出的结果数字**：
  - DeepSeek 官方 Search API（70 题）：Base Macro EM `40.00%`、Format `82.86%`；RL Step 30 `42.86%` / `91.43%`；变化 `+2.86 pp` / `+8.57 pp`。答对从 28 题到 30 题（净增 2 题），格式合规从 58 到 64 题；两次搜索成功率均 `100%`，无超时/429；平均搜索次数 `1.76 → 2.77`。
  - 分 benchmark（Base → Step 30）：2Wiki `50% → 40%`、Bamboogle `70% → 60%`、HotpotQA `70% → 70%`、MuSiQue `10% → 10%`、NQ `20% → 30%`、PopQA `10% → 30%`、TriviaQA `50% → 60%`。
  - 旧 `deepseek-search` Evidence 20-step：Macro EM `44.29% → 50.00%`（`+5.71 pp`，70 题净增 4 题），Format `88.57% → 97.14%`（`+8.57 pp`）；`success_rate` 100%，`error_rate` / `429_rate` / `timeout_rate` 全 0。
  - 旧实验分 benchmark（Base → Step 20）：2Wiki `50% → 60%`、Bamboogle `70% → 70%`、HotpotQA `60% → 70%`、MuSiQue `20% → 30%`、NQ `40% → 40%`、PopQA `20% → 30%`、TriviaQA `50% → 50%`。
  - 成本与资源：原版 Wikipedia corpus 解压后约 `160 GB`、检索服务需约 `180 GB` 内存；DeepSeek Search 每千次有效请求 `5 元`、账号级限速 100 QPS；早期账单评测 `0.44 元` + 训练 `13.30 元`；旧 20-step 当日消费约 `14.06 元`。

- **坑与边界**：
  - 替换了模型、搜索后端、训练框架和部分训练配置，绝对指标不能与论文直接比较，未逐项还原原论文基础设施与 benchmark 分数。
  - 评测只有固定 70 题（每 benchmark 10 题），一条训练 run、单 seed，只能看行为变化和实现问题，不足以给稳定泛化结论。
  - 旧 Evidence 后端与官方 Search API 属于不同搜索环境，Base/checkpoint 必须在同一环境下比较，结果不能混比；Wikipedia 与 DeepSeek 证据范围也不同。
  - 环境质量本身就是实验变量：整组轨迹若因 API 错误一起失败，会白跑一轮昂贵 rollout 且拿不到有效梯度；所以 `reward/correct` 下降时要先看 `search/error_rate` 再解释策略。
  - 一整组 reward 完全相同 → advantage 全 0 → 无策略梯度；整批全 0 则跳过参数更新。
  - 后半段 prompt 已分叉，不能再用 `num_samples=group_size`，否则会错配轨迹。

- **一句话记住它**：让模型自己决定何时搜

## ReTool（代码交织的数学推理）

- **论文**：ReTool: Reinforcement Learning for Strategic Tool Use in LLMs，arXiv 2504.11536。定位：让模型在 long CoT 里边推理边写 Python，用什么工具、调几次完全由结果奖励涌现。

- **解决什么痛点**：long CoT 提升了数学能力，但纯文本推理只能"心算"，开根号、解高次方程、大规模枚举时中间步骤容易算错且不自知。ReTool 的两个关键设定是 code-interlaced rollout（生成到代码就暂停，沙箱执行后把 stdout/stderr 拼回上下文继续生成，解释器返回的 token 不进 loss）与 outcome-only reward（不奖励"用了代码"这个行为）。相对 GRPO，本章多了工具协议、本地沙箱与 ±1 结果奖励，loss 也换成带 clip-higher 的 `ppo`。

- **核心公式**：

结果奖励（对齐官方 `compute_score(strict_box_verify=True)`）：

$$
r =
\begin{cases}
+1, & \text{最后 300 字符中最后一个 } \backslash\text{boxed}\{\} \text{ 与参考答案数学等价} \\
-1, & \text{答案错误，或找不到合法 } \backslash\text{boxed}\{\}
\end{cases}
$$

组内中心化 advantage：

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j,\qquad G = 8
$$

符号：$r_i \in \{+1, -1\}$ 为该轨迹最终 reward；$A_i$ 广播到该轨迹所有 assistant 生成 token，observation token 的 $A$ 恒为 0。

策略损失为 PyTRIO 内置 `ppo`，clip 阈值覆盖为 $\rho \in [0.8,\ 1.28]$，对应官方 recipe 的 $\epsilon_{\text{low}}=0.2$、$\epsilon_{\text{high}}=0.28$（DAPO 的 clip-higher）：

$$
\mathrm{loss} = -\min\big(\rho A,\ \mathrm{clip}(\rho,\ 0.8,\ 1.28)\,A\big),\qquad \rho = \exp(\log\pi_\theta - \log\pi_{\text{old}})
$$

- **reward 设计**：只取回答**最后 300 个字符**，按花括号配平提取**最后一个** `\boxed{}`，用 `math_verify` 的 `parse`/`verify` 做数学等价判定（不是字符串比较）；对 `+1`、错 `-1`，格式非法也按错处理。不奖励工具调用行为，不惩罚搜索/代码次数，不加轮数 shaping —— readme 明确说官方 released 代码里 `score = min(0, score + (num_turns - 2) / 2 * 0.1)` 与论文"outcome-only"不一致，本章跟随论文不加。

- **关键实现细节**：
  - **observation mask（interpreter feedback mask）**：`build_datum` 逐 turn 累加 `delta_observation` 与 `completion_tokens`，`old_logprobs_by_token` 在 observation 段填 `0.0`、`advantages_by_token` 在 observation 段填 `0.0`，只给 assistant token 填 `trajectory.advantage`。与 Search-R1 的差别是 observation 的 **target token 保留真实值**、只把训练信号置零；最后整条序列统一右移一次（`input=full[:-1]`、`target=full[1:]`）。
  - **token-in token-out**：续写用真实采样 token 拼接，不做 text↔token 重编码。readme 引官方实测：重编码不可逆会导致约 100 步后性能崩塌、`grad_norm` NaN。
  - **chat template 两个坑的解法**：Qwen3.5 模板会 strip assistant 内容，且文本含 `</think>` 时会把消息重构成 reasoning/content 两段，所以 `build_next_prompt` 的 canonical 计算一律用占位内容，增量片段只与 tool 消息有关。
  - **group 分叉**：首轮每题 `num_samples=group_size` 一次分叉 8 条，之后每条轨迹单独采样；同 round 内多条轨迹的代码调用用 `asyncio.gather` 并发（代码最长 30s，不能像 HTTP 搜索那样串行）。轨迹内部串行。
  - **沙箱**：`python -B -c <bootstrap>`，list 形式 argv 无 shell、不落脚本文件、每次全新进程；30s wall-clock 超时后用 `os.killpg` 杀整个进程组（`start_new_session=True`），另有 `RLIMIT_CPU` 兜底；stdout/stderr 写匿名临时文件、只读尾部并截断到 4096 字符；子进程把 OMP/OpenBLAS/MKL/VECLIB 线程数压到 1；`BoundedSemaphore(8)` 限并发（不用 `asyncio.Semaphore`，因为每个 step 新建事件循环会跨 loop 报错）。执行前给最后一行非空行补 `print()`（官方 auto-print）；tool 文本里的 `<|im_end|>`、`</tool_response>` 等标记替换成无害写法，防止污染 observation 结构。
  - **loss / 梯度**：`loss_fn="ppo"`、`loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}`；micro-batch 按 padding 后矩形面积做 first-fit decreasing 动态装箱（单批 ≤ 32 个 Datum），提交前把 advantage 乘以 `n_k / N`，保证拆批后累加等价于全局样本均值；每 step 只 `optim_step` 一次。
  - **LoRA / 数据 / 规模**：`Qwen/Qwen3.5-4B`、LoRA rank 32、Adam lr `4e-5`、$\beta=(0.9, 0.95)$；数据 `BytedTsinghua-SIA/DAPO-Math-17k` → `train.jsonl`（另 50 条 dev），DAPO 自带 `"Answer:"` 模板已剥掉以免与 `\boxed{}` 打架。
  - **轨迹预算**：每 step 8 题 × group 8 = 64 条；`max_code_calls=4`、`max_assistant_turns=6`、轨迹 ≤ 8,192、单回合 ≤ 1,024、单次工具回包 ≤ 512；temperature 1.0 / top_p 1.0。
  - **训练步数**：验证跑 20 steps（`--save-every 5`），正式 200 steps（`--save-every 50`），同一份配置只改 `--max-steps`。

- **原文给出的结果数字**：
  - 成本：20-step 训练会话 `¥22.89`（4.34M prefilling / 1.66M train / 1.77M sample），AIME25 评测 `¥4.16 + ¥4.61`，合计 `¥31.66`；200-step 训练会话 `¥224.44`（48.67M prefilling / 15.97M train / 15.84M sample）；训练合计 `22.89 + 224.44 = 247.33`，截图中 4 次评测合计 `18.49`。
  - 20 step 行为变化：`correct` 前 10 步均值 → 后 10 步均值 `0.289 → 0.389`；`code_calls` `1.24 → 1.72`；degenerate group `0`。
  - AIME25（30 题 × 每题 12 次采样，temperature 1.0、top_p 0.7、轨迹 8,192 tokens）：Base `Average@12 23.61%` / `Pass@12 50.00%` / `Format 25.83%` / 平均 code 调用 1.69 / 平均轮数 2.69；Step 20 `32.22%` / `63.33%` / `36.67%` / 1.37 / 2.37；Step 50 `41.11%` / `63.33%` / `52.78%` / 1.70 / 2.70；Step 100 `44.72%` / `66.67%` / `57.78%` / 1.66 / 2.66；Step 150 `46.11%` / `73.33%` / `70.83%` / 2.43 / 3.43；Step 200 `47.50%` / `63.33%` / `76.11%` / 1.98 / 2.98。
  - Step 200 vs Base：Average@12 `+23.89 pp`、Pass@12 `+13.33 pp`、Format `+50.28 pp`；正确 generation 从 `85` 条到 `171` 条，至少答对一次的题目从 `15` 道到 `19` 道（Step 150 时最高 `22` 道）。
  - SwanLab 200-step：`reward/mean -0.5 → +0.3`、`reward/correct 0.2 → 0.6`、`reward/format 0.25 → 0.85`；`valid_tool_call_rate 0.6 → 0.95`、`code_calls 1.0 → 2.0`（step 90 附近峰值约 `2.6`）、`turns 2.0 → 2.8`、`trajectory_tokens` 稳定在 `~2,200`；`sandbox/success_rate 0.68 → 0.80`、`error_rate 0.32 → 0.18`、`timeout_rate` 稳定 `2～3%`、`latency 0.36s → 1.4s`。
  - smoke test：2 题 × 8 条轨迹，工具调用率 `87.5%`（14/16）、合法调用率 `85.4%`。
  - 与官方 recipe 对比：官方 `train_batch_size=512`、`n_resp_per_prompt=16`、max seq 16384、`max_turns=8`、actor_lr `1e-6`、Qwen2.5-32B 全量训练。

- **坑与边界**：
  - 轨迹预算 8k vs 论文 16k 是最大妥协，AIME 长推理被截断，Format 停在 76% 而非更高；预算翻倍是最值得先试的改进。
  - 本地沙箱 ≠ 云沙箱：`python -c` + 30s + `RLIMIT_CPU` 能兜住死循环与超时，但文件读写只靠 system prompt 软约束，macOS 上内存限制不可靠，"不要无人值守跑"。
  - 跳过 cold-start SFT 只对 Qwen3.5-4B 成立（base 工具调用率足够高），换基座不一定成立。
  - 未实现 dual-clip：官方 `clip_ratio_c=10.0` 在 PyTRIO 内置 `ppo` 里不支持，本章忽略。
  - 评测只有 30 题 × 12 次采样，Pass@12 级别天然有 ±1～2 题波动；单次训练、单种子。Pass@12 在 Step 150 → 200 回落（`73.33% → 63.33%`）在该题量下属正常波动。
  - Format 与 correct 的差距对应被 8k 预算截断的长推理；训练中要同时盯 reward、format 与轨迹长度。

- **一句话记住它**：边推理边写代码
