# Agentic RL 全量学习手册

> 由 agentic-rl-lab-local-repro 全量文档合并，供 GPT / 知识库一次性导入。
> 源仓库：https://github.com/lemonbaby2/agentic-rl-lab-local-repro

## 目录

0. 仓库总览（README）
1. 知识点总纲
2. SOTA 汇总
3. 本地 GPU 复现实验记录
4. 原仓库入口可运行性核查
5. 逐章精读笔记（A/B/C/D 四部分）

---


# 0. 仓库总览（README）

# agentic-rl-lab-local-repro

> 在**单张消费级 GPU** 上把 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) 的 Agentic RL 算法真正跑起来,
> 并附带一份把 17 个实验讲透的**知识点总结**与**SOTA 梳理**。

原仓库很优秀,但它有一个前提:**训练不在你机器上跑**。
`pyproject.toml` 把 torch 锁到 `pytorch-cpu` 索引,注释写着"训练和采样由远程 PyTRIO 服务执行"。
于是"用本机 GPU 复现"这件事,必须先补上被交出去的那一半。

本仓库做的就是这件事:**把 PyTRIO 云端的 forward / backward / optimizer / sampler 换成本地 HF + peft 训练器,算法公式一字不改。**

---

## 1. 一张表说清:本仓库 vs 原仓库

| 环节 | 原仓库 | 本仓库 |
|---|---|---|
| base 模型 + LoRA | 云端 PyTRIO | `transformers` + `peft`,本地加载 |
| 采样 (sampling client) | 云端 sampler + `trio://` 权重 | `model.generate(num_return_sequences=G)` |
| 逐 token logprob | 云端 `compute_logprobs` | 一次 teacher-forcing 前向 + `cross_entropy` |
| 反向与更新 | 云端 `forward_backward` | `loss.backward()` + `AdamW.step()` |
| loss 计算 | 内置 loss 在云端 / custom loss 在本地 | **全部本地纯 PyTorch 实现** |
| 实验记录 | SwanLab 云端 | 本地 JSON + matplotlib |
| 需要的东西 | PyTRIO 账号 + 额度 + SwanLab + 各章 API key | **一张 GPU + 网络** |

代价也要说清楚(见 §6 局限):模型从 4B 降到 1.5B、步数从 100+ 降到 10、评测集从 AIME25 降到 GSM8K 子集。
换来的是**可以反复跑、可以拆开看梯度、不用花一分钱额度**。

---

## 2. 本机实测环境

```
GPU    : NVIDIA GeForce RTX 5060 Ti  17.1 GB (sm_120)   ← 训练主卡
         NVIDIA GeForce RTX 4060      8.6 GB (sm_89)
驱动    : 581.57 / CUDA 13.0
Python : 3.12
torch  : 2.11.0+cu128
transformers 5.23.0 / peft 0.21.3 / datasets 5.2.0
模型   : Qwen/Qwen2.5-1.5B-Instruct (LoRA r=16, 可训练 18.46M / 1562M = 1.18%)
```

> 用 `cu128` 是因为 RTX 5060 Ti 是 Blackwell(sm_120),`cu118` 的轮子不支持这个架构。

---

## 3. 复现了什么

### 3.1 五种算法,一套骨架

`local_repro/losses.py` 把原仓库分散在三处的 loss 统一重写为纯 PyTorch 函数:

| 算法 | 公式 | 关键开关 |
|---|---|---|
| **Importance Sampling** | $-\mathbb{E}[\rho A]$ | 无裁剪 |
| **PPO** | $-\mathbb{E}[\min(\rho A,\ \mathrm{clip}(\rho,1-\epsilon_{lo},1+\epsilon_{hi})A)]$ | `clip_low/clip_high` |
| **CISPO** | $-\mathbb{E}[\mathbf{sg}(\mathrm{clip}(\rho))\cdot\log\pi_\theta\cdot A]$ | `detach` 掉 ratio |
| **DAPO** | PPO + Clip-Higher(1.28) + token mean + 动态采样 + 软超长惩罚 | `--dynamic-sampling --enable-overlong` |
| **GSPO** | 序列级 $s_i=\exp(\frac{1}{\|y_i\|}\sum_t\log\frac{\pi_\theta}{\pi_{old}})$,clip 阈值 3e-4 | `--algorithm gspo` |
| **OPD** | $A_t=\mathrm{sg}(\log\pi_T-\log\pi_\theta)$(reverse KL) | `opd_loss()` |

### 3.2 19 条单元测试盯住公式

`tests/test_losses.py` 不只是"能跑通",而是逐条钉住容易写错的细节:

- `ratio` 必须等于 $\exp(\Delta\log p)$;
- **PPO 在 $A>0$ 时不会裁剪低比率一侧**(很多教程画错的地方);
- **被裁剪的一侧梯度必须严格为 0**(PPO),而 **CISPO 梯度限幅但永不为 0**;
- GSPO 的 ratio 必须是长度归一化后的几何平均,而不是 token 级连乘;
- 退化组(全对/全错)的 advantage 必须全 0;
- DAPO 长度惩罚的分段值(3072→0、3584→−0.5、4096→−1.0)。

```bash
python tests/test_losses.py     # 19/19 passed
```

### 3.3 机制图:把"公式差异"画出来

| 图 | 结论 |
|---|---|
| `figures/01-三种loss几何差异.png` | $\rho\in[0.2,3]$ 均匀取值时,**PPO 有 64% 区间梯度被完全掐断**,CISPO 与 IS 为 0% |
| `figures/02-DAPO软超长惩罚.png` | 分段线性坡道:3072 不罚,3584 罚 −0.5,4096 罚 −1.0 |
| `figures/03-ClipHigher区间.png` | GRPO `[0.8,1.2]` vs DAPO `[0.8,1.28]`,只放宽上界 |
| `figures/04-GSPO序列级比率.png` | token 级连乘随长度指数爆炸,序列级几何平均与长度无关 —— 这才是 GSPO 敢把 clip 压到 3e-4 的原因 |

### 3.4 真跑:GSM8K 上的算法对照

用 Qwen2.5-1.5B + LoRA,在 RTX 5060 Ti 上跑 5 组对照(每组 10 step、3 组题 × 每题 4 条轨迹、PPO 内层 2 epoch)。

| 运行 | 算法 | 说明 |
|---|---|---|
| `grpo_ppo` | GRPO + 对称 PPO clip `[0.8,1.2]` | 基线 |
| `grpo_is` | GRPO + 无裁剪 IS | 极限对照组（不加任何限速） |
| `grpo_cispo` | GRPO + CISPO | ratio 当固定砝码 |
| `dapo` | DAPO 四项改动全开 | 含动态采样与长度惩罚 |
| `gspo` | GSPO 序列级 ratio | clip 3e-4 |

结果见 §4。

### 3.5 原仓库 17 个入口的可运行性核查

`scripts/00_verify_repo_chapters.py` 对每个章节跑 `python <入口> --help`,结果写入
[`docs/原仓库可运行性核查.md`](docs/原仓库可运行性核查.md)。

实测:**15 个 Python 入口中 11 个 `--help` 直接通过**;4 个失败全部是环境级原因,与代码本身无关:

| 章节 | 失败原因 |
|---|---|
| alfworld / tempo / agentopsd | `textworld` 依赖 `jericho`,在 Windows 上编译失败(`WinError 2`,缺 cmake/编译器) |
| tournament-grpo | 依赖 `fcntl`,Linux-only |

---

## 4. 复现结果

### 4.1 五组训练：只换 loss 函数，其余全一致

配置：Qwen2.5-1.5B + LoRA(r=16)，10 step，每步 3 组题 × 每题 4 条轨迹，
`max_new_tokens=224`，`ppo_epochs=2`，lr=1e-5，`micro_batch_size=2`，seed=42。

| 运行 | dev 准确率（前→后） | rollout 准确率（前 3 步 → 后 3 步） | 退化组比例 | 平均 loss | 秒/步 |
|---|---|---:|---:|---:|---:|
| `grpo_ppo` | 40% → **60%** | 0.333 → 0.417 | 0.30 | −0.0005 | 100 |
| `grpo_cispo` | 40% → 35% | 0.333 → 0.361 | 0.40 | −0.0378 | 102 |
| `dapo` | 40% → 30% | 0.472 → **0.583** | **0.00** | +0.1749 | **131** |
| `gspo` | 40% → 55% | 0.306 → 0.417 | 0.37 | −0.0120 | 97 |
| `grpo_is` | 40% → 45% | 0.389 → 0.333 | 0.37 | −0.0002 | 49* |

\* **秒/步不可跨会话比较**：前四组运行时机上还有其他负载（97~131 s/步），`grpo_is` 是空闲时补跑的
（45~66 s/步）。它更快**不是**因为 IS 更省算力 —— 五种 loss 的前向/反向开销几乎一样。
`grpo_ppo` 与 `dapo` 的 100 vs 131 s **可以**比，因为那是同一会话内先后跑的。

### 4.2 哪些结论可信，哪些不可信（这段最重要）

| 说法 | 可信？ | 为什么 |
|---|---|---|
| 「PPO 比 CISPO 好 25 个百分点」 | ❌ **不可信** | dev 只有 20 题，**错 1 题 = 5pp**，1SE ≈ ±10pp，所有 dev 变化都在噪声带内（见 `figures/07`） |
| 「DAPO 的 rollout 准确率最高」 | 🟡 弱信号 | 每组只有 12 条轨迹，但它是唯一退化组为 0 的运行，方向可信 |
| 「DAPO 的动态采样把退化组降到 0，代价是多 31% 墙钟」 | ✅ **可信** | 机制与成本都是逐 step 记录的，不依赖统计显著性 |
| 「单轮更新时 IS/PPO 的梯度几乎相同」 | ✅ **可信** | 受控实验：同批数据同权重，梯度范数差 **0.12%** |
| 「loss 值可以跨算法比较」 | ❌ **错** | 实测 loss 从 +0.1314 到 −0.0238，符号相反，梯度范数却只差 0.15% |

**一句话**：本仓库的 10 步实验只能看**机制**，不能给算法排名。
要排名得加大 batch、加多步数、换更大 dev 集 —— 那是工程不是复现。

### 4.3 三个靠实测才敢说的机制发现

**① ratio ≡ 1 时三种 loss 梯度等价，差异只在第 2 个 inner epoch 出现。**
第 1 个 epoch 所有 ratio 精确等于 1.000000（on-policy 的定义结果，不是 bug）；
第 2 个 epoch 才漂到 1.0006~1.0011，PPO 只裁掉 0.39% 的 token。
→ **为什么很多教程写「三种 loss 不一样」，你复现却看不出差别**：差别需要漂移才触发。

**② GSPO 的 3e-4 阈值在小规模下会被撑爆：93% 的序列被裁。**
`gspo` 运行第 2 个 epoch 的序列级 clip 比例高达 **93.0%**，但 dev 仍从 40% 涨到 55% ——
说明梯度几乎全靠第 1 个 epoch。GSPO 论文里单步序列 ratio 漂移在 1e-4 量级，3e-4 是匹配的；
**本实验 lr=1e-5 + 2 个 inner epoch 的漂移（1.0004）已经超出阈值**。
⚠️ 迁移结论：**用 GSPO 时 `ppo_epochs` 要么设 1，要么把 lr 降到漂移 < 3e-4**。

**③ DAPO 的 Dynamic Sampling 是"用算力换信号"，不是免费的。**
退化组 0.30 → 0.00，代价是每步平均丢弃 **1.4 个已生成的候选组**（batch 只有 3 组，等于 +47% 生成量），
墙钟 100s → **131s**。被丢的 completion 已经付过完整生成成本，只是不进 loss。
这与原仓库 DAPO 章"35 step 花了 4.17 小时、作者手动叫停"是同一个现象。

### 4.4 八张图

| 图 | 类型 | 结论 |
|---|---|---|
| `figures/01-三种loss几何差异.png` | 机制（无需 GPU） | ρ∈[0.2,3] 时 **PPO 有 64% 区间梯度被完全掐断**，IS/CISPO 为 0% |
| `figures/02-DAPO软超长惩罚.png` | 机制 | 分段线性坡道：3072 不罚 → 3584 罚 −0.5 → 4096 罚 −1.0 |
| `figures/03-ClipHigher区间.png` | 机制 | GRPO `[0.8,1.2]` vs DAPO `[0.8,1.28]`，**只放宽上界** |
| `figures/04-GSPO序列级比率.png` | 机制 | token 级连乘随长度指数爆炸 vs 序列级几何平均与长度无关，**667 倍阈值差** |
| `figures/05-训练曲线对照.png` | 实测（6 联） | reward / 准确率 / 熵 / 长度 / clip 比例 / ratio 六条曲线 |
| `figures/06-退化组比例.png` | 实测 | DAPO 恒为 0（动态采样丢弃补采），其余三组 0.3~0.4 |
| `figures/07-训练前后对比.png` | 实测 | dev 前后对比 + **±10pp 噪声带**，说明差异不构成结论 |
| `figures/08-五种loss梯度对照.png` | 受控对照 | loss 值差 6.5 倍且符号相反，梯度范数只差 0.15%；GSPO 仅 0.64× |

---

## 5. 快速开始

```bash
git clone <this-repo>
cd agentic-rl-lab-local-repro

# 1) 环境(建议单独 venv)
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe transformers peft accelerate datasets matplotlib numpy sympy tqdm

# 2) 公式单测(不需要 GPU)
python tests/test_losses.py

# 3) 机制图(不需要 GPU)
python scripts/02_mechanism_figures.py

# 4) 训练一组(需要 GPU,首次会下载 Qwen2.5-1.5B ≈ 3GB)
python scripts/03_train.py --algorithm grpo_ppo --tag grpo_ppo \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --steps 10 --groups-per-step 3 --group-size 4 \
    --max-new-tokens 224 --ppo-epochs 2 --micro-batch-size 2 \
    --device cuda:0

# 5) 从日志出图（05~08）
python scripts/05_make_figures.py

# 6) 可选：受控对照 —— 同批 rollout 上五种 loss 的梯度对照（约 2 分钟）
python scripts/06_loss_headtohead.py
python scripts/05_make_figures.py     # 重跑才会生成 figures/08
```

显存不够就把 `--micro-batch-size` 降到 1;序列太长就降 `--max-new-tokens`。

---

## 6. 局限与已知问题(必须读)

| 问题 | 说明 | 处理 |
|---|---|---|
| **模型规模不同** | 原仓库用 Qwen3.5-4B,这里用 Qwen2.5-1.5B(单卡可训 + 迭代快) | 结论只做**相对比较**,不与原仓库数字对照 |
| **步数少** | 10 step 观察趋势,不是收敛结果 | 看曲线形状与机制,不看绝对值 |
| **GSM8K 子集** | 只保留答案 ≤3 位数、题干 ≤400 字符的题,保证小模型有信号 | 数据过滤逻辑在 `local_repro/data.py` |
| **`math_verify` 在 Windows 不可用** | 它的 `verify()` 内部用 multiprocessing + timeout,本机会抛 `WinError 5/6` 并返回 False | 默认改走 sympy 等价判定;要复刻仓库行为设 `ARL_USE_MATH_VERIFY=1` |
| **few-shot 模板不匹配** | 原仓库 few-shot 带 `<think>` 标签(Qwen3 约定),Qwen2.5 不认,实测 0/10 格式合规 | 加了 `plain` 适配版:实测 4/10 正确、7/10 格式合规,可用 `--prompt-style repo` 切回原版 |
| **单卡串行** | 5 组实验顺序跑，纯训练约 80 分钟 | 有第二张卡可改 `--device cuda:1` 并行 |
| **ALFWorld / TEMPO / AgentOPSD 未跑** | `jericho` 在 Windows 编译失败 | 需要 Linux 环境;本仓库只做静态核查 |

---

## 7. 学习资料

| 文档 | 内容 |
|---|---|
| [docs/知识点总结.md](docs/知识点总结.md) | **主文档**。一条主线 + 7 个必掌握知识点 + 17 章逐章精要 + 5 个高频坑 + 迁移到自有任务的四步法 |
| [docs/SOTA汇总.md](docs/SOTA汇总.md) | 各方法在原仓库口径下的 Base→训练后对照、成本表、以及"哪些涨幅是真能力提升"的拆解 |
| [docs/本地复现实验记录.md](docs/本地复现实验记录.md) | 五组训练的全部实测数字、受控对照、以及一次**误判的排查记录** |
| [docs/原仓库可运行性核查.md](docs/原仓库可运行性核查.md) | 17 章（15 个 Python 入口）逐条 `--help` 实测 |
| [docs/raw/](docs/raw) | 942 行逐章原始精读笔记(论文出处、公式、reward、实现细节、原文数字) |

**如果只读三句话**:

```text
1. reward 决定方向,ratio 决定谁被强化,mask 决定谁不被强化。
2. 退化组(全对/全错)是第一诊断指标,不是 reward。
3. 只在固定评测集 + 固定 seed 上宣布结论。
```

---

## 8. 目录结构

```
agentic-rl-lab-repro/
├── local_repro/
│   ├── losses.py      # IS / PPO / CISPO / GSPO / OPD / 长度惩罚 / 组内 advantage
│   ├── rewards.py     # \boxed{} 解析 + sympy 等价判定 + 可验证奖励
│   ├── data.py        # GSM8K 子集 + 两套 few-shot 模板
│   ├── trainer.py     # 本地 HF+peft GRPO/DAPO/GSPO 训练器(梯度累积)
│   └── plots.py       # 机制图 + 训练曲线
├── scripts/
│   ├── 00_verify_repo_chapters.py   # 原仓库 15 个入口的可运行性核查
│   ├── 02_mechanism_figures.py      # 机制图 01~04（无需 GPU）
│   ├── 03_train.py                  # 统一训练 CLI（--algorithm 切五种算法）
│   ├── 05_make_figures.py           # 结果图 05~08（无需 GPU）
│   └── 06_loss_headtohead.py        # 受控对照：同批数据算五种 loss 的梯度
├── tests/test_losses.py             # 19 条公式断言，自带 runner
├── figures/                         # 8 张复现图像
├── logs/                            # 训练日志 JSON（可直接被 05 消费）
│   └── debug/                        # smoke / dbg 等调试运行，不入图
├── outputs/                         # LoRA adapter（体积大，已 gitignore）
└── docs/                            # 知识点总结 / SOTA / 实验记录 / 核查 / 原始笔记
```

---

## 9. 参考与致谢

- 算法与教程来源:[datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)(项目负责人 [宋志学 KMnO4-zx](https://github.com/KMnO4-zx))
- 原仓库依赖的远程训练服务:[PyTRIO](https://pytrio.com) ;实验记录:[SwanLab](https://swanlab.cn)
- 涉及论文:DeepSeekMath (2402.03300)、DAPO (2503.14476)、GSPO (2507.18071)、Search-R1 (2503.09516)、ReTool (2504.11536)、ALFWorld (2010.03768)、OPSD (2601.18734)、AgentOPSD (2608.05987)、Spec-o3 (2601.06498)、Tournament-GRPO (2605.26958)、GKD (2306.13649)、MiniLLM (2306.08543)

本仓库的代码与文档为**学习复现**用途。原仓库内容版权归其作者,遵循原仓库 LICENSE;本仓库新增部分以 Apache-2.0 发布。


# 1. 知识点总纲

# Agentic RL 知识点总纲 —— 一个仓库、17 个实验、一条主线

> 本文是对 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) 全部 17 个实验章节的**知识点系统化总结**，
> 配套本仓库的本地 GPU 复现代码（`local_repro/`）与实测图像（`figures/`）一起读。
> 所有数字均引自原仓库 readme，未作外推；复现相关的数字来自本机实测，会单独标注「本地实测」。

---

## 0. 先把主线立住：这个仓库到底在教你什么

一句话：**教你用「组内相对奖励 + 重要性比率」这一套骨架，把任意可验证任务变成 LLM 的参数更新。**

17 个实验看起来分散（数学、视觉、搜索、代码、家务、光谱、医疗、研究报告），
但它们共享同一套骨架，只是替换了其中一格：

```
            ┌───────────── 任务与可验证性 ─────────────┐
prompt ──► 采样 G 条轨迹 ──► 打分 (reward / teacher) ──► 组内相对 advantage ──► 策略梯度更新
   ▲            │                    │                        │                  │
   │            │                    │                        │                  │
   │      on-policy 约束        reward 设计比算法更关键    critic 被消掉     ratio + clip 的位置
   │      (每步刷新 sampler)    (0/1、±1、格式、长度)      (GRPO 最大简化)   (IS / PPO / CISPO / GSPO)
   │
   └── 多轮工具调用时：observation token 一律不进 loss（mask 为 0）
```

**必须掌握的 7 个知识点**（读代码的最小词汇表）：

| # | 知识点 | 一句话定义 | 在哪一章最清楚 |
|---|---|---|---|
| 1 | **on-policy** | 训练数据必须由「当前权重」采样；所以每个 step 先存权重拿 sampler，采完再更新 | GRPO / OPSD（`sampler_refresh_steps=1`） |
| 2 | **组内相对 advantage** | 同一 prompt 采 G 条，$A_i = r_i - \mathrm{mean}(r)$，用组均值替代 critic | GRPO 第 1 章 |
| 3 | **重要性比率 ratio** | $\rho = \exp(\log\pi_\theta - \log\pi_{old})$，纠正「采样策略 ≠ 训练策略」的偏差 | loss-functions 第 0 章 |
| 4 | **clip 的位置** | PPO 剪 *目标*，CISPO 剪 *比率但当常数*，GSPO 剪 *序列级比率*，DAPO 只放宽上界 | 见 §2 对照表 |
| 5 | **observation mask** | 环境/工具返回的 token 只进上下文、不进 loss（logprob 与 advantage 填 0） | Search-R1 / ReTool / ALFWorld |
| 6 | **reward 是实验的第一变量** | 0/1 vs ±1、格式奖励、长度惩罚、淘汰赛胜场，往往比换算法更影响结果 | 各章 reward 小节 |
| 7 | **训练信号可以不来自 reward** | OPD/OPSD/AgentOPSD 用 teacher logprob 差（reverse KL）当 advantage | OPD 四章 |

**一句话记住整个仓库**：`reward 决定方向，ratio 决定谁被强化，mask 决定谁不被强化。`

---

## 1. 基础篇：三种 loss 的几何差异（本篇有本地实测图）

这一节是后面所有章节的地基。三个 loss 吃同样的输入
（`target_logprobs`、`sampling_logprobs`、`advantages`、`completion mask`），
只在「怎么把 ratio 变成梯度」上不同。

### 1.1 公式与直觉

| Loss | 公式 | 人话 | 梯度行为 |
|---|---|---|---|
| Importance Sampling | $-\mathbb{E}\left[\frac{p_\theta}{q}A\right]$ | 按功劳发奖金 | 无上界，ratio 越大推得越猛 |
| PPO | $-\mathbb{E}\left[\min\left(\rho A,\ \mathrm{clip}(\rho,1-\epsilon_{low},1+\epsilon_{high})A\right)\right]$ | 奖金有预算上限 | **被截断的一侧梯度严格为 0** |
| CISPO | $-\mathbb{E}\left[\mathbf{sg}\left(\mathrm{clip}(\rho)\right)\cdot\log p_\theta\cdot A\right]$ | 给学习力度挂固定砝码 | 限幅但**永不为 0** |

> ⚠️ 最容易误解的一点：`min` **只截断一侧**。
> $A>0$ 时是 $\rho$ 过大（$>1+\epsilon_{high}$）才被截断；$A<0$ 时是 $\rho$ 过小（$<1-\epsilon_{low}$）才被截断。
> 很多教程画成「出区间就归零」，那是错的。本仓库 `tests/test_losses.py` 里
> `test_ppo_does_not_clip_low_side_for_positive_adv` 专门钉住这个细节。

**本地实测**（`figures/01-三种loss几何差异.png`）：令 $\log\pi_{old}=0$、$\rho$ 在 $[0.2,3]$ 均匀取值，
PPO 有 **64%** 的区间梯度被完全掐断，CISPO 与 IS 均为 0%。这就是「CISPO 不让 token 白学」的定量含义。

### 1.2 三种 loss 的取舍

- **IS**：信号最直接，适合小实验里观察 reward 如何推动概率；无裁剪，正式训练容易崩。
- **PPO**：稳定性优先。被裁掉的 token 这一步不再学习——既是保护，也是信息损失。
- **CISPO**：保留每个 token 的学习信号，只给力度设上限。长 CoT 里「少数关键 token 被裁掉」的代价较大时更合适。

---

## 2. 算法谱系：17 个实验的定位

### 2.1 一张谱系图

```
SFT ──► RLVR（可验证奖励）───────────────────────────────────┐
          │                                                  │
          ├─ GRPO (2402.03300)  组均值当 baseline，去 critic   │
          │    ├─ DAPO (2503.14476)  四处工程改动               │
          │    ├─ GSPO (2507.18071)  比率升到 sequence 级        │
          │    ├─ Vision GRPO        图像进上下文                │
          │    ├─ Search-R1 (2503.09516) 多轮搜索               │
          │    ├─ ReTool (2504.11536)   代码解释器              │
          │    ├─ ALFWorld (2010.03768) 具身/文字环境           │
          │    ├─ Spec-o3 (2601.06498)  光谱工具 + 多模态 CoT    │
          │    ├─ TEMPO               macro-step + 生成式 critic │
          │    └─ Tournament-GRPO (2605.26958) 淘汰赛奖励        │
          │                                                  │
          └─ 蒸馏路线（信号不是 reward，是 teacher logprob）      │
               ├─ OPD  (GKD 2306.13649 / MiniLLM 2306.08543)   │
               │    ├─ SAR-OPD / IDT-OPD（双 teacher 保通用能力）  │
               │    ├─ OPSD (2601.18734)  step-0 自己当老师       │
               │    └─ AgentOPSD (2608.05987) turn 级 credit     │
               │                                               │
               └─ PyTRIO-Jev  单 token 受限 softmax（非 RL）      │
                                                                  │
     所有章节共用：PyTRIO 云端训练 + SwanLab 记录 + 组内相对 advantage ◄┘
```

### 2.2 逐章定位表

| 章节 | 论文 / 来源 | 改了什么（一句话） | 记忆锚点 |
|---|---|---|---|
| loss-functions | 原创教学 | 讲清 IS/PPO/CISPO | ratio 决定谁被强化 |
| grpo | DeepSeekMath 2402.03300 | 组均值替代 critic | 组内均值当 baseline |
| vision-grpo | 沿用 GRPO | prompt 里塞图片 chunk | 给 GRPO 装上眼睛 |
| general-opd | GKD 2306.13649 / MiniLLM 2306.08543 | teacher 只打分不写答案 | 学生采样，老师只打分 |
| medical-opd | OPD + 自命名 SAR/IDT | 双 teacher 保领域 + 保通用 | 先学医，再用锚点补通用 |
| search-r1 | Search-R1 2503.09516 | 搜索变成模型动作 | 让模型自己决定何时搜 |
| opsd | OPSD 2601.18734 | 老师 = step-0 的自己 + 参考答案 | 同模型带着答案自评 |
| retool | ReTool 2504.11536 | 代码解释器交织进 CoT | 边推理边写代码 |
| dapo | DAPO 2503.14476 | 四项工程改动 | 四处改动，卡在补采 |
| gspo | GSPO 2507.18071 | ratio 与 clip 升到序列级 | 一条回答，一个 ratio |
| alfworld | ALFWorld 2010.03768 | 同局 K 条轨迹做组 | 同局八条，胜负减均值 |
| tempo | 博客（无论文） | macro-step + 生成式 critic | 切段估值，一参两用 |
| agentopsd | 2608.05987 | teacher 带 Skill 复评 turn | 终局定方向，turn 定强度 |
| spec-o3 | 2601.06498 | 看图 + 调工具重绘光谱 | 边看图边放大光谱再答题 |
| tournament-grpo | 2605.26958 | 组内淘汰赛胜场当 reward | 裁判打淘汰赛，胜场当奖励 |
| pytrio-jev | 无（架构逆向） | 单 token 受限 softmax | 一次前向只吐一个 token |

---

## 3. 逐章知识点精要

### 3.1 GRPO：把 critic 扔掉

**痛点**：PPO 要训 value model 估 baseline，对 LLM 而言显存与计算翻倍；数学题又只有最终答案能打分，让 value 逐 token 估准并不自然。

**做法**：同一题采 $G$ 条，用组内均值当 baseline：

$$A_i = r_i - \mathrm{mean}(r_1,\dots,r_G) \quad\text{（论文版还会除以组内 std）}$$

**必须记住的实现细节**：

- 每个 step 先用**当前权重**导出一个 sampler，再采样；旧 logprob 必须来自这个 sampler，**不能在更新后重算**。
- 组内 reward 全相同（全对或全错）→ advantage 全 0 → 该组白采（`frac_degenerate`）。
- prompt token 的 `logprobs` 与 `advantages` 填 0，只有 completion 进 loss。
- 三处显式简化：reward 只用 rule-based 0/1；advantage 未除 std；无 reference KL 约束。

**指标怎么读**：reward 不动可能是三种情况——没起效、退化组太多、模型转向更长推理。要同时看 `frac_degenerate`、`clip_fraction`、`completion_length`。

### 3.2 DAPO：四处工程改动，和它带来的时间陷阱

| 改动 | 内容 | 解决什么 |
|---|---|---|
| Clip-Higher | 上界 1.2 → **1.28**，下界仍 0.8 | 正 advantage 的低概率 token 有更大上升空间，缓解 entropy collapse |
| Dynamic Sampling | 丢弃全对/全错组，继续补采 | 只训练「处在能力边界」的题 |
| Token-level Loss | sample mean → **token mean** | 长 reasoning 不被序列平均稀释 |
| Soft Overlong Punishment | 接近上限时线性扣分 | 避免硬截断造成的 reward 断崖噪声 |

**本章最值钱的一课不是算法，是成本**：作者实测训练到第 35 step 已耗约 **4.17 小时**，手动叫停。
根因是 Dynamic Sampling 的**补采屏障**——被丢弃的 completion 已经付过完整生成成本，却不进 loss；
当 batch 只差最后一个有效组时，补采变成串行等待，与异步 RL「先完成先流转」的目标结构性冲突。
最坏情况 wave 数 = $1 + (M-1)\times B$（$B$ 组/步，$M$ 候选倍率）。

### 3.3 GSPO：一条回答，一个 ratio

**痛点**：一条 $T$ token 的回答在 GRPO 里有 $T$ 个 ratio、$T$ 次 clip 判断，长序列方差大；而 reward 是按整条回答给的，粒度不匹配。

**做法**：长度归一化后取几何平均，整条序列只留一个 ratio：

$$s_i = \exp\left(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta}{\pi_{old}}\right),\qquad \mathcal{L}= -\frac{1}{G}\sum_i \min\left(s_i A_i,\ \mathrm{clip}(s_i, 1-\epsilon_{low}, 1+\epsilon_{high})A_i\right)$$

**关键**：$\epsilon$ 只有 $3\times10^{-4}$ / $4\times10^{-4}$，比 PPO 的 0.2 小三个数量级——
因为 $s_i$ 已经是几何平均量级，偏离 1 一点点就代表很大的策略漂移。本地实测图见
`figures/04-GSPO序列级比率.png`。

**原文结果**：AIME25 Step 20 的 Pass@12 相对 Base Run 2 从 66.67% → 73.33%（+6.66pp）；
但两次独立 Base 的 Pass@12 相差 10pp，30 题样本下单次波动极大，不能当稳定收益。

### 3.4 Search-R1 / ReTool / ALFWorld：多轮 Agentic RL 的三件套

三章都在做同一件事：把「一次 completion」变成「多轮轨迹」，并解决它带来的三个新问题。

**(1) observation mask**——工具/环境返回的 token 进上下文但不进 loss。
实现方式是给这些位置的 `old_logprob = 0`、`advantage = 0`，而不是额外传 mask。
理由：环境说的话不是模型的动作，不该被强化。

**(2) group 如何分叉**——这是最容易写错的地方：

- 首轮 G 条 prompt 完全相同 → 一次 `num_samples=G` 分叉；
- 首次工具调用后 prompt 已分叉 → 必须「每条未结束轨迹一个独立 prompt × `num_samples=1`」再并发。
  继续用 `num_samples=G` 会直接错配轨迹。

**(3) token-in token-out**——下一轮 prompt 由真实采样 token 拼接，不做 text↔token 重编码。
ReTool 章引用官方实测：重编码不可逆会导致约 100 步后性能崩塌、`grad_norm` 变 NaN。

**各章 reward 设计对比**（reward 设计比算法更能决定结果）：

| 章节 | reward | 特点 |
|---|---|---|
| Search-R1 | 三档 $+1 / 0 / -0.1$ | 格式非法重罚；只奖励最终答案，不奖励「搜了几次」 |
| ReTool | $\pm1$ | 只取最后 300 字符最后一个 `\boxed{}`，数学等价判定；不奖励「用了代码」 |
| ALFWorld | $\mathbb{1}[\text{won}] - 0.1\times n_{invalid}$ | 显式扣非法动作，抑制乱试 |

**本地实测**的机制图见 `figures/02-DAPO软超长惩罚.png`。

### 3.5 OPD 家族：当训练信号不来自 reward

这是一条完全不同的路线：**不给奖励，给逐 token 的「老师批改」**。

$$\text{reverse KL 的 token 形式：}\quad A_t = -\beta\left(\log\pi_\theta(y_t) - \log\pi_T(y_t)\right)$$

学生自己采样（保证 on-policy），teacher 只对这条轨迹逐 token 算 logprob，不重写答案。

| 变体 | teacher 是谁 | 解决什么 |
|---|---|---|
| OPD | 更大的模型（如 27B） | 用密集 token 信号替代稀疏 reward |
| OPSD | **step-0 的自己**，prompt 里多给参考答案 | 不用部署更大 teacher，靠「特权上下文」而非更多参数 |
| SAR-OPD | Medical SFT 的冻结 teacher → 再切原始 Base 当锚点 | 修「学领域破坏通用」 |
| IDT-OPD | 医疗 / 通用两个 teacher 按 1:1 交替 | 同上，但不分段 |
| AgentOPSD | 同一快照带 Skill 复评（Self-Teacher） | 把序列级 advantage 细化成 turn 级 |

**最值得记住的工程约束**：teacher 与 student **必须共享兼容 tokenizer**，
否则无法比较同一 token ID 的 logprob；三者（completion / student logprob / teacher logprob）
长度必须严格相等，不做静默截断。

**最重要的一条认知**：KL / reverse KL 下降只说明「学生更接近老师」，
**不自动等于任务准确率提高**。OPSD 章给出了直接证据——AIME25 上 loss 一路下降，而 checkpoint 准确率反复波动。

### 3.6 TEMPO / AgentOPSD：把 credit 分配到该去的地方

长轨迹的两个硬问题：信号周期太长（一局几小时）、终局 reward 无法判断中间哪步关键。

- **TEMPO**：把优化单位从「整局」降到「macro-step」，用**生成式 critic**（会推理的估值器）给段边界状态估值；
  actor 与 critic **共享同一份参数**，靠 prompt 与上下文切换角色。最硬的工程点是**环境存档读档**
  （存 action_history + token_prefix 双快照，重放后逐字校验 observation）。
- **AgentOPSD**：保留终局 0/1 定方向，用 Self-Teacher 的 turn 级证据修正强度，
  乘子被限制在 $[0.8, 1.2]\times$ 再经 $\lambda=0.5$ 混合——**永远不会把成功轨迹改成负优势**。

### 3.7 Spec-o3 / Tournament-GRPO / Jev：三个「非典型」实验

- **Spec-o3**：SFT 冷启动（942 条示范）→ GRPO。图片以 `ImageChunk` 进入上下文且**训练权重为 0**；
  模型自己决定放大哪个波段，工具重绘局部光谱作为新图继续推理。教训：**SFT 冷启动决定 RL 天花板**。
- **Tournament-GRPO**：无标准答案的长文任务，用组内两两淘汰赛的累计胜场构造 reward，再做 min-max 归一化。
  本章最大的诚实点：**涨幅主要来自「白卷变少」**（59 题共同子集上 Overall 仅 34.49 → 35.25）。
- **PyTRIO-Jev**：不是 RL，是 masked SFT + 受限 softmax。**一次前向只吐一个 token**，
  输出候选符号上的归一化概率，可直接做阈值与校准（温度拟合 + ECE/Brier）。
  教训：**数据分布不匹配会让训练帮倒忙**——短 case 上训完，Hard 长文档档反而从 55.86% 掉到 47.75%。

---

## 4. 五个反复出现的坑（比算法更值得背）

| # | 坑 | 现象 | 处理 |
|---|---|---|---|
| 1 | **退化组** | 组内全对或全错 → advantage 全 0 → 该步无梯度 | 盯 `degenerate_frac`；DAPO 用 Dynamic Sampling 过滤，但要付补采代价 |
| 2 | **格式与正确性混在一起** | 解析不出 `\boxed{}` 也记 0 分 | 同时看 `format_rate` 与条件准确率（Vision GRPO 的 +16pp 主要来自格式） |
| 3 | **observation 泄漏进 loss** | 工具返回的 token 被当模型动作强化 | 一律把 logprob/advantage 置 0 |
| 4 | **重新 tokenize 历史** | text↔token 不可逆，长训练后崩溃 | token-in token-out，严格前缀扩展校验 |
| 5 | **用在线 reward 判断模型变强** | 在线 reward 受题目难度影响剧烈波动 | 固定评测集 + 固定 seed 才是结论来源 |

---

## 5. SOTA 与指标口径（详见 `SOTA汇总.md`）

| 方法 | 评测集 | Base → 训练后 | 关键注意 |
|---|---|---|---|
| Vision GRPO | GeoQA 固定 100 题 | 71.0% → 87.0%（+16.0pp） | 增益主要来自格式合规（条件准确率仅 94.7% → 95.6%） |
| ReTool | AIME25（30 题×12） | Average@12 23.61% → 47.50%（+23.89pp） | 8k 轨迹预算截断长推理；Format 76% |
| GSPO | AIME25 Step 20 | Pass@12 66.67% → 73.33%（+6.66pp） | 30 题单次采样波动可达 ±10pp |
| Search-R1 | 70 题 Macro EM | 40.00% → 42.86%（+2.86pp） | 换过模型/搜索后端，不可与论文直接比 |
| ALFWorld | 274 局 | 52.92% → 56.93%（+4.01pp） | 单 seed 单次训练；任务级收益极不均匀 |
| AgentOPSD | 274 局 | 58.8% → 65.7%（+6.9pp） | 缺少同预算 GRPO 对照，无法归因 |
| OPSD | AIME25 | Average@12 51.67% → 52.78% | 中间 checkpoint 非单调，Step 75 低于 Base |
| Medical SAR-OPD | MedQA-zh / C-Eval | 72.17%→84.33% / 81.67%→84.67% | 单条训练轨迹，无多 seed |
| Spec-o3 | 256 条 dev（Macro F1） | 53.99% → 71.68%（RL epoch 1） | epoch 3 F1 反而回落 |
| Tournament-GRPO | DRB 100 题 Overall | 20.65 → 34.70（+14.05） | 增益 92% 来自白卷减少 |
| PyTRIO-Jev | 五基准平均 | 76.30% → 83.74% | Hard 档倒退（55.86% → 49.55%） |

---

## 6. 从这套骨架出发，怎么迁移到你自己的任务

**四步落地法**（任何「能被程序判对错」的任务都能套）：

1. **先写 reward，再挑算法**。判分函数 ≤ 20 行就够；判不出来说明任务还没准备好做 RL。
2. **确认有组内差异**。同一 prompt 采 8 条，若全对或全错占比 > 80%，先降难度或换数据，
   而不是换算法。
3. **先跑通 20 步**，盯三个指标：`reward_mean`、`degenerate_frac`、`completion_length`。
   三者都正常再谈长训练。
4. **只在固定评测集上宣布结论**。在线 reward 是过程指标，不是效果指标。

**如果算力只有一张消费级卡**（本仓库的处境）：

- 用 1.5B 级小模型 + LoRA，跑 3～4 组算法对照，胜在「能反复试」而不是「冲榜单」；
- 长 CoT 场景优先降 `max_tokens`（截断噪声 vs 成本的权衡）；
- Dynamic Sampling 这类「拿时间换信号质量」的开关，小算力下通常不划算。

---

## 7. 配套复现资产索引

| 资产 | 路径 | 说明 |
|---|---|---|
| 三种 loss 几何差异 | `figures/01-三种loss几何差异.png` | PPO 64% 区间梯度归零的定量图 |
| DAPO 软超长惩罚 | `figures/02-DAPO软超长惩罚.png` | 分段线性坡道 |
| Clip-Higher 区间 | `figures/03-ClipHigher区间.png` | 0.8–1.2 vs 0.8–1.28 |
| GSPO 序列级比率 | `figures/04-GSPO序列级比率.png` | 连乘爆炸 vs 几何平均 |
| 训练曲线对照 | `figures/05-训练曲线对照.png` | 本地 GPU 实测 |
| 退化组比例 | `figures/06-退化组比例.png` | 本地 GPU 实测 |
| 训练前后对比 | `figures/07-训练前后对比.png` | 本地 GPU 实测 |
| loss 单元测试 | `tests/test_losses.py` | 19 条断言，含梯度符号验证 |
| 原仓库可运行性 | `docs/原仓库可运行性核查.md` | 16 个入口逐条 `--help` 实测 |

---

## 8. 一页纸速记

```text
GRPO   = 组均值当 baseline，不要 critic
DAPO   = Clip-Higher + 动态采样 + token mean + 软超长惩罚（代价是补采）
GSPO   = ratio 从 token 级升到 sequence 级，clip 阈值小三个数量级
CISPO  = 剪比率但当常数，梯度限幅不归零
OPD    = 老师逐 token 打分，不写答案
OPSD   = 老师就是 step-0 的自己 + 参考答案（特权信息）
Agent  = observation 不进 loss，prompt 分叉后必须逐条采样
reward = 实验的第一变量；退化组是第一诊断指标
结论   = 只在固定评测集 + 固定 seed 上宣布
```


# 2. SOTA 汇总

# SOTA 视角：这个仓库把每个方法跑到了什么水平

> 数字全部引自 `datawhalechina/agentic-rl-lab` 各章 readme / start.md 与仓库代码，
> 未做任何换算或外推。凡"仓库未给"的一律标注，不用论文数字代替。

## 0. 先说清楚：这份 SOTA 表不能当排行榜用

三条硬约束必须先讲，否则表格会被误读：

| 约束 | 具体表现 |
|---|---|
| **换了模型与训练栈** | 原论文多用 32B 级全参训练（ReTool 官方 Qwen2.5-32B、GSPO 官方 Qwen3-30B-A3B），本仓库统一 Qwen3.5-4B + **LoRA rank 32**、单 batch 一次更新 |
| **换了评测集规模** | AIME25 只有 30 题、GeoQA 固定抽 100 题、Search-R1 固定 70 题、ALFWorld 274 局；单次训练 + 单 seed |
| **换了基础设施** | 搜索用 DeepSeek Search API 而非本地 Wikipedia 语料，judge 用 Jev 而非 Qwen2.5-72B，环境用 ALFWorld 替代 ARC-AGI-3 |

所以正确读法是：**看「同一评测口径下的 Base → 训练后」这个差值，判断"这个算法的信号确实进到参数里了"，而不是拿绝对值去比论文榜单。**

---

## 1. 一张总表：各方法在本仓库口径下的效果

| 方法 | 评测集（规模） | Base | 最佳 checkpoint | 变化 | 是否可信的收益 |
|---|---|---:|---:|---:|---|
| Vision GRPO | GeoQA test（固定 100） | 71.0% | 87.0%（step 100） | **+16.0pp** | 🟡 增益主要来自格式合规，条件准确率仅 +0.9pp |
| ReTool | AIME25（30×12） | Average@12 23.61% | 47.50%（step 200） | **+23.89pp** | 🟢 唯一大幅、且 Pass@12 同步上涨的方法 |
| GSPO | AIME25（30×12） | Pass@12 66.67%（Run 2） | 73.33%（step 20） | +6.66pp | 🟡 两次 Base 相差 10pp，样本太小 |
| Search-R1 | 7 benchmark×10 题（70） | Macro EM 40.00% | 42.86%（step 30） | +2.86pp | 🟡 净增 2 题 |
| ALFWorld GRPO | 274 局（seen 140 + unseen 134） | 52.92% | 56.93%（80 updates） | +4.01pp | 🟡 单 seed，任务级收益极不均匀 |
| AgentOPSD | 274 局 | 58.8% | 65.7%（80 steps） | **+6.9pp** | 🟡 缺同预算 GRPO 对照，无法归因 |
| OPSD | AIME25（30×12） | Average@12 51.67% | 52.78%（step 100） | +1.11pp | 🟡 中间 checkpoint 非单调 |
| Medical SAR-OPD | MedQA-zh 600 / C-Eval 300 | 72.17% / 81.67% | 84.33% / 84.67% | +12.17pp / +3.00pp | 🟡 单条轨迹、无多 seed |
| Medical IDT-OPD | 同上 | 同上 | 79.17% / 86.00%（step 200） | +7.00pp / +4.33pp | 🟡 末端回落，最后一步≠最好 |
| Spec-o3 | dev 256 条（Macro F1） | 53.99% | 71.68%（RL epoch 1） | **+17.69pp** | 🟢 SFT + RL 两级都稳定提升 |
| Tournament-GRPO | DRB 100 题（Overall） | 20.65 | 34.70（step 20） | **+14.05** | 🔴 92% 来自白卷减少，59 题共同子集仅 +0.76 |
| PyTRIO-Jev | 五基准平均 | 76.30% | 83.74%（+温度校准） | +7.44pp | 🟡 Hard 档倒退 6pp，数据问题 |
| DAPO | — | — | — | **仓库未给** | 🔴 作者明说不展示曲线、不比优劣 |
| GRPO（第 1 章） | GSM8K | — | — | **仓库未给** | 🔴 只讲算法与代码 |
| TEMPO | ALFWorld 冒烟 | — | — | 仅机制观察 | 🔴 论文未发布，只做算法级验证 |

🟢 可当作稳定收益 ｜ 🟡 方向正确但证据强度有限 ｜ 🔴 不能据此下结论

---

## 2. 分项细节（只列关键数字）

### 2.1 数学推理类

**ReTool（代码交织）**——本仓库效果最强的一组：

| Checkpoint | Average@12 | Pass@12 | Format | 平均 code 调用 |
|---|---:|---:|---:|---:|
| Base | 23.61% | 50.00% | 25.83% | 1.69 |
| Step 20 | 32.22% | 63.33% | 36.67% | 1.37 |
| Step 100 | 44.72% | 66.67% | 57.78% | 1.66 |
| Step 150 | 46.11% | 73.33% | 70.83% | 2.43 |
| **Step 200** | **47.50%** | 63.33% | **76.11%** | 1.98 |

代价：200-step 训练会话 ¥224.44（48.67M prefill / 15.97M train / 15.84M sample）。
边界：轨迹预算 8k（论文 16k），长推理被截断，Format 卡在 76%。

**GSPO**——注意 Average@12 反而下降，只有 Pass@12 涨：

| Checkpoint | Average@12 | Pass@12 | Format | 平均 tokens |
|---|---:|---:|---:|---:|
| Base (Run 1/2) | 41.11% / 40.56% | 56.67% / 66.67% | 46.67% / 47.50% | 6489.8 / 6442.5 |
| Step 20 | 41.11% | **73.33%** | 62.22% | 5775.4 |
| Step 100 | 33.89% | 66.67% | 99.44% | 3358.1 |

一个被低估的结论：**GSPO 把平均 completion 长度砍掉 48%（6442.5 → 3358.1），格式率从 47.5% 升到 99.44%**——
它更像"让回答变短变规范"，而不是"让推理变强"。封面的"+6pp"指的是 Pass@12 的 Step 20 相对 Base Run 2。

**OPSD（自蒸馏）**：

| Checkpoint | Average@12 | Pass@12 | 正确 generations |
|---|---:|---:|---:|
| Base | 51.67% | 80.00% | 186/360 |
| Step 50 | 50.28% | 86.67% | 181/360 |
| Step 75 | 48.61% | 76.67% | 175/360 |
| **Step 100** | **52.78%** | **86.67%** | **190/360** |

训练侧：`loss_mean 0.0577 → 0.0425`、`reverse_kl_mean 0.0644 → 0.0471`，
但准确率中间下探到 48.61% —— **KL 下降 ≠ 能力上升**的直接证据。

### 2.2 视觉 / 多模态类

**Vision GRPO（GeoQA）**：Base 71.0% → step 100 87.0%，Format 75.0% → 91.0%。
把格式剥开看：条件准确率 94.7% → 95.6%，**+16pp 里只有 0.9pp 是真推理提升**。

**Spec-o3（光谱审核，dev 256 条）**：

| 阶段 | Macro F1 | Accuracy | Format | 答对 |
|---|---:|---:|---:|---:|
| 4B Base | 53.99% | 38.67% | 0.39% | 99/256 |
| 4B SFT ep2 | 63.05% | 53.91% | 85.94% | 138/256 |
| 4B RL ep1 | **71.68%** | 67.97% | 96.09% | 174/256 |
| 4B RL ep3 | 70.93% | 68.36% | 98.05% | 175/256 |
| 9B RL ep3 | 75.94% | 74.61% | 99.61% | 191/256 |

**这是本仓库最值得学的一条曲线**：SFT 先把格式从 0.39% 拉到 85.94%（冷启动），RL 再把 F1 推上去；
而 epoch 3 出现"格式更好、F1 下滑"——多训不一定更好。

### 2.3 Agent / 环境交互类

**ALFWorld（274 局）**：

| 模型 | Seen | Unseen | 总成功率 | 平均非法动作 |
|---|---:|---:|---:|---:|
| Base | 52.86% | 52.99% | 52.92% | 5.41 |
| ckpt 40 | **57.14%** | 52.24% | 54.74% | 5.24 |
| ckpt 80 | 55.00% | **58.96%** | **56.93%** | **4.81** |

成本结构值得抄下来：**prefilling 843.51M ≫ train 62.24M / sample 14.28M**，
账单 ¥1613.71。Agentic RL 的钱主要花在"模型反复读历史"上。

**AgentOPSD**：58.8% → 65.7%（+6.9pp），无效率动作 7.07 → 5.68，¥675.69。
unseen split 提升（+9.7pp）比 seen 更明显，方向上说明学到的是可迁移的策略而非记忆。

**TEMPO**：只做算法级冒烟验证（论文未发布，超参全部脱敏）。观察到的信号：
`segment_reward_mean 0.62 → 0.85`、`degenerate_group_rate 0.19 → 0`、critic 误差 `0.674 → 0.315`。

### 2.4 检索 / 长文 / 决策类

**Search-R1（70 题）**：Macro EM 40.00% → 42.86%，Format 82.86% → 91.43%，平均搜索次数 1.76 → 2.77。
分 benchmark 看更真实：NQ 20%→30%、PopQA 10%→30%、TriviaQA 50%→60%，但 2Wiki 50%→40%、Bamboogle 70%→60%。
**总账是小幅正收益 + 行为变化（更愿意搜）**。

**Tournament-GRPO（DRB 100 题）**：Overall 20.65 → 34.70。但拆开看：

- Base 有 **40 题白卷**（记 0 分），Step 20 只剩 2 题；
- 两边都写出报告的 **59 题**上，Overall 只有 34.49 → 35.25。

即 **涨幅基本等于"从不会写到会写"，单份报告质量几乎没变**。这一条比分数本身有价值。

**PyTRIO-Jev（五基准平均）**：

| 模型 | Easy | Original | Hard | Typed | ToolACE | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B base | 100.00 | 87.50 | 55.86 | 50.40 | 87.74 | 76.30 |
| PyTrio-Jev | 100.00 | 95.83 | 47.75 | **79.20** | 94.52 | 83.46 |
| +温度校准 | 100.00 | 95.83 | 49.55 | 79.10 | 94.19 | **83.74** |
| Intern-Decision-4B（参考） | 100.00 | 98.61 | 73.87 | 80.55 | 96.45 | 89.90 |

**Typed +28.8pp、Hard −8.1pp**：短 case 训完迁移到长文档档反而倒退，因为是数据分布问题不是算法问题。

---

## 3. 横向对比：同一件事，不同方法各付多少代价

这张表是本仓库最实用的部分——**同样提升 1 个百分点，不同路线的成本差几十倍**。

| 方法 | 训练规模 | 墙钟/费用（仓库记录） | 单位收益 |
|---|---|---|---|
| GRPO 最小闭环 | 10 step | "调试花 55 元，只跑 10 步可低到 5 元" | 便宜到可以随便试 |
| GSPO | 100 step，85 s/step | 约 2h22m，¥155.80 | ¥1.56/step |
| OPSD | 100 step | 约 2h06m，¥41.81（+ 5 次评测 ¥83.61） | 评测比训练贵 |
| ReTool | 200 step | 训练 ¥247.33 | 高质量数据 + 沙箱，性价比最高 |
| ALFWorld | 80 updates × 64 条长轨迹 | ¥1613.71 | prefill 主导 |
| AgentOPSD | 80 updates | ¥675.69 | 比同环境 GRPO 便宜（步数少） |
| Spec-o3 | SFT 424 step + RL 1 epoch | SFT ¥27.72 + RL ¥671.96 = ¥699.68 | 冷启动极便宜 |
| Tournament-GRPO | 20 step | 训练 ¥78.92 + 评测 ¥13.12 | 裁判成本仅 ¥1 |
| DAPO | 35 step 未跑完 | 已 4.17 小时 | **Dynamic Sampling 把时间吃掉了** |
| PyTRIO-Jev | 270 step | ¥12.36 | 最便宜 |

---

## 4. 这张表教给你的三件事

1. **"提升"要先拆成格式与能力**。Vision GRPO 的 +16pp、Tournament-GRPO 的 +14 分，拆开后能力提升都不到 1 个点。
   真正能力提升明确的是 ReTool（+23.89pp Average@12，Pass@12 同步涨）与 Spec-o3（SFT 与 RL 两级都涨）。
2. **reward 设计比算法选择更影响结论**。同样是 GRPO：ReTool 的 ±1 + 代码工具 → 大幅提升；
   ALFWorld 的 1−0.1×非法动作 → 小幅提升；Tournament-GRPO 的淘汰赛 → 主要治白卷。
3. **成本结构决定可行性**。Agentic RL 的账单主体是 prefilling（反复读历史），
   DAPO 的墙钟杀手是补采屏障——这两条比任何超参都更早决定"这个实验能不能跑完"。

---

## 5. 想追真正的 SOTA，缺口在哪

| 缺口 | 本仓库现状 | 要补什么 |
|---|---|---|
| 模型规模 | 4B + LoRA | 30B 级全参（GSPO 论文口径）才能验证 MoE 稳定性等结论 |
| 评测规模 | AIME25 30 题 | 至少 8 次重复采样 × 多 seed，否则 Pass@12 波动 ±10pp |
| 对照实验 | 多数章节无 matched baseline | 同模型、同数据、同预算的 GRPO 对照（AgentOPSD 自己就点名缺这个） |
| 长上下文预算 | ReTool 8k vs 论文 16k | 预算翻倍是最值得先试的一项 |
| DAPO 完整配方 | 未完成（35 step 停） | 关掉或改进 Dynamic Sampling 后再比 |

> 结论一句话：**这个仓库给的不是 SOTA 分数，而是"把 SOTA 拆成能跑的 20 行 loss"的能力。**
> 想拿分数，得自己补规模与对照；想理解算法，这套代码已经足够。


# 3. 本地 GPU 复现实验记录

# 本地复现实验记录（实测数据）

> 环境：RTX 5060 Ti 17.1GB（cuda:0）｜torch 2.11.0+cu128｜Qwen2.5-1.5B-Instruct + LoRA(r=16)
> 数据：GSM8K 子集（训练池 300 题、dev 20 题，只保留答案 ≤3 位数、题干 ≤400 字符）
> 时间：2026-11-16，单卡串行 5 组，纯训练 **80.1 分钟**（另加模型加载与前后 eval 约 25 分钟）

---

## 1. 统一配置

| 项 | 值 |
|---|---|
| 每题采样条数 `group_size` | 4 |
| 每组题数 `groups_per_step` | 3（每步 12 条轨迹） |
| 训练步数 | 10 |
| `max_new_tokens` | 224 |
| `ppo_epochs`（内层更新次数） | 2 |
| LoRA | r=16, alpha=32, 7 个投影层 |
| 学习率 | 1e-5, AdamW |
| micro-batch | 2（显存不够就降 1） |
| reward | ±1（对 +1 / 错 −1），DAPO 额外叠长度惩罚 |

> `ppo_epochs=2` 是刻意的：单轮更新时 ratio ≡ 1，三种 loss 会给出**完全相同**的梯度，
> 看不到任何差异。第二轮开始权重已动，ratio 才偏离 1，clip 才有机会生效。

**可比性说明**：题目按 `seed=42` 打乱后**顺序消费**（`trainer._cursor`），10 步共消费 30 题，
远小于训练池，所以各运行吃到的是**完全相同的 30 道题**；dev 也是同一批 20 题。
五次运行的 base dev 准确率全为 **40%**，可反证数据路径一致。

---

## 2. 五组结果

| 运行 | 算法 | dev 训练前 | dev 训练后 | rollout 准确率（前3步→后3步） | 退化组比例 | 平均 loss | 秒/步 |
|---|---|---:|---:|---|---:|---:|---:|
| `grpo_ppo` | GRPO + 对称 PPO clip [0.8,1.2] | 40% | **60%** | 0.333 → **0.417** | 0.30 | −0.0005 | 100 |
| `grpo_cispo` | GRPO + CISPO | 40% | 35% | 0.333 → 0.361 | 0.40 | −0.0378 | 102 |
| `dapo` | DAPO 四项全开 | 40% | 30% | 0.472 → **0.583** | **0.00** | +0.1749 | **131** |
| `gspo` | GSPO 序列级 ratio | 40% | 55% | 0.306 → 0.417 | 0.37 | −0.0120 | 97 |
| `grpo_is` | GRPO + 无裁剪 IS | 40% | 45% | 0.389 → 0.333 | 0.37 | −0.0002 | 49* |

\* 前四组是同一会话内先后跑的（97~131 s/步，含机上其他负载）；`grpo_is` 是空闲时补跑的（45~66 s/步）。
**跨会话的秒/步不可比**，只有同一会话内的 100 vs 131 才能比。 — |

⚠️ **dev 只有 20 题，错 1 题就是 5 个百分点**，所以"训练后"这一列基本是噪声，不能当结论。
真正有信息量的是后三列：rollout 内准确率趋势、退化组比例、时间成本。

---

## 3. 三条可复现的实测结论

### 3.1 DAPO 的 Dynamic Sampling 确实把退化组降到 0，但要多付 31% 墙钟

| 指标 | GRPO (`grpo_ppo`) | DAPO (`dapo`) |
|---|---:|---:|
| 退化组比例 | 0.30 | **0.00** |
| 每步被丢弃的候选组数 | 0 | **1.40** |
| 秒/步 | 100 | **131** |
| rollout 准确率（后3步） | 0.417 | **0.583** |

- 过滤机制按设计生效：`degenerate_frac` 恒为 0，rollout 内准确率也是五组里最高的；
- 代价可量化：**每步平均丢弃 1.4 个已生成的候选组**（batch 只有 3 组，等于 +47% 生成量），
  墙钟从 100s 涨到 131s；
- 这与原仓库 DAPO 章"35 step 花了 4.17 小时、作者手动叫停"的记录是同一个现象：
  **被丢弃的 completion 已经付过完整生成成本，只是不进 loss。**

### 3.2 单轮更新时 ratio ≡ 1，三种 loss 完全等价；差异只在第二轮出现

实测值（10 个 step 的平均，`ratio_mean_e0` / `ratio_mean_e1`）：

| 运行 | inner epoch 1 | inner epoch 2 | clip 比例（epoch 2） |
|---|---:|---:|---:|
| `grpo_is`（无裁剪） | **1.000000** | **1.001318** | —（无 clip） |
| `grpo_ppo` | 1.000000 | 1.001123 | 0.42%（token 级） |
| `dapo` | 1.000000 | 1.000742 | 0.15%（token 级） |
| `grpo_cispo` | 1.000000 | 1.000623 | —（detach 权重） |
| `gspo` | 1.000000 | 1.000365（序列级） | **92.96%（序列级）** |

- 第 1 epoch 所有 ratio **精确等于 1.000000** —— 这是 on-policy 的定义结果，不是 bug；
- **无限速的 `grpo_is` 漂移最大（1.001318）**，与「没有 clip 约束、更新走得更远」的理论一致；
  但其余四组因数据（DAPO 动态采样）与比率定义（GSPO 序列级）不同，**不能横向排序**；
- 第 2 epoch ratio 只漂到 1.0004~1.0013，**token 级 clip 比例不到 0.5%**，
  所以 10 步规模下三种 loss 的差异主要来自 **loss 的量级**，而不是 clip 本身：
  PPO 的 loss 量级 ~1e-4，CISPO ~1e-2（它优化的是 `weight × logp × A`，logp 本身量级就有 1）。
- ⚠️ 唯一例外是 **GSPO**：3e-4 的极窄阈值让 **92.96%** 的序列在第 2 epoch 被裁，
  梯度几乎全靠第 1 epoch —— 详见 §5.3 ④。
- 想真正看到 clip 起作用，必须让 ratio 大幅偏离 1：**更大的 lr、更多 inner epoch、或陈旧 sampler**。

### 3.3 退化组是这套配置的主要瓶颈

- 三组 GRPO 跑下来退化组比例 **0.30~0.40**：每 10 组里有 3~4 组全对或全错，采了白采；
- 1.5B 模型在 GSM8K 子集上 base 准确率约 40%，正处在"一半会一半不会"的区间，
  所以退化主要来自"全错"而不是"全对"；
- 这也解释了为什么 10 步内 rollout 准确率抖动这么大（每步只有 12 条样本）。

---

## 4. 一次误判及其修正：`grpo_is` 的"崩溃"其实是代码 bug

这一节保留的是**排查过程中真踩过的坑**，因为它比顺利的那部分更有价值。

### 4.1 现象

第一次跑 `grpo_is`（无裁剪的重要性采样）10 步时，进程直接退出、没有产出日志文件。
grep 只截到日志尾部，看不到完整 traceback，当时把它归因为
**"无 clip 的重要性采样在 ratio 漂移 + 小 batch 下梯度不稳定"** —— 这个归因是**错的**。

### 4.2 真正的根因

补做取证时拿到了完整 traceback：

```
TypeError: importance_sampling_loss() got an unexpected keyword argument 'clip_low'
  File "local_repro/trainer.py", line 378, in train
    loss, lm = loss_fn(new_lp, old_lp, adv_chunk, new_mk)
  File "local_repro/losses.py", line 381, in _call
    return fn(logp_new, logp_old, advantages, mask, **kwargs)
```

`build_loss` 为了让调用方不必区分参数，把 `clip_low / clip_high / reduction` 统一透传给
所有 loss 函数；但 `importance_sampling_loss` 根本没有 clip 参数，于是直接 `TypeError`。
**这是参数透传 bug，跟数值稳定性毫无关系。**

### 4.3 修复

按函数签名过滤多余的 kwargs，而不是无脑透传：

```python
fn = LOSS_REGISTRY[name]
accepted = set(inspect.signature(fn).parameters)

def _call(logp_new, logp_old, advantages, mask, teacher_logp=None):
    safe_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    ...
```

修完之后 `grpo_is` 正常跑完 10 步，见 §2 表格 —— **它没有崩，只是 loss 量级和 PPO 不同**。

### 4.4 两条教训（迁移到任何 RL 项目都成立）

1. **报错被截断时不要凭现象编根因。** 这次错的不是数值，是 `TypeError`；
   如果当时按"IS 不稳定"去调 lr / 加 clip，就会在错误方向上浪费几小时。
   **先拿完整 traceback，再谈根因。**
2. **"无裁剪必然崩"在这个尺度上本来就站不住。**
   §5 实测：ratio 均值 1.0028 时，无裁剪 IS 与 PPO 的梯度范数只差 **0.12%**，
   PPO 也只裁掉 0.39% 的 token。真正让 IS 危险的是 ratio 大幅偏离 1（大 lr、陈旧 sampler），
   而不是本实验这种 1e-5 lr + 10 步的小规模。

---

## 5. 受控对照：同一批数据上五种 loss 的梯度

数据来自 `logs/loss_headtohead.json`，图见 `figures/08-五种loss梯度对照.png`。

做法：采一批固定 rollout（3 组 × 4 条），**先做 1 次真实更新制造策略漂移**，
然后在同一批数据、同一初始权重上分别计算 IS / PPO / PPO+ClipHigher / CISPO / GSPO 的
loss 值、梯度范数、被裁剪比例。因为它不更新参数，所以是严格受控对比。

### 5.1 实验条件

| 项 | 值 |
|---|---|
| 采样规模 | 3 组题 × 4 条 = **12 条轨迹，1277 个 token**，平均长度 106.4 |
| 这批数据的准确率 | 0.583 |
| 退化组 | **2 组（3 组里）** → 实际只有 1 组贡献梯度信号 |
| 制造漂移 | 先做 1 次真实更新（lr=1e-5），参数已动但本实验不再更新 |
| 漂移幅度 | log-ratio 均值 **+0.00205**、std **0.03734**、max +0.2445、min −0.2114 |
| 硬件 | RTX 5060 Ti 16GB，micro-batch=2 分块，全程 2 分 03 秒 |

### 5.2 结果

| 算法 | loss 值 | 梯度范数 | 相对 IS | 被裁剪比例 | ratio 均值 |
|---|---:|---:|---:|---:|---:|
| 无裁剪 IS | **+0.1314** | 2.8483e-01 | 1.0000× | —（无 clip） | 1.002756 |
| PPO | +0.1314 | 2.8449e-01 | 0.9988× | 0.39%（token 级） | 1.002756 |
| PPO + Clip-Higher | +0.1314 | 2.8485e-01 | 1.0001× | **0.00%** | 1.002756 |
| CISPO | **−0.0238** | 2.8486e-01 | 1.0001× | —（detach 权重） | 1.002756 |
| GSPO | +0.0535 | **1.8280e-01** | **0.6418×** | **73.0%**（序列级） | 1.002786 |

### 5.3 四条结论

**① loss 的绝对值不能跨算法比较 —— 这是最容易踩的坑。**
IS/PPO/ClipHigher 都是 +0.1314，CISPO 是 **−0.0238**，符号都相反，量级差 6.5 倍；
但它们的**梯度范数全部落在 2.8449e-01 ~ 2.8486e-01**，相对差 **不到 0.15%**。
原因是这几种 loss 只差一个常数倍或一个 `detach`，`d/dθ` 之后被约掉。
所以「我的 loss 比你的小」毫无意义，**要比就比梯度范数、比更新后的 reward**。

**② ratio ≡ 1 附近，PPO 和 IS 的梯度几乎完全一致。**
PPO 只裁掉 **0.39%** 的 token，所以它的梯度范数只比无裁剪 IS 低 0.12%。
这说明在小规模、单轮更新下，「加不加 clip」不是主要矛盾 ——
**原仓库 `loss-functions` 章说「三种 loss 在同一 batch 上梯度相同」，这次是实测坐实了**。

**③ Clip-Higher 在这批数据上完全没生效（clip_frac = 0）。**
DAPO 把上界从 1.2 放宽到 1.28，但本批数据里 ratio 最大只到 **1.2075**，
刚好卡在 1.2 和 1.28 之间 —— 于是对称 PPO 会裁掉的那部分，Clip-Higher 全放过了。
**这解释了为什么 Clip-Higher 在 10 步小实验里"看起来没用"**：它不是没用，
是**需要更大的漂移才能触发**（论文里是几十到上百步、更大 lr 的场景）。
小实验得不出「Clip-Higher 无效」的结论，只能得出「这个尺度下触发不到」。

**④ GSPO 的梯度只有别人的 64%，因为它裁掉了 73% 的序列。**
GSPO 用序列级几何平均 ratio，阈值只有 **3e-4**（比 PPO 的 0.2 小 667 倍）。
一次更新后序列 ratio 均值 1.002786，已经**超出阈值 9 倍**，
于是 73% 的序列被裁、梯度被置零，只剩 27% 在贡献。
（完整 10 步训练里这个比例更高，平均 **92.96%** —— 样本量 12 条时是 73%，100+ 条时收敛到 93%，同一现象。）
这不是 bug —— 论文里 GSPO 用于大规模训练，单步序列 ratio 漂移在 1e-4 量级，
3e-4 是匹配的；**本实验 lr=1e-5 + 2 个 inner epoch 的漂移更大，把它撑爆了**。
⚠️ 迁移结论：**用 GSPO 时 inner epoch 必须设 1，或者把 lr 调小到序列 ratio 漂移 < 3e-4**。

---

## 6. 与原仓库的对照关系

| 维度 | 原仓库 | 本仓库 | 可对照性 |
|---|---|---|---|
| 模型 | Qwen3.5-4B | Qwen2.5-1.5B | ❌ 绝对值不可比 |
| 训练栈 | PyTRIO 云端 | 本地 HF+peft | ✅ 算法等价 |
| reward | ±1 + math_verify | ±1 + sympy 判定 | 🟡 判定实现不同，口径一致 |
| 步数 | 100~300 | 10 | ❌ 未收敛 |
| 评测 | AIME25（30×12）/ DRB（100） | GSM8K dev（20） | ❌ 规模不可比 |
| 可复现性 | 需账号额度 | 一张卡即可 | ✅ 本仓库优势 |


# 4. 原仓库入口可运行性核查

# 原仓库可运行性核查（本机实测）

> 用 `python <入口> --help` 探测：能出帮助说明脚本级依赖齐备；
> 真正跑训练还需要 PyTRIO 账号额度与各章数据，这一层在本机未执行。

| 章节 | 入口 | --help | 额外依赖 | 备注 |
| --- | --- | --- | --- | --- |
| getting-started | `-` | n/a | - | 环境与概念（纯文档） |
| loss-functions | `-` | n/a | - | loss 直觉（纯文档） |
| grpo | `01-demo-sync.py` | ✅ | PyTRIO（GSM8K 可自动下载） |                         SwanLab project |
| vision-grpo | `train.py` | ✅ | PyTRIO + GeoQA 图片数据 |   --show-samples |
| general-opd | `01-demo-sync.py` | ✅ | PyTRIO + DeepMath-103K |   --swanlab-mode {online,local,offline,disabled} |
| medical-opd | `02-medical-sft.py` | ✅ | PyTRIO + 医疗数据集（ModelScope/HF） |                         SwanLab 运行模式 |
| dapo | `train.py` | ✅ | PyTRIO（DAPO-Math-17k 可自动下载） |                         SwanLab 实验记录模式。 |
| gspo | `train.py` | ✅ | PyTRIO（数学数据自动下载） |                         SwanLab 实验记录模式。 |
| opsd | `01-opsd-sync.py` | ✅ | PyTRIO + OPSD/AIME25 数据 |   --swanlab-mode {online,local,offline,disabled} |
| agentopsd | `train.py` | ❌ | PyTRIO + ALFWorld + SkillBank | ModuleNotFoundError: No module named 'textworld' |
| search-r1 | `train.py` | ✅ | PyTRIO + DEEPSEEK_API_KEY（在线搜索后端） |                         SwanLab 日志模式 |
| retool | `train.py` | ✅ | PyTRIO + 本地代码沙箱 |                         SwanLab 日志模式 |
| alfworld | `train.py` | ❌ | PyTRIO + alfworld==0.4.2 + spacy（TextWorld 环境） | ModuleNotFoundError: No module named 'textworld' |
| tempo | `train.py` | ❌ | PyTRIO + TextWorld 存档读档 | ModuleNotFoundError: No module named 'textworld' |
| spec-o3 | `train_rl.py` | ✅ | PyTRIO + 光谱数据 + 图片 token 支持 |   --swanlab-mode {online,local,disabled} |
| tournament-grpo | `train.py` | ❌ | PyTRIO + judge 模型 key | ModuleNotFoundError: No module named 'fcntl' |
| pytrio-jev | `train.py` | ✅ | PyTRIO + Jev 决策数据 |                         默认跟随 --base-model |

## 结论

原仓库所有入口脚本都依赖 `pytrio` 客户端，训练与采样在云端执行，
本机只承担数据、reward 与 loss 计算。因此「用本机 GPU 复现」必须补上训练器，
这正是本仓库 `local_repro/trainer.py` 的作用。


# 5. 逐章精读笔记（索引）

# 原始精读笔记（17 章）

这四份文件是逐章精读原仓库 readme / start.md / 源码后的结构化笔记，
字段统一为：**论文出处 → 痛点 → 核心公式（逐符号解释）→ reward/teacher 信号设计 →
关键实现细节 → 原文结果数字 → 坑与边界 → 一句话记忆锚点**。

主文档 [../知识点总结.md](../知识点总结.md) 与 [../SOTA汇总.md](../SOTA汇总.md) 就是从这里合成的。
如果想核对某个数字的出处，直接在这四份笔记里搜。

| 文件 | 覆盖章节 |
|---|---|
| [A-grpo系列.md](A-grpo系列.md) | GRPO、Vision GRPO、Search-R1、ReTool |
| [B-OPD蒸馏系列.md](B-OPD蒸馏系列.md) | General OPD、Medical OPD（SAR/IDT）、OPSD、AgentOPSD |
| [C-DAPO等.md](C-DAPO等.md) | DAPO、GSPO、TEMPO、ALFWorld |
| [D-多模态等.md](D-多模态等.md) | Spec-o3、Tournament-GRPO、PyTRIO-Jev、基础篇（IS/PPO/CISPO + 概念词汇表） |

> 约定：笔记里出现的所有数字都来自原仓库原文，未做换算；原文没给定量结果的章节会明确写"该章未给定量结果"。


---

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


---

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


---

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


---

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
