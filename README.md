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
