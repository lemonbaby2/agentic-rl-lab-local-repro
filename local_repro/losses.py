"""策略梯度损失函数库（纯 PyTorch 实现，逐条对应论文公式）。

设计原则
--------
原仓库 datawhalechina/agentic-rl-lab 的 loss 有三种来源：

1. PyTRIO 内置：``importance_sampling``、``ppo`` —— 在云端算，本地看不到代码；
2. 本地 custom loss：``cispo``、``gspo``、DAPO 变体 —— 在本地 autograd 计算；
3. 蒸馏类：``opd`` / ``opsd`` —— advantage 不由 reward 给出，而由 teacher logprob 给出。

为了能在**本机 GPU** 上复现，这里把上述所有 loss 统一重写为纯 PyTorch 函数，
使训练循环可以完全跑在本地，不依赖任何远端 Train API。

统一符号（与仓库 experiments/loss-functions/readme.md 一致）
-----------------------------------------------------------
- ``logp_new``      : 当前策略 π_θ 对 token 的 logprob（可求导，target_logprobs）
- ``logp_old``      : 采样时策略 π_old 的 logprob（不可求导，sampling_logprobs）
- ``advantages``    : 逐 token 的优势 A（由序列级 advantage 广播而来）
- ``mask``          : completion 掩码，1 表示"这是模型生成的 token"

    ratio_t = exp(logp_new_t - logp_old_t)

所有函数都返回 ``(loss, metrics)``，loss 已经带上负号（PyTorch 最小化）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

LossReturn = Tuple[torch.Tensor, Dict[str, float]]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _safe_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在 mask=1 的位置求均值，避免 padding 稀释分母。"""
    denom = mask.sum().clamp(min=1.0)
    return (x * mask).sum() / denom


def _reduce_token_level(
    token_obj: torch.Tensor, mask: torch.Tensor, reduction: str
) -> torch.Tensor:
    """把一个 batch 的逐 token 目标聚合成标量 loss 目标（未加负号）。

    reduction 语义直接对应 DAPO 论文第 3 项改动：

    - ``"sequence"``：先对每条回答内的 token 求均值，再对回答求均值
      （GRPO / PPO 的朴素做法，长短回答权重相同）；
    - ``"token"``   ：把所有 completion token 拉平后统一求均值
      （DAPO 的 token-level policy gradient loss，长回答权重更大）。
    """
    if reduction == "token":
        return _safe_mean(token_obj, mask)
    if reduction == "sequence":
        per_seq = (token_obj * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return per_seq.mean()
    raise ValueError(f"unknown reduction: {reduction}")


def compute_ratio(logp_new: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
    """重要性采样比率 ratio = exp(logp_new - logp_old)。

    数值上先 clamp 差值再 exp，避免极端值把梯度炸到 inf。
    """
    log_diff = (logp_new - logp_old).clamp(-20.0, 20.0)
    return torch.exp(log_diff)


# --------------------------------------------------------------------------- #
# 1. Importance Sampling
# --------------------------------------------------------------------------- #
def importance_sampling_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "token",
) -> LossReturn:
    r"""最朴素的重要性采样策略梯度。

    .. math::
        \mathcal{L}_{\text{IS}}(\theta) =
        -\mathbb{E}\left[\frac{p_\theta(x)}{q(x)} A(x)\right]

    advantage > 0 的 token 概率被推高，< 0 的被压低；ratio 只用于纠正
    "数据由旧策略采样、但正在训练新策略"这一偏差。没有任何限速。
    """
    ratio = compute_ratio(logp_new, logp_old)
    token_obj = ratio * advantages
    objective = _reduce_token_level(token_obj, mask, reduction)
    metrics = {
        "ratio_mean": _safe_mean(ratio.detach(), mask).item(),
        "ratio_max": (ratio.detach() * mask).max().item(),
        "ratio_min": ((ratio.detach() + (1 - mask) * 1e9) * mask + (1 - mask) * 0).min().item()
        if mask.sum() > 0
        else 1.0,
    }
    return -objective, metrics


# --------------------------------------------------------------------------- #
# 2. PPO（支持 Clip-Higher 的非对称裁剪）
# --------------------------------------------------------------------------- #
def ppo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    reduction: str = "token",
) -> LossReturn:
    r"""PPO 的 clipped surrogate objective。

    .. math::
        \mathcal{L} = -\mathbb{E}\Big[\min\big(\rho A,\;
        \mathrm{clip}(\rho, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}) A\big)\Big]

    ``clip_low == clip_high`` 时就是 GRPO 常用的对称区间 [0.8, 1.2]；
    DAPO 的 **Clip-Higher** 取 ``clip_low=0.2, clip_high=0.28``，
    即区间 [0.8, 1.28]，给正 advantage 且概率很低的 token 更多上升空间，
    缓解 entropy collapse。
    """
    ratio = compute_ratio(logp_new, logp_old)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    clipped = clipped_ratio * advantages
    token_obj = torch.min(unclipped, clipped)
    objective = _reduce_token_level(token_obj, mask, reduction)

    with torch.no_grad():
        was_clipped = ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)).float()
        clip_frac = _safe_mean(was_clipped, mask).item()
    metrics = {
        "ratio_mean": _safe_mean(ratio.detach(), mask).item(),
        "clip_frac": clip_frac,
    }
    return -objective, metrics


# --------------------------------------------------------------------------- #
# 3. CISPO
# --------------------------------------------------------------------------- #
def cispo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    reduction: str = "token",
) -> LossReturn:
    r"""CISPO：裁剪 ratio 但不裁剪梯度。

    .. math::
        \mathcal{L} = -\mathbb{E}\Big[\mathbf{sg}\big(\mathrm{clip}(\rho,
        1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big)\cdot
        \log p_\theta(x)\cdot A\Big]

    ``sg`` 即 ``detach()``：被裁剪后的 ratio 只当**固定砝码**决定每个 token
    的学习力度，真正参与梯度回传的是 ``logp_new`` 本身。

    这与 PPO 的本质区别：PPO 里被裁掉的 token 梯度直接归零（min 取到常数项），
    CISPO 里被裁掉的 token 仍然保留一个**有上限**的梯度。
    """
    ratio = compute_ratio(logp_new, logp_old)
    with torch.no_grad():
        weight = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    token_obj = weight * logp_new * advantages
    objective = _reduce_token_level(token_obj, mask, reduction)
    metrics = {"ratio_mean": _safe_mean(ratio.detach(), mask).item()}
    return -objective, metrics


# --------------------------------------------------------------------------- #
# 4. GSPO（序列级重要性比率）
# --------------------------------------------------------------------------- #
def gspo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 3e-4,
    reduction: str = "token",
) -> LossReturn:
    r"""GSPO：把重要性比率从 token 级提升到 sequence 级。

    .. math::
        s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}
        \right)^{1/|y_i|}
        = \exp\left(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta}{\pi_{\text{old}}}\right)

        \mathcal{L} = -\frac{1}{G}\sum_i \min\big(s_i A_i,\;
        \mathrm{clip}(s_i, 1-\epsilon, 1+\epsilon) A_i\big)

    三个关键点：

    1. **长度归一化**：先对 log-ratio 在序列内取平均再 exp，所以 s_i 的量级
       与序列长度无关（普通 token 级 ratio 的连乘会随长度指数爆炸）；
    2. **裁剪阈值极小**：官方用 eps=3e-4（token 级 GSPO-token 用 4e-4），
       因为 s_i 已经是几何平均意义下的比率，偏离 1 一点点就代表很大的策略漂移；
    3. **梯度仍按 token 分配**：s_i 对每个 token 的梯度带有 1/|y_i| 系数，
       长序列被自动降权。

    ``reduction`` 在这里只影响 metrics 的加权方式，序列级 loss 本身已按样本求均值。
    """
    # 每条序列的 log-ratio 均值（长度归一化）
    seq_log_ratio = (logp_new - logp_old).clamp(-20.0, 20.0) * mask
    seq_log_ratio = seq_log_ratio.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    seq_ratio = torch.exp(seq_log_ratio)  # (B,)

    # 序列级 advantage：取该序列内任意一个 mask=1 的 token 上的值即可
    seq_adv = (advantages * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

    unclipped = seq_ratio * seq_adv
    clipped = torch.clamp(seq_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * seq_adv
    objective = torch.min(unclipped, clipped).mean()

    with torch.no_grad():
        clip_frac = (
            ((seq_ratio < 1.0 - clip_eps) | (seq_ratio > 1.0 + clip_eps))
            .float()
            .mean()
            .item()
        )
    metrics = {
        "seq_ratio_mean": seq_ratio.detach().mean().item(),
        "seq_clip_frac": clip_frac,
    }
    return -objective, metrics


# --------------------------------------------------------------------------- #
# 5. On-Policy Distillation（OPD / OPSD / AgentOPSD 共用）
# --------------------------------------------------------------------------- #
def opd_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    teacher_logp: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "token",
    kl_coef: float = 1.0,
) -> LossReturn:
    r"""On-Policy Distillation：把 teacher 的逐 token 打分当 advantage。

    OPD 的信号不是 reward，而是"老师会怎么写这段回答"。对 student 自己采样
    出的轨迹，逐 token 计算

    .. math::
        A_t = \mathrm{sg}\big(\log \pi_{\text{teacher}}(y_t) -
        \log \pi_{\text{student}}(y_t)\big)

    这正是 reverse KL 的逐 token 形式：teacher 概率高于 student 的 token 拿到
    正 advantage（该被推高），低于的拿到负 advantage（该被压低）。
    student 采样保证数据始终 on-policy，teacher 只提供"批改意见"。

    对 OPSD / AgentOPSD，``teacher_logp`` 来自：
    - OPSD：step-0 的自己 + prompt 里附带的参考解答；
    - AgentOPSD：同一模型带 Skill 复评同一条轨迹（Self-Teacher），
      turn 级证据递推后再做有界微调。
    """
    with torch.no_grad():
        adv = (teacher_logp - logp_new.detach()).clamp(-20.0, 20.0)

    ratio = compute_ratio(logp_new, logp_old)
    token_obj = ratio * adv * kl_coef
    objective = _reduce_token_level(token_obj, mask, reduction)

    with torch.no_grad():
        reverse_kl = _safe_mean(adv, mask).item()  # E[log p_t - log p_s]
    metrics = {"reverse_kl_est": reverse_kl, "ratio_mean": _safe_mean(ratio.detach(), mask).item()}
    return -objective, metrics


# --------------------------------------------------------------------------- #
# 6. 长度惩罚（DAPO Soft Overlong Punishment）
# --------------------------------------------------------------------------- #
def soft_overlong_penalty(
    lengths: torch.Tensor, max_tokens: int, overlong_cache: int
) -> torch.Tensor:
    r"""DAPO 的软超长惩罚。

    .. code-block:: text

        length <= penalty_start          : 0
        penalty_start < length <= max    : -(length - penalty_start) / cache
        length > max_tokens              : -1

    动机：如果 completion 在 max_tokens 处被硬截断，往往来不及输出 ``\boxed{}``，
    reward 会从"可能正确"突变成"错误"，这个断崖式噪声会误导优化。线性坡道让
    模型提前收到"别再长了"的信号。
    """
    penalty_start = max_tokens - overlong_cache
    lengths = lengths.float()
    ramp = -(lengths - penalty_start) / float(overlong_cache)
    penalty = torch.where(lengths <= penalty_start, torch.zeros_like(ramp), ramp)
    penalty = torch.where(lengths > max_tokens, torch.full_like(penalty, -1.0), penalty)
    return penalty.clamp(min=-1.0, max=0.0)


# --------------------------------------------------------------------------- #
# 7. 组内相对 advantage（GRPO 核心）
# --------------------------------------------------------------------------- #
def group_relative_advantage(
    rewards: torch.Tensor, group_size: int, normalize_std: bool = False, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""把 (B*G,) 的 reward 变成组内相对 advantage。

    .. math::
        A_i = r_i - \mathrm{mean}(r_1..r_G)
        \quad\text{或}\quad
        A_i = \frac{r_i - \mathrm{mean}}{\mathrm{std}}

    ``normalize_std=True`` 对应论文版（DeepSeekMath），本仓库 demo 为了逻辑清晰
    只减均值。返回 ``(advantages, degenerate_mask)``，后者标记"整组 reward 完全相同"
    的退化组 —— 这些组 advantage 全 0，采了也白采，正是 DAPO Dynamic Sampling
    要剔除的对象。
    """
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    centered = grouped - mean
    if normalize_std:
        std = grouped.std(dim=1, keepdim=True, unbiased=False)
        centered = centered / (std + eps)
    degenerate = (grouped.max(dim=1).values - grouped.min(dim=1).values) < eps
    return centered.reshape(-1), degenerate


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
LOSS_REGISTRY = {
    "importance_sampling": importance_sampling_loss,
    "ppo": ppo_loss,
    "cispo": cispo_loss,
    "gspo": gspo_loss,
    "opd": opd_loss,
}


@dataclass
class LossConfig:
    """一次性描述某个算法的 loss 开关，对应仓库里的 preset。"""

    name: str
    clip_low: float = 0.2
    clip_high: float = 0.2
    reduction: str = "token"
    clip_eps: float = 3e-4  # GSPO 专用


PRESETS: Dict[str, LossConfig] = {
    # GRPO + 对称 PPO clip，序列级平均（仓库 grpo/03-grpo.py 的默认口径）
    "grpo_ppo": LossConfig("ppo", 0.2, 0.2, "sequence"),
    # GRPO + 无裁剪 IS
    "grpo_is": LossConfig("importance_sampling", reduction="sequence"),
    # GRPO + CISPO
    "grpo_cispo": LossConfig("cispo", 0.2, 0.2, "sequence"),
    # DAPO 四项改动：Clip-Higher + token mean
    "dapo": LossConfig("ppo", 0.2, 0.28, "token"),
    # GSPO：序列级 ratio + 极窄 clip
    "gspo": LossConfig("gspo", clip_eps=3e-4),
}


def build_loss(name: str, **kwargs):
    """按名字取 loss 函数，方便训练脚本切换算法。

    会自动按函数签名过滤多余的 kwargs —— 例如统一传入
    ``clip_low/clip_high/reduction`` 时，``importance_sampling_loss`` 只接
    ``reduction``，``gspo_loss`` 只接 ``clip_eps``，多传的直接丢掉而不是报错。
    """
    import inspect

    if name not in LOSS_REGISTRY:
        raise KeyError(f"unknown loss {name}, available: {list(LOSS_REGISTRY)}")
    fn = LOSS_REGISTRY[name]
    accepted = set(inspect.signature(fn).parameters)

    def _call(logp_new, logp_old, advantages, mask, teacher_logp=None):
        safe_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        if name == "opd":
            return fn(logp_new, logp_old, teacher_logp, mask, **safe_kwargs)
        return fn(logp_new, logp_old, advantages, mask, **safe_kwargs)

    return _call
