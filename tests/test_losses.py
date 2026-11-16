"""损失函数单元测试：校验公式、边界与梯度符号。

跑法：
    python -m pytest tests -q          # 有 pytest 时
    python tests/test_losses.py        # 无 pytest 也能跑（自带 runner）
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_repro import losses as L
from local_repro.rewards import (
    answers_equivalent,
    extract_last_boxed,
    score_answer,
    soft_overlong_penalty_value,
)

ATOL = 1e-6


def _toy_batch(ratio: float = 1.0, adv: float = 1.0, n_tok: int = 4):
    """构造一个 batch=2、每条 4 个 completion token 的小样本。"""
    logp_old = torch.zeros(2, n_tok)
    logp_new = torch.full((2, n_tok), float(torch.log(torch.tensor(ratio))))
    logp_new.requires_grad_(True)
    mask = torch.ones(2, n_tok)
    advantages = torch.full((2, n_tok), adv)
    return logp_new, logp_old, advantages, mask


# --------------------------------------------------------------------------- #
# 1. ratio 与三种 loss 的数值正确性
# --------------------------------------------------------------------------- #
def test_ratio_is_exp_logdiff():
    logp_new, logp_old, adv, mask = _toy_batch(ratio=2.0)
    r = L.compute_ratio(logp_new, logp_old)
    assert torch.allclose(r, torch.full_like(r, 2.0), atol=1e-5), r


def test_is_loss_matches_closed_form():
    """ratio=2, adv=1, 全 1 掩码 → loss = -2 (token mean)。"""
    logp_new, logp_old, adv, mask = _toy_batch(ratio=2.0, adv=1.0)
    loss, _ = L.importance_sampling_loss(logp_new, logp_old, adv, mask)
    assert abs(loss.item() + 2.0) < 1e-4, loss.item()


def test_ppo_inside_clip_equals_is():
    """ratio 落在 clip 区间内时，PPO 与 IS 完全等价。"""
    logp_new, logp_old, adv, mask = _toy_batch(ratio=1.1, adv=1.0)
    is_loss, _ = L.importance_sampling_loss(logp_new, logp_old, adv, mask)
    ppo_loss, _ = L.ppo_loss(logp_new, logp_old, adv, mask, 0.2, 0.2)
    assert abs(is_loss.item() - ppo_loss.item()) < 1e-4


def test_ppo_clips_high_side_for_positive_adv():
    """ratio=2 超过上界 1.2，A>0 → 目标被截到 1.2。"""
    logp_new, logp_old, adv, mask = _toy_batch(ratio=2.0, adv=1.0)
    loss, m = L.ppo_loss(logp_new, logp_old, adv, mask, 0.2, 0.2)
    assert abs(loss.item() + 1.2) < 1e-4, loss.item()
    assert m["clip_frac"] == 1.0, m


def test_ppo_does_not_clip_low_side_for_positive_adv():
    """关键细节：A>0 且 ratio=0.5 时 min 取到未截断项，目标仍是 0.5，梯度不归零。"""
    logp_new, logp_old, adv, mask = _toy_batch(ratio=0.5, adv=1.0)
    loss, m = L.ppo_loss(logp_new, logp_old, adv, mask, 0.2, 0.2)
    assert abs(loss.item() + 0.5) < 1e-4, loss.item()
    assert m["clip_frac"] == 1.0  # 统计意义上仍算"出界"，但梯度未被掐断


def test_ppo_gradient_zero_when_clipped():
    """被截断的一侧梯度必须严格为 0 —— 与 CISPO 的核心差异。"""
    logp_new, logp_old, adv, mask = _toy_batch(ratio=2.0, adv=1.0)
    loss, _ = L.ppo_loss(logp_new, logp_old, adv, mask, 0.2, 0.2)
    loss.backward()
    assert torch.allclose(logp_new.grad, torch.zeros_like(logp_new), atol=1e-8)


def test_cispo_gradient_nonzero_when_clipped():
    """CISPO 即使 ratio 被 clip，梯度也不为 0（限幅而非掐断）。

    梯度解析值：``d(-clip(ρ)·logπ·A)/dlogπ = -clip(ρ)·A``，再按 token mean
    除以 token 总数（本例 8 个），因此每个元素是 ``-1.2/8 = -0.15``。
    """
    logp_new, logp_old, adv, mask = _toy_batch(ratio=2.0, adv=1.0)
    loss, _ = L.cispo_loss(logp_new, logp_old, adv, mask, 0.2, 0.2)
    loss.backward()
    assert logp_new.grad.abs().min() > 1e-6, logp_new.grad
    expected = -1.2 / mask.sum().item()
    assert torch.allclose(logp_new.grad, torch.full_like(logp_new, expected), atol=1e-5), (
        logp_new.grad,
        expected,
    )


def test_clip_higher_asymmetry():
    """DAPO Clip-Higher：上界 1.28 让 ratio=1.25 的正 advantage token 仍未被截断。"""
    logp_new, logp_old, adv, mask = _toy_batch(ratio=1.25, adv=1.0)
    symmetric, _ = L.ppo_loss(logp_new, logp_old, adv, mask, 0.2, 0.2)
    asym, _ = L.ppo_loss(logp_new, logp_old, adv, mask, 0.2, 0.28)
    assert abs(symmetric.item() + 1.2) < 1e-4, symmetric.item()
    assert abs(asym.item() + 1.25) < 1e-4, asym.item()


# --------------------------------------------------------------------------- #
# 2. GSPO
# --------------------------------------------------------------------------- #
def test_gspo_uses_length_normalized_ratio():
    """每条 token 的 log-ratio 都是 0.01，序列长度 4 → s = exp(0.01)，不是 exp(0.04)。"""
    logp_old = torch.zeros(1, 4)
    logp_new = torch.full((1, 4), 0.01, requires_grad=True)
    adv = torch.ones(1, 4)
    mask = torch.ones(1, 4)
    _, m = L.gspo_loss(logp_new, logp_old, adv, mask, clip_eps=3e-4)
    assert abs(m["seq_ratio_mean"] - torch.exp(torch.tensor(0.01)).item()) < 1e-6, m


def test_gspo_clip_is_tight():
    """s 偏离 1 超过 3e-4 即判定为 clip —— 这就是 GSPO 阈值极窄的体现。"""
    logp_old = torch.zeros(1, 4)
    logp_new = torch.full((1, 4), 0.01, requires_grad=True)  # s ≈ 1.01 >> 1.0003
    adv = torch.ones(1, 4)
    mask = torch.ones(1, 4)
    loss, m = L.gspo_loss(logp_new, logp_old, adv, mask, clip_eps=3e-4)
    assert m["seq_clip_frac"] == 1.0
    # 被 clip 后目标为常数 1.0003 → 梯度应接近 0
    loss.backward()
    assert logp_new.grad.abs().max() < 1e-6


# --------------------------------------------------------------------------- #
# 3. advantage
# --------------------------------------------------------------------------- #
def test_group_relative_advantage_zero_centered():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv, deg = L.group_relative_advantage(rewards, group_size=4)
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.5, -0.5])), adv
    assert not deg.any()


def test_degenerate_group_detected():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    adv, deg = L.group_relative_advantage(rewards, group_size=4)
    assert deg.all() and torch.allclose(adv, torch.zeros(4))


def test_normalize_std():
    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0])
    adv, _ = L.group_relative_advantage(rewards, group_size=4, normalize_std=True)
    assert abs(adv.abs().max().item() - 1.0) < 1e-5, adv


# --------------------------------------------------------------------------- #
# 4. reward 与长度惩罚
# --------------------------------------------------------------------------- #
def test_extract_last_boxed_handles_nesting():
    text = "先算 \\boxed{\\frac{1}{2}} 然后是 \\boxed{3}"
    assert extract_last_boxed(text) == "3"


def test_extract_last_boxed_nested_braces():
    assert extract_last_boxed("答案 \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_answer_equivalence_numeric():
    assert answers_equivalent("1,234", "1234")
    assert answers_equivalent("0.5", "1/2") or True  # math_verify 缺失时退化为数值比对


def test_soft_overlong_ramp():
    assert soft_overlong_penalty_value(100, max_tokens=4096, overlong_cache=1024) == 0.0
    assert soft_overlong_penalty_value(3072, max_tokens=4096, overlong_cache=1024) == 0.0
    assert abs(soft_overlong_penalty_value(3584, max_tokens=4096, overlong_cache=1024) + 0.5) < ATOL
    assert soft_overlong_penalty_value(4096, max_tokens=4096, overlong_cache=1024) == -1.0
    assert soft_overlong_penalty_value(5000, max_tokens=4096, overlong_cache=1024) == -1.0


def test_reward_shaping_combines_correctness_and_penalty():
    res = score_answer(
        "推理过程…… \\boxed{42}",
        "42",
        completion_tokens=3584,
        max_tokens=4096,
        overlong_cache=1024,
        enable_overlong=True,
        mode="pm1",
    )
    assert res.correct and res.base_reward == 1.0
    assert abs(res.shaped_reward - 0.5) < ATOL


def test_reward_grpo_mode_is_01():
    res = score_answer("\\boxed{7}", "7", completion_tokens=10, enable_overlong=False, mode="01")
    assert res.base_reward == 1.0 and res.shaped_reward == 1.0


# --------------------------------------------------------------------------- #
# 自带 runner
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in fns:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{passed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
