"""可视化：既复刻仓库里那些"机制示意图"，也把本地训练日志画成对照曲线。

分三类
------
**机制图**（不需要训练，纯数学即可画）

- ``plot_loss_geometry``    ：三种 loss 在 ratio 变化时的目标值与梯度形态
- ``plot_soft_overlong``    ：DAPO 软超长惩罚的坡道
- ``plot_clip_higher``      ：对称 clip 与 Clip-Higher 的区间差异
- ``plot_gspo_ratio_scale`` ：token 级连乘 ratio 与序列级长度归一化 ratio 的差别

**结果图**（读 ``logs/*.json``）

- ``plot_training_curves`` ：reward / accuracy / entropy / length / clip_frac 对照
- ``plot_advantage_dist``  ：退化组比例
- ``plot_before_after``    ：训练前后 dev 准确率

**受控对照**（读 ``logs/loss_headtohead.json``）

- ``plot_headtohead``      ：同一批数据上五种 loss 的梯度范数、裁剪比例、平均 ratio

说明：轴标签刻意不用 LaTeX（``\\rho`` 之类在不同后端易出错），中文 + 纯文本符号更稳。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 统一风格：白底、无衬线、网格淡；中文字体按平台优先级回退
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
        "font.sans-serif": [
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
    }
)

PALETTE = {
    "importance_sampling": "#d94a45",
    "ppo": "#2f6fb3",
    "cispo": "#258f4b",
    "dapo": "#8e44ad",
    "gspo": "#e08b1a",
}


def color_for(tag: str):
    """按日志名猜颜色：grpo_is / grpo_ppo / grpo_cispo / dapo / gspo。"""
    low = tag.lower()
    if "cispo" in low:
        return PALETTE["cispo"]
    if low.endswith("_is") or "-is" in low or low == "is":
        return PALETTE["importance_sampling"]
    if "dapo" in low:
        return PALETTE["dapo"]
    if "gspo" in low:
        return PALETTE["gspo"]
    if "ppo" in low:
        return PALETTE["ppo"]
    return None


def _finish(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")
    return str(out)


# --------------------------------------------------------------------------- #
# 机制图
# --------------------------------------------------------------------------- #
def plot_loss_geometry(out, clip_low: float = 0.2, clip_high: float = 0.2) -> str:
    """ratio 从 0.2 到 3.0 时，三种 loss 的目标值与对 logprob 的梯度。

    令 ``logp_old = 0``，则 ``logp_new = log(ratio)``，于是

    - IS：``obj = ratio*A``，梯度 ``= ratio*A``，无上界；
    - PPO：``obj = min(ratio*A, clip(ratio)*A)``。``min`` 只截断一侧：
      A>0 时 ratio 过大（>1+eps）才被截断，A<0 时 ratio 过小（<1-eps）才被截断，
      被截断一侧梯度严格为 0；
    - CISPO：梯度 ``= clip(ratio)*A``，限幅但永不为 0。
    """
    ratios = np.linspace(0.2, 3.0, 1200)
    logp = np.log(ratios)
    lo, hi = 1.0 - clip_low, 1.0 + clip_high

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax = axes[0]

    for adv, ls in ((1.0, "-"), (-1.0, "--")):
        is_obj = ratios * adv
        ppo_obj = np.minimum(ratios * adv, np.clip(ratios, lo, hi) * adv)
        cispo_obj = np.clip(ratios, lo, hi) * logp * adv
        ax.plot(ratios, is_obj, ls, color=PALETTE["importance_sampling"], lw=2, alpha=0.9)
        ax.plot(ratios, ppo_obj, ls, color=PALETTE["ppo"], lw=2, alpha=0.9)
        ax.plot(ratios, cispo_obj, ls, color=PALETTE["cispo"], lw=2, alpha=0.9)

    ax.plot([], [], color=PALETTE["importance_sampling"], lw=2, label="IS")
    ax.plot([], [], color=PALETTE["ppo"], lw=2, label="PPO")
    ax.plot([], [], color=PALETTE["cispo"], lw=2, label="CISPO")
    ax.plot([], [], color="#555", lw=1.5, ls="-", label="A=+1")
    ax.plot([], [], color="#555", lw=1.5, ls="--", label="A=-1")
    ax.axvline(1.0, color="#999", lw=1, ls=":")
    ax.axvspan(lo, hi, color="#2f6fb3", alpha=0.07)
    ax.set_title("① 目标值：三个 loss 各自在优化什么")
    ax.set_xlabel("重要性比率 ratio")
    ax.set_ylabel("objective（未取负）")
    ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)

    ax = axes[1]
    for adv, ls in ((1.0, "-"), (-1.0, "--")):
        is_grad = ratios * adv
        ppo_grad = np.where(
            ((adv > 0) & (ratios > hi)) | ((adv < 0) & (ratios < lo)), 0.0, ratios * adv
        )
        cispo_grad = np.clip(ratios, lo, hi) * adv
        ax.plot(ratios, is_grad, ls, color=PALETTE["importance_sampling"], lw=2, alpha=0.9)
        ax.plot(ratios, ppo_grad, ls, color=PALETTE["ppo"], lw=2, alpha=0.9)
        ax.plot(ratios, cispo_grad, ls, color=PALETTE["cispo"], lw=2, alpha=0.9)

    ax.axhline(0, color="#333", lw=0.8)
    ax.axvline(1.0, color="#999", lw=1, ls=":")
    ax.axvspan(lo, hi, color="#2f6fb3", alpha=0.07)
    ax.set_title("② 梯度：PPO 出界即 0，CISPO 只限幅")
    ax.set_xlabel("重要性比率 ratio")
    ax.set_ylabel("d(objective)/d(logp)")
    ax.annotate(
        "PPO 在 A>0 时\n这一整段梯度为 0",
        xy=(2.4, 0.05),
        xytext=(1.35, 1.55),
        arrowprops=dict(arrowstyle="->", color="#2f6fb3"),
        fontsize=8.5,
        color="#2f6fb3",
    )

    ax = axes[2]
    vals = [0.0, float((ratios > hi).mean()), 0.0]
    bars = ax.bar(
        ["IS", "PPO", "CISPO"],
        vals,
        color=[PALETTE["importance_sampling"], PALETTE["ppo"], PALETTE["cispo"]],
    )
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v * 100:.0f}%", ha="center", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("③ ratio 在 [0.2, 3] 均匀分布时，梯度恒为 0 的区间占比")
    ax.set_ylabel("比例")

    fig.suptitle(
        "三种 RL loss 的几何差异（clip 区间 [%.1f, %.1f]，A>0 视角）" % (lo, hi),
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return _finish(fig, out)


def plot_soft_overlong(out, max_tokens: int = 4096, overlong_cache: int = 1024) -> str:
    """DAPO Soft Overlong Punishment 坡道（复刻仓库 dapo/images/04 那张图）。"""
    lengths = np.arange(0, max_tokens + 512)
    start = max_tokens - overlong_cache
    penalty = np.where(
        lengths <= start,
        0.0,
        np.where(lengths <= max_tokens, -(lengths - start) / overlong_cache, -1.0),
    )

    fig, ax = plt.subplots(figsize=(8.2, 4))
    ax.plot(lengths, penalty, color=PALETTE["dapo"], lw=2.4)
    ax.axvline(start, color="#999", ls="--", lw=1)
    ax.axvline(max_tokens, color="#d94a45", ls="--", lw=1)
    ax.annotate(
        f"惩罚起点 {start}",
        (start, 0),
        xytext=(start - 900, -0.18),
        arrowprops=dict(arrowstyle="->", color="#666"),
        fontsize=9,
    )
    ax.annotate(
        f"硬截断 {max_tokens}",
        (max_tokens, -1),
        xytext=(max_tokens - 1500, -0.78),
        arrowprops=dict(arrowstyle="->", color="#666"),
        fontsize=9,
    )
    ax.fill_between(lengths, penalty, 0, color=PALETTE["dapo"], alpha=0.12)
    ax.set_title("DAPO Soft Overlong Punishment：接近上限时线性扣分，而不是断崖判错")
    ax.set_xlabel("completion token 数")
    ax.set_ylabel("length penalty")
    ax.set_ylim(-1.1, 0.1)
    return _finish(fig, out)


def plot_clip_higher(out) -> str:
    """对称 clip（GRPO）与 Clip-Higher（DAPO）的区间对比。"""
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    rows = [
        ("DAPO Clip-Higher", 0.8, 1.28, PALETTE["dapo"]),
        ("GRPO 对称 clip", 0.8, 1.2, PALETTE["ppo"]),
    ]
    for i, (name, lo, hi, color) in enumerate(rows):
        ax.barh(i, hi - lo, left=lo, height=0.46, color=color, alpha=0.9)
        ax.text(lo - 0.012, i, f"{lo:.2f}", ha="right", va="center", fontsize=10)
        ax.text(hi + 0.012, i, f"{hi:.2f}", ha="left", va="center", fontsize=10, fontweight="bold")
        ax.text(
            (lo + hi) / 2,
            i,
            name,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="white",
        )
    ax.axvline(1.0, color="#333", lw=1, ls=":")
    ax.text(1.005, -0.62, "ratio = 1", fontsize=8.5, color="#333")
    ax.set_yticks([])
    ax.set_ylim(-0.8, 1.6)
    ax.set_xlim(0.7, 1.42)
    ax.set_xlabel("重要性比率 ratio 的可信区间")
    ax.set_title("Clip-Higher：只放宽上界，给正 advantage 的低概率 token 更多上升空间", pad=10)
    return _finish(fig, out)


def plot_gspo_ratio_scale(out, eps: float = 3e-4) -> str:
    """token 级连乘 ratio vs 序列级长度归一化 ratio，以及阈值尺度差异。

    若每个 token 的 log-ratio 都是常量 d，则

    - token 级序列概率比 = exp(d*L)，随长度指数放大；
    - GSPO 的 s = exp(d)，与长度无关。

    这就是 GSPO 敢把 clip 阈值压到 3e-4 的原因。
    """
    lengths = np.arange(1, 513)
    deltas = [0.001, 0.005, 0.02]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4))

    for d, color in zip(deltas, ["#2f6fb3", "#258f4b", "#d94a45"]):
        axes[0].plot(lengths, np.exp(d * lengths), lw=2, color=color, label=f"d={d}")
        axes[0].axhline(np.exp(d), color=color, ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_title("序列概率比：token 级连乘（实线）vs GSPO 几何平均（虚线）")
    axes[0].set_xlabel("序列长度 L")
    axes[0].set_ylabel("pi_theta / pi_old")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].annotate(
        "虚线 = exp(d)，与 L 无关\n（三条几乎重合在 1 附近）",
        xy=(430, 1.05),
        xytext=(215, 12),
        fontsize=8.5,
        color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.9),
    )

    # 阈值尺度：用点 + 竖线，不用 barh —— 条形在 log 轴上没有合法起点
    names = ["PPO（token 级）", "GSPO（sequence 级）"]
    vals = [0.2, eps]
    ys = [1.0, 0.0]
    cols = [PALETTE["ppo"], PALETTE["gspo"]]
    axes[1].set_xscale("log")
    for y, v, c in zip(ys, vals, cols):
        axes[1].vlines(v, y - 0.22, y + 0.22, color=c, lw=10, alpha=0.85)
        axes[1].text(v * 1.25, y, f"eps = {v:g}", va="center", fontsize=11, color=c)
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels(names)
    axes[1].set_ylim(-0.6, 1.6)
    axes[1].set_xlim(1e-4, 1.0)
    axes[1].set_xlabel("clip 阈值 eps（对数轴）")
    axes[1].set_title(f"阈值尺度：PPO 比 GSPO 大约 {0.2 / eps:.0f} 倍")
    axes[1].grid(axis="x", which="both", color="#eeeeee")

    return _finish(fig, out)


# --------------------------------------------------------------------------- #
# 结果图
# --------------------------------------------------------------------------- #
MAIN_RUNS = ("grpo_ppo", "grpo_cispo", "grpo_is", "dapo", "gspo")


def _load_logs(log_dir, only_main: bool = True) -> Dict[str, dict]:
    """读取 logs/*.json。

    ``only_main=True`` 时只保留排版用的正式对照运行（``MAIN_RUNS``），
    把 smoke / dbg / dbg2 这类调试运行排除掉，否则图上全是噪声线。
    """
    logs = {}
    for p in sorted(Path(log_dir).glob("*.json")):
        if p.stem in {"loss_headtohead"}:
            continue
        if only_main and p.stem not in MAIN_RUNS:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "history" in data:
            logs[p.stem] = data
    return logs


def plot_training_curves(log_dir, out) -> Optional[str]:
    """把 logs/*.json 画成 6 联对照图（含 3 点滑动平均）。"""
    logs = _load_logs(log_dir)
    if not logs:
        print("[fig] 没有找到日志，跳过训练曲线")
        return None

    panels = [
        ("reward_mean", "① 组内平均 reward"),
        ("accuracy", "② 采样准确率（rollout 内）"),
        ("policy_entropy", "③ 策略熵（-logp 的蒙特卡洛估计）"),
        ("length_mean", "④ completion 平均长度"),
        ("clip_frac", "⑤ 内层第 2 epoch 被 clip 的 token 比例"),
        ("ratio_mean", "⑥ 内层第 2 epoch 的平均 ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.2))
    for ax, (key, title) in zip(axes.ravel(), panels):
        drew = False
        for tag, data in logs.items():
            hist = data.get("history", [])
            # 优先取内层第 2 epoch 的指标（此时 ratio 才偏离 1）
            for cand in (f"{key}_e1", f"{key}_e0", key):
                if hist and cand in hist[-1]:
                    use = cand
                    break
            else:
                continue
            xs = [h["step"] for h in hist if use in h]
            ys = [h[use] for h in hist if use in h]
            if not xs:
                continue
            drew = True
            color = color_for(tag)
            ax.plot(xs, ys, marker="o", ms=3, lw=1, alpha=0.4, color=color)
            if len(ys) >= 3:
                smooth = np.convolve(ys, np.ones(3) / 3, mode="valid")
                ax.plot(xs[2:], smooth, lw=2.2, label=tag, color=color)
            else:
                ax.plot(xs, ys, lw=2.2, label=tag, color=color)
        ax.set_title(title)
        ax.set_xlabel("step")
        if not drew:
            ax.text(0.5, 0.5, "该指标未记录", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("本地 GPU 复现：GRPO / DAPO / GSPO 训练过程对照", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _finish(fig, out)


def plot_advantage_dist(log_dir, out) -> Optional[str]:
    """退化组比例 —— DAPO Dynamic Sampling 的动机图。"""
    logs = _load_logs(log_dir)
    if not logs:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4))
    drew = False
    for tag, data in logs.items():
        hist = [h for h in data.get("history", []) if "degenerate_frac" in h]
        if not hist:
            continue
        drew = True
        ax.plot(
            [h["step"] for h in hist],
            [h["degenerate_frac"] for h in hist],
            marker="s",
            ms=3,
            lw=1.8,
            label=tag,
            color=color_for(tag),
        )
    if not drew:
        plt.close(fig)
        return None
    ax.set_title("退化组比例：组内全对或全错的题目不产生任何梯度")
    ax.set_xlabel("step")
    ax.set_ylabel("degenerate group fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    return _finish(fig, out)


def plot_before_after(log_dir, out) -> Optional[str]:
    """训练前后 dev 集准确率对比。"""
    logs = _load_logs(log_dir)
    rows = []
    for tag, data in logs.items():
        base = data.get("base_eval", {}).get("accuracy")
        final = data.get("final_eval", {}).get("accuracy")
        if base is None or final is None:
            continue
        rows.append((tag, base, final))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 4))
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [r[1] for r in rows], w, label="训练前", color="#9aa5b1")
    ax.bar(x + w / 2, [r[2] for r in rows], w, label="训练后", color="#2f6fb3")
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r[1] + 0.01, f"{r[1] * 100:.1f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, r[2] + 0.01, f"{r[2] * 100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=15, fontsize=9)
    ax.set_ylabel("dev accuracy")

    # 噪声带：n=20 时 1 个标准差 ≈ 11 个百分点，训练前后的差异基本都在带内
    n = int(logs[rows[0][0]].get("base_eval", {}).get("n", 20) or 20)
    p = float(np.mean([r[1] for r in rows]))
    se = float(np.sqrt(max(p * (1 - p), 1e-9) / n))
    ax.axhspan(p - se, p + se, color="#ffd166", alpha=0.28, zorder=0)
    ax.axhline(p, color="#c99a2e", ls=":", lw=1, zorder=1)
    ax.set_title("训练前后 dev 集准确率（贪心解码，%d 题）" % n)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(r[1] for r in rows), max(r[2] for r in rows)) * 1.45 + 0.04)
    # 说明文字放在顶部空白区，避免压住柱子上的数值标签
    ax.text(
        0.012,
        0.985,
        f"黄色带 = 基线 ±1SE ≈ ±{se * 100:.0f}pp（n={n}）：带内差异不构成结论",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#8a6d1f",
        va="top",
        bbox=dict(boxstyle="round,pad=0.28", fc="#fff8e1", ec="#e0c169", lw=0.7),
    )
    return _finish(fig, out)


# --------------------------------------------------------------------------- #
# 受控对照
# --------------------------------------------------------------------------- #
def plot_headtohead(json_path, out) -> Optional[str]:
    """同一批 rollout 上五种 loss 的梯度范数、裁剪比例与平均 ratio。"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    losses = data["losses"]
    names = list(losses)
    pretty = {
        "importance_sampling": "IS",
        "ppo": "PPO",
        "ppo_cliphigher": "PPO+ClipHigher",
        "cispo": "CISPO",
        "gspo": "GSPO",
    }
    labels = [pretty.get(n, n) for n in names]
    colors = [
        PALETTE["importance_sampling"],
        PALETTE["ppo"],
        PALETTE["dapo"],
        PALETTE["cispo"],
        PALETTE["gspo"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    grads = [losses[n]["grad_norm"] for n in names]
    vals = [losses[n]["loss"] for n in names]
    clip_fracs = [losses[n].get("clip_frac", losses[n].get("seq_clip_frac", 0.0)) for n in names]
    ref = grads[0]  # 以 IS 为 1.0 基准

    # ① 梯度范数：归一化到 IS，才能看出"几乎相等"和"GSPO 明显小"
    rel = [g / ref for g in grads]
    bars = axes[0].bar(labels, rel, color=colors)
    for b, v, g in zip(bars, rel, grads):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.4f}×\n{g:.3e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].axhline(1.0, color="#333", ls=":", lw=1)
    axes[0].set_ylim(0, max(rel) * 1.35)
    axes[0].set_title("① 梯度范数（以 IS 为 1.0 归一化）")
    axes[0].set_ylabel("grad norm / IS grad norm")
    axes[0].tick_params(axis="x", labelrotation=15)

    # ② loss 数值：量级、符号都不同 —— 证明 loss 绝对值不可跨算法比较
    bars = axes[1].bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:+.4f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
        )
    axes[1].axhline(0.0, color="#333", lw=1)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.25
    axes[1].set_ylim(lo - pad, hi + pad)
    axes[1].set_title("② loss 值：量级与符号都不同")
    axes[1].set_ylabel("loss")
    axes[1].tick_params(axis="x", labelrotation=15)

    # ③ 裁剪比例
    bars = axes[2].bar(labels, clip_fracs, color=colors)
    for b, v in zip(bars, clip_fracs):
        axes[2].text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[2].set_ylim(0, max(max(clip_fracs) * 1.4, 0.05))
    axes[2].set_title("③ 被裁剪的比例（token 级 / GSPO 为序列级）")
    axes[2].set_ylabel("clip fraction")
    axes[2].tick_params(axis="x", labelrotation=15)

    b0 = data["base"]
    d = data.get("log_ratio_after_one_update", {})
    fig.suptitle(
        "受控对照：同一批 %d 条轨迹 / %d tokens，其中 %.0f 个退化组（无梯度），"
        "一次更新后 log-ratio std=%.4f"
        % (
            b0["n_rollouts"],
            b0["n_tokens"],
            b0.get("degenerate_groups", 0.0),
            d.get("std", 0.0),
        ),
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    return _finish(fig, out)


__all__ = [
    "PALETTE",
    "color_for",
    "plot_advantage_dist",
    "plot_before_after",
    "plot_clip_higher",
    "plot_gspo_ratio_scale",
    "plot_headtohead",
    "plot_loss_geometry",
    "plot_soft_overlong",
    "plot_training_curves",
]
