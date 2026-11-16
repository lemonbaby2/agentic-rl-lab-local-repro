"""同一批 rollout 上,五种 loss 的梯度行为对照(受控实验)。

为什么需要这个实验
------------------
最小闭环里"一步一更新"意味着 ratio ≡ 1,此时 IS / PPO / CISPO 给出**完全相同**的梯度
——这不是 bug,而是 on-policy 的定义决定的。所以"三种 loss 有区别"只能在两种情况下观察到:

1. 一轮 rollout 内做多个 inner epoch(权重已经动了,ratio ≠ 1);
2. sampler 陈旧(异步 RL 或刷新间隔 > 1)。

本脚本用**同一批采样数据**、**同一个初始权重**,分别计算五种算法的
loss 值、梯度范数、ratio 分布、被裁剪 token 占比,把差异量化出来。
全程不更新参数,所以是严格的受控对比。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_repro import losses as L
from local_repro.data import load_gsm8k
from local_repro.trainer import LocalGRPOTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--groups", type=int, default=3, help="题目组数")
    p.add_argument("--group-size", type=int, default=4, help="每题采样条数")
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--out", default=str(ROOT / "logs" / "loss_headtohead.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        device=args.device,
        groups_per_step=args.groups,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.lr,
        micro_batch_size=1,
        seed=7,
    )
    problems = load_gsm8k("train", limit=64, seed=7)
    trainer = LocalGRPOTrainer(cfg, problems, dev=None)

    # ---------- 采一批固定数据 ----------
    groups, _ = trainer._collect_groups()
    trainer._assign_advantages(groups)
    flat = [r for g in groups for r in g.rollouts]
    print(f"[batch] {len(groups)} 组 × {args.group_size} 条 = {len(flat)} 条轨迹")

    adv_vec = torch.tensor([r.advantage for r in flat], device=trainer.device)

    # 按 micro-batch 切块：152k 词表的 cross_entropy 一次放不下全部序列
    chunks = [list(range(i, min(i + 2, len(flat)))) for i in range(0, len(flat), 2)]
    chunk_cache = []
    for idxs in chunks:
        sub = trainer._batch_tensors([flat[i] for i in idxs])
        with torch.no_grad():
            lp, mk = trainer._token_logprobs(sub, grad=False)
        chunk_cache.append((sub, lp.detach(), mk, idxs))

    n_tok = sum(c[2].sum().item() for c in chunk_cache)
    seq_len = float(sum(c[2].sum(dim=1).sum().item() for c in chunk_cache) / len(flat))
    base_metrics = {
        "n_rollouts": len(flat),
        "n_tokens": int(n_tok),
        "mean_completion_len": seq_len,
        "reward_mean": sum(r.reward for r in flat) / len(flat),
        "accuracy": sum(float(r.correct) for r in flat) / len(flat),
        "degenerate_groups": sum(float(g.degenerate) for g in groups),
    }
    print(f"[batch] tokens={int(n_tok)} avg_len={seq_len:.0f} acc={base_metrics['accuracy']:.3f}")

    # ---------- 制造策略漂移:做 1 次真实更新,让 ratio 有变化 ----------
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
    loss_fn_ppo = L.build_loss("ppo", clip_low=0.2, clip_high=0.2, reduction="token")
    trainer.model.train()
    opt.zero_grad(set_to_none=True)
    for sub, old_lp, mk, idxs in chunk_cache:
        adv_chunk = adv_vec[idxs].unsqueeze(1).expand_as(old_lp) * mk
        new_lp, new_mk = trainer._token_logprobs(sub, grad=True)
        loss, _ = loss_fn_ppo(new_lp, old_lp, adv_chunk, new_mk)
        (loss * mk.sum() / n_tok).backward()
    opt.step()

    # 更新后再算一次当前 logprob,作为"训练侧"logp_new
    drift_parts = []
    with torch.no_grad():
        for sub, old_lp, mk, idxs in chunk_cache:
            lp, _ = trainer._token_logprobs(sub, grad=False)
            drift_parts.append((lp - old_lp)[mk.bool()])
    drift = torch.cat(drift_parts)
    print(
        f"[drift] 一次更新后的 log-ratio: mean={drift.mean().item():+.5f} "
        f"std={drift.std().item():.5f} max={drift.max().item():+.4f} min={drift.min().item():+.4f}"
    )

    # ---------- 五种 loss 在同一数据上的梯度 ----------
    results = {}
    presets = {
        "importance_sampling": dict(clip_low=0.2, clip_high=0.2, reduction="token"),
        "ppo": dict(clip_low=0.2, clip_high=0.2, reduction="token"),
        "ppo_cliphigher": dict(clip_low=0.2, clip_high=0.28, reduction="token"),
        "cispo": dict(clip_low=0.2, clip_high=0.2, reduction="token"),
        "gspo": dict(clip_eps=3e-4),
    }
    for name, kwargs in presets.items():
        base = "ppo" if name.startswith("ppo") else name
        fn = L.build_loss(base, **kwargs)
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        agg = {}
        for sub, old_lp, mk, idxs in chunk_cache:
            adv_chunk = adv_vec[idxs].unsqueeze(1).expand_as(old_lp) * mk
            lp, new_mk = trainer._token_logprobs(sub, grad=True)
            loss, metrics = fn(lp, old_lp, adv_chunk, new_mk)
            (loss * mk.sum() / n_tok).backward()
            loss_val += float(loss.item()) * (mk.sum().item() / n_tok)
            for k, v in metrics.items():
                agg.setdefault(k, 0.0)
                agg[k] += v * (mk.sum().item() / n_tok)

        grad_sq = 0.0
        zero_grad_params = 0
        total_params = 0
        for p in params:
            if p.grad is None:
                continue
            grad_sq += float(p.grad.detach().pow(2).sum().item())
            total_params += 1
            if float(p.grad.detach().abs().max().item()) == 0.0:
                zero_grad_params += 1

        results[name] = {
            "loss": loss_val,
            "grad_norm": grad_sq**0.5,
            "params_with_zero_grad": zero_grad_params,
            "params_touched": total_params,
            **{k: float(v) for k, v in agg.items()},
        }
        print(
            f"[{name:18s}] loss={results[name]['loss']:+.6f} |g|={results[name]['grad_norm']:.4e} "
            f"| {agg}"
        )

    out = {"base": base_metrics, "log_ratio_after_one_update": {
        "mean": float(drift.mean()), "std": float(drift.std()),
        "max": float(drift.max()), "min": float(drift.min()),
    }, "losses": results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
