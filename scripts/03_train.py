"""统一训练入口：一条命令跑一种算法，日志落到 logs/<tag>.json。

用法示例
--------
    # GRPO + 对称 PPO clip
    python scripts/03_train.py --algorithm grpo_ppo --tag grpo_ppo --steps 12

    # GRPO + 无裁剪 IS / CISPO
    python scripts/03_train.py --algorithm grpo_is    --tag grpo_is    --steps 12
    python scripts/03_train.py --algorithm grpo_cispo --tag grpo_cispo --steps 12

    # DAPO：Clip-Higher + token mean + 动态采样 + 软超长惩罚
    python scripts/03_train.py --algorithm dapo --tag dapo --steps 12 \\\\
        --dynamic-sampling --enable-overlong

    # GSPO：序列级 ratio
    python scripts/03_train.py --algorithm gspo --tag gspo --steps 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_repro.data import load_gsm8k
from local_repro.losses import PRESETS
from local_repro.trainer import LocalGRPOTrainer, TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="本地 GPU GRPO / DAPO / GSPO 复现训练")
    p.add_argument("--algorithm", default="grpo_ppo", choices=list(PRESETS) + ["opd"])
    p.add_argument("--tag", default=None, help="日志文件名，默认等于 algorithm")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt-style", default="plain", choices=["plain", "repo"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--groups-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=512, help="DAPO 长度惩罚的硬上限")
    p.add_argument("--overlong-cache", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--micro-batch-size", type=int, default=2, help="每次前向处理几条序列（控制显存）")
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--reward-mode", default="pm1", choices=["pm1", "01"])
    p.add_argument("--normalize-std", action="store_true")
    p.add_argument("--dynamic-sampling", action="store_true")
    p.add_argument("--max-candidate-multiplier", type=int, default=2)
    p.add_argument("--enable-overlong", action="store_true")
    p.add_argument("--train-limit", type=int, default=400)
    p.add_argument("--dev-limit", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(ROOT / "logs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or args.algorithm

    print("[data] loading GSM8K subset ...")
    train = load_gsm8k("train", limit=args.train_limit, seed=args.seed)
    dev = load_gsm8k("test", limit=args.dev_limit, seed=args.seed)
    print(f"[data] train {len(train)} | dev {len(dev)} | math_verify="
          f"{__import__('local_repro.rewards', fromlist=['HAS_MATH_VERIFY']).HAS_MATH_VERIFY}")

    cfg = TrainConfig(
        algorithm=args.algorithm,
        reward_mode=args.reward_mode,
        normalize_std=args.normalize_std,
        ppo_epochs=args.ppo_epochs,
        dynamic_sampling=args.dynamic_sampling,
        max_candidate_multiplier=args.max_candidate_multiplier,
        enable_overlong=args.enable_overlong,
        max_tokens=args.max_tokens,
        overlong_cache=args.overlong_cache,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        lora_rank=args.lora_rank,
        micro_batch_size=args.micro_batch_size,
        steps=args.steps,
        groups_per_step=args.groups_per_step,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        prompt_style=args.prompt_style,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.lr,
        seed=args.seed,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        train_size=len(train),
        dev_size=len(dev),
        tag=tag,
        out_dir=args.out_dir,
    )

    trainer = LocalGRPOTrainer(cfg, train, dev)
    trainer.train()
    adapter_dir = ROOT / "outputs" / f"{tag}-lora"
    trainer.save_adapter(str(adapter_dir))


if __name__ == "__main__":
    main()
