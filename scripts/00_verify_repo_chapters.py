"""核查原仓库每个章节在本机能否 import / --help，输出一张证据表。

这不是"跑通训练"的证明，而是明确区分三件事：

1. **本地完全可跑**：纯本地逻辑（loss、reward、loss 曲线、机制图）；
2. **需要 PyTRIO 凭据**：脚本能 import、能出 --help，但真正训练要云端 key 与额度；
3. **需要额外环境/数据**：ALFWorld 需要 TextWorld，Search-R1 需要 DeepSeek key。

结果写入 docs/原仓库可运行性核查.md。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "agentic-rl-lab"

CHAPTERS = {
    "getting-started": "环境与概念（纯文档）",
    "loss-functions": "loss 直觉（纯文档）",
    "grpo": "01-demo-sync.py",
    "vision-grpo": "train.py",
    "general-opd": "01-demo-sync.py",
    "medical-opd": "02-medical-sft.py",
    "dapo": "train.py",
    "gspo": "train.py",
    "opsd": "01-opsd-sync.py",
    "agentopsd": "train.py",
    "search-r1": "train.py",
    "retool": "train.py",
    "alfworld": "train.py",
    "tempo": "train.py",
    "spec-o3": "train_rl.py",
    "tournament-grpo": "train.py",
    "pytrio-jev": "train.py",
}

NEEDS = {
    "search-r1": "PyTRIO + DEEPSEEK_API_KEY（在线搜索后端）",
    "tournament-grpo": "PyTRIO + judge 模型 key",
    "alfworld": "PyTRIO + alfworld==0.4.2 + spacy（TextWorld 环境）",
    "medical-opd": "PyTRIO + 医疗数据集（ModelScope/HF）",
    "spec-o3": "PyTRIO + 光谱数据 + 图片 token 支持",
    "vision-grpo": "PyTRIO + GeoQA 图片数据",
    "retool": "PyTRIO + 本地代码沙箱",
    "tempo": "PyTRIO + TextWorld 存档读档",
    "agentopsd": "PyTRIO + ALFWorld + SkillBank",
    "opsd": "PyTRIO + OPSD/AIME25 数据",
    "grpo": "PyTRIO（GSM8K 可自动下载）",
    "dapo": "PyTRIO（DAPO-Math-17k 可自动下载）",
    "gspo": "PyTRIO（数学数据自动下载）",
    "general-opd": "PyTRIO + DeepMath-103K",
    "pytrio-jev": "PyTRIO + Jev 决策数据",
}


def probe(script: Path, python: str) -> tuple[bool, str]:
    """用 --help 探测脚本层面的可运行性（不触发云端调用）。"""
    try:
        proc = subprocess.run(
            [python, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(script.parent),
        )
        ok = proc.returncode == 0
        tail = (proc.stdout or proc.stderr).strip().splitlines()
        msg = tail[-1][:140] if tail else ""
        return ok, msg
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    python = sys.executable
    rows = []
    for chapter, entry in CHAPTERS.items():
        if entry.endswith(".py"):
            ok, msg = probe(REPO / "experiments" / chapter / entry, python)
            rows.append(
                {
                    "章节": chapter,
                    "入口": entry,
                    "help 可运行": "✅" if ok else "❌",
                    "备注": msg,
                    "额外依赖": NEEDS.get(chapter, "-"),
                }
            )
        else:
            rows.append(
                {"章节": chapter, "入口": "-", "help 可运行": "n/a", "备注": entry, "额外依赖": "-"}
            )

    lines = [
        "# 原仓库可运行性核查（本机实测）",
        "",
        "> 用 `python <入口> --help` 探测：能出帮助说明脚本级依赖齐备；",
        "> 真正跑训练还需要 PyTRIO 账号额度与各章数据，这一层在本机未执行。",
        "",
        "| 章节 | 入口 | --help | 额外依赖 | 备注 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['章节']} | `{r['入口']}` | {r['help 可运行']} | {r['额外依赖']} | {r['备注']} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "原仓库所有入口脚本都依赖 `pytrio` 客户端，训练与采样在云端执行，",
        "本机只承担数据、reward 与 loss 计算。因此「用本机 GPU 复现」必须补上训练器，",
        "这正是本仓库 `local_repro/trainer.py` 的作用。",
    ]
    out = ROOT / "docs" / "原仓库可运行性核查.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\n[done] 写入 {out}")


if __name__ == "__main__":
    main()
