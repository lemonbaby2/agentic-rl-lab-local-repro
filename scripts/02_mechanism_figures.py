"""生成机制图：三种 loss 的几何差异、DAPO 长度惩罚、Clip-Higher、GSPO 比率尺度。

这些图不需要训练、不需要模型，纯 NumPy/Matplotlib 即可，用来替代仓库里
loss-functions 与 dapo 章节的机制示意图，并给出可复算的定量依据。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_repro import plots

FIG = ROOT / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plots.plot_loss_geometry(FIG / "01-三种loss几何差异.png")
    plots.plot_soft_overlong(FIG / "02-DAPO软超长惩罚.png", max_tokens=4096, overlong_cache=1024)
    plots.plot_clip_higher(FIG / "03-ClipHigher区间.png")
    plots.plot_gspo_ratio_scale(FIG / "04-GSPO序列级比率.png")
    print("\n[done] 4 张机制图已生成到 figures/")


if __name__ == "__main__":
    main()
