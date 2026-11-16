"""把 logs/*.json 渲染成训练曲线与前后对比图。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_repro import plots


def main() -> None:
    fig = ROOT / "figures"
    log_dir = ROOT / "logs"
    plots.plot_training_curves(log_dir, fig / "05-训练曲线对照.png")
    plots.plot_advantage_dist(log_dir, fig / "06-退化组比例.png")
    plots.plot_before_after(log_dir, fig / "07-训练前后对比.png")
    h2h = log_dir / "loss_headtohead.json"
    if h2h.exists():
        plots.plot_headtohead(h2h, fig / "08-五种loss梯度对照.png")
    else:
        print(f"[skip] 缺少 {h2h}，先跑 scripts/06_loss_headtohead.py")
    print("\n[done] 结果图已生成到 figures/")


if __name__ == "__main__":
    main()
