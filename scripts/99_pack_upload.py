"""把复现仓库打包成可直接拖进 GPT 知识库的 zip。

排除 .git / __pycache__ / outputs（LoRA 权重体积大）。
用法：python scripts/99_pack_upload.py
"""
from pathlib import Path
import zipfile

SRC = Path(__file__).resolve().parents[1]
DST = SRC.parent / "agentic-rl-lab-repro-upload-pack.zip"
PREFIX = "agentic-rl-lab-local-repro/"
INCLUDE_FILES = ("README.md", "LICENSE", "pyproject.toml", ".gitignore")
INCLUDE_DIRS = ("local_repro", "scripts", "tests", "docs", "figures", "logs")


def main() -> None:
    n = 0
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for name in INCLUDE_FILES:
            p = SRC / name
            if p.exists():
                z.write(p, PREFIX + name)
                n += 1
        for d in INCLUDE_DIRS:
            for p in sorted((SRC / d).rglob("*")):
                if not p.is_file() or "__pycache__" in p.parts:
                    continue
                rel = p.relative_to(SRC).as_posix()
                z.write(p, PREFIX + rel)
                n += 1
    print(f"[ok] {DST.name}: {DST.stat().st_size / 1024 / 1024:.2f} MB, {n} files")


if __name__ == "__main__":
    main()
