"""Zip a GPU-built Chroma index so it can be served from a CPU-only container.

On Kaggle:
    python scripts/export_index.py --out /kaggle/working/index.zip

Locally, after downloading:
    python scripts/export_index.py --import index.zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402


def export(index_dir: Path, out: Path) -> None:
    if not index_dir.exists():
        raise SystemExit(f"nothing to export: {index_dir} does not exist")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in index_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(index_dir))
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB)")
    print("Download this before the session ends — /kaggle/working is deleted with the container.")


def import_(archive: Path, index_dir: Path, replace: bool) -> None:
    if index_dir.exists() and replace:
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(index_dir)
    print(f"extracted {archive} -> {index_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--out", default="index.zip")
    parser.add_argument("--import", dest="import_path", default=None, help="import a zip instead of exporting")
    parser.add_argument("--replace", action="store_true", help="wipe the local index before importing")
    args = parser.parse_args()

    index_dir = load_config(args.config).index_path
    if args.import_path:
        import_(Path(args.import_path), index_dir, args.replace)
    else:
        export(index_dir, Path(args.out))


if __name__ == "__main__":
    main()
