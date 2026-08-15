"""Build the index for one or all chunking strategies.

    python scripts/ingest.py --config configs/local.yaml
    python scripts/ingest.py --all-chunkers      # build all three collections
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig, load_config  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--all-chunkers", action="store_true", help="build one collection per chunking strategy")
    args = parser.parse_args()

    app = load_config(args.config)
    chunkers = ["fixed", "recursive", "semantic"] if args.all_chunkers else [app.pipeline.chunker]

    for name in chunkers:
        cfg = PipelineConfig(**app.pipeline.to_dict())
        cfg.chunker = name
        pipeline = RAGPipeline(app, cfg)
        print(f"\n=== {name} -> {cfg.collection_name} ===")
        print(json.dumps(pipeline.ingest(reset=args.reset), indent=2))


if __name__ == "__main__":
    main()
