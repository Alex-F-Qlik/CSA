"""CLI entry point for the sentiment signal pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import process_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Process customer feedback into normalized signals.")
    parser.add_argument("batch", type=Path, help="Path to the normalized batch file (CSV/Parquet).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write normalized outputs.")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet", help="Output format.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"), help="Directory for YAML configs.")
    parser.add_argument("--log-level", default="INFO", choices=logging._nameToLevel.keys(), help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    config_paths = {
        "fluff": args.config_dir / "fluff.yaml",
        "weights": args.config_dir / "weights.yaml",
    }
    missing = [path for path in config_paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing config files: {missing}")

    process_batch(
        source_path=args.batch,
        output_dir=args.output_dir,
        config_paths=config_paths,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
