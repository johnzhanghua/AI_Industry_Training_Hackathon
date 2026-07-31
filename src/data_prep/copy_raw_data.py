#!/usr/bin/env python3
"""Stage a local raw copy of the RBA/ASX/AFR datasets from 'data set/'.

Read-only against 'data set/' -- only ever writes to --target. Optional: clean_datasets.py
already reads 'data set/' directly, so this step is only needed if you want an explicit
local snapshot (e.g. to work off a branch/copy without touching the approved source).

Usage:
    python src/data_prep/copy_raw_data.py
    python src/data_prep/copy_raw_data.py --target data/raw
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data set"
DATASET_DIRS = ("RBA Rates", "ASX", "AFR")


def copy_raw(target: Path):
    target.mkdir(parents=True, exist_ok=True)
    for name in DATASET_DIRS:
        src = SOURCE / name
        dst = target / name
        if dst.exists():
            print(f"skip (already exists): {dst}")
            continue
        shutil.copytree(src, dst)
        print(f"copied {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=str(ROOT / "data" / "raw"),
                         help="destination for the raw copy (default: data/raw)")
    args = parser.parse_args()
    copy_raw(Path(args.target))


if __name__ == "__main__":
    main()
