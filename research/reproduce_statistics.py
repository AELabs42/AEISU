"""Run the archived paired analysis on compressed packaged scores in a new workspace."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def actual_file(path):
    return path if path.exists() else path.with_suffix(path.suffix + ".gz")


def records(path):
    path = actual_file(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new analysis directory; packaged evidence is immutable")
    shutil.copytree(ROOT / "quality", args.output)
    source = ROOT / "archival_sources/paper_bench/final_benchmark_analysis.py"
    spec = importlib.util.spec_from_file_location("archived_analysis", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only I/O changes: estimator, bootstrap seeds and scientific calculations are unchanged.
    module.records = records
    module.sha256 = lambda path: hashlib.sha256(actual_file(path).read_bytes()).hexdigest()
    sys.argv = [str(source), "--output", str(args.output), "--draws", str(args.draws)]
    module.main()


if __name__ == "__main__":
    main()
