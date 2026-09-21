"""Command-line audit for a converted external benchmark root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import audit_benchmark_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_benchmark_dataset(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


