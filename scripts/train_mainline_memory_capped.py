"""Run the unchanged mainline CLI with a process-local CUDA allocator cap."""

import argparse
import runpy
import sys
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise RuntimeError("run this launcher from its selected repository root")
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--memory-cap-gib", type=float, required=True)
    args, mainline_args = parser.parse_known_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one GPU explicitly before using the capped launcher")
    if not 0 < args.memory_cap_gib <= 22:
        raise ValueError("memory-cap-gib must be in (0,22]")
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = args.memory_cap_gib * 1024**3 / total
    if fraction >= 1:
        raise ValueError("allocator cap must leave device/context headroom")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(
        f"[mainline-process-cap] cuda_allocator_cap_gib={args.memory_cap_gib} "
        "shared_gpu=true core_graph_and_optimizer=unchanged",
        flush=True,
    )
    sys.argv = ["clearvla.mainline.train", *mainline_args]
    runpy.run_module("clearvla.mainline.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
