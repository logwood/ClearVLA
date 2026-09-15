"""Construction-vs-invocation call-graph audit (do_before_v78 §14.6 / M1 quarantine proof).

A PyTorch model is two overlaid call graphs: the CONSTRUCTION graph
(__init__ chain -> parameter ownership, state_dict paths, RNG order) and the
INVOCATION graph (forward chain -> dataflow, gradients).  Their differences
are exactly this project's documented failure taxonomy:

  constructed but never invoked  -> dead-weight modules (A8 class)
  invoked in train but grad-less -> frozen/metric-only paths (probe class)
  invoked only in sample         -> eval-only machinery
  invoked but owned elsewhere    -> cross-generation edges

Usage (diagnostic run, one batch, then exits):
    CALLGRAPH_AUDIT=1 bash scripts/current_v77_*.sh
Report lands in <OUT_DIR>/callgraph_audit/ as report.md + report.json.
The mainline closure in report.json is the authoritative quarantine list for
the do_before_v78 function ledger; the hand-traced ledger is its review
baseline and every discrepancy must be investigated, not overwritten.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch
from torch import nn


class CallGraphAuditor:
    """Records which modules fire per phase and which parameters receive grads."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._phases: dict[str, set[str]] = {}
        self._current: str | None = None
        self.grad_received: set[str] = set()
        self.grad_nonzero: set[str] = set()

    # -- lifecycle -----------------------------------------------------------
    def attach(self, first_phase: str = "train") -> None:
        for name, module in self.model.named_modules():
            handle = module.register_forward_hook(partial(self._fire, name or "<root>"))
            self._handles.append(handle)
        self.begin_phase(first_phase)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_phase(self, tag: str) -> None:
        self._phases.setdefault(tag, set())
        self._current = tag

    def _fire(self, name: str, _module: nn.Module, _inputs, _output) -> None:
        if self._current is not None:
            self._phases[self._current].add(name)

    def capture_gradients(self) -> None:
        """Call right after loss.backward(), before any zero_grad."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.grad_received.add(name)
                if bool(param.grad.detach().abs().sum().item() > 0.0):
                    self.grad_nonzero.add(name)

    # -- report --------------------------------------------------------------
    @staticmethod
    def _owner(param_name: str) -> str:
        return param_name.rsplit(".", 1)[0] if "." in param_name else "<root>"

    @staticmethod
    def _prefix(name: str, depth: int = 3) -> str:
        return ".".join(name.split(".")[:depth])

    def write_report(self, out_dir: Path, context_note: str = "") -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        params = {name: int(p.numel()) for name, p in self.model.named_parameters()}
        buffers = {name: int(b.numel()) for name, b in self.model.named_buffers()}
        fired_any: set[str] = set().union(*self._phases.values()) if self._phases else set()

        rows: dict[str, dict[str, object]] = {}
        for name, numel in params.items():
            owner = self._owner(name)
            rows[name] = {
                "numel": numel,
                "owner_fired": {tag: owner in fired for tag, fired in self._phases.items()},
                "grad_received": name in self.grad_received,
                "grad_nonzero": name in self.grad_nonzero,
            }
        dead = {n: r for n, r in rows.items() if not any(r["owner_fired"].values())}
        frozen = {
            n: r
            for n, r in rows.items()
            if any(r["owner_fired"].values()) and not r["grad_received"]
        }
        sample_only = {
            n: r
            for n, r in rows.items()
            if not r["owner_fired"].get("train", False)
            and any(v for t, v in r["owner_fired"].items() if t != "train")
        }

        def _mb(entries: dict[str, dict[str, object]]) -> float:
            return sum(int(r["numel"]) for r in entries.values()) * 4 / 1e6

        def _prefix_table(entries: dict[str, dict[str, object]]) -> list[tuple[str, int, float]]:
            agg: dict[str, int] = {}
            for name, r in entries.items():
                agg[self._prefix(name)] = agg.get(self._prefix(name), 0) + int(r["numel"])
            return sorted(((k, v, v * 4 / 1e6) for k, v in agg.items()), key=lambda t: -t[1])[:20]

        report = {
            "context": context_note,
            "phases": {tag: sorted(fired) for tag, fired in self._phases.items()},
            "totals": {
                "parameters": len(params),
                "parameter_mb": round(sum(params.values()) * 4 / 1e6, 2),
                "buffers": len(buffers),
                "dead_parameters": len(dead),
                "dead_mb": round(_mb(dead), 2),
                "frozen_or_metric_only": len(frozen),
                "sample_only": len(sample_only),
                "grad_received": len(self.grad_received),
                "grad_nonzero": len(self.grad_nonzero),
            },
            "dead": sorted(dead),
            "frozen_or_metric_only": sorted(frozen),
            "sample_only": sorted(sample_only),
            "params": rows,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=1))

        lines = [
            "# Call-graph audit report",
            "",
            f"context: {context_note}",
            "",
            "| 指标 | 值 |",
            "|---|---|",
        ]
        for key, value in report["totals"].items():
            lines.append(f"| {key} | {value} |")
        lines += [
            "",
            "## 死重 top-20 前缀（构造了、任何相位都未开火）",
            "",
            "| 前缀 | 参数量 | MB |",
            "|---|---|---|",
        ]
        for prefix, numel, mb in _prefix_table(dead):
            lines.append(f"| {prefix} | {numel} | {mb:.2f} |")
        lines += [
            "",
            "## 开火但无梯度 top-20 前缀（冻结/纯仪表路径, 应与已知探针清单对账）",
            "",
            "| 前缀 | 参数量 | MB |",
            "|---|---|---|",
        ]
        for prefix, numel, mb in _prefix_table(frozen):
            lines.append(f"| {prefix} | {numel} | {mb:.2f} |")
        lines += [
            "",
            "## 仅采样相位开火 top-20 前缀",
            "",
            "| 前缀 | 参数量 | MB |",
            "|---|---|---|",
        ]
        for prefix, numel, mb in _prefix_table(sample_only):
            lines.append(f"| {prefix} | {numel} | {mb:.2f} |")
        lines.append("")
        (out_dir / "report.md").write_text("\n".join(lines))
        return out_dir / "report.md"
