"""Install the opt-in batched raw-flow sampler in a version-local snapshot.

The active mainline already owns the implementation.  Older Pen/RDT snapshots
can receive the same narrow interface without importing the current model or
changing persistent state: a non-persistent stencil buffer, one helper method,
and a guarded branch around the historical per-offset loop.  This script is
intentionally fail-closed and is meant for an explicitly authorized lab copy,
never for the production checkout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
from pathlib import Path


_BUFFER = """\
        # The fixed stencil is non-persistent so the opt-in path is capture
        # safe without adding checkpoint keys or optimizer state.
        self.register_buffer(
            "_batched_offset_values",
            torch.tensor(
                [
                    (dx, dy)
                    for dy in range(-self.radius, self.radius + 1)
                    for dx in range(-self.radius, self.radius + 1)
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
"""


_HELPER = """\
    def _batched_samples(
        self,
        second_feature: Tensor,
        center: Tensor,
        offsets: list[tuple[int, int]],
        search_scale: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        \"\"\"Sample all local offsets in one launch while retaining order.\"\"\"

        if second_feature.ndim != 4 or center.ndim != 4:
            raise ValueError("batched raw sampling received invalid feature geometry")
        batch, channels, side, side_b = second_feature.shape
        if side != side_b or tuple(center.shape) != (batch, side, side, 2):
            raise ValueError("batched raw sampling geometry does not align")
        if not offsets:
            raise ValueError("batched raw sampling requires at least one offset")
        offset_values = self._batched_offset_values
        if offset_values.device != second_feature.device:
            raise RuntimeError(
                "batched raw sampling stencil must be on the feature device before capture"
            )
        if tuple(offset_values.shape) != (len(offsets), 2):
            raise RuntimeError("batched raw sampling stencil does not match radius")
        offset_field = offset_values.view(1, len(offsets), 2, 1, 1)
        if self.preserve_uncertain_seed:
            offset_field = offset_field * search_scale.unsqueeze(1)
        else:
            offset_field = offset_field.expand(batch, -1, -1, side, side)
        coordinates = center.unsqueeze(1) + offset_field.permute(0, 1, 3, 4, 2)
        valid = (
            (coordinates[..., 0] >= 0.0)
            & (coordinates[..., 0] <= float(side - 1))
            & (coordinates[..., 1] >= 0.0)
            & (coordinates[..., 1] <= float(side - 1))
        )
        flat_coordinates = coordinates.reshape(batch * len(offsets), side, side, 2)
        flat_grid = _normalize_grid(flat_coordinates, side, side)
        flat_value = (
            second_feature.unsqueeze(1)
            .expand(-1, len(offsets), -1, -1, -1)
            .reshape(batch * len(offsets), channels, side, side)
        )
        with torch.autocast(device_type=second_feature.device.type, enabled=False):
            sampled = F.grid_sample(
                flat_value.float(),
                flat_grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        return (
            sampled.reshape(batch, len(offsets), channels, side, side),
            valid,
            offset_field,
        )

"""


_LOOP = """\
        batched_rows = None
        if getattr(self, "_batched_offset_sampling", False):
            batched_rows = self._batched_samples(
                second_feature,
                center,
                offsets,
                search_scale,
            )
        for index, (dx, dy) in enumerate(offsets):
            if batched_rows is None:
                if self.preserve_uncertain_seed:
                    offset = torch.cat(
                        (float(dx) * search_scale, float(dy) * search_scale), dim=1
                    )
                else:
                    offset = up_flow.new_tensor((float(dx), float(dy)))[None, :, None, None]
                    offset = offset.expand(batch, -1, side, side)
                coordinates = center + offset.permute(0, 2, 3, 1)
                sampled, valid = self._sample(second_feature, coordinates)
            else:
                sampled = batched_rows[0][:, index]
                valid = batched_rows[1][:, index]
                offset = batched_rows[2][:, index]
"""


def _newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_text(source: str) -> tuple[str, dict[str, object]]:
    tree = ast.parse(source)
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "_DenseRawFlowRefiner"
    ]
    if len(classes) != 1:
        raise RuntimeError(
            f"expected one _DenseRawFlowRefiner class, found {len(classes)}"
        )
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    class_start = offsets[classes[0].lineno - 1]
    class_end = offsets[classes[0].end_lineno]
    prefix, body, suffix = (
        source[:class_start],
        source[class_start:class_end],
        source[class_end:],
    )
    if re.search(r"^\s+def _batched_samples\(", body, re.MULTILINE):
        return source, {"already_applied": True, "changed": False}

    buffer_anchor = "        self.activation_checkpoint = bool(activation_checkpoint)\n"
    body = _replace_once(
        body,
        buffer_anchor,
        buffer_anchor + _BUFFER,
        label="stencil buffer anchor",
    )

    helper_anchor = "    def _forward_tensors(\n"
    body = _replace_once(
        body,
        helper_anchor,
        _HELPER + helper_anchor,
        label="forward helper anchor",
    )

    loop_pattern = re.compile(
        r"        for dx, dy in offsets:\n"
        r".*?"
        r"            if self\.normalization_floor is None:\n",
        re.DOTALL,
    )
    matches = list(loop_pattern.finditer(body))
    if len(matches) != 1:
        raise RuntimeError(f"offset loop anchor: expected one match, found {len(matches)}")
    match = matches[0]
    old_loop = match.group(0)
    replacement = _LOOP + "            if self.normalization_floor is None:\n"
    body = body[: match.start()] + replacement + body[match.end() :]
    return prefix + body + suffix, {
        "already_applied": False,
        "changed": True,
        "old_loop_sha256": hashlib.sha256(old_loop.encode()).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    for path in args.paths:
        raw = path.read_bytes()
        before = hashlib.sha256(raw).hexdigest()
        newline = _newline(raw)
        # Work on LF text so the exact anchors are independent of the
        # snapshot's checkout line-ending policy; restore it below.
        source = raw.decode("utf-8").replace("\r\n", "\n")
        updated, report = patch_text(source)
        updated = updated.replace("\n", newline)
        after = hashlib.sha256(updated.encode("utf-8")).hexdigest()
        print(
            f"path={path.resolve()} write={int(args.write)} "
            f"changed={int(bool(report.get('changed')))} "
            f"before_sha256={before} after_sha256={after} "
            f"newline={'crlf' if newline == chr(13) + chr(10) else 'lf'}"
        )
        if args.write and updated.encode("utf-8") != raw:
            temp = path.with_name(path.name + ".codex-batched.tmp")
            temp.write_bytes(updated.encode("utf-8"))
            temp.replace(path)


if __name__ == "__main__":
    main()
