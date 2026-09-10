"""Run the training profiler with the compiler-visible factual lab operator.

This entry point is intentionally laboratory-only.  It replaces the symbol
that the existing ``--atomic-factual-contraction`` adapter imports, then
delegates to the ordinary profiler.  Model source and serialized state are not
changed, and the accepted factual-atomic implementation remains the default.
"""

from __future__ import annotations

import os

from probe_factual_atomic_layout import (
    visible_matrix_contraction,
    visible_native_bmm_contraction,
)

from clearvla.mainline.training import factual_contraction
from clearvla.tools.profile_mainline_training import main


def _main() -> None:
    mode = os.environ.get(
        "CLEARVLA_FACTUAL_LAYOUT_CANDIDATE",
        "opaque-matrix",
    ).strip().lower()
    candidates = {
        "opaque-matrix": visible_matrix_contraction,
        "native-bmm": visible_native_bmm_contraction,
    }
    try:
        candidate = candidates[mode]
    except KeyError as error:
        raise ValueError(
            "CLEARVLA_FACTUAL_LAYOUT_CANDIDATE must be one of "
            f"{tuple(candidates)!r}; received {mode!r}"
        ) from error
    factual_contraction.atomic_factual_rgb_detail_contraction = candidate
    main()


if __name__ == "__main__":
    _main()
