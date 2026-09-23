"""Resolve real CALVIN annotation endpoints without manufacturing success labels."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .hdf5_episode import RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING, LoadedEpisode


@dataclass(frozen=True)
class AnnotationEndpointSource:
    index: int | None
    status: str


def resolve_annotation_endpoint(episode: LoadedEpisode) -> AnnotationEndpointSource:
    """Unknown/censored labels keep BC examples; contradictory metadata is an error.

    This deliberately does not use valid_center_end, window length, action
    horizon, filename, task text or a simulated success oracle.
    """
    values = (
        episode.source_annotation_index,
        episode.context_start,
        episode.source_start,
        episode.source_end,
        episode.terminal_state_index,
    )
    for value in values:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        ):
            raise ValueError("annotation endpoint provenance must use nonnegative integer indices")
    if any(value is None for value in values):
        return AnnotationEndpointSource(None, "unknown-provenance")
    annotation, context, start, end, terminal = values
    assert (
        annotation is not None
        and context is not None
        and start is not None
        and end is not None
        and terminal is not None
    )
    if not context <= start < end:
        raise ValueError("annotation endpoint source_start/source_end/context_start contradict")
    local = end - context
    if episode.terminal_padding_mode != RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING:
        return AnnotationEndpointSource(None, "unverified-terminal-convention")
    if terminal < local or local >= episode.length:
        # A stored/cropped prefix is not a labelled goal observation.
        return AnnotationEndpointSource(None, "censored-before-annotation-end")
    if terminal != local:
        raise ValueError("annotation endpoint does not match declared last real observation")
    if local >= episode.cached_frame_count:
        raise ValueError(
            "annotation endpoint visual cache does not contain the real end observation"
        )
    if episode.states_raw is None or local >= len(episode.states_raw):
        raise ValueError("annotation endpoint state observation is absent")
    return AnnotationEndpointSource(int(local), "annotated-end-observation")
