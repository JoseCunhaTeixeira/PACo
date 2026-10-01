"""Positions along the line, as a person gives them (metres), to the run's windows: each the
nearest window's, the mapping said back; a position beyond the line's windows refused."""

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, cast

from sigpipe.masw.runs import RunError, RunManifest


def shown_xmid(xmid: float) -> str:
    """A window's xmid as its folder and the gates' summaries write it: to the centimetre."""
    return f"{xmid:.2f}".rstrip("0").rstrip(".")


def at_positions(manifest: RunManifest, positions: Sequence[float]) -> tuple[list[str], str]:
    """The windows of the run nearest `positions` (m), each once, in line order, and how each
    position was read ("30 m: xmid 30.75"). Refuses a position more than a window step beyond
    the first or last window."""
    windows = sorted(manifest.windows, key=lambda window: window.xmid)
    if not windows:
        raise RunError(f"Run '{manifest.run_id}' has no window.")
    xmids = [window.xmid for window in windows]
    step = min((b - a for a, b in pairwise(xmids)), default=0.0)
    reach = (xmids[0] - step, xmids[-1] + step)
    if outside := [one for one in positions if not reach[0] <= one <= reach[1]]:
        raise RunError(
            f"{', '.join(f'{one:g} m' for one in outside)} "
            f"{'is' if len(outside) == 1 else 'are'} off the line of run '{manifest.run_id}': "
            f"its windows are at xmid {shown_xmid(xmids[0])} to {shown_xmid(xmids[-1])} m."
        )
    chosen: dict[str, float] = {}
    said: list[str] = []
    for position in positions:
        nearest = min(windows, key=lambda window: abs(window.xmid - position))
        chosen[nearest.folder] = nearest.xmid
        said.append(f"{position:g} m: xmid {shown_xmid(nearest.xmid)}")
    units = sorted(chosen, key=lambda unit: chosen[unit])
    return units, "; ".join(said)


def in_receivers(
    overrides: Mapping[str, Any] | None, spacing_m: float
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """`overrides` with a window's length and step given in metres (masw.length_m, step_m) in
    receivers, the nearest count (a length of n receivers spans n - 1 spacings), and notes
    saying each conversion. Refused when a length leaves fewer than two receivers."""
    if not overrides:
        return None if overrides is None else dict(overrides), ()
    masw = overrides.get("masw")
    if not isinstance(masw, Mapping) or not {"length_m", "step_m"} & set(masw):
        return dict(overrides), ()
    converted = dict(cast(Mapping[str, Any], masw))
    notes: list[str] = []
    if (length_m := converted.pop("length_m", None)) is not None:
        length = round(float(length_m) / spacing_m) + 1
        if length < 2:
            raise ValueError(
                f"masw length_m must be over the receivers' spacing ({spacing_m:g} m): "
                f"{length_m:g} m holds fewer than two receivers."
            )
        converted["length"] = length
        notes.append(
            f"masw length_m {length_m:g} m: {length} receivers ({(length - 1) * spacing_m:g} m)."
        )
    if (step_m := converted.pop("step_m", None)) is not None:
        step = max(1, round(float(step_m) / spacing_m))
        converted["step"] = step
        notes.append(f"masw step_m {step_m:g} m: {step} receivers ({step * spacing_m:g} m).")
    return {**overrides, "masw": converted}, tuple(notes)
