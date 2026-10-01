"""Positions along the line, as a person gives them (metres), to the run's windows: each the
nearest window's, the mapping said back; a position beyond the line's windows refused."""

from collections.abc import Sequence
from itertools import pairwise

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
