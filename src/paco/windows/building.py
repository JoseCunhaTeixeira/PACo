"""Building the MASW windows of a profile, as PAC's build_windows does."""

import logging
from pathlib import Path

from sigpipe.base import Coordinate, LinearAcquisition

from paco.profiles import Profile
from paco.windows.models import Exclusions, MASWParameters, MASWWindow

logger = logging.getLogger(__name__)


def build_windows(profile: Profile, params: MASWParameters) -> list[MASWWindow]:
    n_receivers = len(profile.receivers)
    if params.length > n_receivers:
        raise ValueError(
            f"length ({params.length}) exceeds the {n_receivers} receivers "
            f"of profile '{profile.name}'."
        )

    windows: list[MASWWindow] = []

    for start in range(0, n_receivers - params.length + 1, params.step):
        stop = start + params.length
        receivers = profile.receivers[start:stop]
        xmin, xmax = receivers[0].x, receivers[-1].x

        # sigpipe finds the window middle by arc length along (x, z), as PAC does. The source
        # plays no part in it, so the first receiver fills that slot.
        xmid = LinearAcquisition(source=receivers[0], receivers=receivers).xmid

        selected_files: list[Path] = []
        acquisitions: list[LinearAcquisition] = []

        for record in profile.records:
            source: Coordinate
            if record.source is None:
                # Passive: every record serves every window. LinearAcquisition needs a known
                # source, so the first receiver stands in for it; cross-correlation redefines
                # the source later anyway.
                source = receivers[0]
            else:
                source = record.source

                # The source must be outside the window...
                if xmin < source.x < xmax:
                    continue

                # ...and at a usable distance from its middle.
                distance = abs(source.x - xmid)
                if distance <= params.distance_min or distance >= params.distance_max:
                    continue

            selected_files.append(record.path)
            acquisitions.append(LinearAcquisition(source=source, receivers=receivers))

        if not selected_files:
            logger.warning("No valid shots for xmid=%.2f", xmid)
            continue

        windows.append(
            MASWWindow(
                xmid=xmid,
                selected_files=selected_files,
                receiver_indices=list(range(start, stop)),
                acquisitions=acquisitions,
            )
        )

    logger.info("Built %d valid MASW windows", len(windows))

    return windows


def apply_exclusions(window: MASWWindow, exclusions: Exclusions) -> MASWWindow | None:
    """`window` without the records and traces G1 excluded; None when fewer than 3 receivers or
    no record are left. The xmid stays: an exclusion never moves a window."""
    kept = [
        index
        for index, path in enumerate(window.selected_files)
        if path.name not in exclusions.records
    ]
    files = [window.selected_files[index] for index in kept]
    dropped = {trace for path in files for trace in exclusions.traces.get(path.name, ())}
    positions = [
        position
        for position, receiver in enumerate(window.receiver_indices)
        if receiver not in dropped
    ]
    if not files or len(positions) < 3:
        return None
    acquisitions = [
        LinearAcquisition(
            source=window.acquisitions[index].source,
            receivers=tuple(
                window.acquisitions[index].receivers[position] for position in positions
            ),
        )
        for index in kept
    ]
    return MASWWindow(
        xmid=window.xmid,
        selected_files=files,
        receiver_indices=[window.receiver_indices[position] for position in positions],
        acquisitions=acquisitions,
    )
