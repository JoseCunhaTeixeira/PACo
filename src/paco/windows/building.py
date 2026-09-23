"""Building the MASW windows of a profile, as PAC's build_windows does."""

import logging
from pathlib import Path

from paco.profiles import Profile
from paco.windows.models import MASWParameters, MASWWindow
from sigpipe.base import Coordinate, LinearAcquisition

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
