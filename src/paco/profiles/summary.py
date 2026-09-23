"""The short description of a profile that the agent reads."""

import statistics
from itertools import pairwise

from paco.profiles.loading import load_profile
from paco.profiles.models import Profile, ProfileSummary
from paco.settings import Settings


def inspect_profile(name: str, settings: Settings) -> ProfileSummary:
    return summarize(load_profile(name, settings))


def summarize(profile: Profile) -> ProfileSummary:
    receiver_xs = [receiver.x for receiver in profile.receivers]
    durations = [record.duration_s for record in profile.records]
    source_xs = [record.source.x for record in profile.records if record.source is not None]

    return ProfileSummary(
        name=profile.name,
        kind=profile.kind,
        n_records=len(profile.records),
        n_receivers=len(profile.receivers),
        receiver_x_range_m=(receiver_xs[0], receiver_xs[-1]),
        receiver_spacing_m=statistics.median(b - a for a, b in pairwise(receiver_xs)),
        sampling_rate_hz=profile.sampling_rate_hz,
        nyquist_hz=profile.nyquist_hz,
        record_duration_range_s=(round(min(durations), 3), round(max(durations), 3)),
        source_x_range_m=(min(source_xs), max(source_xs)) if source_xs else None,
    )
