"""Processing settings compared on a sample of the line's windows (L10 of PACo's agent
guidelines): the model chooses the variants and the metric; the code runs each variant on the
same spread of windows (its records preprocessed with it, imaged, picked, judged by G3, as the
window-length and mute trials do) and computes the metric. Nothing of a run changes: the trials
are written apart, under the profile's compare/ folder.

The metrics, over the curves G3 passed (medians):
- depth: half the longest wavelength, MASW's depth of investigation (the priors' max_depth);
- band: the band the curves cover, Hz;
- curve_length: the wavelengths they span (longest minus shortest), m;
- windows_passing: the trial windows G3 passed."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.presets import apply_overrides, make_preset, resolve_preset
from sigpipe.masw.profiles import load_profile
from sigpipe.masw.runs.processing import RECORDS_FOLDER, preprocess_records
from sigpipe.masw.windows import build_windows

from paco import stopping
from paco.qc.coherence import TrialJudge, receiver_spacing, trial_indices, try_windows
from paco.qc.config import QCConfig
from paco.qc.positions import in_receivers
from paco.settings import Settings

COMPARE_FOLDER = "compare"  # in the profile's output folder: one folder per comparison
# The windows each variant is tried on, spread along the line.
TRIAL_WINDOWS = 9
type CompareMetric = Literal["depth", "band", "curve_length", "windows_passing"]
# What each metric measures, in words, and its unit.
METRICS: dict[CompareMetric, tuple[str, str]] = {
    "depth": ("depth of investigation, half the longest wavelength", "m"),
    "band": ("band the curves cover", "Hz"),
    "curve_length": ("wavelengths the curves span", "m"),
    "windows_passing": ("trial windows G3 passed", ""),
}


class Variant(BaseModel):
    """One variant on its trial windows: its settings, the windows G3 passed, what the passed
    curves reach, and the metric's value (None: no curve passed)."""

    model_config = ConfigDict(frozen=True)

    label: str
    settings: dict[str, Any]
    length: int  # receivers of its windows
    passed: int
    windows: int
    depth_m: float | None
    band_hz: tuple[float, float] | None
    wavelengths_m: tuple[float, float] | None
    value: float | None


class Comparison(BaseModel):
    """The variants side by side on one metric, and the best (the highest value; ties to the
    earlier)."""

    model_config = ConfigDict(frozen=True)

    profile: str
    metric: CompareMetric
    did: str  # the comparison in one line, for the answer
    variants: tuple[Variant, ...]
    best: str | None  # the best variant's label; None when no variant gave a curve
    table: tuple[str, ...]  # one line each, for the agent
    next: str


def compare_settings(
    profile: str,
    variants: Sequence[Mapping[str, Any]],
    metric: CompareMetric,
    settings: Settings,
    config: QCConfig,
) -> Comparison:
    """`variants` (run_processing's overrides each: a length in metres converted) of `profile`,
    compared on `metric` over TRIAL_WINDOWS windows each."""
    loaded = load_profile(profile, settings)
    folder = settings.output_dir / profile / COMPARE_FOLDER / f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    judge = TrialJudge(config.coherence, config.curve, config.picking, config.image)
    spacing = receiver_spacing(loaded)
    tried: list[Variant] = []
    for index, given in enumerate(variants, start=1):
        stopping.check()
        overrides, _ = in_receivers(given, spacing)
        overrides = dict(overrides or {})
        mode = str(overrides.pop("mode", loaded.kind))
        preset = resolve_preset(make_preset(mode, overrides), loaded)
        trial_preset = resolve_preset(apply_overrides(preset, {"masw": {"step": 1}}), loaded)
        windows = build_windows(loaded, trial_preset.masw)
        label = f"variant {index}"
        if not windows:
            tried.append(_empty(label, dict(given), preset.masw.length))
            continue
        chosen = [windows[at] for at in trial_indices(len(windows), TRIAL_WINDOWS)]
        needed = sorted({path.name for window in chosen for path in window.selected_files})
        own = folder / str(index)
        records = preprocess_records(
            trial_preset,
            loaded,
            own,
            settings.workers,
            presets=dict.fromkeys(needed, trial_preset),
            stop=stopping.current(),
        )
        result = try_windows(
            trial_preset,
            loaded,
            chosen,
            records,
            own / RECORDS_FOLDER,
            own / "windows",
            judge,
            settings.workers,
        )
        depth = (
            round(config.priors.max_depth * result.wavelengths_m[1], 1)
            if result.wavelengths_m is not None
            else None
        )
        values: dict[CompareMetric, float | None] = {
            "depth": depth,
            "band": _span(result.band_hz),
            "curve_length": _span(result.wavelengths_m),
            "windows_passing": float(result.passed),
        }
        tried.append(
            Variant(
                label=label,
                settings=dict(given),
                length=trial_preset.masw.length,
                passed=result.passed,
                windows=len(result.verdicts),
                depth_m=depth,
                band_hz=result.band_hz,
                wavelengths_m=result.wavelengths_m,
                value=values[metric],
            )
        )
    scored = [variant for variant in tried if variant.value is not None]
    best = max(scored, key=lambda variant: cast(float, variant.value)) if scored else None
    what, unit = METRICS[metric]
    others = [variant for variant in scored if variant is not best]
    did = (
        f"Compared {len(tried)} variants of {profile} on the {what}, {TRIAL_WINDOWS} trial windows each: "
        + (
            f"{best.label} best, {best.value:g}{f' {unit}' if unit else ''}"
            + (" against " + ", ".join(f"{one.value:g}" for one in others) if others else "")
            + "."
            if best is not None and best.value is not None
            else "no variant gave a curve G3 passes."
        )
    )
    return Comparison(
        profile=profile,
        metric=metric,
        did=did,
        variants=tuple(tried),
        best=best.label if best is not None else None,
        table=tuple(_line(variant, metric, best) for variant in tried),
        next=(
            f"Say which is best and why. Processing the line with {best.label}'s settings is "
            f'the user\'s to ask: run_processing(profile="{profile}", overrides={best.settings}).'
            if best is not None
            else "No variant gave a curve G3 passes: other settings, or the data."
        ),
    )


def _span(values: tuple[float, float] | None) -> float | None:
    return round(values[1] - values[0], 1) if values is not None else None


def _empty(label: str, given: dict[str, Any], length: int) -> Variant:
    """A variant whose windows hold no shot: nothing to judge."""
    return Variant(
        label=label,
        settings=given,
        length=length,
        passed=0,
        windows=0,
        depth_m=None,
        band_hz=None,
        wavelengths_m=None,
        value=None,
    )


def _line(variant: Variant, metric: CompareMetric, best: Variant | None) -> str:
    """A variant in words: its settings, its windows, the metric, what its curves reach."""
    what, unit = METRICS[metric]
    value = (
        "no curve passed"
        if variant.value is None
        else f"{what} {variant.value:g}{f' {unit}' if unit else ''}"
    )
    reach = []
    if variant.wavelengths_m is not None:
        reach.append(f"wavelengths {variant.wavelengths_m[0]:g} to {variant.wavelengths_m[1]:g} m")
    if variant.band_hz is not None:
        reach.append(f"{variant.band_hz[0]:g} to {variant.band_hz[1]:g} Hz")
    mark = " (best)" if best is not None and variant.label == best.label else ""
    return (
        f"{variant.label}{mark} {variant.settings}: windows of {variant.length} receivers, "
        f"{variant.passed} of {variant.windows} trial windows passed G3; {value}"
        + (f"; {', '.join(reach)}" if reach else "")
    )
