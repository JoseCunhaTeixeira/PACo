"""The mute trial: a mute kept only where it gives more windows passing G3 than no mute, without
losing the long wavelengths; never over a muting the user gave, nor on a passive line."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sigpipe.base.acquisition import LinearAcquisition
from sigpipe.base.coordinate import Coordinate
from sigpipe.base.stream import Stream
from sigpipe.masw.presets import make_preset
from sigpipe.masw.runs import find_run, list_runs, load_manifest

from paco.evaluation.defects import build_inputs
from paco.qc import LINE, QCConfig, compare_settings, latest, process_line, read_attempts
from paco.qc.g1_signal import SignalThresholds, judge_fast_arrivals
from paco.qc.models import GateResult, Override
from paco.qc.muting import (
    MuteChoice,
    MuteRules,
    MuteTrial,
    _decide,  # pyright: ignore[reportPrivateUsage]
    describe_mutes,
    given_muting,
    mutable,
    read_mute_choice,
)
from paco.settings import Settings

CONE: dict[str, Any] = {"method": "mute", "vmin": 113.5, "vmax": 332.7, "width": 0.05, "taper": 20}
WIDER: dict[str, Any] = {"method": "mute", "vmin": 75.6, "vmax": 499.1, "width": 0.05, "taper": 20}
STANDARD: dict[str, Any] = {
    "method": "mute",
    "vmin": 80.0,
    "vmax": 1500.0,
    "width": 0.05,
    "taper": 20,
}


def _trial(candidate: str, muting: dict[str, Any] | None, passed: int, longest: float) -> MuteTrial:
    return MuteTrial(
        candidate=candidate,  # pyright: ignore[reportArgumentType]
        muting=muting,
        verdicts=("pass",) * passed + ("reject",) * (15 - passed),
        passed=passed,
        wavelengths_m=(5.0, longest),
    )


def test_a_mute_is_kept_only_with_enough_more_windows_passing() -> None:
    rules = MuteRules()
    none = _trial("none", None, 12, 39.5)

    # +1 is within what the trial windows vary by: no mute kept.
    kept, checked = _decide([none, _trial("cone", CONE, 13, 55.5)], rules)
    assert kept.candidate == "none"
    assert checked[1].refused == "+1 trial windows passing against no mute, 2 more needed"
    # +2, and the earlier of equals, the less cutting: the standard mute before the wider cone.
    kept, _ = _decide(
        [none, _trial("standard", STANDARD, 14, 60), _trial("wider", WIDER, 14, 59),
         _trial("cone", CONE, 13, 53)],
        rules,
    )  # fmt: skip
    assert (kept.candidate, kept.muting) == ("standard", STANDARD)


def test_a_mute_that_loses_the_long_wavelengths_is_refused() -> None:
    # Over-muting: more windows pass, but the curves no longer reach the long wavelengths the
    # far offsets and the low band give.
    none = _trial("none", None, 10, 40.0)
    shorter = _trial("cone", CONE, 14, 30.0)

    kept, checked = _decide([none, shorter], MuteRules())

    assert kept.candidate == "none"
    assert checked[1].refused is not None and "its longest wavelengths (30 m)" in checked[1].refused
    # No mute passing a single window says nothing of what the line reaches: a gather swamped
    # by refractions, which the tight cone takes out.
    swamped = _trial("none", None, 1, 16.0)
    tight = _trial("tight", CONE, 7, 11.0)
    assert _decide([swamped, tight], MuteRules())[0].candidate == "tight"


def test_the_trial_says_each_candidate_and_why_it_was_not_kept(tmp_path: Path) -> None:
    trials = [
        _trial("none", None, 13, 17.0),
        _trial("cone", CONE, 14, 17.0).model_copy(
            update={"refused": "+1 trial windows passing against no mute, 2 more needed"}
        ),
    ]
    choice = MuteChoice(
        chosen="none",
        muting=None,
        cone_m_s=(170.2, 185.0, 221.8),
        length=9,
        xmids=(1.0, 2.0),
        trials=tuple(trials),
        notes=("muting none: the mute trial.",),
    )
    (tmp_path / "mute.json").write_text(choice.model_dump_json())

    assert describe_mutes(choice) == (
        "no mute (kept): 13 of 15 trial windows passed G3, wavelengths 5 to 17 m",
        "the cone 113.5 to 332.7 m/s: 14 of 15 trial windows passed G3, wavelengths 5 to 17 m; "
        "not kept: +1 trial windows passing against no mute, 2 more needed",
    )
    assert read_mute_choice(tmp_path) == choice
    assert read_mute_choice(tmp_path / "elsewhere") is None


@pytest.mark.parametrize(
    ("mode", "can"), [("active", True), ("passive-active", True), ("passive", False)]
)
def test_only_a_shots_line_is_muted(mode: str, can: bool) -> None:
    assert mutable(make_preset(mode)) == can


def test_a_muting_the_user_gave_has_no_trial() -> None:
    assert given_muting({"muting": {"method": "mute", "vmin": 100, "vmax": 600}})
    assert not given_muting({"masw": {"length": 24}})
    assert not given_muting(None)


def _shot(refraction: float) -> Stream:
    """A surface wave at 200 m/s from a source 2 m before the first of 24 receivers 1 m apart,
    and with `refraction`, a head wave at 1,200 m/s of that amplitude relative to it."""
    rng = np.random.default_rng(0)
    ts = np.arange(1000) / 1000.0
    receivers = tuple(Coordinate(2.0 + i, 0.0, 0.0) for i in range(24))
    acquisition = LinearAcquisition(source=Coordinate(0.0, 0.0, 0.0), receivers=receivers)
    xt = rng.standard_normal((24, ts.size)) * 0.01

    def wavelet(tau: np.ndarray) -> np.ndarray:
        return np.where(tau >= 0, tau * np.exp(-40 * tau) * np.cos(40 * np.pi * tau), 0.0) * 108.7

    for i, offset in enumerate(acquisition.offsets):
        xt[i] += wavelet(ts - offset / 200.0) + refraction * wavelet(ts - offset / 1200.0)
    return Stream(xt=xt, ts=ts, sampling_freq=1000.0, acquisition=acquisition)


def test_g1_mutes_a_record_whose_arrivals_outrun_the_cone() -> None:
    thresholds = SignalThresholds()
    passed = GateResult(gate="G1", unit="1.mseed", verdict="pass")

    clean = judge_fast_arrivals(passed, _shot(0.0), 0.0, CONE, thresholds, 3.0)
    refracted = judge_fast_arrivals(passed, _shot(1.0), 0.0, CONE, thresholds, 3.0)

    (measured,) = clean.metrics
    assert clean.verdict == "pass" and measured.name == "fast_arrivals" and measured.passed
    # A head wave as strong as the surface wave: the record muted around the cone.
    assert refracted.verdict == "retry"
    (flag,) = refracted.flags
    assert flag.name == "fast_arrivals" and isinstance(flag.action, Override)
    assert flag.action.overrides == {"muting": CONE}


def test_the_demo_line_is_tried_with_mutes_and_kept_unmuted(
    demo_input_dir: Path, tmp_path: Path
) -> None:
    settings = Settings(input_dir=demo_input_dir, output_dir=tmp_path, workers=2)

    report = process_line("active_p1", {"masw": {"length": 24, "step": 24}}, settings, QCConfig())

    run_folder = find_run(report.run_id, settings)
    choice = read_mute_choice(run_folder)
    # Five candidates on the same trial windows; no mute gave 2 more windows passing G3 than
    # none on the demo's clean shots: the line stays unmuted, as it was.
    assert choice is not None and choice.chosen == "none" and choice.muting is None
    assert [trial.candidate for trial in choice.trials] == [
        "none", "standard", "wider", "cone", "tight",
    ]  # fmt: skip
    assert load_manifest(report.run_id, settings).preset.muting.method == "none"  # pyright: ignore[reportAttributeAccessIssue]
    # Nothing changed: no note among the line's changes; the trial's candidates in mute.json.
    line = latest(read_attempts(run_folder), LINE, "phase_shift")
    assert line is not None and not any(note.startswith("muting") for note in line.notes)
    # Given by the user, a muting is theirs: no trial.
    muted = {
        "masw": {"length": 24, "step": 24},
        "muting": {"method": "mute", "vmin": 100.0, "vmax": 900.0},
    }
    given = process_line("active_p1", muted, settings, QCConfig())
    assert read_mute_choice(find_run(given.run_id, settings)) is None


def test_compare_tries_settings_apart_from_the_runs(demo_input_dir: Path, tmp_path: Path) -> None:
    settings = Settings(input_dir=demo_input_dir, output_dir=tmp_path, workers=2)
    variants = [{"masw": {"length_m": 2.75}}, {"masw": {"length": 24}}]

    compared = compare_settings("active_p1", variants, "depth", settings, QCConfig())

    first, second = compared.variants
    # The length in metres converted: 2.75 m are 12 receivers 0.25 m apart.
    assert (first.length, second.length) == (12, 24)
    assert first.windows == second.windows == 9
    depths = {variant.label: variant.depth_m for variant in compared.variants}
    assert compared.best == max(depths, key=lambda label: depths[label] or 0.0)
    assert compared.did.startswith("Compared 2 variants of active_p1 on the depth of investigation")
    assert compared.table[0].startswith("variant 1") and "(best)" in " ".join(compared.table)
    # No run made or changed: the trials sit in the profile's compare folder.
    assert list_runs(settings) == []
    assert (tmp_path / "active_p1" / "compare").is_dir()


def test_strong_refractions_are_muted_with_the_tight_cone(
    demo_input_dir: Path, tmp_path: Path
) -> None:
    # The evaluation's active_refracted: a head wave as strong as the surface waves, which
    # pulls the gathers' fast quartile up; their median holds, and its tight cone takes the
    # head wave out.
    inputs = build_inputs(demo_input_dir, tmp_path / "inputs")
    settings = Settings(input_dir=inputs, output_dir=tmp_path / "outputs", workers=2)

    report = process_line("active_refracted", None, settings, QCConfig())

    choice = read_mute_choice(find_run(report.run_id, settings))
    assert choice is not None and choice.chosen == "tight"
    none, *_, tight = choice.trials
    assert tight.passed >= none.passed + MuteRules().min_gain
