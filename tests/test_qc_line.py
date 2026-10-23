"""The coherence rules for S2 and a line processed the QC way: the band capped by what the
records keep usable, the window length from the ladder of trial windows, the line judged up to
G4 with the rules' changes in the log."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters
from sigpipe.masw.pipelines.common import stage_kwargs
from sigpipe.masw.presets import make_preset
from sigpipe.masw.profiles import Profile
from sigpipe.masw.runs import find_run, load_manifest

from paco.qc import QCConfig, coherence, latest, read_attempts, summarize_report
from paco.qc.coherence import (
    COHERENCE_FILE,
    CoherenceRules,
    LengthChoice,
    LengthTrial,
    TrialJudge,
    cap_band,
    choose_length,
    describe_lengths,
    given_length,
    length_hint,
    trial_indices,
)
from paco.qc.g3_curve import CurveThresholds
from paco.qc.line import process_line
from paco.qc.report import QCReport
from paco.qc.segments import SEGMENTS_FILE
from paco.settings import Settings

# Three rungs and three trial windows, up to half the line: the ladder of the test line in a few
# seconds.
QUICK = QCConfig(
    coherence=CoherenceRules(
        lengths=(5, 24, 32), trials=3, min_pass_share=2 / 3, max_line_share=0.5
    )
)


def test_the_band_is_capped_by_the_records_and_nyquist_never_widened() -> None:
    preset = make_preset("active", {"dispersion": {"fmin": 0.0, "fmax": 500.0}})

    # The records' median usable band, not the worst record's.
    overrides, notes = cap_band(preset, [(0.0, 349.5), (0.0, 324.0), None], nyquist=1000.0)
    assert overrides == {"dispersion": {"fmax": 336.8}}
    assert notes == (
        "dispersion fmax 500 Hz is above the records' median usable band (up to 336.8 Hz): set "
        "to 336.8.",
    )
    # One record usable only from 24 Hz among many from 2: the line is not cut there.
    many = [(2.0, 300.0)] * 8 + [(24.0, 86.0)]
    assert cap_band(preset, many, 1000.0)[0] == {"dispersion": {"fmin": 2.0, "fmax": 300.0}}
    # PAC's 100 Hz is within: nothing changes, though the records go higher.
    assert cap_band(make_preset("active", None), [(0.0, 324.0)], 1000.0) == ({}, ())
    # A usable band starting above fmin raises it; Nyquist caps when no record says more.
    overrides, _ = cap_band(preset, [(5.0, 800.0)], nyquist=400.0)
    assert overrides == {"dispersion": {"fmax": 400.0, "fmin": 5.0}}
    # A band wholly below the usable one would be empty: the one case fmax goes up.
    overrides, notes = cap_band(make_preset("active", None), [(122.4, 480.0)], 2400.0)
    assert overrides == {"dispersion": {"fmin": 122.4, "fmax": 480.0}}
    assert notes[-1] == (
        "dispersion fmax 100 Hz lies below the records' median usable band (122.4-480.0 Hz): set to "
        "480.0."
    )


def test_the_ladder_defaults_to_27_trials_and_four_fifths() -> None:
    # 80 % of the trials, 27 of them over the whole line, the autonomous search's lengths up to a
    # quarter of the line: a line where none passes takes the one that passed most.
    rules = CoherenceRules()
    assert (rules.trials, rules.min_pass_share, rules.max_line_share) == (27, 0.8, 0.25)
    assert rules.lengths == (5, 7, 9, 11, 13, 15, 17, 21, 25, 33, 41, 47)


def test_the_trial_windows_cover_the_whole_line() -> None:
    # 27 spread evenly, the ends included: their share passing G3 is the line's.
    assert CoherenceRules().trials == 27
    assert trial_indices(92, 9) == [0, 11, 23, 34, 46, 57, 68, 80, 91]
    assert len(trial_indices(92, 27)) == 27
    assert trial_indices(4, 9) == [0, 1, 2, 3]


def test_the_users_length_is_read_from_the_overrides() -> None:
    assert given_length({"masw": {"length": 24, "step": 4}}) == 24
    assert given_length({"masw": {"step": 4}}) is None
    assert given_length(None) is None


def _line(
    demo_input_dir: Path, root: Path, overrides: dict[str, object], config: QCConfig
) -> tuple[Path, QCReport]:
    settings = Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        report = process_line("active_test_profile", overrides, settings, config)
    return find_run(report.run_id, settings), report


@pytest.mark.slow
def test_the_ladder_keeps_the_most_precise_length_that_passes(
    demo_input_dir: Path, tmp_path: Path
) -> None:
    # The step keeps the line to four windows; the length is left to the ladder.
    run_folder, report = _line(demo_input_dir, tmp_path, {"masw": {"step": 24}}, QUICK)

    choice = json.loads((run_folder / COHERENCE_FILE).read_text())
    # At 5 receivers (6 m) the trial windows keep too few points: none of the 3 passes. 24
    # passes 2 of 3, the share asked, with picks within 8 %, under the 20 % at which the climb
    # stops: 24, the most precise length that passed and the first within 20 %, is kept; 32 is
    # tried too, for the agent to compare the depth it would reach.
    assert [trial["length"] for trial in choice["trials"]] == [5, 24, 32]
    assert [trial["compared"] for trial in choice["trials"]] == [False, False, True]
    five, twenty_four, _ = choice["trials"]
    assert (five["passed"], len(five["xmids"])) == (0, 3)
    assert (twenty_four["passed"], twenty_four["uncertainty"]) == (2, pytest.approx(0.076))
    assert choice["length"] == 24
    # What the agent reads to choose another length: the line, then each length tried.
    assert choice["receivers"] == 96 and choice["spacing_m"] == 1.5
    assert (five["metres"], twenty_four["metres"]) == (6.0, 34.5)
    assert (five["windows"], twenty_four["windows"]) == (4, 4)  # at the run's step of 24
    assert twenty_four["wavelengths_m"] is not None
    # The line is processed with it, and the rules' change is logged with its reason.
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    manifest = load_manifest(report.run_id, settings)
    assert (manifest.preset.masw.length, manifest.preset.masw.step) == (24, 24)
    assert len(manifest.windows) == 4
    line = latest(read_attempts(run_folder), "line", "phase_shift")
    # The farthest shot a window stacks, from the line's reach at 2 dB (G1, its windows from
    # each record's shot, 20 ms in): xmid 17.25 stacks 149.dat, 54 m away, not 165.dat, 78 m
    # away. The mute trial at the line's length keeps the tight cone (test_qc_muting.py).
    assert line is not None
    assert line.parameters == {
        "masw": {"distance_max": 57.79, "length": 24},
        "muting": {"method": "mute", "vmin": 134.5, "vmax": 302.7, "width": 0.05, "taper": 20},
        "near_field": {"distance_m": 8.5},
    }
    # In the order of PAC's page, the windows' (the farthest shot, the length, the near field),
    # then the preprocessing's (the mute); the line loop's tries after them.
    far, note, near, muted = [one for one in line.notes if not one.startswith("Tried for the line")]
    assert far == (
        "masw distance_max 57.79 m: beyond it from the shot, the traces' median SNR falls under "
        "2 dB (G1), so the windows stack no farther shot."
    )
    assert note.startswith(
        "masw length 24 for the whole line, the first that passed with picks within 20%: "
        "trial windows G3 passed 0/3 at 5, 2/3 at 24 (picks "
    )
    assert note.endswith("1/3 at 32 (picks 5%; compared).")
    assert muted.startswith("muting mute 134.5 to 302.7 m/s: the mute trial ")
    # Out of the near field: half the longest wavelength of the trial curves from a window's
    # nearest receiver, where it has a farther shot whose record G1 keeps. xmid 53.25 has none
    # (101.dat, 117.dat and 165.dat left out): it keeps its near shot, 149.dat.
    assert near.startswith("near_field distance_m 8.5 m: a window stacks no shot nearer than")
    assert near.endswith("; 1 window with near shots only keeps them.")
    # The gates of the processing ran on it (the picking's are pick's), and the summary gives
    # the change.
    assert set(report.counts) == {"G1", "G2"}
    summary = summarize_report(report)
    assert "Changes at phase_shift, line: masw distance_max 57.79 m" in summary
    assert "masw length 24 for the whole line" in summary


@pytest.mark.slow
def test_a_given_length_is_kept_as_it_is(demo_input_dir: Path, tmp_path: Path) -> None:
    # Given, the length stays whatever its trial windows give (the ladder proposes, a length
    # given decides), with nothing to compare: 32 receivers, where 1 trial window of the 3 the
    # rules ask passes, a length the ladder never tries.
    strict = QCConfig(coherence=CoherenceRules(lengths=(5, 24), trials=3, min_pass_share=1.0))
    run_folder, report = _line(
        demo_input_dir, tmp_path, {"masw": {"length": 32, "step": 24}}, strict
    )

    choice = json.loads((run_folder / COHERENCE_FILE).read_text())
    assert [(trial["length"], trial["passed"]) for trial in choice["trials"]] == [(32, 1)]
    assert choice["length"] == 32
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    assert load_manifest(report.run_id, settings).preset.masw.length == 32
    line = latest(read_attempts(run_folder), "line", "phase_shift")
    assert line is not None
    assert line.notes[1].startswith(
        "masw length 32 for the whole line, as given: trial windows G3 passed 1/3 at 32 (picks "
    )
    # At 32 receivers, the mute trial keeps a mute; in the order of PAC's page, the windows'
    # near field before it, the line loop's tries after them.
    assert line.notes[2].startswith("near_field distance_m ")
    assert line.notes[3].startswith("muting mute ")


@pytest.mark.slow
def test_a_record_g1_rejects_goes_into_no_window(demo_input_dir: Path, tmp_path: Path) -> None:
    # 101.dat's median SNR is 4.6 dB, under G1's 6 dB; the traces of 117.dat, and of 133.dat and
    # 165.dat once muted, correlate under its 0.5: a filter or a mute would measure the same, so
    # each record is rejected at once and left out of every window.
    run_folder, report = _line(
        demo_input_dir, tmp_path, {"masw": {"length": 24, "step": 24}}, QUICK
    )

    # The records' verdicts: the line's own G1 result (its receivers) is not one of them.
    assert report.counts["G1"] == {"pass": 3, "reject": 4}
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    assert load_manifest(report.run_id, settings).exclusions.records == (
        "101.dat",
        "117.dat",
        "133.dat",
        "165.dat",
    )
    assert (
        "Changes at preprocessing, 101.dat, 117.dat, 133.dat, 165.dat: left out of every window\n"
    ) in summarize_report(report)


@pytest.mark.slow
def test_an_active_profile_is_processed_passive_active(
    demo_input_dir: Path, tmp_path: Path
) -> None:
    # PAC's third mode: interferometry on the shots, chosen with "mode". G1 judges the records
    # as shots (their trigger corrected in this mode too), and each window images the stack of
    # its shots' correlations.
    run_folder, report = _line(
        demo_input_dir,
        tmp_path,
        {"mode": "passive-active", "masw": {"length": 24, "step": 24}},
        QUICK,
    )

    # The records judged as in active mode: 3 kept; 101.dat, 117.dat, 133.dat and 165.dat left
    # out (test_a_record_g1_rejects_goes_into_no_window).
    assert report.counts["G1"] == {"pass": 3, "reject": 4}
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    manifest = load_manifest(report.run_id, settings)
    assert manifest.preset.mode == "passive-active"
    assert [window.status for window in manifest.windows] == ["succeeded"] * 4
    for window in manifest.windows:
        folder = run_folder / window.folder
        assert (folder / "Stream_0000.hdf5").exists()  # the stacked correlation gather
        assert (folder / "DispersionImage_0000.hdf5").exists()


def test_the_ladder_climbs_while_its_picks_are_loose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Picks at the array's 40 % on short windows, no length more precise than the one before:
    # the climb goes on to a quarter of the line, 24 receivers, not to 48.
    passes = {5: 1, 7: 3, 9: 3, 16: 3, 48: 3}
    uncertainties = {5: 0.37, 7: 0.39, 9: 0.4, 16: 0.36, 48: 0.27}

    def tried(*args: Any) -> LengthTrial:  # noqa: ANN401
        length = cast(int, args[6])
        return LengthTrial(
            length=length,
            xmids=(1.0, 2.0, 3.0),
            verdicts=("pass",) * passes[length] + ("retry",) * (3 - passes[length]),
            flags=(),
            passed=passes[length],
            uncertainty=uncertainties[length],
        )

    monkeypatch.setattr(coherence, "_try_length", tried)
    line = cast(
        Profile, SimpleNamespace(receivers=[SimpleNamespace(x=0.25 * k) for k in range(96)])
    )
    judge = TrialJudge(
        CoherenceRules(lengths=(5, 7, 9, 16, 48), trials=3), CurveThresholds(), PickingParameters()
    )

    choice = choose_length(line, make_preset("active", None), (), tmp_path, judge, 1)

    # None within 20 %: the most precise that passed, 16, every length up to it tried.
    assert choice.length == 16
    assert [(trial.length, trial.compared) for trial in choice.trials] == [
        (5, False),
        (7, False),
        (9, False),
        (16, False),
    ]


def test_the_hint_keeps_the_ladders_length_whatever_the_request() -> None:
    # The test line's ladder with no settings: 11 receivers chosen among 5 to 16. The best curves
    # first: a request for depth or lateral detail keeps it.
    def trial(length: int, passed: int, longest: float | None = None) -> LengthTrial:
        return LengthTrial(
            length=length,
            xmids=tuple(float(x) for x in range(27)),
            verdicts=("pass",) * passed + ("reject",) * (27 - passed),
            flags=(),
            passed=passed,
            wavelengths_m=(5.0, longest) if longest is not None else None,
        )

    choice = LengthChoice(
        length=11,
        trials=tuple(trial(n, k, w) for n, k, w in ((5, 17, 9.0), (11, 23, 16.4), (16, 25, 22.0))),
        notes=(),
    )
    then = "pick comes next for run_id 20260926-050000-abcd."

    assert length_hint(choice, then) == (
        "The window length is the ladder's (11 receivers), the best curves of the lengths tried, "
        "its trial curves reaching about 8.2 m deep: kept, whatever depth or detail the request "
        "asks. pick comes next for run_id 20260926-050000-abcd."
    )
    # No wavelength known: the depth left out.
    alone = choice.model_copy(update={"length": 5, "trials": (trial(5, 27),)})
    assert length_hint(alone, then).startswith(
        "The window length is the ladder's (5 receivers), the best curves of the lengths tried: "
        "kept"
    )


def test_the_lengths_tried_are_said_for_the_agent_to_choose() -> None:
    def trial(length: int, passed: int, wavelengths: tuple[float, float] | None) -> LengthTrial:
        return LengthTrial(
            length=length,
            xmids=(1.0, 2.0, 3.0),
            verdicts=("pass",) * passed + ("retry",) * (3 - passed),
            flags=(),
            passed=passed,
            metres=(length - 1) * 0.25,
            windows=97 - length,
            wavelengths_m=wavelengths,
            compared=length == 24,
        )

    choice = LengthChoice(
        length=16,
        trials=(trial(5, 0, None), trial(16, 3, (5.0, 22.0)), trial(24, 3, (5.0, 26.0))),
        notes=(),
        receivers=96,
        spacing_m=0.25,
        longest=48,
    )

    assert describe_lengths(choice) == (
        "line: 96 receivers 0.25 m apart (23.75 m); windows of up to 48 receivers (half the line)",
        "5 receivers (1.00 m): 0/3 trial windows passed G3, 92 windows on the line",
        "16 receivers (3.75 m): 3/3 trial windows passed G3, wavelengths 5.0-22.0 m (models down to about 11.0 m), 81 windows "
        "on the line (proposed)",
        "24 receivers (5.75 m): 3/3 trial windows passed G3, wavelengths 5.0-26.0 m (models down to about 13.0 m), 73 windows "
        "on the line",
    )


def test_the_ladder_keeps_the_best_proposes_the_shortest_and_keeps_what_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passes = {5: 1, 7: 1, 9: 3, 11: 3, 16: 3, 24: 2}
    uncertainties = {5: 0.4, 7: 0.3, 9: 0.3, 11: 0.25, 16: 0.15, 24: 0.1}

    def tried(*args: Any) -> LengthTrial:  # noqa: ANN401
        length = cast(int, args[6])
        passed = passes[length]
        verdicts = ("pass",) * passed + ("retry",) * (3 - passed)
        return LengthTrial(
            length=length,
            xmids=(1.0, 2.0, 3.0),
            verdicts=verdicts,
            flags=(),
            passed=passed,
            uncertainty=uncertainties[length],
        )

    monkeypatch.setattr(coherence, "_try_length", tried)
    line = cast(
        Profile, SimpleNamespace(receivers=[SimpleNamespace(x=0.25 * k) for k in range(96)])
    )

    def choose(lengths: tuple[int, ...], first: int | None = None) -> LengthChoice:
        judge = TrialJudge(
            CoherenceRules(lengths=lengths, trials=3), CurveThresholds(), PickingParameters()
        )
        return choose_length(line, make_preset("active", None), (), tmp_path, judge, 1, first)

    # None reaches 3 of 3: the one that passed most, the shortest on a tie; nothing compared.
    assert [trial.length for trial in choose((5, 7, 24)).trials] == [5, 7, 24]
    assert choose((5, 7, 24)).length == 24
    assert choose((5, 7)).length == 5
    # Up the ladder to the first length that passes with picks within 20 %; one length more is
    # tried to compare.
    precise = choose((5, 7, 9, 11, 16, 24))
    assert precise.length == 16
    assert [trial.compared for trial in precise.trials] == [False] * 5 + [True]
    # None passes that precise: the most precise that passed, the longer lengths tried kept to
    # compare.
    loose = choose((5, 7, 9, 11, 24))
    assert loose.length == 11
    assert [trial.length for trial in loose.trials] == [5, 7, 9, 11, 24]
    assert [trial.compared for trial in loose.trials] == [False] * 4 + [True]
    # A length given is kept, alone, whatever it gives.
    assert [trial.length for trial in choose((5, 7, 9), first=7).trials] == [7]
    assert choose((5, 7, 9), first=7).length == 7


def test_a_passive_ladder_slices_each_length_as_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sliced: dict[int, object] = {}

    def tried(*args: Any) -> LengthTrial:  # noqa: ANN401
        length = cast(int, args[6])
        sliced[length] = args[8]
        return LengthTrial(length=length, xmids=(1.0,), verdicts=("retry",), flags=(), passed=0)

    monkeypatch.setattr(coherence, "_try_length", tried)
    line = cast(
        Profile, SimpleNamespace(receivers=[SimpleNamespace(x=0.25 * k) for k in range(96)])
    )
    judge = TrialJudge(
        CoherenceRules(lengths=(5, 9), trials=1), CurveThresholds(), PickingParameters()
    )

    choose_length(
        line, make_preset("passive", None), (), tmp_path, judge, 1, None, None,
        lambda length: {"slicing": {"segment_duration": length / 10}},
    )  # fmt: skip

    # Each length its own segments, as the window asks.
    assert sliced == {
        5: {"slicing": {"segment_duration": 0.5}},
        9: {"slicing": {"segment_duration": 0.9}},
    }


@pytest.mark.slow
def test_a_passive_line_keeps_the_segments_its_length_was_tried_with(
    demo_input_dir: Path, tmp_path: Path
) -> None:
    # One rung, 24 receivers 1.5 m apart (34.5 m): ten crossings at 80 m/s make 4.31 s segments.
    config = QCConfig(coherence=CoherenceRules(lengths=(24,), trials=3))
    settings = Settings(input_dir=demo_input_dir, output_dir=tmp_path / "outputs", workers=2)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        report = process_line("passive_test_profile", {"masw": {"step": 24}}, settings, config)

    run_folder = find_run(report.run_id, settings)
    sliced = stage_kwargs(load_manifest(report.run_id, settings).preset, "slicing")
    assert (sliced["segment_duration"], sliced["segment_step"]) == (4.31, 4.31)
    line = latest(read_attempts(run_folder), "line", "phase_shift")
    assert line is not None
    assert line.parameters["slicing"] == {"segment_duration": 4.31, "segment_step": 4.31}
    assert (
        "slicing segment_duration 4.31 s: the segments the window length was chosen with, 10 "
        "crossings of the window by the slowest wave (80 m/s)"
    ) in line.notes
    # No segments trial: the length's own segments are the line's.
    assert not (run_folder / SEGMENTS_FILE).exists()
