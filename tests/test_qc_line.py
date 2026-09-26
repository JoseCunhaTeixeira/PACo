"""The coherence rules for S2 and a line processed the QC way: the band capped by what the
records keep usable, the window length from the ladder of trial windows, the line judged up to
G4 with the rules' changes in the log."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from paco.picking import PickingParameters
from paco.presets import make_preset
from paco.profiles import Profile
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
from paco.qc.g1_signal import SignalThresholds
from paco.qc.g3_curve import CurveThresholds
from paco.qc.line import process_line
from paco.qc.models import Budgets
from paco.qc.report import QCReport
from paco.runs import find_run, load_manifest
from paco.settings import Settings

# Three rungs and three trial windows: the ladder of the demo in a few seconds.
QUICK = QCConfig(coherence=CoherenceRules(lengths=(5, 24, 32), trials=3, min_pass_share=2 / 3))


def test_the_band_is_capped_by_the_records_and_nyquist_never_widened() -> None:
    preset = make_preset("active", {"dispersion": {"fmin": 0.0, "fmax": 500.0}})

    # The records' median usable band, not the worst record's (the user, 2026-09-25).
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
    # The user's choices: 80 % of the trials (milestone 13), 27 of them over the whole line, the
    # short lengths first, then longer ones for a line where none passes (2026-09-25).
    rules = CoherenceRules()
    assert (rules.trials, rules.min_pass_share) == (27, 0.8)
    assert rules.lengths[:6] == (5, 7, 9, 11, 16, 24)


def test_the_trial_windows_cover_the_whole_line() -> None:
    # 27 spread evenly, the ends included: their share passing G3 is the line's (the user's
    # decision of 2026-09-25).
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
        report = process_line("active_p1", overrides, settings, config)
    return find_run(report.run_id, settings), report


def test_the_ladder_keeps_the_shortest_length_that_passes(
    demo_input_dir: Path, tmp_path: Path
) -> None:
    # The step keeps the line to four windows; the length is left to the ladder.
    run_folder, report = _line(demo_input_dir, tmp_path, {"masw": {"step": 24}}, QUICK)

    choice = json.loads((run_folder / COHERENCE_FILE).read_text())
    # At 5 receivers the trial windows at the line's ends keep too few points: 1 of 3 passes.
    # 24 is kept; 32 is tried too, for the agent to compare the depth it would reach.
    assert [trial["length"] for trial in choice["trials"]] == [5, 24, 32]
    assert [trial["compared"] for trial in choice["trials"]] == [False, False, True]
    five, twenty_four, _ = choice["trials"]
    assert (five["passed"], len(five["xmids"])) == (1, 3)
    assert choice["length"] == 24
    # What the agent reads to choose another length: the line, then each length tried.
    assert choice["receivers"] == 96 and choice["spacing_m"] == 0.25
    assert (five["metres"], twenty_four["metres"]) == (1.0, 5.75)
    assert (five["windows"], twenty_four["windows"]) == (4, 4)  # at the run's step of 24
    assert twenty_four["wavelengths_m"] is not None
    # The line is processed with it, and the rules' change is logged with its reason.
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    manifest = load_manifest(report.run_id, settings)
    assert (manifest.preset.masw.length, manifest.preset.masw.step) == (24, 24)
    assert len(manifest.windows) == 4
    line = latest(read_attempts(run_folder), "line", "phase_shift")
    # The farthest shot a window stacks, from the line's reach at 2 dB (G1): the demo's far
    # shot, 21.6 m from the end windows, stays in their stack.
    assert line is not None
    assert line.parameters == {"masw": {"distance_max": 24.34, "length": 24}}
    far, note = line.notes
    assert far == (
        "masw distance_max 24.34 m: beyond it from the shot, the traces' median SNR falls under "
        "2 dB (G1), so the windows stack no farther shot."
    )
    assert note.startswith("masw length 24 for the whole line: trial windows G3 passed, by")
    assert "1/3 at 5" in note and "at 32 (compared)" in note
    # The gates of the processing ran on it (the picking's are pick's), and the summary gives
    # the change.
    assert set(report.counts) == {"G1", "G2"}
    summary = summarize_report(report)
    assert "Changes at phase_shift, line: masw distance_max 24.34 m" in summary
    assert "masw length 24 for the whole line" in summary


def test_a_given_length_is_kept_as_it_is(demo_input_dir: Path, tmp_path: Path) -> None:
    # Given, the length stays whatever its trial windows give (the user's decision of
    # 2026-09-25: the ladder proposes, a length given decides), with nothing to compare.
    strict = QCConfig(coherence=CoherenceRules(lengths=(5, 24), trials=3, min_pass_share=1.0))
    run_folder, report = _line(
        demo_input_dir, tmp_path, {"masw": {"length": 5, "step": 24}}, strict
    )

    choice = json.loads((run_folder / COHERENCE_FILE).read_text())
    assert [(trial["length"], trial["passed"]) for trial in choice["trials"]] == [(5, 1)]
    assert choice["length"] == 5
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    assert load_manifest(report.run_id, settings).preset.masw.length == 5
    line = latest(read_attempts(run_folder), "line", "phase_shift")
    assert line is not None
    assert line.notes[1:] == (
        "masw length 5 for the whole line: trial windows G3 passed, as given: 1/3 at 5.",
    )


def test_a_record_g1_rejects_goes_into_no_window(demo_input_dir: Path, tmp_path: Path) -> None:
    # 2.dat's median SNR is 9.8 dB: under a limit of 10 dB, the filter G1 asks does not raise
    # it, and once its retry is spent the record is rejected and left out of every window (it
    # was stacked all the same before 2026-09-25).
    strict = QUICK.model_copy(
        update={
            "signal": SignalThresholds(min_snr_db=10.0),
            "budgets": Budgets(per_gate_and_unit=1),
        }
    )
    run_folder, report = _line(
        demo_input_dir, tmp_path, {"masw": {"length": 24, "step": 24}}, strict
    )

    assert report.counts["G1"] == {"pass": 1, "reject": 1}
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    assert load_manifest(report.run_id, settings).exclusions.records == ("2.dat",)
    assert "Changes at preprocessing, 2.dat: left out of every window" in (summarize_report(report))


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

    assert report.counts["G1"] == {"pass": 2}
    settings = Settings(input_dir=demo_input_dir, output_dir=run_folder.parents[1])
    manifest = load_manifest(report.run_id, settings)
    assert manifest.preset.mode == "passive-active"
    assert [window.status for window in manifest.windows] == ["succeeded"] * 4
    for window in manifest.windows:
        folder = run_folder / window.folder
        assert (folder / "Stream_0000.hdf5").exists()  # the stacked correlation gather
        assert (folder / "DispersionImage_0000.hdf5").exists()


def test_the_hint_names_the_lengths_to_change_to() -> None:
    # The demo's ladder with no settings: 11 receivers proposed, 16 tried for comparison, 5 the
    # shortest whose trials passed half. Qwen3-8B never turned "longer" or "shorter" into a
    # length of the table: the hint names them.
    def trial(length: int, passed: int) -> LengthTrial:
        return LengthTrial(
            length=length,
            xmids=tuple(float(x) for x in range(27)),
            verdicts=("pass",) * passed + ("reject",) * (27 - passed),
            flags=(),
            passed=passed,
        )

    choice = LengthChoice(
        length=11,
        trials=tuple(trial(n, k) for n, k in ((5, 17), (7, 19), (9, 19), (11, 23), (16, 25))),
        notes=(),
    )

    then = "pick comes next for run_id 20260926-050000-abcd."

    # The lengths first: Qwen3-8B follows the first step it reads.
    assert length_hint(choice, then) == (
        "The window length is the ladder's proposal (11 receivers). If the request needs more "
        "depth, first run_processing again with masw.length 16 (the longest tried). If it needs "
        "more lateral detail, first run_processing again with masw.length 5 (the shortest whose "
        "trials passed at least half). Say why. Otherwise, pick comes next for run_id "
        "20260926-050000-abcd."
    )
    # Proposed at the ladder's first rung with nothing longer tried: nothing to name.
    alone = choice.model_copy(update={"length": 5, "trials": (trial(5, 27),)})
    assert length_hint(alone, then) == (
        "pick comes next for run_id 20260926-050000-abcd. The window length is the ladder's "
        "proposal (5 receivers)."
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
        "16 receivers (3.75 m): 3/3 trial windows passed G3, wavelengths 5.0-22.0 m, 81 windows "
        "on the line (proposed)",
        "24 receivers (5.75 m): 3/3 trial windows passed G3, wavelengths 5.0-26.0 m, 73 windows "
        "on the line",
    )


def test_the_ladder_keeps_the_best_proposes_the_shortest_and_keeps_what_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passes = {5: 1, 7: 1, 9: 3, 11: 3, 16: 2}

    def tried(*args: Any) -> LengthTrial:  # noqa: ANN401
        length = cast(int, args[6])
        passed = passes[length]
        verdicts = ("pass",) * passed + ("retry",) * (3 - passed)
        return LengthTrial(
            length=length, xmids=(1.0, 2.0, 3.0), verdicts=verdicts, flags=(), passed=passed
        )

    monkeypatch.setattr(coherence, "_try_length", tried)
    line = cast(
        Profile, SimpleNamespace(receivers=[SimpleNamespace(x=0.25 * k) for k in range(48)])
    )

    def choose(lengths: tuple[int, ...], first: int | None = None) -> LengthChoice:
        judge = TrialJudge(
            CoherenceRules(lengths=lengths, trials=3), CurveThresholds(), PickingParameters()
        )
        return choose_length(line, make_preset("active", None), (), tmp_path, judge, 1, first)

    # None reaches 3 of 3: the one that passed most, the shortest on a tie; nothing compared.
    assert [trial.length for trial in choose((5, 7, 16)).trials] == [5, 7, 16]
    assert choose((5, 7, 16)).length == 16
    assert choose((5, 7)).length == 5
    # The shortest that passes is proposed, and one length more is tried to compare.
    proposed = choose((5, 7, 9, 11, 16))
    assert proposed.length == 9
    assert [trial.compared for trial in proposed.trials] == [False, False, False, True]
    # A length given is kept, alone, whatever it gives.
    assert [trial.length for trial in choose((5, 7, 9), first=7).trials] == [7]
    assert choose((5, 7, 9), first=7).length == 7
