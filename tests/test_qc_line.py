"""The coherence rules for S2 and a line processed the QC way: the band capped by what the
records keep usable, the window length from the ladder of trial windows, the line judged up to
G4 with the rules' changes in the log."""

import json
from pathlib import Path

import pytest

from paco.presets import make_preset
from paco.qc import QCConfig, latest, read_attempts, summarize_report
from paco.qc.coherence import COHERENCE_FILE, CoherenceRules, cap_band, given_length
from paco.qc.line import process_line
from paco.qc.report import QCReport
from paco.runs import find_run, load_manifest
from paco.settings import Settings

# Two rungs and three trial windows: the ladder of the demo in a few seconds.
QUICK = QCConfig(coherence=CoherenceRules(lengths=(5, 24), trials=3, min_pass_share=2 / 3))


def test_the_band_is_capped_by_the_records_and_nyquist_never_widened() -> None:
    preset = make_preset("active", {"dispersion": {"fmin": 0.0, "fmax": 500.0}})

    overrides, notes = cap_band(preset, [(0.0, 349.5), (0.0, 324.0), None], nyquist=1000.0)
    assert overrides == {"dispersion": {"fmax": 324.0}}
    assert notes == (
        "dispersion fmax 500 Hz is above the records' usable band (up to 324.0 Hz): set to 324.0.",
    )
    # PAC's 100 Hz is within: nothing changes, though the records go higher.
    assert cap_band(make_preset("active", None), [(0.0, 324.0)], 1000.0) == ({}, ())
    # A usable band starting above fmin raises it; Nyquist caps when no record says more.
    overrides, _ = cap_band(preset, [(5.0, 800.0)], nyquist=400.0)
    assert overrides == {"dispersion": {"fmax": 400.0, "fmin": 5.0}}
    # A band wholly below the usable one would be empty: the one case fmax goes up.
    overrides, notes = cap_band(make_preset("active", None), [(122.4, 480.0)], 2400.0)
    assert overrides == {"dispersion": {"fmin": 122.4, "fmax": 480.0}}
    assert notes[-1] == (
        "dispersion fmax 100 Hz lies below the records' usable band (122.4-480.0 Hz): set to 480.0."
    )


def test_the_ladder_defaults_to_nine_trials_and_four_fifths() -> None:
    # The user's choice of milestone 13: on active_p1 they keep 24 receivers, where 5 trials at
    # two thirds kept 16 and left 38 % of the line to fix.
    rules = CoherenceRules()
    assert (rules.trials, rules.min_pass_share, rules.lengths[:5]) == (9, 0.8, (5, 8, 12, 16, 24))


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
    assert [trial["length"] for trial in choice["trials"]] == [5, 24]
    five, twenty_four = choice["trials"]
    assert five["passed"] == 0 and len(five["xmids"]) == 3
    assert twenty_four["passed"] >= 2
    assert choice["length"] == 24
    # The line is processed with it, and the rules' change is logged with its reason.
    manifest = load_manifest(report.run_id, Settings(output_dir=run_folder.parents[1]))
    assert (manifest.preset.masw.length, manifest.preset.masw.step) == (24, 24)
    assert len(manifest.windows) == 4
    line = latest(read_attempts(run_folder), "line", "phase_shift")
    assert line is not None and line.parameters == {"masw": {"length": 24}}
    (note,) = line.notes
    assert note.startswith("masw length 24 for the whole line: trial windows G3 passed, by")
    assert "0/3 at 5" in note
    # The gates of the processing ran on it (the picking's are pick's), and the summary gives
    # the change.
    assert set(report.counts) == {"G1", "G2"}
    assert "Changes at phase_shift, line: masw length 24 for the whole line" in (
        summarize_report(report)
    )


def test_the_users_length_is_where_the_ladder_starts(demo_input_dir: Path, tmp_path: Path) -> None:
    run_folder, _ = _line(demo_input_dir, tmp_path, {"masw": {"length": 24, "step": 24}}, QUICK)

    choice = json.loads((run_folder / COHERENCE_FILE).read_text())
    assert [trial["length"] for trial in choice["trials"]] == [24]
    assert choice["length"] == 24


def test_when_no_length_passes_the_best_tried_is_kept(demo_input_dir: Path, tmp_path: Path) -> None:
    config = QCConfig(coherence=CoherenceRules(lengths=(5, 8), trials=3))
    run_folder, _ = _line(demo_input_dir, tmp_path, {"masw": {"step": 24}}, config)

    choice = json.loads((run_folder / COHERENCE_FILE).read_text())
    assert [trial["passed"] for trial in choice["trials"]] == [0, 0]
    assert choice["length"] == 5  # a tie: the shortest
