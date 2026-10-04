"""The line loop (paco.qc.line_loop): the changes of the line's settings the gates ask, tried for
the line and kept when more windows pass G3; a change kept made on the whole line."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from sigpipe.masw.runs import find_run, load_manifest

from paco.qc import QCConfig, latest, processing_used, read_attempts, read_report
from paco.qc import line as line_module
from paco.qc.g4_profile import LINE
from paco.qc.line import process_line
from paco.qc.line_loop import (
    LINE_LOOP_FILE,
    Candidate,
    LineRules,
    LineTrial,
    candidates,
    kept_on_line,
)
from paco.qc.line_loop import _said as said  # pyright: ignore[reportPrivateUsage]
from paco.qc.line_loop import _sample as sample  # pyright: ignore[reportPrivateUsage]
from paco.qc.log import LINE_CHANGE
from paco.qc.models import Flag, GateResult, Keep, Override
from paco.settings import Settings

MUTE = {"muting": {"method": "mute", "vmin": 80.0, "vmax": 1500.0}}
KEPT_NOTE = (
    "dispersion vmax 900 m/s: the line loop, asked by 1 windows (G2:ridge_at_vmax): 2 of 2 trial "
    "windows passing G3 against 0 with the line's settings before it."
)
WIDER = {"dispersion": {"vmax": 1500.0}}
CURRENT = {"muting": {"method": "none"}, "dispersion": {"vmax": 1000.0}}


def _asking(gate: str, unit: str, name: str, stage: str, overrides: dict[str, Any]) -> GateResult:
    flag = Flag(
        name=name,
        message=name,
        stage=stage,  # pyright: ignore[reportArgumentType]
        action=Override(stage=stage, overrides=overrides),  # pyright: ignore[reportArgumentType]
    )
    return GateResult(gate=gate, unit=unit, verdict="retry", flags=(flag,))


def _verdict(unit: str, verdict: str) -> GateResult:
    return GateResult(gate="G3", unit=unit, verdict=verdict)  # pyright: ignore[reportArgumentType]


UNITS = ("xmid_1.00", "xmid_2.00", "xmid_3.00", "xmid_4.00")
USES = {
    "xmid_1.00": frozenset({"1.dat", "2.dat"}),
    "xmid_2.00": frozenset({"2.dat", "3.dat"}),
    "xmid_3.00": frozenset({"3.dat", "4.dat"}),
    "xmid_4.00": frozenset({"4.dat", "5.dat"}),
}


def test_the_change_the_most_failing_windows_ask_is_tried_first() -> None:
    images = {
        "xmid_1.00": _asking("G2", "xmid_1.00", "ridge_at_vmax", "phase_shift", WIDER),
        "xmid_2.00": _asking("G2", "xmid_2.00", "weak_coherence", "preprocessing", MUTE),
        "xmid_3.00": _asking("G2", "xmid_3.00", "weak_coherence", "preprocessing", MUTE),
    }
    picks = {
        "xmid_1.00": _verdict("xmid_1.00", "reject"),
        "xmid_2.00": _verdict("xmid_2.00", "reject"),
        "xmid_3.00": _verdict("xmid_3.00", "reject"),
        "xmid_4.00": _verdict("xmid_4.00", "pass"),
    }

    found = candidates({}, images, picks, USES, CURRENT, {}, frozenset())

    assert [(one.overrides, one.failing) for one in found] == [(MUTE, 2), (WIDER, 1)]
    mute, _ = found
    assert mute.flag == "G2:weak_coherence" and mute.stage == "preprocessing"
    assert mute.windows == ("xmid_2.00", "xmid_3.00")


def test_the_records_trigger_delays_go_as_one_change_their_median() -> None:
    # A record's ask is the windows using it: 2.dat's both windows, 4.dat's two.
    records = {
        name: _asking("G1", name, "shifted_trigger", "preprocessing", {"trigger": {"t0": t0}})
        for name, t0 in (("2.dat", 0.004), ("4.dat", 0.008), ("5.dat", 0.009))
    }
    picks = {unit: _verdict(unit, "reject") for unit in UNITS}

    (delay,) = candidates(records, {}, picks, USES, CURRENT, {}, frozenset())

    assert delay.overrides == {"trigger": {"t0": 0.008}}
    assert delay.flag == "G1:shifted_trigger"
    assert delay.windows == UNITS


def test_no_change_the_line_has_the_user_gave_or_was_tried_is_asked() -> None:
    images = {unit: _asking("G2", unit, "weak_coherence", "preprocessing", MUTE) for unit in UNITS}
    picks = {unit: _verdict(unit, "reject") for unit in UNITS}
    asked = candidates(images, {}, picks, {}, CURRENT, {}, frozenset())
    assert asked == []  # images given as records: no window uses them
    (mute,) = candidates({}, images, picks, USES, CURRENT, {}, frozenset())

    # Already the line's; a muting the user gave (theirs, U2); tried before.
    assert not candidates({}, images, picks, USES, {**CURRENT, **MUTE}, {}, frozenset())
    given = {"muting": {"method": "none"}}
    assert not candidates({}, images, picks, USES, CURRENT, given, frozenset())
    assert not candidates({}, images, picks, USES, CURRENT, {}, frozenset({mute.key}))


def test_a_change_only_passing_windows_ask_is_not_tried() -> None:
    images = {unit: _asking("G2", unit, "ridge_at_vmax", "phase_shift", WIDER) for unit in UNITS}
    picks = {unit: _verdict(unit, "pass") for unit in UNITS}

    assert candidates({}, images, picks, USES, CURRENT, {}, frozenset()) == []
    # A window with no image (no trial pick) fails.
    del picks["xmid_4.00"]
    (wider,) = candidates({}, images, picks, USES, CURRENT, {}, frozenset())
    assert wider.failing == 1


def test_the_trial_windows_spread_among_those_asking_and_the_others() -> None:
    order = [f"xmid_{index}.00" for index in range(20)]
    asking = set(order[5:9])

    chosen = sample(order, asking, 6)

    # Half among the four asking, half among the others, each spread: the line's ends too.
    assert len(chosen) == 6 and len(set(chosen) & asking) == 3
    assert chosen[0] == "xmid_0.00" and chosen[-1] == "xmid_19.00"
    assert set(sample(order, asking, 8)) >= asking
    # Fewer others than half: the rest of the trials go to those asking.
    assert sample(order[:3], set(order[:3]), 6) == order[:3]


def test_a_change_is_worded_as_the_lines_rules_word_their_settings() -> None:
    mute = {"muting": {**MUTE["muting"], "width": 0.0485}}
    assert said(mute) == "muting mute 80 to 1500 m/s, width 0.0485 s"
    assert said({"dispersion": {"vmax": 1500.0}}) == "dispersion vmax 1500 m/s"
    assert said({"trigger": {"t0": 0.0029}}) == "trigger t0 0.0029 s"
    assert said({"selection": {"threshold": 0.15}}) == "selection threshold 0.15"


def test_an_ask_the_line_did_not_keep_is_kept_as_a_note() -> None:
    asked = _asking("G2", "xmid_1.00", "weak_coherence", "preprocessing", MUTE)
    alias = Flag(
        name="aliasing",
        message="alias",
        stage="picking",
        action=Override(stage="picking", overrides={"fmax": 30.0}),
    )
    asked = asked.model_copy(update={"flags": (*asked.flags, alias)})

    kept = kept_on_line(asked)

    assert kept.verdict == "pass"
    weak, still = kept.flags
    assert isinstance(weak.action, Keep) and "line's settings" in weak.action.note
    # The picking's limit stays the window's.
    assert still == alias


def _demo(demo_input_dir: Path, root: Path) -> Settings:
    return Settings(input_dir=demo_input_dir, output_dir=root / "outputs", workers=2)


def test_a_change_kept_is_made_on_the_whole_line(
    demo_input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gates' ask and its trial stand in: the loop's own work, the line made again with it.
    asked = Candidate(
        stage="phase_shift",
        overrides={"dispersion": {"vmax": 900.0}},
        flag="G2:ridge_at_vmax",
        windows=("xmid_2.88",),
        failing=1,
    )
    calls: list[int] = []

    def found(*_: Any) -> list[Candidate]:  # noqa: ANN401
        calls.append(1)
        return [asked] if len(calls) == 1 else []

    def kept(candidate: Candidate, *_: Any) -> LineTrial:  # noqa: ANN401
        return LineTrial(
            stage=candidate.stage,
            overrides=candidate.overrides,
            flag=candidate.flag,
            asked_by=1,
            xmids=(2.88,),
            before=0,
            after=2,
            kept=True,
            note=KEPT_NOTE,
        )

    monkeypatch.setattr(line_module, "candidates", found)
    monkeypatch.setattr(line_module, "try_candidate", kept)
    monkeypatch.chdir(tmp_path)
    settings = _demo(demo_input_dir, tmp_path)

    report = process_line("active_p1", {"masw": {"length": 24, "step": 24}}, settings, QCConfig())

    run_folder = find_run(report.run_id, settings)
    manifest = load_manifest(report.run_id, settings)
    assert manifest.preset.model_dump()["dispersion"]["vmax"] == 900
    attempts = read_attempts(run_folder)
    for window in manifest.windows:
        attempt = latest(attempts, window.folder, "phase_shift")
        assert attempt is not None and attempt.triggered_by == LINE_CHANGE
        assert not attempt.parameters and "G2" in attempt.results
    line = latest(attempts, LINE, "phase_shift")
    assert line is not None and line.parameters["dispersion"] == {"vmax": 900.0}
    assert line.notes[-1] == KEPT_NOTE
    # The parameters used say whose the velocity range is: the line loop's.
    report = read_report(run_folder)
    line_notes = next(unit for unit in report.units if unit.unit == LINE).notes
    used = processing_used(manifest, None, [one for notes in line_notes.values() for one in notes])
    velocities = next(one for one in used if one.startswith("velocities"))
    assert velocities.startswith("velocities 1 to 900 m/s")
    assert velocities.endswith(
        ": the line loop, asked by 1 windows (G2:ridge_at_vmax): 2 of 2 "
        "trial windows passing G3 against 0 with the line's settings before it"
    )
    trials = json.loads((run_folder / LINE_LOOP_FILE).read_text())["trials"]
    assert [trial["kept"] for trial in trials] == [True]
    # Read again after it, the asks gave nothing more: the loop ended.
    assert len(calls) == 2


def test_the_loop_keeps_at_most_its_changes(
    demo_input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = Candidate(
        stage="phase_shift",
        overrides={"dispersion": {"vmax": 900.0}},
        flag="G2:ridge_at_vmax",
        windows=("xmid_2.88",),
        failing=1,
    )
    asks = iter(range(10))

    def found(*_: Any) -> list[Candidate]:  # noqa: ANN401
        vmax = 900.0 + 100 * next(asks)
        return [replace(asked, overrides={"dispersion": {"vmax": vmax}})]

    def kept(candidate: Candidate, *_: Any) -> LineTrial:  # noqa: ANN401
        return LineTrial(
            stage=candidate.stage,
            overrides=candidate.overrides,
            flag=candidate.flag,
            asked_by=1,
            xmids=(2.88,),
            before=0,
            after=2,
            kept=True,
            note=f"{candidate.overrides}, kept.",
        )

    monkeypatch.setattr(line_module, "candidates", found)
    monkeypatch.setattr(line_module, "try_candidate", kept)
    monkeypatch.chdir(tmp_path)
    settings = _demo(demo_input_dir, tmp_path)
    config = QCConfig(line=LineRules(max_changes=2))

    report = process_line("active_p1", {"masw": {"length": 24, "step": 24}}, settings, config)

    assert load_manifest(report.run_id, settings).preset.model_dump()["dispersion"]["vmax"] == 1000
    trials = json.loads((find_run(report.run_id, settings) / LINE_LOOP_FILE).read_text())
    assert len(trials["trials"]) == 2


def test_an_ask_of_a_setting_the_user_gave_rejects_its_window(
    demo_input_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ground is faster than the 250 m/s given: G2 asks a wider range, the user's to give (U2);
    # the line keeps it, and the windows asking are rejected with the change they ask (L6).
    monkeypatch.chdir(tmp_path)
    settings = _demo(demo_input_dir, tmp_path)
    overrides = {"masw": {"length": 24, "step": 24}, "dispersion": {"vmax": 250}}

    report = process_line("active_p1", overrides, settings, QCConfig())

    manifest = load_manifest(report.run_id, settings)
    assert manifest.preset.model_dump()["dispersion"]["vmax"] == 250
    attempts = read_attempts(find_run(report.run_id, settings))
    refused = [
        g2
        for window in manifest.windows
        if (attempt := latest(attempts, window.folder, "phase_shift")) is not None
        and (g2 := attempt.results.get("G2")) is not None
        and g2.verdict == "reject"
    ]
    assert refused
    assert all(g2.flags[0].name == "locked" for g2 in refused)
    assert "asks dispersion vmax 375 (given: 250)" in refused[0].flags[0].message
