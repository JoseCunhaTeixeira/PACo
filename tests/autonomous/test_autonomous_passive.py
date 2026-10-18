"""Passive and passive-active lines (docs/autonomous_plan.md §16, step 4): a passive line surveyed for its
records of noise, its levers tried (the segments, the FK selection) with G2 on their correlations,
committed, read on the card and in the report, looked at, and decided without G3's near-field note
(its windows' virtual source is their first receiver); PACo's segment trial as advice; and the
passive-active mode of an active line, a mode change a new run, the user's mode kept. On PAC's local
passive_p2, 16 records of 120 s on active_p2's 96 receivers, and active_p2's 28 shots."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sigpipe.masw.runs import find_run, load_manifest

from paco.autonomous.host import answer
from paco.autonomous.host.card import card
from paco.autonomous.tools import decide, given
from paco.autonomous.tools.advise import NO_MUTE, NO_SEGMENTS, advise
from paco.autonomous.tools.commit import Committed, commit
from paco.autonomous.tools.look import look
from paco.autonomous.tools.survey import Survey, describe_survey, survey
from paco.autonomous.tools.trials import (
    Experiment,
    Study,
    TrialRun,
    describe,
    new_study,
    try_settings,
)
from paco.qc.config import QCConfig
from paco.qc.log import latest, read_attempts
from paco.settings import Settings

LINE = {"masw": {"length": 24}}
# The passive levers: longer segments, a lower FK threshold.
VARIANTS = [
    {"slicing": {"segment_duration": 2.0, "segment_step": 2.0}},
    {"selection": {"method": "fk", "threshold": 0.1}},
]
PNG = b"\x89PNG"


@dataclass(frozen=True)
class Passive:
    """A study of passive_p2: its survey, its first experiment, its first commit."""

    settings: Settings
    study: Study
    found: Survey
    experiment: Experiment
    first: Committed

    @property
    def run(self) -> Path:
        return find_run(self.first.line.run_id, self.settings)


@pytest.fixture(scope="module")
def passive(passive_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Passive]:
    root = tmp_path_factory.mktemp("passive")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.chdir(root)
        settings = Settings(input_dir=passive_dir, output_dir=root / "output", workers=4)
        config = QCConfig()
        study = new_study("passive_p2", settings)
        found = survey(study, settings, config)
        experiment = try_settings(
            study, VARIANTS, "spread:3", current=LINE, settings=settings, config=config
        )
        first = commit(study, LINE, {"masw length": "a test"}, experiment.id, settings, config)
        yield Passive(settings, study, found, experiment, first)


def test_the_survey_says_a_passive_lines_records_of_noise(passive: Passive) -> None:
    found = passive.found

    assert (found.kind, found.modes, found.segment_s) == ("passive", ("passive",), 1.0)
    assert (found.reach_m, found.cone_m_s, found.frequency_step_hz) == (None, None, 1.0)
    lines = describe_survey(found).splitlines()
    assert lines[0] == (
        "passive_p2: 96 receivers 1.5 m apart, 142.5 m long; 16 records of ambient noise, 120 s "
        "long, no shot."
    )
    assert lines[3].startswith("Reach and cone: none, a passive line has no shot")
    assert "Modes: passive (each window correlates the noise's segments and stacks them)." in lines
    assert lines[-1] == "- 16 records: pass, no flag"


def test_passive_trials_are_judged_on_their_correlations(passive: Passive) -> None:
    runs = [
        TrialRun.model_validate_json(
            (passive.study.folder / "trials" / one.trial / "result.json").read_text()
        )
        for one in passive.experiment.tried
    ]

    assert {run.mode for run in runs} == {"passive"}
    # G2 measured each window's correlations and its FK selection.
    measured = {
        metric.name
        for run in runs
        for window in run.windows
        if window.g2 is not None
        for metric in window.g2.metrics
    }
    assert {"fk_kept", "lateral_coherence"} <= measured
    # No shot, no near field: neither scored nor said.
    assert all(one.scorecard.near_field is None for one in passive.experiment.tried)
    assert "near field" not in describe(passive.experiment)


def test_a_passive_line_is_committed_and_said_as_such(passive: Passive) -> None:
    assert str(load_manifest(passive.first.line.run_id, passive.settings).preset.mode) == "passive"
    lines = card(passive.study, passive.settings).splitlines()
    (now,) = (line for line in lines if line.startswith("Line now"))
    assert ": passive mode, 24 receivers (34.5 m), segments 1 s every 1 s, FK selection 0.2" in now
    assert "near field" not in "\n".join(lines) and "shot" not in now
    assert (
        "Survey: 96 receivers 1.5 m apart (142.5 m); 16 records of ambient noise, segments of 1 s "
        "(the preset's)"
    ) in lines[2]
    used = answer.parameters_used(passive.study, passive.settings)
    assert used[0] == (
        "passive mode (each window correlates the ambient noise's segments and stacks them): the "
        "preset's default"
    )
    assert any(
        line.startswith("slicing: segment_duration 1.0, segment_step 1.0: ") for line in used
    )
    assert not any(line.startswith("shots ") for line in used)


def test_look_shows_a_passive_windows_correlations_and_selection(passive: Passive) -> None:
    run = passive.run

    for what in ("correlations", "selection"):
        (figure,) = look(run, what, "53", vision=True).figures
        assert figure.startswith(PNG)
    said = look(run, "selection", "53", vision=False).text
    assert said.startswith("xmid_53.25, its FK selection (vision is off): ")
    assert said.endswith("at a threshold of 0.2, 0 flipped.") or "flipped." in said
    assert look(run, "correlations", "53", vision=False).text.startswith(
        "xmid_53.25, G2's measures in its correlations' stead (vision is off): 53.25: "
    )


def test_g3s_near_field_note_on_a_passive_window_needs_no_reason(passive: Passive) -> None:
    attempts = read_attempts(passive.run)
    noted = [
        path.name
        for path in sorted(passive.run.glob("xmid_*"))
        if (picked := latest(attempts, path.name, "picking")) is not None
        and "G3" in picked.results
        and picked.results["G3"].verdict == "pass"
        and any(flag.name == "near_field" for flag in picked.results["G3"].flags)
    ]

    assert noted, "G3 raises its near-field note on every passive curve"
    for unit in noted:
        problem, notes = decide.acceptable(passive.run, attempts, unit)
        assert problem is None and "near_field" not in notes


def test_advise_on_a_passive_line(passive: Passive) -> None:
    # One length and one threshold, on one window: PACo's trial in seconds.
    quick = QCConfig.model_validate(
        {
            "segments": {
                "crossings": [10],
                "thresholds": [0.2],
                "bands": [[None, None]],
                "trial_windows": 1,
            }
        }
    )

    assert advise(passive.study, "mute", LINE, passive.settings, quick) == NO_MUTE
    said = advise(passive.study, "segments", LINE, passive.settings, quick).splitlines()
    assert said[0] == "PACo's segment trial, on trial windows spread along the line (not applied):"
    assert said[-1].startswith("It would keep")


# ---------------------------------------------------------------- passive-active


def test_passive_active_is_a_mode_of_an_active_line(input_dir: Path, tmp_path: Path) -> None:
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=4)
    config = QCConfig()
    study = new_study("active_p2", settings)
    current = {"masw": {"length": 48}}

    made = try_settings(
        study,
        [{"mode": "passive-active"}],
        "spread:2",
        current=current,
        settings=settings,
        config=config,
    )

    modes = [
        TrialRun.model_validate_json(
            (study.folder / "trials" / one.trial / "result.json").read_text()
        ).mode
        for one in made.tried
    ]
    assert modes == ["active", "passive-active"]
    for variant, said in (
        ({"mode": "passive"}, "mode: active or passive-active on this active line, not passive"),
        ({"mode": "passive-active", "near_field_m": 3.0}, "near_field_m: the active mode's"),
    ):
        with pytest.raises(ValueError, match=said):
            try_settings(study, [variant], current=current, settings=settings, config=config)
    # A change of mode is a new run: another preset.
    first = commit(study, current, {"masw length": "a test"}, None, settings, config)
    second = commit(study, {"mode": "passive-active"}, {"mode": "a test"}, None, settings, config)
    assert second.line.commits[-1].new_run
    assert second.line.run_id != first.line.run_id
    assert str(load_manifest(second.line.run_id, settings).preset.mode) == "passive-active"


def test_the_users_mode_is_kept(input_dir: Path, tmp_path: Path) -> None:
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=4)
    study = new_study("active_p2", settings)
    given.give(study, given.Given(mode="passive-active"))

    with pytest.raises(ValueError, match=re.escape("mode is the user's (passive-active)")):
        try_settings(study, [{"mode": "active"}], settings=settings, config=QCConfig())
    assert given.given_of(study).said() == ["mode passive-active"]
    assert NO_SEGMENTS.startswith("PACo's segment trial is a passive line's")
