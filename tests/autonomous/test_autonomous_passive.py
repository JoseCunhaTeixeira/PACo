"""Passive and passive-active lines: a passive line surveyed for its
records of noise, its levers tried (the segments, the FK selection) with G2 on their correlations,
committed, read on the card and in the report, looked at, and decided without G3's near-field note
(its windows' virtual source is their first receiver); PACo's segments as advice; and the
passive-active mode of an active line, a mode change a new run, the user's mode kept. On PAC's test
profiles: the first 2 of the passive one's records of 120 s, on the active one's 96 receivers, and
the active one's 25 shots."""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sigpipe.masw.runs import find_run, load_manifest

from paco.autonomous.host import answer
from paco.autonomous.host.card import card
from paco.autonomous.tools import given
from paco.autonomous.tools.advise import NO_MUTE, NO_SEGMENTS, advise
from paco.autonomous.tools.commit import Committed, commit
from paco.autonomous.tools.look import look, look_experiment, look_survey
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
    """A study of passive_test_profile: its survey, its first experiment, its first commit."""

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
        study = new_study("passive_test_profile", settings)
        found = survey(study, settings, config)
        experiment = try_settings(
            study, VARIANTS, "spread:3", current=LINE, settings=settings, config=config
        )
        first = commit(study, LINE, {"masw length": "a test"}, experiment.id, settings, config)
        yield Passive(settings, study, found, experiment, first)


@pytest.mark.slow
def test_the_survey_says_a_passive_lines_records_of_noise(passive: Passive) -> None:
    found = passive.found

    assert (found.kind, found.modes, found.segment_s) == ("passive", ("passive",), 1.0)
    assert (found.reach_m, found.cone_m_s, found.frequency_step_hz) == (None, None, 1.0)
    lines = describe_survey(found).splitlines()
    assert lines[0] == (
        "passive_test_profile: 96 receivers 1.5 m apart, 142.5 m long; 2 records of ambient noise "
        "(1001.dat, 1002.dat), 120 s long, no shot."
    )
    assert lines[3].startswith("Reach and cone: none, a passive line has no shot")
    assert "Modes: passive (each window correlates the noise's segments and stacks them)." in lines
    assert lines[-1] == "- 2 records: pass, no flag"


@pytest.mark.slow
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


@pytest.mark.slow
def test_a_passive_line_is_committed_and_said_as_such(passive: Passive) -> None:
    assert str(load_manifest(passive.first.line.run_id, passive.settings).preset.mode) == "passive"
    lines = card(passive.study, passive.settings).splitlines()
    (now,) = (line for line in lines if line.startswith("Line now"))
    assert ": passive mode, 24 receivers (34.5 m), segments 1 s every 1 s, FK selection 0.2" in now
    assert "near field" not in "\n".join(lines) and "shot" not in now
    assert (
        "Survey: 96 receivers 1.5 m apart (142.5 m); 2 records of ambient noise, segments of 1 s "
        "(the preset's)"
    ) in lines[2]
    used = answer.parameters_used(passive.study, passive.settings)
    assert used[0] == (
        "passive mode (each window correlates the ambient noise's segments and stacks them): the "
        "profile's kind"
    )
    assert any(
        line.startswith("slicing: segment_duration 1.0, segment_step 1.0: ") for line in used
    )
    assert not any(line.startswith("shots ") for line in used)


@pytest.mark.slow
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


@pytest.mark.slow
def test_a_passive_experiment_shows_its_correlations_and_no_shot(passive: Passive) -> None:
    seen = look_experiment(passive.study, f"{passive.experiment.id}:current,1,2", vision=True)
    record = next(unit for unit in passive.found.g1 if unit != "line")

    # Its images, then its windows' stacked correlations at the same windows.
    assert [figure.startswith(PNG) for figure in seen.figures] == [True, True]
    assert seen.text.endswith(
        "; then the stacked correlations of current, 1, 2 there, their virtual shot gathers."
    )
    # A record of noise has no shot to draw a mute or velocities from, on the survey or the line.
    for looked in (
        lambda: look_survey(passive.study, "gather", record, True, velocities=[300.0]),
        lambda: look(passive.run, "gather", record, False, velocities=[300.0]),
    ):
        with pytest.raises(ValueError, match="A passive line's records have no shot"):
            looked()


@pytest.mark.slow
def test_g3_raises_no_near_field_note_on_a_passive_window(passive: Passive) -> None:
    attempts = read_attempts(passive.run)
    judged = [
        picked.results["G3"]
        for path in sorted(passive.run.glob("xmid_*"))
        if (picked := latest(attempts, path.name, "picking")) is not None and "G3" in picked.results
    ]

    # The nearest shot is reported, never flagged: a passive line has no shot.
    assert judged
    assert not any(flag.name == "near_field" for result in judged for flag in result.flags)


@pytest.mark.slow
def test_advise_on_a_passive_line(passive: Passive) -> None:
    # One threshold, on one window: PACo's FK selection trial in seconds.
    quick = QCConfig.model_validate(
        {"coherence": {"trials": 1}, "segments": {"thresholds": [0.2], "bands": [[None, None]]}}
    )

    assert advise(passive.study, "mute", LINE, passive.settings, quick) == NO_MUTE
    # Segments as long as the line's windows ask, as its ladder tries them: 24 receivers 1.5 m
    # apart, ten crossings at 80 m/s.
    sized = advise(passive.study, "segments", LINE, passive.settings, quick).splitlines()
    assert sized == [
        "PACo's segments for windows of 24 receivers (not applied): 4.31 s end to end, 10 "
        "crossings of the window by the slowest wave (80 m/s), as its ladder tried that length; "
        "the FK selection the line's own.",
        'It would keep: {"slicing": {"segment_duration": 4.31, "segment_step": 4.31}}.',
    ]
    # The line's own segments kept, their FK selection tried.
    line = {**LINE, **VARIANTS[0]}
    said = advise(passive.study, "segments", line, passive.settings, quick).splitlines()
    assert said[0] == (
        "PACo's FK selection trial, on trial windows spread along the line (not applied):"
    )
    assert said[-1].startswith("It would keep")


# ---------------------------------------------------------------- passive-active


@pytest.mark.slow
def test_passive_active_is_a_mode_of_an_active_line(input_dir: Path, tmp_path: Path) -> None:
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=4)
    config = QCConfig()
    study = new_study("active_test_profile", settings)
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
    with pytest.raises(ValueError, match="mode: active or passive-active on this active line"):
        try_settings(
            study, [{"mode": "passive"}], current=current, settings=settings, config=config
        )
    # A change of mode is a new run: another preset.
    first = commit(study, current, {"masw length": "a test"}, None, settings, config)
    second = commit(study, {"mode": "passive-active"}, {"mode": "a test"}, None, settings, config)
    assert second.line.commits[-1].new_run
    assert second.line.run_id != first.line.run_id
    assert str(load_manifest(second.line.run_id, settings).preset.mode) == "passive-active"


def test_the_users_mode_is_kept(input_dir: Path, tmp_path: Path) -> None:
    settings = Settings(input_dir=input_dir, output_dir=tmp_path / "output", workers=4)
    study = new_study("active_test_profile", settings)
    given.give(study, given.Given(mode="passive-active"))

    with pytest.raises(ValueError, match=re.escape("mode is the user's (passive-active)")):
        try_settings(study, [{"mode": "active"}], settings=settings, config=QCConfig())
    assert given.given_of(study).said() == ["mode passive-active"]
    assert NO_SEGMENTS.startswith("PACo's segments are a passive line's")
