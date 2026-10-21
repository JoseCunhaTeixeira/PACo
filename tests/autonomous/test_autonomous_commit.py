"""commit: the line made with the model's settings in PAC's layout;
its settings changed in the same run, as PACo logs a change of the line's; a change of the windows
in a new run. Each setting's reason goes where PAC's run card reads it, and paco-replay makes the
line again. On active_test_profile with windows of 90 receivers on a coarse grid: 7 windows, a commit in a few
seconds."""

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pytest
from sigpipe.masw.presets import make_preset
from sigpipe.masw.runs import find_run, load_manifest

from paco.autonomous.tools import commit as committing
from paco.autonomous.tools.commit import Committed, commit, describe_commit, line_of, line_state
from paco.autonomous.tools.trials import G1, Study, new_study, try_settings
from paco.qc.config import QCConfig
from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt
from paco.qc.replay import replay_run
from paco.settings import Settings

FIRST = {
    "masw": {"length": 90, "distance_max": 80.0},
    "dispersion": {"nv": 250},
    "near_field_m": 8.0,
}
FIRST_REASONS = {
    "masw length": "90 receivers, for a test",
    "masw distance_max": "every shot off the windows' ends, 80 m",
    "dispersion nv": "a coarse grid, for a test",
    "near_field_m": "half the longest wavelength the trials reached.",
}
# The records band-passed and the pick bounded: a change of the line's records, and of its picking.
SECOND = {"filtering": {"method": "iir", "fmin": 5.0, "fmax": 90.0}, "picking": {"fmax": 90.0}}
SECOND_REASONS = {"filtering": "no signal outside 5-90 Hz (G1)", "picking fmax": "the band G2 kept"}
# Other windows: a new run.
THIRD = {"masw": {"length": 94}}


@dataclass(frozen=True)
class Commits:
    """Three commits of a study of active_test_profile, and the first run's log after the first one."""

    settings: Settings
    study: Study
    first: Committed
    second: Committed
    third: Committed
    after_first: tuple[Attempt, ...]

    @property
    def run(self) -> Path:
        """The run the first two commits made and changed."""
        return find_run(self.first.line.run_id, self.settings)


@pytest.fixture(scope="module")
def commits(input_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Commits]:
    root = tmp_path_factory.mktemp("commit")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.chdir(root)
        settings = Settings(input_dir=input_dir, output_dir=root / "output", workers=2)
        config = QCConfig()
        study = new_study("active_test_profile", settings)
        first = commit(study, FIRST, FIRST_REASONS, None, settings, config)
        run = find_run(first.line.run_id, settings)
        after_first = read_attempts(run)
        experiment = try_settings(
            study,
            [{"picking": {"fmax": 90.0}}],
            "spread:2",
            current=first.line.settings,
            line=line_state(run),
            settings=settings,
            config=config,
        )
        second = commit(study, SECOND, SECOND_REASONS, experiment.id, settings, config)
        third = commit(study, THIRD, {"masw length": "shorter windows"}, None, settings, config)
        yield Commits(settings, study, first, second, third, after_first)


@pytest.mark.slow
def test_the_first_commit_makes_a_run_in_pacs_layout(commits: Commits) -> None:
    run = commits.run

    assert run.parent == commits.settings.output_dir / "active_test_profile"
    for name in ("run.json", "qc_log.jsonl", "qc_report.json", "qc_config.json"):
        assert (run / name).is_file()
    assert len(list(run.glob("xmid_*"))) == 7
    # Every record, window and pick an attempt of its own, made by the commit.
    stages = {(attempt.stage, attempt.triggered_by) for attempt in commits.after_first}
    assert stages == {
        ("preprocessing", "initial"),
        ("phase_shift", "initial"),
        ("picking", "initial"),
    }


@pytest.mark.slow
def test_each_reason_goes_where_pacs_run_card_reads_it(commits: Commits) -> None:
    line = latest(read_attempts(commits.run), "line", "phase_shift")

    assert line is not None
    assert line.parameters == {
        "masw": {"length": 90, "distance_max": 80.0},
        "dispersion": {"nv": 250},
        "filtering": {"method": "iir", "fmin": 5.0, "fmax": 90.0},
        "near_field": {"distance_m": 8.0},
    }
    # `<stage> <key>: <reason>`, with the commit and the experiment; a stage's reason for each of
    # its keys; the near field under the name PACo logs it by.
    assert line.notes == (
        "masw length: 90 receivers, for a test (L1)",
        "masw distance_max: every shot off the windows' ends, 80 m (L1)",
        "dispersion nv: a coarse grid, for a test (L1)",
        "near_field distance_m: half the longest wavelength the trials reached (L1)",
        "filtering method: no signal outside 5-90 Hz (G1) (L2, E1)",
        "filtering fmin: no signal outside 5-90 Hz (G1) (L2, E1)",
        "filtering fmax: no signal outside 5-90 Hz (G1) (L2, E1)",
        "picking fmax: the band G2 kept (L2, E1)",
        f"Committed by PACo's autonomous agent (paco {version('paco')}) in study "
        f"{commits.study.id}, with its experiments.",
    )


@pytest.mark.slow
def test_a_later_commit_changes_the_line_in_its_run(commits: Commits) -> None:
    run = commits.run
    attempts = read_attempts(run)

    assert commits.second.line.run_id == commits.first.line.run_id
    assert {(a.stage, a.triggered_by) for a in attempts if a.unit != "line"} == {
        ("preprocessing", "initial"),
        ("preprocessing", "line change"),
        ("phase_shift", "initial"),
        ("phase_shift", "line change"),
        ("picking", "initial"),
        ("picking", "line change"),
    }
    line = latest(attempts, "line", "phase_shift")
    assert line is not None
    assert (line.attempt, line.triggered_by) == (2, "line change")
    # What the change replaced kept in each window's attempts/.
    assert all((window / "attempts").is_dir() for window in run.glob("xmid_*"))


@pytest.mark.slow
def test_paco_replay_makes_the_changed_line_again(commits: Commits) -> None:
    replay = replay_run(commits.first.line.run_id, commits.settings)

    assert replay.same
    assert {one.stage for one in replay.compared} == {"preprocessing", "phase_shift", "picking"}


@pytest.mark.slow
def test_a_change_of_the_windows_makes_a_new_run(commits: Commits) -> None:
    line = line_of(commits.study)

    assert line is not None
    assert line.run_id == commits.third.line.run_id != commits.first.line.run_id
    assert [(one.number, one.new_run) for one in line.commits] == [(1, True), (2, False), (3, True)]
    assert line.settings["masw"] == {"length": 94, "distance_max": 80.0}
    assert len(list(find_run(line.run_id, commits.settings).glob("xmid_*"))) == 3


@pytest.mark.slow
def test_g1_is_made_once_for_each_preprocessing(commits: Commits) -> None:
    # The first commit made G1 on the records as the preset preprocesses them (the trial took it),
    # the second on the band-passed ones, which the third's new run took.
    assert len(list((commits.study.folder / G1).iterdir())) == 2
    second, third = commits.run, find_run(commits.third.line.run_id, commits.settings)

    # The third run's G1, given back, as the second's made: each record's result and notes, the
    # line's, the traces and records left out.
    assert _g1(third) == _g1(second)
    second_line, third_line = (
        load_manifest(one.line.run_id, commits.settings) for one in (commits.second, commits.third)
    )
    assert third_line.exclusions == second_line.exclusions
    # Its records' spectra copied beside them, as PAC's pages show them.
    assert len(list(third.glob("records/*/Spectrum_0000.png"))) == 25


@pytest.mark.slow
def test_the_commit_as_the_model_reads_it(commits: Commits) -> None:
    run_id = commits.first.line.run_id

    assert describe_commit(commits.first).startswith(f"L1 made run {run_id}.\nG1: ")
    assert describe_commit(commits.second).startswith(
        f"L2 changed the line's settings in run {run_id}.\n"
    )


@pytest.mark.slow
def test_the_line_as_a_trial_set_reads_it(commits: Commits) -> None:
    state = line_state(commits.run)

    assert len(state.verdicts) == 7
    assert set(state.verdicts.values()) <= {"pass", "retry", "reject", "no curve"}
    assert set(state.flags) == set(state.verdicts)


def test_a_setting_given_at_its_value_is_no_change_and_needs_no_reason() -> None:
    # The preset's own values given again, as a model may write a whole stage: no change.
    changes = {
        "masw": {"length": 12},
        "dispersion": {"fmin": 16.7, "fmax": 100, "vmin": 1, "nv": 1000},
    }

    same = committing._unchanged(changes, make_preset("active"))  # pyright: ignore[reportPrivateUsage]

    assert same == {"dispersion fmax", "dispersion vmin", "dispersion nv"}
    assert committing._without(changes, same) == {  # pyright: ignore[reportPrivateUsage]
        "masw": {"length": 12},
        "dispersion": {"fmin": 16.7},
    }


def test_a_stages_method_stays_while_its_parameters_change() -> None:
    # The FK selection kept, its threshold changed: the method goes with it, as the override
    # needs; the whole stage at its values, no change.
    passive = make_preset("passive")
    changes = {"selection": {"method": "fk", "threshold": 0.3}}

    assert committing._unchanged(changes, passive) == set()  # pyright: ignore[reportPrivateUsage]
    assert committing._unchanged(  # pyright: ignore[reportPrivateUsage]
        {"selection": {"method": "fk", "threshold": 0.2}}, passive
    ) == {"selection method", "selection threshold"}


def test_a_method_and_its_parameters_take_one_reason() -> None:
    mute = {"muting": {"method": "mute", "vmin": 80, "vmax": 1500}}
    reasons = {"muting vmin": "80 m/s, PACo's", "muting vmax": "1500 m/s, the fast arrivals"}

    committing._check_reasons(mute, reasons)  # pyright: ignore[reportPrivateUsage]

    said = committing._by_key(mute, reasons)  # pyright: ignore[reportPrivateUsage]
    assert said["muting method"] == "80 m/s, PACo's; 1500 m/s, the fast arrivals"
    # Settings of a stage without a method: each its own reason.
    with pytest.raises(ValueError, match="missing masw distance_max"):
        committing._check_reasons(  # pyright: ignore[reportPrivateUsage]
            {"masw": {"length": 12, "distance_max": 63}}, {"masw length": "why"}
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("changes", "reasons", "experiment", "said"),
    [
        ({}, {}, None, "none given"),
        (
            FIRST,
            {"masw length": "why"},
            None,
            "missing masw distance_max, dispersion nv, near_field_m",
        ),
        ({"masw": {"length": 9}}, {"masw length": "a", "dispersion": "b"}, None, "not change"),
        ({"masw": {"length": 9}}, {"masw length": "two\nlines"}, None, "one line"),
        ({"masw": {"length": 9}}, {"masw length": "why"}, "E9", "No experiment 'E9'"),
        ({"masw": {"length": 9, "step": 2}}, {"masw": "why"}, None, "one receiver apart"),
    ],
)
def test_a_commit_is_refused_saying_why(
    commits: Commits,
    changes: dict[str, object],
    reasons: dict[str, str],
    experiment: str | None,
    said: str,
) -> None:
    study = new_study("active_test_profile", commits.settings)

    with pytest.raises(ValueError, match=said):
        commit(study, changes, reasons, experiment, commits.settings, QCConfig())
    assert line_of(study) is None


def _g1(run: Path) -> dict[str, tuple[object, tuple[str, ...]]]:
    """G1 in `run`'s log: each unit's latest result, and the notes on records it left out."""
    attempts = read_attempts(run)
    found: dict[str, tuple[object, tuple[str, ...]]] = {}
    for unit in sorted({attempt.unit for attempt in attempts if attempt.stage == "preprocessing"}):
        attempt = latest(attempts, unit, "preprocessing")
        if attempt is not None:
            left = tuple(note for note in attempt.notes if "left out" in note)
            found[unit] = (attempt.results.get("G1"), left)
    return found
