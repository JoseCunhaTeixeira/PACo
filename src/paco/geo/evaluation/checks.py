"""The geophysicist agent's checks (docs/geo_plan.md §12), on what its play left: the study's
records and the line's runs. PACo's checks (paco.evaluation.checks) that apply to it are used as
they are; where PACo's refuse a change of the settings (the ladder's length kept, no setting
invented), these expect the changes the new agent makes: an experiment before the first commit,
every commit from an experiment, window lengths compared, the objective read, and every phase gone
through."""

import json
from collections.abc import Iterator
from typing import Any

from sigpipe.masw.runs import RunManifest

from paco.evaluation.checks import Check, Trial
from paco.evaluation.models import CheckResult
from paco.geo.host.card import objective_of
from paco.geo.host.phases import phase_of
from paco.geo.tools import journal, models
from paco.geo.tools.commit import line_of
from paco.geo.tools.freeze import frozen
from paco.geo.tools.invert import inversions_of
from paco.geo.tools.trials import STUDIES, STUDY_ID, Study


def study_of(trial: Trial) -> Study | None:
    """The study the play worked on: the newest of its outputs."""
    found = sorted(
        (path for path in trial.output_dir.glob(f"*/{STUDIES}/*") if STUDY_ID.fullmatch(path.name)),
        key=lambda path: path.name,
    )
    return Study(found[-1].parent.parent.name, found[-1]) if found else None


def report_given() -> Check:
    """The study ended with its report (P5): every phase gone through, nothing asked on the way."""
    name = "the report given"

    def check(trial: Trial) -> CheckResult:
        study = study_of(trial)
        if study is None:
            return CheckResult(name=name, passed=False, detail="no study")
        phase = phase_of(study)
        detail = "" if phase == "P5" else f"stopped at {phase}"
        return CheckResult(name=name, passed=phase == "P5", detail=detail)

    return check


def curves_frozen() -> Check:
    """The curves were frozen, one accepted at least."""
    name = "the curves frozen"

    def check(trial: Trial) -> CheckResult:
        study = study_of(trial)
        made = frozen(study) if study is not None else None
        if made is None:
            return CheckResult(name=name, passed=False, detail="nothing frozen")
        detail = "" if made.curves else f"all {len(made.rejected)} windows rejected"
        return CheckResult(name=name, passed=bool(made.curves), detail=detail)

    return check


def models_decided() -> Check:
    """Every window inverted has its model accepted or rejected, none pending."""
    name = "every model decided"

    def check(trial: Trial) -> CheckResult:
        study = study_of(trial)
        made = frozen(study) if study is not None else None
        if study is None or made is None:
            return CheckResult(name=name, passed=False, detail="nothing frozen")
        inverted = {one.unit for one in inversions_of(study, made.run_id)}
        if not inverted:
            return CheckResult(name=name, passed=False, detail="nothing inverted")
        run_folder = trial.output_dir / study.profile / made.run_id
        pending = sorted(inverted - set(models.statuses(study, run_folder, made.run_id)))
        detail = f"pending: {', '.join(pending)}" if pending else ""
        return CheckResult(name=name, passed=not pending, detail=detail)

    return check


def experimented() -> Check:
    """An experiment came before the line's first commit: its settings tried, not guessed."""
    name = "an experiment before the first commit"

    def check(trial: Trial) -> CheckResult:
        study = study_of(trial)
        found = journal.entries(study) if study is not None else []
        kinds = [entry.kind for entry in found]
        if "commit" not in kinds:
            return CheckResult(name=name, passed=False, detail="no commit")
        passed = "experiment" in kinds[: kinds.index("commit")]
        return CheckResult(name=name, passed=passed, detail="" if passed else "committed first")

    return check


def committed_from_experiments() -> Check:
    """Every commit names the experiment its settings came from (commit asks a reason for each),
    but a first commit of the user's settings alone."""
    name = "every commit from an experiment"

    def check(trial: Trial) -> CheckResult:
        study = study_of(trial)
        line = line_of(study) if study is not None else None
        if study is None or line is None:
            return CheckResult(name=name, passed=False, detail="no commit")
        goal = objective_of(study)
        given = goal.data.get("given") if goal is not None else None
        bare = [
            f"L{one.number}"
            for one in line.commits
            if one.experiment is None and not (one.number == 1 and one.changes == given)
        ]
        detail = f"no experiment named by {', '.join(bare)}" if bare else ""
        return CheckResult(name=name, passed=not bare, detail=detail)

    return check


def lengths_tried(count: int) -> Check:
    """The experiments and commits tried `count` window lengths at least: the length chosen by
    comparing, for the objective."""
    name = f"{count} window lengths tried at least"

    def check(trial: Trial) -> CheckResult:
        study = study_of(trial)
        lengths: set[int] = set()
        for entry in journal.entries(study) if study is not None else []:
            changes: list[Any] = []
            if entry.kind == "experiment":
                changes = [one.get("changes") for one in entry.data.get("variants", ())]
            elif entry.kind == "commit":
                changes = [entry.data.get("changes")]
            for one in changes:
                masw = one.get("masw") if isinstance(one, dict) else None
                if isinstance(masw, dict) and isinstance(length := masw.get("length"), int):
                    lengths.add(length)
        passed = len(lengths) >= count
        detail = "" if passed else f"tried {sorted(lengths) or 'none'}"
        return CheckResult(name=name, passed=passed, detail=detail)

    return check


def objective_read(priority: str) -> Check:
    """The objective the host read from the message puts `priority` first."""
    name = f"the objective read as {priority} first"

    def check(trial: Trial) -> CheckResult:
        study = study_of(trial)
        goal = objective_of(study) if study is not None else None
        read = goal.data.get("priority") if goal is not None else None
        detail = "" if read == priority else f"read as {read}"
        return CheckResult(name=name, passed=read == priority, detail=detail)

    return check


def every_run_with(*path: str, value: object) -> Check:
    """Every run of the play was made with `value` at `path` of its preset (e.g. "dispersion",
    "vmax"): a setting the user gave, kept in each commit of the line."""
    shown = " ".join(path)
    name = f"every run with {shown} {value}"

    def check(trial: Trial) -> CheckResult:
        presets = list(_presets(trial))
        if not presets:
            return CheckResult(name=name, passed=False, detail="no run on disk")
        found = [_at(one, path) for one in presets]
        others = sorted({json.dumps(one) for one in found if not _same(one, value)})
        detail = f"made with {', '.join(others)}" if others else ""
        return CheckResult(name=name, passed=not others, detail=detail)

    return check


def _presets(trial: Trial) -> Iterator[dict[str, Any]]:
    for path in sorted(trial.output_dir.glob("*/*/run.json")):
        yield RunManifest.model_validate_json(path.read_text()).preset.model_dump(mode="json")


def _at(values: Any, path: tuple[str, ...]) -> Any:  # noqa: ANN401
    for key in path:
        values = values.get(key) if isinstance(values, dict) else None
    return values


def _same(found: object, value: object) -> bool:
    """Whether a preset's value is `value`: numbers as numbers (250 is 250.0)."""
    if isinstance(found, int | float) and isinstance(value, int | float):
        return float(found) == float(value)
    return found == value
