"""The journal (docs/geo_plan.md §7.1): the study's memory, journal.jsonl in its folder,
append-only, a line an entry. Code writes what happened, with its exact numbers: the objective (O),
the user's settings a later message changes (U), each experiment (E, try_settings' own ids), each
commit of the line (L), the freeze (F) and the report (R), and each inversion job (I). The model
writes short hypotheses (H), conclusions (C) and decisions (D), 600 characters at most, each naming
the entries it rests on. The host rebuilds a study's phase, its budgets and its state card from the
journal and the study's other records, never from memory (§14)."""

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paco.geo.tools.commit import Committed
from paco.geo.tools.freeze import Frozen
from paco.geo.tools.trials import Experiment, Study

JOURNAL = "journal.jsonl"  # in the study's folder
MOST_TEXT = 600  # characters, a model's entry
type Kind = Literal[
    "objective",
    "settings",
    "experiment",
    "commit",
    "freeze",
    "inversion",
    "report",
    "hypothesis",
    "conclusion",
    "decision",
]
type ModelKind = Literal["hypothesis", "conclusion", "decision"]
PREFIXES: dict[str, str] = {
    "objective": "O",
    "settings": "U",
    "experiment": "E",
    "commit": "L",
    "freeze": "F",
    "inversion": "I",
    "report": "R",
    "hypothesis": "H",
    "conclusion": "C",
    "decision": "D",
}
# The settings a reading names, as the experiments' and commits' changes hold them.
SETTING = "a setting, '<stage> <key>', or a whole stage, '<stage>'"


class Entry(BaseModel):
    """One entry: its id and kind, when, what it says in a line, the entries it rests on, a
    hypothesis's expected outcome, and code's numbers."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Kind
    at: datetime
    text: str
    of: tuple[str, ...] = ()
    expect: str | None = None
    data: dict[str, Any] = Field(default_factory=dict[str, Any])


def entries(study: Study) -> list[Entry]:
    """The study's journal, in its order."""
    path = study.folder / JOURNAL
    if not path.exists():
        return []
    return [Entry.model_validate_json(line) for line in path.read_text().splitlines() if line]


def append(study: Study, entry: Entry) -> Entry:
    """`entry` appended to the study's journal."""
    with (study.folder / JOURNAL).open("a") as file:
        file.write(entry.model_dump_json() + "\n")
    return entry


def write(
    study: Study, kind: ModelKind, text: str, of: Sequence[str] = (), expect: str | None = None
) -> Entry:
    """The model's entry: a hypothesis (what it expects, `expect`), a conclusion or a decision, in
    one paragraph of MOST_TEXT characters at most, resting on the entries `of`."""
    said = " ".join(text.split())
    if not said:
        raise ValueError("text: what the entry says, one paragraph.")
    if len(said) > MOST_TEXT:
        raise ValueError(f"text: {MOST_TEXT} characters at most ({len(said)} given): say less.")
    known = {entry.id for entry in entries(study)}
    if unknown := [one for one in of if one not in known]:
        raise ValueError(
            f"of: no entry {', '.join(unknown)} in the journal; read it for the ids (E3, H2)."
        )
    if kind == "hypothesis" and not expect:
        raise ValueError("expect: what a hypothesis expects the experiment to show.")
    return append(
        study,
        Entry(
            id=next_id(study, kind),
            kind=kind,
            at=datetime.now(UTC),
            text=said,
            of=tuple(of),
            expect=" ".join(expect.split()) if expect else None,
        ),
    )


def next_id(study: Study, kind: Kind) -> str:
    """The id `kind`'s next entry takes: its letter and the count of such entries, plus one."""
    count = sum(1 for entry in entries(study) if entry.kind == kind)
    return f"{PREFIXES[kind]}{count + 1}"


def objective(study: Study, fields: dict[str, Any], said: str) -> Entry:
    """The user's objective, as the host read it from their message (§4.0)."""
    return append(
        study,
        Entry(id="O1", kind="objective", at=datetime.now(UTC), text=said, data=fields),
    )


def settings(study: Study, changed: Sequence[str]) -> Entry:
    """The settings the user gave that a later message changed, as they now stand (§8, rule 1)."""
    return append(
        study,
        Entry(
            id=next_id(study, "settings"),
            kind="settings",
            at=datetime.now(UTC),
            text="The user now gives " + "; ".join(changed) + ".",
            data={"changed": list(changed)},
        ),
    )


def experiment(study: Study, made: Experiment, hypothesis: str | None = None) -> Entry | None:
    """An experiment try_settings made, its numbers as its scorecards say them; none for one made
    before (its entry is there)."""
    if made.repeated or any(entry.id == made.id for entry in entries(study)):
        return None
    variants = [
        {
            "label": one.label,
            "changes": one.changes,
            "passing": len(one.scorecard.covered),
            "windows": len(one.scorecard.windows),
            "gained": list(one.against.gained) if one.against is not None else [],
            "lost": list(one.against.lost) if one.against is not None else [],
            "depth_m": one.scorecard.depth_m,
            "near_field": one.scorecard.near_field,
            "dominated_by": one.dominated_by,
            "seconds": one.seconds,
        }
        for one in made.tried
    ]
    return append(
        study,
        Entry(
            id=made.id,
            kind="experiment",
            at=made.made_at,
            text=experiment_line(made.windows, variants),
            of=(hypothesis,) if hypothesis else (),
            data={"windows": made.windows, "variants": variants},
        ),
    )


def commit(study: Study, committed: Committed) -> Entry:
    """A commit of the line: its run, its changes and reasons, the experiment they came from."""
    made = committed.line.commits[-1]
    how = "made run" if made.new_run else "changed the line in run"
    changes = json.dumps(made.changes, separators=(",", ":"))
    return append(
        study,
        Entry(
            id=f"L{made.number}",
            kind="commit",
            at=made.made_at,
            text=f"L{made.number} {how} {made.run_id}: {changes}",
            of=(made.experiment,) if made.experiment else (),
            data={
                "run_id": made.run_id,
                "changes": made.changes,
                "reasons": made.reasons,
                "new_run": made.new_run,
                "reapplied": len(committed.reapplied),
            },
        ),
    )


def freeze(study: Study, frozen: Frozen) -> Entry:
    """The freeze: the curves kept and the windows rejected."""
    return append(
        study,
        Entry(
            id="F1",
            kind="freeze",
            at=frozen.frozen_at,
            text=(
                f"Frozen from L{frozen.commit}: {len(frozen.curves)} curves accepted, "
                f"{len(frozen.rejected)} windows rejected"
            ),
            data={"run_id": frozen.run_id, "accepted": len(frozen.curves)},
        ),
    )


def report(study: Study, answer: str) -> Entry:
    """The report, the run's last answer (P5)."""
    return append(
        study,
        Entry(
            id=next_id(study, "report"),
            kind="report",
            at=datetime.now(UTC),
            text="The report, given to the user",
            data={"answer": answer},
        ),
    )


def read(
    study: Study, entry: str | None = None, last: int | None = None, setting: str | None = None
) -> str:
    """The journal as the model reads it: entry `entry` whole; the experiments and commits that
    changed `setting`; or the `last` entries (5 by default), one a line."""
    found = entries(study)
    if entry is not None:
        one = next((each for each in found if each.id == entry), None)
        if one is None:
            raise ValueError(f"No entry {entry} in the journal.")
        return one.model_dump_json(indent=1, exclude_defaults=True)
    if setting is not None:
        changed = [each for each in found if _changes(each, setting)]
        if not changed:
            return f"No experiment or commit changed {setting}."
        return "\n".join(line(each) for each in changed)
    if last is not None and last < 1:
        raise ValueError("last: 1 entry at least.")
    shown = found[-(last or 5) :]
    if not shown:
        return "The journal is empty."
    return "\n".join(line(each) for each in shown)


def line(entry: Entry) -> str:
    """An entry in one line: its id, what it says, what it rests on."""
    rests = f" (of {', '.join(entry.of)})" if entry.of else ""
    expects = f"; expects: {entry.expect}" if entry.expect else ""
    head = entry.text if entry.text.startswith(entry.id) else f"{entry.id} {entry.text}"
    return f"{head}{rests}{expects}"


def experiment_line(windows: str, variants: Iterable[dict[str, Any]]) -> str:
    """An experiment's numbers in a line: each variant's change, its passes and the windows it
    gained and lost against the current settings."""
    said: list[str] = []
    for one in variants:
        passing = f"{one['passing']}/{one['windows']}"
        if one["label"] == "current":
            said.append(f"current {passing}")
            continue
        change = json.dumps(one["changes"], separators=(",", ":"))
        marks = f" (+{len(one['gained'])}, -{len(one['lost'])})"
        if one["dominated_by"]:
            marks += f", dominated by {one['dominated_by']}"
        said.append(f"{one['label']} {change} {passing}{marks}")
    return f"on {windows}: " + "; ".join(said)


def _changes(entry: Entry, setting: str) -> bool:
    """Whether `entry`, an experiment or a commit, changes `setting` ('<stage> <key>', or a
    stage)."""
    stage, _, key = setting.partition(" ")
    changed: list[dict[str, Any]] = []
    if entry.kind == "commit":
        changed = [entry.data.get("changes", {})]
    elif entry.kind == "experiment":
        changed = [one.get("changes", {}) for one in entry.data.get("variants", [])]
    for changes in changed:
        value = changes.get(stage)
        if value is not None and (not key or (isinstance(value, dict) and key in value)):
            return True
    return False
