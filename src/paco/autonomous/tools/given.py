"""The settings the user gave (docs/autonomous_plan.md §8, rule 1), as the objective form read them from
their messages (§4.0): the line's, its mode and by preset stage as commit takes its changes (the
windows' length, the images' velocities and band), and the inversion's (its iterations, the models'
Vs bounds). The latest message giving a setting wins, as in PACo's paco.qc.given, which this
follows. The study keeps them in given.json.

Code keeps them. Every trial and commit runs with the line's, every inversion with the
inversion's, which sigpipe's checks before S4 keep as given where they fail, with a note of what
they ask. Both are logged in the run as PACo logs a user's settings (paco.qc.given.give): PACo's
G2 then rejects a window it could fix only by changing one of the line's, saying the change it
asks (keep_line_asks). A call changing one is refused, saying it is the user's: the model
proposes the change to them."""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field
from sigpipe.masw.inversion.priors import PriorRules, layering_of
from sigpipe.masw.runs.writing import write_atomic

from paco.qc import given as paco_given
from paco.qc.config import CONFIG_FILE, QCConfig, read_qc_config
from paco.qc.log import latest, read_attempts
from paco.qc.loops import deep_merge
from paco.qc.models import Override, Reject, Stage
from paco.qc.report import stretches, xmid_of

if TYPE_CHECKING:  # trials keeps the user's settings in its trials
    from paco.autonomous.tools.trials import Study

GIVEN = "given.json"  # in the study's folder
MODE = "mode"  # the line's mode, among its settings (paco.autonomous.tools.trials)
# A user's setting's reason, as the line's notes and PAC's run card say it.
REASON = "given by the user"
# The inversion's settings a user gives, as the form reads them.
ITERATIONS = "n_iterations"
BOUNDS = ("vs_min", "vs_max")  # m/s, every layer's least and greatest Vs
# PACo's flag on a unit rejected for a change of a setting the user gave (paco.qc.budgets), and
# sigpipe's note on a locked value that fails a check before S4 (priors._ending).
LOCKED = "locked"
_KEPT = re.compile(r"^(?P<field>[\w.]+) .*: kept as given \(the check \w+ (?P<asked>[^)]*)\)\.?$")


class Given(BaseModel):
    """The user's settings: the line's mode and its stages (commit's changes), and the
    inversion's (ITERATIONS and BOUNDS)."""

    model_config = ConfigDict(frozen=True)

    mode: str | None = None  # active, passive or passive-active (an active line's shots correlated)
    line: dict[str, dict[str, Any]] = Field(default_factory=dict[str, dict[str, Any]])
    inversion: dict[str, Any] = Field(default_factory=dict[str, Any])

    def said(self) -> list[str]:
        """Each setting in words: "mode passive-active", "dispersion vmax 350", "inversion vs_max
        300"."""
        return [f"{name} {_shown(value)}" for name, value in self.named().items()]

    def named(self) -> dict[str, Any]:
        """Each setting by its name: "mode", "dispersion vmax", "inversion n_iterations"."""
        found: dict[str, Any] = {MODE: self.mode} if self.mode is not None else {}
        found |= {
            f"{stage} {key}": value
            for stage, values in self.line.items()
            for key, value in values.items()
        }
        return found | {f"inversion {key}": value for key, value in self.inversion.items()}

    def changes(self) -> dict[str, Any]:
        """The line's settings the user gave, as commit takes changes: the mode, then by
        stage."""
        return ({MODE: self.mode} if self.mode is not None else {}) | self.line


def given_of(study: Study) -> Given:
    """The settings the user gave in the study's messages; none given: empty. A study begun before
    given.json was kept: its objective's window length."""
    path = study.folder / GIVEN
    if path.exists():
        return Given.model_validate_json(path.read_text())
    journal = study.folder / "journal.jsonl"
    for line in journal.read_text().splitlines() if journal.exists() else ():
        entry = json.loads(line) if line else {}
        if entry.get("kind") == "objective":
            older = entry.get("data", {}).get("given")
            if isinstance(older, dict) and not {"line", "inversion"} & set(older):
                return Given(line=cast(dict[str, dict[str, Any]], older))
            return Given.model_validate(older) if isinstance(older, dict) else Given()
    return Given()


def give(study: Study, given: Given) -> list[str]:
    """`given`, from the user's latest message, kept over what they gave before: what changed, in
    words ("dispersion vmax 600 (was 350)")."""
    before = given_of(study)
    after = Given(
        mode=given.mode or before.mode,
        line=deep_merge(before.line, given.line),
        inversion={**before.inversion, **given.inversion},
    )
    if after != before or not (study.folder / GIVEN).exists():
        write_atomic(study.folder / GIVEN, after.model_dump_json(indent=2))
    old, new = before.named(), after.named()
    return [
        f"{name} {_shown(value)}" + (f" (was {_shown(old[name])})" if name in old else "")
        for name, value in new.items()
        if old.get(name) != value
    ]


def check_line(changes: Mapping[str, Any], given: Given) -> None:
    """Refused, saying why, when `changes` (of the line's settings, by stage) change one the user
    gave."""
    changed = [
        f"{stage} {key} is the user's ({_shown(value)})"
        for stage, values in given.line.items()
        for key, value in values.items()
        if isinstance(asked := changes.get(stage), Mapping)
        and (other := cast(Mapping[str, Any], asked).get(key)) is not None
        and other != value
    ]
    if given.mode is not None and changes.get(MODE) not in (None, given.mode):
        changed.insert(0, f"{MODE} is the user's ({given.mode})")
    if changed:
        raise ValueError(f"{'; '.join(changed)}: propose a change to them, never make it.")


def with_line(settings: Mapping[str, Any], given: Given) -> dict[str, Any]:
    """The line's `settings` with the user's."""
    return deep_merge(dict(settings), given.changes())


def without(changes: Mapping[str, Any], given: Given) -> dict[str, Any]:
    """`changes` (of the line's settings, by stage) without those the user gave."""
    kept: dict[str, Any] = {}
    for stage, values in changes.items():
        if stage == MODE and given.mode is not None:
            continue
        mine = given.line.get(stage, {})
        if isinstance(values, Mapping) and mine:
            rest = {
                key: value
                for key, value in cast(Mapping[str, Any], values).items()
                if key not in mine
            }
            if rest:
                kept[stage] = rest
        else:
            kept[stage] = values
    return kept


def missing(settings: Mapping[str, Any], given: Given) -> dict[str, Any]:
    """The user's line settings that `settings` (the line's) do not hold as they gave them."""
    found: dict[str, Any] = {}
    if given.mode is not None and settings.get(MODE) != given.mode:
        found[MODE] = given.mode
    for stage, values in given.line.items():
        held = settings.get(stage)
        for key, value in values.items():
            if not isinstance(held, Mapping) or cast(Mapping[str, Any], held).get(key) != value:
                found.setdefault(stage, {})[key] = value
    return found


def check_inversion(parameters: Mapping[str, Any], given: Given) -> None:
    """Refused, saying why, when a group's `parameters` (InversionParameters' fields) change an
    inversion setting the user gave: its iterations, or a Vs bound of any layer."""
    changed: list[str] = []
    mine = given.inversion.get(ITERATIONS)
    if mine is not None and parameters.get(ITERATIONS) not in (None, mine):
        changed.append(f"{ITERATIONS} is the user's ({mine})")
    free = cast(Mapping[str, Any], parameters.get("free") or {})
    layers = cast(list[Mapping[str, Any]], parameters.get("vs_layers") or [])
    for key in BOUNDS:
        mine = given.inversion.get(key)
        asked = [free.get(key), *(layer.get(key) for layer in layers)]
        if mine is not None and any(one not in (None, mine) for one in asked):
            changed.append(f"{key} is the user's ({_shown(mine)} m/s, every layer's)")
    if changed:
        raise ValueError(f"{'; '.join(changed)}: propose a change to them, never make it.")


def with_inversion(
    parameters: Mapping[str, Any], given: Given, rules: PriorRules
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A window's `parameters` with the user's inversion settings, and those locked (sigpipe's
    checks before S4 keep them where they fail, with a note): the iterations as given; the Vs
    bounds as the layering runs them, every layer's under the free layering (free's), one range
    standing for every layer under the fixed one (vs_layers)."""
    merged = dict(parameters)
    locked: dict[str, Any] = {}
    if (iterations := given.inversion.get(ITERATIONS)) is not None:
        merged[ITERATIONS] = locked[ITERATIONS] = iterations
    bounds = {key: given.inversion[key] for key in BOUNDS if given.inversion.get(key) is not None}
    if bounds and layering_of(merged, rules) == "free":
        merged["free"] = {**cast(Mapping[str, Any], merged.get("free") or {}), **bounds}
        locked["free"] = bounds
    elif bounds:
        merged["vs_layers"] = locked["vs_layers"] = [bounds]
    return merged, locked


def asks(run_folder: Path, given: Given) -> list[str]:
    """What PACo's gates and sigpipe's checks asked of the user's settings on the line in
    `run_folder`, each with its windows, for the model to propose and the report to say: the
    units G1 and G2 rejected for a change of the line's (keep_line_asks), G5's changes of the
    inversion's held back, and the bounds the checks before S4 kept though they fail."""
    if not given.line and not given.inversion:
        return []
    config = read_qc_config(run_folder) if (run_folder / CONFIG_FILE).exists() else QCConfig()
    mine = with_inversion({}, given, config.priors)[0] if given.inversion else {}
    attempts = read_attempts(run_folder)
    found: dict[str, list[str]] = {}
    checked: dict[str, list[tuple[str, str]]] = {}
    gates: tuple[tuple[Stage, str], ...] = (
        ("preprocessing", "G1"),
        ("phase_shift", "G2"),
        ("inversion", "G5"),
    )
    for unit in sorted({attempt.unit for attempt in attempts}):
        for stage, gate in gates:
            attempt = latest(attempts, unit, stage)
            result = attempt.results.get(gate) if attempt is not None else None
            for flag in result.flags if result is not None else ():
                held: Mapping[str, Any] = {}
                if gate == "G5" and isinstance(flag.action, Override) and mine:
                    held = paco_given.unlocked(flag.action.overrides, mine)[1]
                if flag.name == LOCKED and isinstance(flag.action, Reject):
                    said = flag.action.reason.removeprefix(f"{LOCKED}, asks ")
                    found.setdefault(f"{gate} rejected them, asking {said}", []).append(unit)
                elif held:
                    said = paco_given.said(held, mine)
                    found.setdefault(f"G5 asks {said}", []).append(unit)
        inverted = latest(attempts, unit, "inversion")
        for note in inverted.notes if inverted is not None else ():
            if (kept := _KEPT.match(note)) is not None:
                checked.setdefault(kept["field"], []).append((unit, kept["asked"]))
    lines = [f"{_where(units)}: {said}" for said, units in found.items()]
    for name, pairs in checked.items():
        values = sorted({asked for _, asked in pairs}, key=_number)
        lines.append(
            f"{_where([unit for unit, _ in pairs])}: the checks before S4 kept {name} as given, "
            f"asking {' to '.join(dict.fromkeys((values[0], values[-1])))}"
        )
    return lines


def _number(text: str) -> float:
    """The number a check's ask starts with ("572 m/s": 572)."""
    found = re.match(r"[\d.]+", text)
    return float(found.group()) if found is not None else 0.0


def _where(units: list[str]) -> str:
    """Windows as their xmids' stretches, records by name."""
    xmids = [x for unit in units if (x := xmid_of(unit)) is not None]
    records = [unit for unit in units if xmid_of(unit) is None]
    return ", ".join(
        part for part in (stretches(xmids) if xmids else "", ", ".join(records)) if part
    )


def _shown(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, str):
        return value
    return f"{value:g}" if isinstance(value, int | float) else json.dumps(value)
