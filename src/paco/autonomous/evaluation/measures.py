"""What a play gives and costs (docs/autonomous_plan.md §12), read from what it left, the same way for both
agents where both leave it: the line's latest run in PAC's layout (its QC log, its curves and
models), the autonomous agent's study, and the play's transcript; its CPU and wall-clock time as
its clock measured them (paco.autonomous.evaluation.clock)."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sigpipe.masw.inversion.measuring import InversionMeasures
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.runs import RunManifest, window_length

from paco.agent.record import AnswerStep, ModelStep, ScopeStep, ToolStep, Transcript
from paco.autonomous.evaluation.clock import Clock
from paco.autonomous.evaluation.meter import Meter
from paco.autonomous.host import budgets
from paco.autonomous.sensors.checks import failing_runs, median_known
from paco.autonomous.sensors.scorecard import scorecard
from paco.autonomous.tools import decide, journal
from paco.autonomous.tools.commit import line_of
from paco.autonomous.tools.freeze import frozen
from paco.autonomous.tools.trials import NEAR_FIELD, Study
from paco.qc.config import read_qc_config
from paco.qc.inverting import MEASURES_FILE, investigation_depth
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.models import GateResult

type Agent = Literal["workflow", "autonomous"]
# The calls that change the line's settings, by agent: one after the first pick goes back.
_LINE_CALLS: dict[Agent, frozenset[str]] = {
    "workflow": frozenset({"run_processing", "redo"}),
    "autonomous": frozenset({"commit"}),
}


class Measures(BaseModel):
    """A play's measures (§12); None where the play left nothing to measure."""

    model_config = ConfigDict(frozen=True)

    # Results, on the line's latest run.
    windows: int | None = None  # the line's windows
    accepted: int | None = None  # those whose curve went on to the inversion
    gaps: int | None = None  # the stretches of consecutive windows without one
    depth_m: float | None = None  # the accepted curves' median λmax/2
    uncertainty: float | None = None  # their median relative uncertainty
    outliers: int | None = None  # G4's outliers among the curves
    models: int | None = None  # the windows inverted
    g5_pass: float | None = None  # the share of their latest models G5 passed
    misfit: float | None = None  # the models' median misfit (about 1 fits the uncertainties)
    depth_informed_m: float | None = None  # their median depth informed
    # Cost.
    cpu_s: float = 0.0
    wall_s: float = 0.0
    model_calls: int = 0  # the model's calls and the forms it filled
    prompt_tokens: int = 0  # every prompt token, the cached ones included
    cached_tokens: int = 0  # those the API read from its cache
    completion_tokens: int = 0
    # Process.
    experiments: int | None = None  # the autonomous agent's (the workflow agent runs none)
    # The line's settings made: PACo's runs processed, the autonomous agent's commits.
    commits: int = 0
    returns: int = 0  # the line's settings changed after the first pick
    refused: int = 0  # calls the host did not make
    failed: int = 0  # calls made that failed
    # The choices whose spread over the plays says how stable the agent is.
    length: int | None = None  # the windows' receivers
    distance_min_m: float | None = None  # the nearest shot allowed: the near field's, or masw's
    distance_max_m: float | None = None  # the farthest

    @property
    def coverage(self) -> float | None:
        """The share of the line's windows accepted."""
        if not self.windows or self.accepted is None:
            return None
        return self.accepted / self.windows


def measure(
    agent: Agent,
    outputs: Path,
    transcript: Transcript,
    clock: Clock,
    study: Study | None = None,
    since: datetime | None = None,
    meter: Meter | None = None,
) -> Measures:
    """The measures of a play of `agent` that wrote into `outputs`; `study`, the autonomous agent
    agent's, its journal and usage counted `since` the play began (a known-truth scenario's line
    was made before it, not by the agent); `meter`, the agent's model calls as the play counted
    them, the cached tokens too (without it, as the transcript and the usage log say them)."""
    steps = transcript.tool_steps
    picked = next(
        (index for index, step in enumerate(steps) if step.name == "pick" and _made(step)),
        len(steps),
    )
    returns = sum(
        1 for step in steps[picked + 1 :] if step.name in _LINE_CALLS[agent] and _made(step)
    )
    found = Measures(
        cpu_s=clock.cpu_s,
        wall_s=clock.wall_s,
        returns=returns,
        refused=sum(1 for step in steps if not step.called),
        failed=sum(1 for step in steps if step.called and step.is_error),
        commits=sum(1 for step in steps if step.name == "run_processing" and _made(step)),
        **_calls(transcript),
    )
    if study is not None:
        entries = [one for one in journal.entries(study) if since is None or one.at >= since]
        used = [one for one in budgets.uses(study) if since is None or one.at >= since]
        found = found.model_copy(
            update={
                "experiments": sum(1 for entry in entries if entry.kind == "experiment"),
                "commits": sum(1 for entry in entries if entry.kind == "commit"),
                "model_calls": sum(1 for one in used if one.kind in ("model", "form")),
                "prompt_tokens": sum(one.prompt_tokens or 0 for one in used),
                "completion_tokens": sum(one.completion_tokens or 0 for one in used),
            }
        )
    if meter is not None:
        found = found.model_copy(
            update={
                "model_calls": meter.calls,
                "prompt_tokens": meter.prompt_tokens,
                "cached_tokens": meter.cached_tokens,
                "completion_tokens": meter.completion_tokens,
            }
        )
    run_folder = latest_run(outputs)
    if run_folder is None:
        return found
    return found.model_copy(update=_line(run_folder, study))


def latest_run(outputs: Path) -> Path | None:
    """The newest run under `outputs`: the line as the play left it."""
    paths = sorted(outputs.glob("*/*/run.json"), key=lambda path: path.parent.name)
    return paths[-1].parent if paths else None


def _line(run_folder: Path, study: Study | None) -> dict[str, object]:
    """The line's results and settings, in `run_folder`: the curves accepted (the autonomous agent
    agent's frozen or accepted ones, PACo's passing G3 and G4), their scorecard, and the models."""
    manifest = RunManifest.model_validate_json((run_folder / "run.json").read_text())
    attempts = read_attempts(run_folder)
    g3: dict[float, GateResult | None] = {}
    g4: list[GateResult] = []
    passing: list[float] = []
    for window in manifest.windows:
        picked = latest(attempts, window.folder, "picking")
        results = picked.results if picked is not None else {}
        g3[window.xmid] = results.get("G3")
        if "G4" in results:
            g4.append(results["G4"])
        if all(
            results.get(gate) is not None and results[gate].verdict == "pass"
            for gate in ("G3", "G4")
        ):
            passing.append(window.xmid)
    accepted = passing if study is None else _accepted(study, run_folder, manifest)
    card = scorecard(
        g3,
        window_length(run_folder) or 0.0,
        covered=accepted,
        g4=g4,
        shots=str(manifest.preset.mode) != "passive",
    )
    config = read_qc_config(run_folder)
    models: list[tuple[str | None, InversionMeasures, float]] = []
    for window in manifest.windows:
        inverted = latest(attempts, window.folder, "inversion")
        path = run_folder / window.folder / MEASURES_FILE
        curve = saved_m0(run_folder / window.folder / CURVES_FILE)
        if inverted is None or not path.exists() or curve is None:
            continue
        g5 = inverted.results.get("G5")
        measures = InversionMeasures.model_validate_json(path.read_text())
        models.append(
            (g5.verdict if g5 is not None else None, measures, investigation_depth(curve, config))
        )
    preset = manifest.preset
    line = line_of(study) if study is not None else None
    near = line.settings.get(NEAR_FIELD) if line is not None else None
    return {
        "windows": len(card.windows),
        "accepted": len(card.covered),
        "gaps": len(failing_runs({xmid: xmid in card.covered for xmid in card.windows})),
        "depth_m": _rounded(card.depth_m, 2),
        "uncertainty": _rounded(card.precision, 3),
        "outliers": card.outliers,
        "models": len(models) if models else None,
        "g5_pass": sum(verdict == "pass" for verdict, _, _ in models) / len(models)
        if models
        else None,
        "misfit": _rounded(median_known(one.fits[0].misfit for _, one, _ in models if one.fits), 3),
        "depth_informed_m": _rounded(
            median_known(
                one.useful_depth_m if one.useful_depth_m is not None else reach
                for _, one, reach in models
            ),
            2,
        ),
        "length": preset.masw.length,
        "distance_min_m": float(near)
        if isinstance(near, int | float)
        else preset.masw.distance_min,
        "distance_max_m": preset.masw.distance_max,
    }


def _accepted(study: Study, run_folder: Path, manifest: RunManifest) -> list[float]:
    """The autonomous agent's accepted windows on the run: the frozen curves, or before the
    freeze, the windows decided accepted."""
    made = frozen(study)
    if made is not None and made.run_id == manifest.run_id:
        return [curve.xmid for curve in made.curves]
    statuses = decide.statuses(study, run_folder, manifest.run_id)
    return [window.xmid for window in manifest.windows if statuses.get(window.folder) == "accepted"]


def _calls(transcript: Transcript) -> dict[str, int]:
    """The model's calls and tokens as the transcript's steps say them: each call, each form
    filled (a form tried twice, two calls)."""
    calls = prompt = completion = 0
    for step in transcript.steps:
        if isinstance(step, ModelStep):
            calls += 1
        elif isinstance(step, ScopeStep | AnswerStep):
            calls += step.tries
        else:
            continue
        prompt += step.prompt_tokens or 0
        completion += step.completion_tokens or 0
    return {"model_calls": calls, "prompt_tokens": prompt, "completion_tokens": completion}


def _made(step: ToolStep) -> bool:
    return step.called and not step.is_error


def _rounded(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def spread(values: Sequence[float | None]) -> tuple[float, float] | None:
    """The smallest and largest of the known `values`: their spread over the plays."""
    known = [value for value in values if value is not None]
    return (min(known), max(known)) if known else None
