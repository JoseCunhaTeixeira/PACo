"""From PACo, src/paco/qc/judging.py, `judge_line` (G4 over the line), at commit e5f1a24.

Why: judge_line reads the whole QC log twice for each window to log its verdict (its
`_judge_latest`, then `record_result`), and logs every verdict again on every call. On a line of
73 windows a call costs 6 s, more with each pick as the log grows. pick (paco.geo.tools.pick) runs
G4 after every call: this copy reads the log once and appends only the verdicts that changed, so
that the log reads the same. Otherwise as it is."""

from datetime import UTC, datetime
from pathlib import Path

from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.quality.line import Series
from sigpipe.masw.runs import RunManifest

from paco.qc.config import QCConfig
from paco.qc.g4_profile import LINE, judge_profile
from paco.qc.judging import saved_m0
from paco.qc.log import append_attempt, attempts_of, latest, read_attempts
from paco.qc.models import Attempt, GateResult
from paco.qc.origin import run_work


def judge_line(run_folder: Path, manifest: RunManifest, config: QCConfig) -> tuple[GateResult, ...]:
    """PACo's `judge_line`: G4 over the line, the saved M0 curve of every window whose latest pick
    passed G3, each against its neighbours, a person's curves among them as trusted references
    (never judged). A window's verdict goes to its latest picking attempt, the line's own (the
    coverage) to an attempt of the unit "line", each only when it changed. No automatic curve on
    the line (a run picked by hand): no G4."""
    attempts = read_attempts(run_folder)
    work = run_work(run_folder, manifest, attempts)
    curves: list[Series] = []
    trusted: set[str] = set()
    without: list[float] = []
    for window in manifest.windows:
        state = work[window.folder].m0
        path = run_folder / window.folder / CURVES_FILE
        picked = latest(attempts, window.folder, "picking")
        g3 = picked.results.get("G3") if picked else None
        curve = (
            saved_m0(path)
            if state == "user" or (state == "judged" and g3 is not None and g3.verdict == "pass")
            else None
        )
        if curve is None:
            without.append(window.xmid)
            continue
        curves.append(Series.from_curve(window.folder, window.xmid, curve))
        if state == "user":
            trusted.add(window.folder)
    if not any(one.m0 in ("judged", "unjudged") for one in work.values()):
        return ()
    started_at = datetime.now(UTC)
    results = judge_profile(curves, config.profile, without, frozenset(trusted))
    for result in results:
        if result.unit != LINE:
            attempt = latest(attempts, result.unit, "picking")
            if attempt is not None and attempt.results.get(result.gate) != result:
                judged = attempt.model_copy(
                    update={"results": {**attempt.results, result.gate: result}}
                )
                append_attempt(run_folder, judged, "verdict")
            continue
        line = latest(attempts, LINE, "picking")
        if line is not None and line.results.get(result.gate) == result:
            continue
        append_attempt(
            run_folder,
            Attempt(
                unit=LINE,
                stage="picking",
                attempt=len(attempts_of(attempts, LINE, "picking")) + 1,
                parameters={},
                triggered_by="initial",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status="succeeded",
                results={result.gate: result},
            ),
        )
    return results
