"""From PACo, src/paco/qc/line.py, `_log` and `_line_settings`, at commit e5f1a24.

Why: private functions of PACo's line processing. commit (paco.geo.tools.commit) composes PACo's
public steps of the line anew, without its automatic choices (docs/geo_plan.md §13 M4), and logs the
attempts as PACo does, for PAC's pages and paco-replay to read them. Copied as they are."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sigpipe.masw.runs import RecordOutcome

from paco.qc.g4_profile import LINE
from paco.qc.log import append_attempt
from paco.qc.models import Attempt, Stage


def log_attempt(
    run_folder: Path,
    unit: str,
    stage: Stage,
    number: int,
    parameters: dict[str, Any],
    trigger: str,
    started_at: datetime,
    outcome: RecordOutcome,
) -> None:
    """PACo's `_log`: a record's stage, as its attempt."""
    append_attempt(
        run_folder,
        Attempt(
            unit=unit,
            stage=stage,
            attempt=number,
            parameters=parameters,
            triggered_by=trigger,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status=outcome.status,
            error=outcome.error,
        ),
    )


def line_settings(
    run_folder: Path, parameters: dict[str, Any], notes: tuple[str, ...], started_at: datetime
) -> None:
    """PACo's `_line_settings`: the line's settings the processing chose, and why (`notes`), as
    the line's phase-shift attempt; logged again when the line loop changes them: an attempt's
    last line is its current one."""
    append_attempt(
        run_folder,
        Attempt(
            unit=LINE,
            stage="phase_shift",
            attempt=1,
            parameters=parameters,
            triggered_by="initial",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="succeeded",
            notes=notes,
        ),
    )
