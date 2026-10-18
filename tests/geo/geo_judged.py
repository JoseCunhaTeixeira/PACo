"""A judged line for the tests: active_p2 (the shots the tests keep) processed in four 24-receiver
windows, then picked and judged by G1 to G4, as PACo's own tests make theirs (its test_qc.py)."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from sigpipe.masw.runs import find_run, run_processing, window_length

from paco.qc import QCConfig, judge_run
from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt, GateResult
from paco.qc.report import xmid_of
from paco.settings import Settings

# Four windows of 24 receivers, one every 24: real curves in seconds.
FOUR_WINDOWS = {"masw": {"length": 24, "step": 24}}


@dataclass(frozen=True)
class JudgedLine:
    """A processed, picked and judged run of active_p2."""

    folder: Path  # the run's folder
    length_m: float  # its windows' length, from their first receiver to their last
    attempts: tuple[Attempt, ...]  # its QC log
    g3: dict[float, GateResult]  # each window's latest G3 result, by xmid
    g4: dict[float, GateResult]  # each curve's latest G4 result, by xmid
    line: GateResult  # G4's result on the line


def make_judged_line(input_dir: Path, root: Path) -> JudgedLine:
    """The line, made in `root` from active_p2 in `input_dir`."""
    with pytest.MonkeyPatch.context() as patch:
        # PACo's settings from these arguments and its defaults alone, never a developer's .env.
        patch.setitem(Settings.model_config, "env_file", None)
        patch.chdir(root)
        settings = Settings(input_dir=input_dir, output_dir=root / "output", workers=4)
        summary = run_processing("active_p2", "active", FOUR_WINDOWS, settings)
        judge_run(summary.run_id, settings, QCConfig())
        folder = find_run(summary.run_id, settings)
    attempts = read_attempts(folder)
    g3: dict[float, GateResult] = {}
    g4: dict[float, GateResult] = {}
    for unit in sorted({attempt.unit for attempt in attempts}):
        picked = latest(attempts, unit, "picking")
        xmid = xmid_of(unit)
        if picked is None or xmid is None:
            continue
        g3[xmid] = picked.results["G3"]
        if "G4" in picked.results:
            g4[xmid] = picked.results["G4"]
    length_m = window_length(folder)
    line = latest(attempts, "line", "picking")
    assert length_m is not None and line is not None
    return JudgedLine(folder, length_m, attempts, g3, g4, line.results["G4"])
