"""Evaluation of the agent: scenarios played against PACo's real server, scored by rule checks
and, optionally, a judge model.

Each scenario runs in its own output directory; its transcript, and the report of the whole
suite, are saved for review.
"""

from .checks import Check, Trial
from .judging import judge, read_score, render
from .models import CheckResult, EvaluationReport, JudgeScore, Kind, ScenarioResult
from .report import format_report
from .running import run_evaluation, run_scenario
from .scenarios import SCENARIOS, Scenario

__all__ = [
    "SCENARIOS",
    "Check",
    "CheckResult",
    "EvaluationReport",
    "JudgeScore",
    "Kind",
    "Scenario",
    "ScenarioResult",
    "Trial",
    "format_report",
    "judge",
    "read_score",
    "render",
    "run_evaluation",
    "run_scenario",
]
