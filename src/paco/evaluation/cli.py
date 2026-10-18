"""paco-evaluate [--repeat N] [scenario ...]: play the suite (or the named scenarios) with the
model in .env, and print the report; paco-evaluate --scopes: score the model's reading of the
labelled messages (paco.evaluation.scope_set)."""

import argparse
import logging

import anyio
from pydantic import ValidationError

from paco import logs
from paco.agent import AgentSettings
from paco.agent.settings import OutputSettings
from paco.evaluation.report import format_history, format_report, read_reports
from paco.evaluation.running import run_evaluation
from paco.evaluation.scenarios import SCENARIOS
from paco.evaluation.scope_set import format_scope_report, read_scopes
from paco.llm.chat import chat_model
from paco.llm.settings import NO_MODEL, missing


def main() -> None:
    """paco-evaluate: the suite, or the scenarios named on the command line."""
    parser = argparse.ArgumentParser(
        prog="paco-evaluate",
        description="Play PACo's evaluation scenarios with the model in .env, and print the "
        "report.",
    )
    parser.add_argument("scenarios", nargs="*", help="scenarios to play (default: all of them)")
    parser.add_argument(
        "-n",
        "--repeat",
        type=int,
        default=5,
        metavar="N",
        help="plays of each scenario, for pass rates, since the model samples (default: 5)",
    )
    parser.add_argument(
        "--scopes",
        action="store_true",
        help="score the model's scope forms on the labelled messages instead of the scenarios",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="print each scenario's pass rate by model and prompt version, from the reports kept",
    )
    parser.add_argument("--model", help="with --history: only this model's evaluations")
    parser.add_argument(
        "--full-inversion",
        action="store_true",
        help="every play's inversions at PAC's effort (by default quick, but in the scenarios "
        "that judge them)",
    )
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")
    if arguments.history:
        print(format_history(read_reports(OutputSettings().evaluations), arguments.model))
        return
    # PACo's server runs in this process, and its SDK logs every request at INFO level: the
    # scenarios' own lines (calls, failures, progress) say what matters.
    logs.setup(logging.WARNING)
    anyio.run(
        evaluate, arguments.scenarios, arguments.repeat, arguments.scopes, arguments.full_inversion
    )


async def evaluate(
    names: list[str], repeat: int = 1, scopes: bool = False, full_inversion: bool = False
) -> None:
    try:
        settings = AgentSettings()  # pyright: ignore[reportCallIssue]  # fields come from .env
    except ValidationError as error:
        print(NO_MODEL.format(missing=missing(error)))
        return
    scenarios = [scenario for scenario in SCENARIOS if not names or scenario.name in names]
    if unknown := set(names) - {scenario.name for scenario in SCENARIOS}:
        known = ", ".join(scenario.name for scenario in SCENARIOS)
        print(f"Unknown scenario(s): {', '.join(sorted(unknown))}. Scenarios: {known}.")
        return

    # The model, and another client of it as the judge.
    model, judge = chat_model(settings), chat_model(settings)
    if scopes:
        read = await read_scopes(model, settings.llm_model, settings.evaluations)
        print()
        print(format_scope_report(read))
        return
    report = await run_evaluation(
        scenarios,
        model,
        settings.llm_model,
        settings.evaluations,
        judge,
        settings.llm_model,
        repeat,
        full_inversion=full_inversion,
    )
    print()
    print(format_report(report))
    print(f"\nReport and transcripts in {settings.evaluations / report.eval_id}")
