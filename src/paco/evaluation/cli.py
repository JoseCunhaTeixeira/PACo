"""paco-evaluate [--repeat N] [scenario ...]: play the suite (or the named scenarios) with the
model in .env, and print the report; paco-evaluate --scopes: score the model's reading of the
labelled messages (paco.evaluation.scope_set)."""

import argparse
import contextlib
import logging

import anyio
from openai import AsyncOpenAI
from pydantic import ValidationError

from paco import logs
from paco.agent import AgentSettings, ChatModel, OpenAIChat
from paco.evaluation.report import format_history, format_report, read_reports
from paco.evaluation.running import run_evaluation
from paco.evaluation.scenarios import SCENARIOS
from paco.evaluation.scope_set import format_scope_report, read_scopes


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
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")
    if arguments.history:
        root = AgentSettings.model_fields["evaluation_dir"].default
        with contextlib.suppress(ValidationError):  # the model's settings missing: the default
            root = AgentSettings().evaluation_dir  # pyright: ignore[reportCallIssue]
        print(format_history(read_reports(root), arguments.model))
        return
    # PACo's server runs in this process, and its SDK logs every request at INFO level: the
    # scenarios' own lines (calls, failures, progress) say what matters.
    logs.setup(logging.WARNING)
    anyio.run(evaluate, arguments.scenarios, arguments.repeat, arguments.scopes)


async def evaluate(names: list[str], repeat: int = 1, scopes: bool = False) -> None:
    try:
        settings = AgentSettings()  # pyright: ignore[reportCallIssue]  # fields come from .env
    except ValidationError as error:
        missing = ", ".join(f"PACO_{problem['loc'][0]}".upper() for problem in error.errors())
        print(f"Set {missing} in .env (vLLM's address and model name).")
        return
    scenarios = [scenario for scenario in SCENARIOS if not names or scenario.name in names]
    if unknown := set(names) - {scenario.name for scenario in SCENARIOS}:
        known = ", ".join(scenario.name for scenario in SCENARIOS)
        print(f"Unknown scenario(s): {', '.join(sorted(unknown))}. Scenarios: {known}.")
        return

    model = OpenAIChat(
        AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key.get_secret_value()
        ),
        settings.llm_model,
        settings.llm_temperature,
        settings.llm_seed,
    )
    if scopes:
        read = await read_scopes(model, settings.llm_model, settings.evaluation_dir)
        print()
        print(format_scope_report(read))
        return
    judge_model: ChatModel | None = None
    if settings.judge_base_url and settings.judge_model:
        client = AsyncOpenAI(
            base_url=settings.judge_base_url, api_key=settings.judge_api_key.get_secret_value()
        )
        judge_model = OpenAIChat(client, settings.judge_model)
    else:
        print("No judge model (PACO_JUDGE_BASE_URL, PACO_JUDGE_MODEL): rule checks only.")

    report = await run_evaluation(
        scenarios,
        model,
        settings.llm_model,
        settings.evaluation_dir,
        judge_model,
        settings.judge_model,
        repeat,
    )
    print()
    print(format_report(report))
    print(f"\nReport and transcripts in {settings.evaluation_dir / report.eval_id}")
