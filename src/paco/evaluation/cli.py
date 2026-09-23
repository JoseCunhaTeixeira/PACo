"""paco-evaluate [--repeat N] [scenario ...]: play the suite (or the named scenarios) with the
model in .env, and print the report."""

import argparse
import logging

import anyio
from openai import AsyncOpenAI
from pydantic import ValidationError

from paco.agent import AgentSettings, ChatModel, OpenAIChat
from paco.evaluation.report import format_report
from paco.evaluation.running import run_evaluation
from paco.evaluation.scenarios import SCENARIOS


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
        default=1,
        metavar="N",
        help="plays of each scenario, for pass rates, since the model samples (default: 1)",
    )
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")
    # PACo's server runs in this process, and its SDK logs every request at INFO level: the
    # scenarios' own lines (calls, failures, progress) say what matters.
    logging.getLogger().setLevel(logging.WARNING)
    anyio.run(evaluate, arguments.scenarios, arguments.repeat)


async def evaluate(names: list[str], repeat: int = 1) -> None:
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
    )
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
