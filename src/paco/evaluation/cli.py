"""paco-evaluate [scenario ...]: play the suite (or the named scenarios) with the model in .env,
and print the report."""

import logging
import sys

import anyio
from openai import AsyncOpenAI
from pydantic import ValidationError

from paco.agent import AgentSettings, ChatModel, OpenAIChat
from paco.evaluation.report import format_report
from paco.evaluation.running import run_evaluation
from paco.evaluation.scenarios import SCENARIOS


def main() -> None:
    """paco-evaluate: the suite, or the scenarios named on the command line."""
    # PACo's server runs in this process, and its SDK logs every request at INFO level: the
    # scenarios' own lines (calls, failures, progress) say what matters.
    logging.getLogger().setLevel(logging.WARNING)
    anyio.run(evaluate, sys.argv[1:])


async def evaluate(names: list[str]) -> None:
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
    )
    print()
    print(format_report(report))
    print(f"\nReport and transcripts in {settings.evaluation_dir / report.eval_id}")
