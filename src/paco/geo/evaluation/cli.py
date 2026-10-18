"""paco-ab [scenario ...]: the evaluation of docs/geo_plan.md §12, PACo's two agents on the same
scenarios (all of them by default, the known-truth set's too): the guided agent as it is and the
autonomous geophysicist, each scenario played --repeat times by each, then the A/B report. Both
agents on the model of .env (PACO_LLM_*, any provider), so that the agents are compared, not their
models; the judge on its own (PACO_JUDGE_*) when set. With --list, each agent's tools and the
scenarios; with --smoke, the model client's smoke test (M1); with --pac, PAC's pages on runs the
agent made (M4)."""

import argparse
import logging
import sys
from collections.abc import Sequence
from functools import partial
from importlib.metadata import version
from pathlib import Path

import anyio
from pydantic import ValidationError

from paco import logs
from paco import server as paco_server
from paco.geo import pins
from paco.geo import server as geo_server
from paco.geo.evaluation import pac, smoke
from paco.geo.evaluation.inputs import offered
from paco.geo.evaluation.report import format_ab
from paco.geo.evaluation.running import VARIANTS, Inversions, Models, Variant, run_ab
from paco.geo.evaluation.scenarios import PAIRS, TRUTH_PAIRS, pairs_named
from paco.geo.host.loop import Limits, short_context
from paco.geo.settings import GeoSettings
from paco.llm.chat import chat_model, judge_model, seeing
from paco.llm.settings import NO_MODEL, JudgeSettings, ModelSettings, missing
from paco.settings import get_settings

# What the evaluation says when the judge's settings are wrong, or it has none.
NO_JUDGE_MODEL = "Set {missing} in .env (the judge's model), or leave PACO_JUDGE_MODEL unset."
NO_JUDGE = "No judge model (PACO_JUDGE_MODEL): rule checks only."
NO_VISION = (
    "The vision variant needs PACO_LLM_VISION=true in .env: the model's provider then receives "
    "the figures."
)


def main(argv: Sequence[str] | None = None) -> None:
    """paco-ab: the A/B evaluation, which exits with status 1 when it cannot be played; --list,
    --smoke and --pac as their help says."""
    parser = argparse.ArgumentParser(
        prog="paco-ab",
        description="Play PACo's guided agent and its autonomous geophysicist on the same "
        "scenarios, and compare them.",
    )
    parser.add_argument(
        "scenarios", nargs="*", help="scenarios to play (default: all of them; see --list)"
    )
    parser.add_argument(
        "-n",
        "--repeat",
        type=int,
        default=5,
        metavar="N",
        help="plays of each scenario by each agent, since the models sample (default: 5)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=[],
        help="the geophysicist agent played also without advise, with vision, or at an effort "
        "whatever the message asks (the agent as it is is always played)",
    )
    parser.add_argument(
        "--inversions",
        choices=("scenario", "quick", "full"),
        default="scenario",
        help="the plays' inversions: quick, but where a scenario judges the models (scenario, "
        "the default); every play's quick; or every play's at PAC's effort",
    )
    parser.add_argument(
        "--price",
        nargs="+",
        type=float,
        metavar="DOLLARS",
        help="the models' price, in dollars a million tokens: prompt, completion, and cached "
        "prompt (a tenth of the prompt's when left out), for the money each play cost",
    )
    parser.add_argument("--geo-only", action="store_true", help="play the geophysicist agent alone")
    parser.add_argument(
        "--list", action="store_true", help="list each agent's tools and the scenarios"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"make {smoke.CALLS} calls to the model in .env (PACO_LLM_*), each kind of "
        "call in turn, and print their usage",
    )
    parser.add_argument(
        "--pac",
        nargs="+",
        metavar="RUN",
        help="check that PAC's pages open these runs (<profile>/<run id> in PACO_OUTPUT_DIR) "
        "with no error, PAC's API asked in PAC's own environment, and print each run card's "
        "settings with their reasons",
    )
    parser.add_argument(
        "--pac-dir",
        type=Path,
        default=pac.PAC_DIR,
        help="PAC's checkout, with its .venv (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    # Both servers run in this process, and their SDK logs every request at INFO level.
    logs.setup(logging.WARNING)
    if arguments.list:
        print(anyio.run(listing))
        return
    if arguments.smoke:
        if not anyio.run(smoked):
            sys.exit(1)
        return
    if arguments.pac:
        try:
            found = pac.pac_pages(arguments.pac, get_settings(), arguments.pac_dir)
        except ValueError as error:
            parser.error(str(error))
        print("\n\n".join(pac.describe_pages(pages) for pages in found))
        if not all(pages.opened for pages in found):
            sys.exit(1)
        return
    if arguments.repeat < 1:
        parser.error("--repeat must be at least 1")
    prices: tuple[float, float, float] | None = None
    if arguments.price:
        if len(arguments.price) not in (2, 3):
            parser.error("--price takes PROMPT COMPLETION [CACHED], dollars a million tokens")
        given = [*arguments.price, arguments.price[0] / 10][:3]
        prices = (given[0], given[1], given[2])
    try:
        pairs = pairs_named(arguments.scenarios, offered(get_settings().input_dir))
    except ValueError as error:
        parser.error(str(error))
    if not pairs:
        parser.error(f"None of these scenarios' lines is in {get_settings().input_dir}.")
    played = partial(
        evaluate,
        arguments.scenarios,
        arguments.repeat,
        [VARIANTS[key] for key in arguments.variants],
        arguments.inversions,
        prices,
        arguments.geo_only,
    )
    if not anyio.run(played):
        sys.exit(1)


async def evaluate(
    names: list[str],
    repeat: int,
    variants: list[Variant],
    inversions: Inversions,
    prices: tuple[float, float, float] | None,
    geo_only: bool,
) -> bool:
    """The A/B evaluation of the scenarios `names` (all when none) with the models of .env,
    printed; whether it could be played."""
    try:
        settings = ModelSettings()  # pyright: ignore[reportCallIssue]  # fields come from .env
    except ValidationError as error:
        print(NO_MODEL.format(missing=missing(error)))
        return False
    if (short := short_context(settings.context())) is not None:
        print(short)
        return False
    if any(variant.vision for variant in variants) and not settings.llm_vision:
        print(NO_VISION)
        return False
    try:
        judging = JudgeSettings()
    except ValidationError as error:
        print(NO_JUDGE_MODEL.format(missing=missing(error)))
        return False
    try:
        # Each agent its own client, on the same model.
        paco_model = None if geo_only else chat_model(settings)
    except ValueError as error:  # a cloud's library or credentials missing
        print(error)
        return False
    judge = judge_model(judging)
    if judge is None:
        print(NO_JUDGE)
    source = get_settings().input_dir
    try:
        geo = chat_model(settings)
    except ValueError as error:  # a cloud's library or credentials missing
        print(error)
        return False
    if any(variant.vision for variant in variants) and (said := await seeing(geo)) is not None:
        print(f"{said} The vision variant needs a model that takes them.")
        return False
    models = Models(
        geo=geo,
        geo_name=settings.llm_model,
        paco=paco_model,
        paco_name=None if geo_only else settings.llm_model,
        judge=judge,
        judge_name=judging.judge_model,
        limits=Limits.of(settings),
    )
    root = GeoSettings().evaluation_dir or get_settings().output_dir / "evaluations"
    report = await run_ab(
        pairs_named(names, offered(source)),
        models,
        root,
        source,
        repeat,
        [Variant(), *variants],
        inversions,
    )
    print()
    print(format_ab(report, prices))
    print(f"\nReport and transcripts in {root / report.eval_id}")
    return True


async def smoked() -> bool:
    """The smoke test with the model of .env, printed; whether every call succeeded."""
    try:
        settings = ModelSettings()  # pyright: ignore[reportCallIssue]  # fields come from .env
    except ValidationError as error:
        print(NO_MODEL.format(missing=missing(error)))
        return False
    try:
        model = chat_model(settings)
    except ValueError as error:
        print(error)
        return False
    if (said := await seeing(model)) is not None:
        print(said)
    where = settings.llm_provider + (
        f" at {settings.llm_base_url}" if settings.llm_base_url else ""
    )
    print(
        f"{smoke.CALLS} calls to {settings.llm_model} ({where}), vision {_on(model.vision)}, "
        f"thinking {_on(model.thinking)} unless a call says otherwise:"
    )
    calls = await smoke.smoke(model)
    print(smoke.format_calls(calls))
    return not any(call.error for call in calls)


async def listing() -> str:
    """Each agent the evaluation plays, what it runs on, and the tools its server offers; then
    the scenarios, their lines and messages."""
    upstream = f"PACo's quality control {pins.paco_code()[:7]}, sigpipe {_short('sigpipe')}"
    agents = (
        (f"PACo's guided agent as it is ({upstream})", paco_server.server),
        (
            f"The autonomous geophysicist (paco {version('paco')}, on {upstream})",
            geo_server.server,
        ),
    )
    lines: list[str] = []
    for name, server in agents:
        tools = await server.list_tools()
        lines.append(f"{name}: {len(tools)} tools")
        lines.extend(f"  {tool.name}" for tool in tools)
    pairs = (*PAIRS, *TRUTH_PAIRS)
    lines.append(f"Scenarios, both agents on the same messages: {len(pairs)}")
    lines.extend(f"  {pair.name} ({pair.profile}): {' / '.join(pair.questions)}" for pair in pairs)
    return "\n".join(lines)


def _short(name: str) -> str:
    """The commit distribution `name` runs at, shortened as git shows it."""
    commit = pins.commit(name)
    return commit[:7] if commit else "(not installed from git)"


def _on(switch: bool) -> str:
    return "on" if switch else "off"
