"""From PACo, src/paco/evaluation/running.py, `_server_settings`, `_quick_config` and `_lost`, at
commit e5f1a24.

Why: private, and the A/B evaluation (paco.autonomous.evaluation.running) plays the autonomous agent
as PACo's `run_scenario` plays PACo's: its in-process server pointed at the play's folders with
the evaluation's workers, quick inversions judged by the same QC configuration, and a play the
model's server cut short told from a request it refused. Copied as they are."""

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import openai

from paco.evaluation.running import EVALUATION_WORKERS, QUICK_MODEL
from paco.settings import get_settings


@contextmanager
def server_settings(
    outputs: Path, inputs: Path | None, cache: Path | None = None, quick: Path | None = None
) -> Generator[None]:
    """PACo's `_server_settings`: the in-process server reads its settings from the environment:
    point its outputs (and inputs, and images' cache) here, with the evaluation's workers;
    `quick`, the QC configuration of quick inversions (quick_config), with their effort."""
    values = {"PACO_OUTPUT_DIR": str(outputs), "PACO_WORKERS": str(EVALUATION_WORKERS)}
    if inputs is not None:
        values["PACO_INPUT_DIR"] = str(inputs)
    if cache is not None:
        values["PACO_CACHE_DIR"] = str(cache)
    if quick is not None:
        values |= {"PACO_INVERSION_EFFORT": "quick", "PACO_QC_CONFIG": str(quick)}
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                del os.environ[name]
            else:
                os.environ[name] = value
        get_settings.cache_clear()


def quick_config(folder: Path) -> Path:
    """PACo's `_quick_config`: the QC configuration of a play's quick inversions, written in its
    `folder`: G5 takes their short chains as converged (QUICK_MODEL), so that it does not sample
    them longer."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "quick_qc.json"
    path.write_text(json.dumps({"model": QUICK_MODEL}))
    return path


def lost(failed: openai.APIError) -> str | None:
    """PACo's `_lost`: why `failed` lost the play, when no model's server judged the request: the
    server out of reach or failing (a timeout too), or a 404 that is not its JSON, a page from
    whatever answers at its address (a proxy's while the server behind it is down). None for a
    refusal, the server's own 404 among them (a model it does not serve, a path it does not
    have)."""
    name = type(failed).__name__
    if isinstance(failed, openai.NotFoundError) and not isinstance(failed.body, dict):
        return f"{name}: a 404 page at {failed.request.url}, no model's server behind it"
    if isinstance(failed, (openai.APIConnectionError, openai.InternalServerError)):
        return f"{name}: {failed}"
    return None
