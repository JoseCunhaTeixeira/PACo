"""Finding a run from its ID, and reading what it holds, for the tools that work on a run once
it is processed."""

import re
from pathlib import Path

from paco.runs.models import RunError, RunManifest
from paco.settings import Settings
from sigpipe.base import DispersionImage
from sigpipe.transformers import Load

# Only names shaped like run IDs are looked up, so an ID can never reach outside the output
# directory (e.g. "../x").
_RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
_LISTED_RUNS = 5
IMAGE_FILE = "DispersionImage_0000.hdf5"


def find_run(run_id: str, settings: Settings) -> Path:
    """The folder of run `run_id`, in `<output_dir>/<profile>/<run_id>/`."""
    matches = (
        sorted(settings.output_dir.glob(f"*/{run_id}/run.json"))
        if _RUN_ID.fullmatch(run_id)
        else []
    )
    if not matches:
        latest = sorted(settings.output_dir.glob("*/*/run.json"), key=lambda path: path.parent.name)
        listed = ", ".join(
            f"{path.parent.name} ({path.parent.parent.name})"
            for path in reversed(latest[-_LISTED_RUNS:])
        )
        raise RunError(f"Unknown run '{run_id}'. Latest runs: {listed or 'none'}.")
    return matches[0].parent


def load_manifest(run_id: str, settings: Settings) -> RunManifest:
    return RunManifest.model_validate_json((find_run(run_id, settings) / "run.json").read_text())


def load_image(folder: Path) -> DispersionImage:
    """The dispersion image a window's pipeline saved in `folder`."""
    path = folder / IMAGE_FILE
    (image,) = Load(file_paths=[path], data_type="dispersion_image").transform()
    if not isinstance(image, DispersionImage):
        raise TypeError(f"{path} did not load as a dispersion image")
    return image
