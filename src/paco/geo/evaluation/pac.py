"""PAC's pages on a run (docs/geo_plan.md §2.3, §13 M4): what PAC's quality and dispersion pages ask
of a run, asked of PAC's own API in PAC's environment (its .venv), with PACo's folders as PAC's
(MASW_INPUT_DIR, MASW_OUTPUT_DIR). The run opens with no error when every page answers; its run
card's settings come back, with the reasons PAC shows for them."""

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paco.settings import Settings

# PAC's checkout, next to this one's as the plan sets them side by side (§2).
PAC_DIR = Path("../PAC")
# Where the script's answer starts in what it prints: PAC may log before it.
ANSWER = "PAC'S PAGES: "
# Run in PAC's Python: PAC's app asked through FastAPI's test client (no server started), every page
# of a run without an inversion, the line's, each record's and each window's.
SCRIPT = r"""
import json
import sys

from fastapi.testclient import TestClient
from masw.api.main import app
from masw.io.paths import OUTPUT_DIR

client = TestClient(app)
found = {}
for folder in sys.argv[1:]:
    manifest = json.loads((OUTPUT_DIR / folder / "run.json").read_text())
    failed = {}
    asked = 0

    def ask(path):
        global asked
        asked += 1
        response = client.get(path)
        if response.status_code != 200:
            failed[path] = f"{response.status_code} {response.text[:300]}"
            return None
        return response.json() if "json" in response.headers.get("content-type", "") else None

    runs = ask("/quality/runs") or []
    listed = any(entry.get("folder") == folder for one in runs for entry in one.get("runs", []))
    ask("/output_folders")
    card = ask(f"/quality/run/{folder}") or {}
    for page in ("records", "dispersion", "inversion", "petro"):
        ask(f"/quality/{page}/overview/{folder}")
    ask(f"/xmids/{folder}")
    ask(f"/dispersion_picks_by_position/{folder}")
    ask(f"/inversion/status/{folder}")
    if any(
        (OUTPUT_DIR / folder / window["folder"] / "SeismicInversion_Parameters_0000.json").exists()
        for window in manifest["windows"]
    ):
        ask(f"/inversion/velocity_section/{folder}")
    for label in ask(f"/dispersion_image_labels/{folder}") or {}:
        ask(f"/dispersion_pseudo_section/{folder}/{label}")
    for name in ask(f"/quality/run_figures/{folder}") or []:
        ask(f"/quality/run_figure/{folder}/{name}")
    for record in manifest["records"]:
        for page in ("card", "gather", "spectrum"):
            ask(f"/quality/records/{page}/{folder}/{record['name']}")
    for window in manifest["windows"]:
        xmid = window["folder"].removeprefix("xmid_")
        ask(f"/quality/dispersion/card/{folder}/{xmid}")
        if window.get("status") == "failed":
            # A window whose processing failed has no image, nor its shots' page.
            continue
        if manifest["preset"]["mode"] != "active":
            # A passive window's own: its correlations, FK selection and spectra (PACo's active
            # runs have none either).
            for page in ("gather", "selection", "spectrum"):
                ask(f"/quality/dispersion/{page}/{folder}/{xmid}")
        ask(f"/quality/sources/{folder}/{xmid}")
        ask(f"/dispersion_images/{folder}/{xmid}")
        if (OUTPUT_DIR / folder / window["folder"] / "SeismicInversion_Parameters_0000.json").exists():
            # A window's model: PAC's inversion card and its chains.
            ask(f"/quality/inversion/card/{folder}/{xmid}")
            ask(f"/quality/inversion/chains/{folder}/{xmid}")
    found[folder] = {
        "asked": asked,
        "failed": failed,
        "listed": listed,
        "settings": card.get("settings", []),
    }
print("PAC'S PAGES: " + json.dumps(found))
"""


@dataclass(frozen=True)
class Pages:
    """What PAC's pages answered of run `folder` (<profile>/<run id>): how many were asked, those
    that failed, whether PAC lists the run, and its run card's settings."""

    folder: str
    asked: int
    failed: dict[str, str]
    listed: bool
    settings: tuple[dict[str, Any], ...] = field(default=())

    @property
    def opened(self) -> bool:
        return self.listed and not self.failed


def pac_pages(folders: Sequence[str], settings: Settings, pac: Path = PAC_DIR) -> list[Pages]:
    """PAC's pages on each run of `folders` (<profile>/<run id>), PAC at `pac`, its folders
    PACo's (`settings`)."""
    python = pac / ".venv" / "bin" / "python"
    if not python.exists():
        raise ValueError(f"No PAC environment at {python}: give PAC's checkout (--pac-dir).")
    done = subprocess.run(
        [str(python), "-c", SCRIPT, *folders],
        cwd=pac,
        env={
            **os.environ,
            "MASW_INPUT_DIR": str(settings.input_dir.resolve()),
            "MASW_OUTPUT_DIR": str(settings.output_dir.resolve()),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    answer = next(
        (line for line in done.stdout.splitlines() if line.startswith(ANSWER)),
        None,
    )
    if done.returncode != 0 or answer is None:
        raise RuntimeError(f"PAC's pages could not be asked:\n{done.stderr[-3000:]}")
    found = json.loads(answer.removeprefix(ANSWER))
    return [
        Pages(
            folder=folder,
            asked=found[folder]["asked"],
            failed=found[folder]["failed"],
            listed=found[folder]["listed"],
            settings=tuple(found[folder]["settings"]),
        )
        for folder in folders
    ]


def describe_pages(pages: Pages) -> str:
    """What PAC's pages answered of a run, for a person: the pages that failed, then the run
    card's settings, each with where it comes from and why."""
    head = (
        f"PAC's pages on {pages.folder}: {pages.asked - len(pages.failed)} of {pages.asked} "
        "answered" + ("" if pages.listed else "; the run is not in PAC's list of runs") + "."
    )
    lines = [head, *(f"  {path}: {said}" for path, said in pages.failed.items())]
    lines.append("Its run card's settings:")
    for setting in pages.settings:
        detail = f" ({setting['detail']})" if setting.get("detail") else ""
        lines.append(
            f"  {setting['label']}: {setting['value']}{detail}; {setting['origin']}: "
            f"{setting['why']}"
        )
    return "\n".join(lines)
