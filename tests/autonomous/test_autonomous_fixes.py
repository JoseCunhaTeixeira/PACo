"""pick's fixes: every window G3 asked a change of picked again with it, in one call, each window
with its own change as its recipe; the windows it asked nothing of left as they were. On
active_test_profile in 24-receiver windows, as the picking tests make it."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
from mcp import Client
from sigpipe.masw.runs import find_run

from paco.autonomous import server
from paco.autonomous.tools.pick import picks_of
from paco.autonomous.tools.trials import Study
from paco.call import shown
from paco.qc.log import latest, read_attempts
from paco.qc.loops import stage_changes
from paco.settings import Settings, get_settings

L1 = {"masw": {"length": 24, "distance_max": 30.0}, "near_field_m": 8.25}
L1_REASONS = {
    "masw length": "a middle length between the survey's limits",
    "masw distance_max": "the shots near each window, for a quick test",
    "near_field_m": "half the longest wavelength PACo's trials reached",
}


@dataclass(frozen=True)
class Fixed:
    """The line before the fixes, what G3 asked of each window then, and the pick's answer."""

    asked: dict[str, dict[str, object]]
    answer: str
    refused: bool
    study: Study


@pytest.fixture(scope="module")
def fixed(
    input_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    searched: None,  # noqa: ARG001
) -> Iterator[Fixed]:
    root = tmp_path_factory.mktemp("fixes")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(Settings.model_config, "env_file", None)
        patch.setenv("PACO_INPUT_DIR", str(input_dir))
        patch.setenv("PACO_OUTPUT_DIR", str(root / "output"))
        patch.setenv("PACO_WORKERS", "4")
        get_settings.cache_clear()
        profile = {"profile": "active_test_profile"}

        async def play() -> tuple[dict[str, dict[str, object]], bool, str]:
            async with Client(server.server) as client:
                await client.call_tool("survey", profile)
                await client.call_tool("commit", {**profile, "changes": L1, "reasons": L1_REASONS})
                (folder,) = (
                    get_settings().output_dir / "active_test_profile" / "studies"
                ).iterdir()
                line = json.loads((folder / "line.json").read_text())
                run = find_run(line["run_id"], get_settings())
                attempts = read_attempts(run)
                asked: dict[str, dict[str, object]] = {}
                for unit in sorted(path.name for path in run.glob("xmid_*")):
                    attempt = latest(attempts, unit, "picking")
                    g3 = attempt.results.get("G3") if attempt is not None else None
                    if g3 is not None and (changes := stage_changes(g3, "picking")) is not None:
                        asked[unit] = changes[0]
                result = await client.call_tool(
                    "pick", {**profile, "ways": [{"windows": "all", "how": "fixes"}]}
                )
                return asked, bool(result.is_error), shown(result)

        asked, refused, answer = anyio.run(play)
        (folder,) = (get_settings().output_dir / "active_test_profile" / "studies").iterdir()
        yield Fixed(asked, answer, refused, Study("active_test_profile", folder))
    get_settings.cache_clear()


@pytest.mark.slow
def test_every_window_g3_asked_a_change_of_is_picked_with_it_in_one_call(fixed: Fixed) -> None:
    picked = {one.unit: one for one in picks_of(fixed.study) if one.recipe.how != "line"}

    assert not fixed.refused, fixed.answer
    assert fixed.asked, "G3 asks no window's picking to change: nothing to test"
    # Each window G3 asked a change of: picked with that change, the tracker its way; the others
    # left as they were, and said.
    assert set(picked) == set(fixed.asked)
    for unit, changes in fixed.asked.items():
        assert picked[unit].recipe.how == "auto"
        assert picked[unit].recipe.picking == changes
    assert fixed.answer.startswith(f"Picked {len(fixed.asked)} window")
    assert "Left as they were (no change G3 asks, or its fix tried): " in fixed.answer
