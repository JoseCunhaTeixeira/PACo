"""Properties over many generated inputs (E6 of PACo's agent guidelines), from a seeded generator:
the conversions of metres to receivers, the mapping of positions to windows, and the QC log read
as the run stands after any sequence of attempts and resets."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sigpipe.masw.runs import RunError, RunManifest
from sigpipe.masw.runs.history import LOG_FILE, forget

from paco.qc import Attempt, Stage, append_attempt, read_attempts
from paco.qc.positions import at_positions, in_receivers

CASES = 200


def _manifest(xmids: list[float]) -> RunManifest:
    return RunManifest.model_validate(
        {
            "run_id": "20260924-100000-abcd",
            "profile": {
                "name": "line",
                "kind": "active",
                "n_records": 1,
                "n_receivers": 96,
                "receiver_x_range_m": [0.0, 95.0],
                "receiver_spacing_m": 1.0,
                "sampling_rate_hz": 1000.0,
                "nyquist_hz": 500.0,
                "record_duration_range_s": [1.0, 1.0],
                "source_x_range_m": [0.0, 0.0],
            },
            "preset": {"mode": "active"},
            "versions": {},
            "started_at": "2026-09-24T10:00:00Z",
            "finished_at": "2026-09-24T10:00:10Z",
            "n_positions": len(xmids),
            "windows": [
                {"xmid": xmid, "folder": f"xmid_{xmid:.2f}", "status": "succeeded"}
                for xmid in xmids
            ],
        }
    )


def test_a_length_in_metres_is_the_nearest_count_of_receivers() -> None:
    rng = np.random.default_rng(1)
    for _ in range(CASES):
        spacing = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, 5.0]))
        metres = float(rng.uniform(spacing, 60 * spacing))

        converted, notes = in_receivers({"masw": {"length_m": metres}}, spacing)

        assert converted is not None
        length = converted["masw"]["length"]
        # n receivers span n - 1 spacings: the nearest count, within half a spacing.
        assert abs((length - 1) * spacing - metres) <= spacing / 2 + 1e-9
        assert length >= 2 and len(notes) == 1
        # The receivers' own count goes through as it is.
        assert in_receivers({"masw": {"length": length}}, spacing)[0] == {
            "masw": {"length": length}
        }


def test_each_position_maps_to_its_nearest_window_or_is_refused() -> None:
    rng = np.random.default_rng(2)
    for _ in range(CASES):
        step = float(rng.choice([0.5, 1.0, 1.5, 3.0]))
        first = float(rng.uniform(0, 10))
        xmids = [round(first + step * index, 2) for index in range(int(rng.integers(1, 40)))]
        manifest = _manifest(xmids)
        position = float(rng.uniform(xmids[0] - 3 * step, xmids[-1] + 3 * step))
        reach = (xmids[0] - step, xmids[-1] + step) if len(xmids) > 1 else (xmids[0], xmids[0])

        if not reach[0] <= position <= reach[1]:
            with pytest.raises(RunError, match="off the line"):
                at_positions(manifest, [position])
            continue
        (unit,), said = at_positions(manifest, [position])

        nearest = min(abs(xmid - position) for xmid in xmids)
        assert abs(float(unit.removeprefix("xmid_")) - position) == pytest.approx(nearest)
        assert said.startswith(f"{position:g} m: xmid ")


def test_the_log_read_after_any_resets_holds_what_came_after_each(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    units = ("xmid_1.00", "xmid_2.00")
    stages: tuple[Stage, ...] = ("phase_shift", "picking", "inversion")
    for case in range(40):
        run = tmp_path / f"run{case}"
        run.mkdir()
        expected: dict[tuple[str, str], int] = {}  # attempts since each unit's stage's reset
        numbers: dict[tuple[str, str], int] = {}
        appended = resets = 0
        for _ in range(int(rng.integers(5, 30))):
            unit, stage = units[int(rng.integers(2))], stages[int(rng.integers(3))]
            if rng.random() < 0.2 and (run / LOG_FILE).exists():
                forget(run, unit, stage, results=False, later=False, folder=run / unit)
                expected[unit, stage] = 0
                numbers[unit, stage] = 0
                resets += 1
                continue
            appended += 1
            numbers[unit, stage] = numbers.get((unit, stage), 0) + 1
            expected[unit, stage] = expected.get((unit, stage), 0) + 1
            append_attempt(
                run,
                Attempt(
                    unit=unit,
                    stage=stage,
                    attempt=numbers[unit, stage],
                    parameters={},
                    triggered_by="initial" if numbers[unit, stage] == 1 else "G3:mode_jump",
                    started_at=datetime(2026, 9, 24, tzinfo=UTC),
                    status="succeeded",
                ),
            )

        read = read_attempts(run)

        for (unit, stage), count in expected.items():
            own = [one for one in read if (one.unit, one.stage) == (unit, stage)]
            assert [one.attempt for one in own] == list(range(1, count + 1))
        # The log only grew: every line ever appended is there, the resets among them.
        lines = [json.loads(line) for line in (run / LOG_FILE).read_text().splitlines()]
        assert sum(line.get("event") == "stage" for line in lines) == appended
        assert sum(line.get("event") == "reset" for line in lines) == resets
