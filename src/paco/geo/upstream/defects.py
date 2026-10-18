"""From PACo, src/paco/evaluation/defects.py, `_dead_trace` and `_refracted`, at commit e5f1a24.

Why: private, and PACo's `build_inputs` copies a passive demo profile too, which this repository
does not keep (the geophysicist agent studies active lines, docs/geo_plan.md §16). The A/B
evaluation (paco.geo.evaluation) builds the active profiles PACo's suite plays on, with these.
Copied as they are."""

import shutil
import warnings
from pathlib import Path

import numpy as np
import obspy
import yaml

from paco.evaluation.defects import DEMO_SHOT_S


def dead_trace(source: Path, folder: Path, trace: int) -> None:
    """PACo's `_dead_trace`: a copy of profile `source` whose first record has trace `trace`
    zeroed, the records rewritten as MiniSEED (obspy writes no SEG-2)."""
    folder.mkdir(parents=True)
    shutil.copy(source / "receiver_positions.yaml", folder / "receiver_positions.yaml")
    sources = yaml.safe_load((source / "source_positions.yaml").read_text())
    renamed = {f"{Path(name).stem}.mseed": point for name, point in sources.items()}
    (folder / "source_positions.yaml").write_text(yaml.safe_dump(renamed))
    for index, path in enumerate(
        sorted(path for path in source.iterdir() if path.suffix == ".dat")
    ):
        with warnings.catch_warnings():  # obspy doubts the demo's SEG-2 delay header
            warnings.simplefilter("ignore")
            stream = obspy.read(str(path))
        if index == 0:
            stream[trace].data = np.zeros_like(stream[trace].data)
        for one in stream:
            one.data = np.asarray(one.data, dtype=np.float32)
        stream.write(str(folder / f"{path.stem}.mseed"), format="MSEED", encoding="FLOAT32")


def refracted(source: Path, folder: Path, velocity: float, ratio: float) -> None:
    """PACo's `_refracted`: a copy of active profile `source` with a head wave added to every
    trace, a 40 Hz Ricker wavelet leaving each shot at DEMO_SHOT_S at `velocity`, `ratio` times
    the trace's peak, the records rewritten as MiniSEED."""
    folder.mkdir(parents=True)
    shutil.copy(source / "receiver_positions.yaml", folder / "receiver_positions.yaml")
    sources = yaml.safe_load((source / "source_positions.yaml").read_text())
    renamed = {f"{Path(name).stem}.mseed": point for name, point in sources.items()}
    (folder / "source_positions.yaml").write_text(yaml.safe_dump(renamed))
    receivers = [
        float(point["x"])
        for point in yaml.safe_load((source / "receiver_positions.yaml").read_text())
    ]
    for path in sorted(path for path in source.iterdir() if path.suffix == ".dat"):
        with warnings.catch_warnings():  # obspy doubts the demo's SEG-2 delay header
            warnings.simplefilter("ignore")
            stream = obspy.read(str(path))
        shot_x = float(sources[path.name]["x"])
        for one, x in zip(stream, receivers, strict=True):
            data = np.asarray(one.data, dtype=float)
            times = np.arange(data.size) / one.stats.sampling_rate
            tau = np.pi * 40.0 * (times - DEMO_SHOT_S - abs(x - shot_x) / velocity)
            ricker = (1 - 2 * tau**2) * np.exp(-(tau**2))
            one.data = (data + ratio * np.abs(data).max() * ricker).astype(np.float32)
        stream.write(str(folder / f"{path.stem}.mseed"), format="MSEED", encoding="FLOAT32")
