"""The profiles an evaluation plays on: PACo's demo profiles, and synthetic variants of them with
a known defect, written once per evaluation (docs/qc_workflow.md: "synthetic variants of the
demo profiles with known defects", in the data where the defect is physical).

- active_dead: active_p1 with trace 40 of its first record zeroed, a dead geophone for G1.

The other defects of the scenarios come from the demo itself (its records' 20 ms trigger delay,
the mode jump of a 24-receiver window at xmid 14.88, a passive line with no curve) or from
settings the user types (a velocity range too narrow, too few iterations, bounds too tight).
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import obspy
import yaml

DEMO_PROFILES = ("active_p1", "passive_p1")
DEAD_TRACE = ("active_dead", "active_p1", 40)  # the variant, its source profile, the trace


def build_inputs(demo_dir: Path, target: Path) -> Path:
    """`target`, holding the demo profiles of `demo_dir` and the synthetic variants; built once,
    kept when it exists."""
    for name in DEMO_PROFILES:
        if not (target / name).exists():
            shutil.copytree(demo_dir / name, target / name)
    variant, source, trace = DEAD_TRACE
    if not (target / variant).exists():
        _dead_trace(demo_dir / source, target / variant, trace)
    return target


def _dead_trace(source: Path, folder: Path, trace: int) -> None:
    """A copy of profile `source` whose first record has trace `trace` zeroed, the records
    rewritten as MiniSEED (obspy writes no SEG-2)."""
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
