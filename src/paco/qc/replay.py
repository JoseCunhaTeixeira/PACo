"""paco-replay RUN_ID: a run made again from its inputs and its QC log (S7), in a folder of its
own, and compared with the run within PACO_REPLAY_TOLERANCE: its records preprocessed with each
record's latest parameters, its windows imaged with theirs, its curves picked with theirs on the
images made again. The run is only read.

Not replayed: what a person made in PAC (a record or an image made again there, a curve changed
by hand or picked again by PAC's picking), whose parameters the log does not hold; what makes a
replayed window from a record of theirs is the run's own record. Nor the inversions: a random
search without a seed, made again it draws other models; the soils come from those models."""

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sigpipe.algorithms.picking.dispersion.tracking import PickingParameters, pick_modes
from sigpipe.base.stream import Stream
from sigpipe.masw.picks import CURVES_FILE, save_pick
from sigpipe.masw.pipelines import PREPROCESSED, record_folder
from sigpipe.masw.presets import ActivePreset, PassivePreset, apply_overrides, resolve_preset
from sigpipe.masw.profiles import Profile, load_profile
from sigpipe.masw.runs import RunError, RunManifest, find_run, load_image, load_manifest
from sigpipe.masw.runs.caching import using
from sigpipe.masw.runs.history import log_entries
from sigpipe.masw.runs.origin import JUDGED, checks_current
from sigpipe.masw.runs.processing import (
    RECORDS_FOLDER,
    input_files,
    package_versions,
    preprocess_records,
    process_windows,
)
from sigpipe.masw.windows import MASWWindow, build_windows
from sigpipe.transformers import Load

from paco.qc.coherence import near_field_windows
from paco.qc.g4_profile import LINE
from paco.qc.judging import saved_m0
from paco.qc.log import latest, read_attempts
from paco.qc.models import Attempt, Stage
from paco.runs import PACKAGES
from paco.settings import Settings, get_settings

# Where the replays go, in the output folder: on disk, beside the runs.
REPLAYS = "replays"


@dataclass(frozen=True)
class Compared:
    """One output of the run against the replay's: the largest difference, relative; None, with
    `said`, when they cannot be set side by side."""

    unit: str
    stage: Stage
    difference: float | None
    said: str = ""

    def within(self, tolerance: float) -> bool:
        return self.difference is not None and self.difference <= tolerance


@dataclass(frozen=True)
class Skipped:
    unit: str
    stage: Stage
    reason: str


@dataclass(frozen=True)
class Replay:
    run_id: str
    compared: tuple[Compared, ...]
    skipped: tuple[Skipped, ...]
    notes: tuple[str, ...]
    folder: Path | None  # the replay's outputs, when kept
    tolerance: float  # the settings' replay_tolerance

    @property
    def same(self) -> bool:
        return all(one.within(self.tolerance) for one in self.compared)

    def lines(self) -> list[str]:
        """What the replay found, for a person: by stage, what matched, what did not and why
        some were not replayed."""
        out = [*self.notes]
        for stage in ("preprocessing", "phase_shift", "picking"):
            ones = [one for one in self.compared if one.stage == stage]
            if not ones:
                continue
            what = {"preprocessing": "records", "phase_shift": "images", "picking": "curves"}
            off = [one for one in ones if not one.within(self.tolerance)]
            largest = max((one.difference or 0.0 for one in ones if one not in off), default=0.0)
            out.append(
                f"{what[stage]}: {len(ones) - len(off)} of {len(ones)} the same within "
                f"{self.tolerance:g} (largest difference {largest:.1e})."
            )
            out += [f"  {one.unit}: {_differs(one)}" for one in off]
        reasons = Counter((one.stage, one.reason) for one in self.skipped)
        out += [
            f"Not replayed: {stage} of {count} unit(s), {reason}."
            for (stage, reason), count in reasons.items()
        ]
        if self.folder is not None:
            out.append(f"The replay's outputs: {self.folder}")
        verdict = "the same" if self.same else "different"
        out.append(f"Run {self.run_id} replayed: {verdict}.")
        return out


def replay_run(run_id: str, settings: Settings, keep: bool = False) -> Replay:
    """Run `run_id` made again from its inputs and its QC log's parameters, in a folder of
    `settings.output_dir`/replays/ (removed after, unless `keep`), and compared with the run.
    Raises RunError when its inputs changed since it ran, or the run changed meanwhile."""
    run_folder = find_run(run_id, settings)
    manifest = load_manifest(run_id, settings)
    profile = load_profile(manifest.profile.name, settings)
    notes = [*_inputs(run_id, manifest, profile), *_versions(manifest)]
    entries = log_entries(run_folder)
    attempts = read_attempts(run_folder)
    replays = settings.output_dir / REPLAYS
    replays.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=replays))
    try:
        # Everything made again: no image taken from a cache.
        with using(None):
            windows, skipped = _windows(manifest, profile, attempts, entries)
            used = sorted({path.name for window in windows for path in window.selected_files})
            records, out = _records(
                run_folder, folder, manifest, profile, attempts, entries, used, settings.workers
            )
            skipped += out
            _images(folder, manifest, profile, attempts, windows, settings.workers)
        compared = [
            *_compare_records(run_folder, folder, profile, records),
            *_compare_images(run_folder, folder, windows),
        ]
        picked, out = _picks(run_folder, folder, attempts, windows)
        compared += picked
        skipped += out
        skipped += _not_replayed(attempts)
        if log_entries(run_folder) != entries:
            raise RunError(
                f"Run {run_id} changed during its replay: replay it when nothing works on it."
            )
    finally:
        if not keep:
            shutil.rmtree(folder, ignore_errors=True)
    kept = folder if keep else None
    return Replay(
        run_id, tuple(compared), tuple(skipped), tuple(notes), kept, settings.replay_tolerance
    )


def _inputs(run_id: str, manifest: RunManifest, profile: Profile) -> list[str]:
    """Refused when the run's inputs changed since it ran: a replay makes its outputs from the
    same inputs. A run.json without their hashes, said."""
    if not manifest.inputs:
        return ["run.json holds no hash of the inputs: replayed on the files as they are now."]
    now = {one.name: one.sha256 for one in input_files(profile)}
    if changed := [one.name for one in manifest.inputs if now.get(one.name) != one.sha256]:
        raise RunError(
            f"Run {run_id}'s inputs changed since it ran: {', '.join(changed)}. A replay makes "
            "the run's outputs from the inputs it read."
        )
    return []


def _versions(manifest: RunManifest) -> list[str]:
    # The packages the run recorded: a run of PAC's pages names sigpipe alone.
    now = package_versions(tuple(name for name in PACKAGES if name in manifest.versions))
    if now == manifest.versions:
        return []
    then = ", ".join(f"{name} {value}" for name, value in manifest.versions.items())
    replayed = ", ".join(f"{name} {value}" for name, value in now.items())
    return [f"Made with {then}; replayed with {replayed}: a difference may come from the code."]


def _redone_in_pac(entries: Sequence[Mapping[str, Any]], unit: str, stage: Stage) -> bool:
    """Whether `unit`'s `stage` was started afresh in PAC with no attempt of PACo's after it."""
    reset = False
    for entry in entries:
        if entry.get("unit") != unit:
            continue
        if entry.get("event") == "reset" and stage in (entry.get("stages") or ()):
            reset = entry.get("actor") == "user"
        elif entry.get("stage") == stage and "attempt" in entry:
            reset = False
    return reset


def _windows(
    manifest: RunManifest,
    profile: Profile,
    attempts: Sequence[Attempt],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[MASWWindow], list[Skipped]]:
    """The windows to image again, as the run built them: from the preset's windows, without the
    shots its near field left out; those a person imaged in PAC left out."""
    windows = build_windows(profile, manifest.preset.masw)
    line = latest(attempts, LINE, "phase_shift")
    near = line.parameters.get("near_field", {}) if line is not None else {}
    if isinstance(near, Mapping) and isinstance(distance := near.get("distance_m"), int | float):
        windows, _ = near_field_windows(windows, float(distance))
    by_folder = {f"xmid_{window.xmid:.2f}": window for window in windows}
    kept: list[MASWWindow] = []
    skipped: list[Skipped] = []
    for outcome in manifest.windows:
        if outcome.status != "succeeded" or outcome.folder not in by_folder:
            continue
        if _redone_in_pac(entries, outcome.folder, "phase_shift"):
            skipped.append(Skipped(outcome.folder, "phase_shift", "imaged again in PAC"))
            continue
        kept.append(by_folder[outcome.folder])
    return kept, skipped


def _records(
    run_folder: Path,
    folder: Path,
    manifest: RunManifest,
    profile: Profile,
    attempts: Sequence[Attempt],
    entries: Sequence[Mapping[str, Any]],
    used: Sequence[str],
    workers: int,
) -> tuple[list[str], list[Skipped]]:
    """The records the windows use, preprocessed again into `folder`, each with its latest
    parameters on the run's preset; a record preprocessed again in PAC copied from the run.
    Returns the records made again."""
    by_name = {record.path.name: record for record in profile.records}
    presets: dict[str, ActivePreset | PassivePreset] = {}
    skipped: list[Skipped] = []
    for name in used:
        if _redone_in_pac(entries, name, "preprocessing"):
            skipped.append(Skipped(name, "preprocessing", "preprocessed again in PAC"))
            shutil.copytree(
                record_folder(run_folder / RECORDS_FOLDER, by_name[name]),
                record_folder(folder / RECORDS_FOLDER, by_name[name]),
            )
            continue
        attempt = latest(attempts, name, "preprocessing")
        own = attempt.parameters if attempt is not None else {}
        presets[name] = resolve_preset(apply_overrides(manifest.preset, own), profile)
    if presets:
        preprocess_records(manifest.preset, profile, folder, workers, presets=presets)
    return sorted(presets), skipped


def _images(
    folder: Path,
    manifest: RunManifest,
    profile: Profile,
    attempts: Sequence[Attempt],
    windows: Sequence[MASWWindow],
    workers: int,
) -> None:
    """The windows imaged again into `folder`, each with its latest parameters on the run's
    preset, on the records made again."""
    groups: dict[str, tuple[dict[str, Any], list[MASWWindow]]] = {}
    for window in windows:
        attempt = latest(attempts, f"xmid_{window.xmid:.2f}", "phase_shift")
        parameters = attempt.parameters if attempt is not None else {}
        key = json.dumps(parameters, sort_keys=True)
        groups.setdefault(key, (parameters, []))[1].append(window)
    for parameters, group in groups.values():
        preset = resolve_preset(apply_overrides(manifest.preset, parameters), profile)
        process_windows(
            preset, group, manifest.records, folder, workers, exclusions=manifest.exclusions
        )


def _compare_records(
    run_folder: Path, folder: Path, profile: Profile, names: Iterable[str]
) -> list[Compared]:
    by_name = {record.path.name: record for record in profile.records}
    compared: list[Compared] = []
    for name in names:
        record = by_name[name]
        paths = [
            record_folder(one / RECORDS_FOLDER, record) / PREPROCESSED
            for one in (run_folder, folder)
        ]
        if not all(path.exists() for path in paths):
            compared.append(Compared(name, "preprocessing", None, "no stream on one side"))
            continue
        run, replayed = (_stream(path) for path in paths)
        if run.xt.shape != replayed.xt.shape:
            said = f"{run.xt.shape} samples against {replayed.xt.shape}"
            compared.append(Compared(name, "preprocessing", None, said))
            continue
        compared.append(Compared(name, "preprocessing", _relative(run.xt, replayed.xt)))
    return compared


def _compare_images(
    run_folder: Path, folder: Path, windows: Sequence[MASWWindow]
) -> list[Compared]:
    compared: list[Compared] = []
    for window in windows:
        unit = f"xmid_{window.xmid:.2f}"
        run_window = (run_folder / unit / "window.json").read_text()
        replayed_window = folder / unit / "window.json"
        if not replayed_window.exists() or MASWWindow.model_validate_json(
            run_window
        ) != MASWWindow.model_validate_json(replayed_window.read_text()):
            compared.append(Compared(unit, "phase_shift", None, "its receivers or shots differ"))
            continue
        try:
            run, replayed = load_image(run_folder / unit), load_image(folder / unit)
        except (OSError, ValueError) as error:
            compared.append(Compared(unit, "phase_shift", None, f"no image: {error}"))
            continue
        if run.fv_map.shape != replayed.fv_map.shape or not (
            np.array_equal(run.fs, replayed.fs) and np.array_equal(run.vs, replayed.vs)
        ):
            said = "its frequencies or velocities differ"
            compared.append(Compared(unit, "phase_shift", None, said))
            continue
        compared.append(Compared(unit, "phase_shift", _relative(run.fv_map, replayed.fv_map)))
    return compared


def _picks(
    run_folder: Path, folder: Path, attempts: Sequence[Attempt], windows: Sequence[MASWWindow]
) -> tuple[list[Compared], list[Skipped]]:
    """The curves picked again on the images made again, with each window's latest picking
    parameters: those the run holds as PACo picked them, no person's change after."""
    compared: list[Compared] = []
    skipped: list[Skipped] = []
    for window in windows:
        unit = f"xmid_{window.xmid:.2f}"
        attempt = latest(attempts, unit, "picking")
        run_curve = saved_m0(run_folder / unit / CURVES_FILE)
        if attempt is None or run_curve is None:
            continue
        if attempt.triggered_by == JUDGED or attempt.status != "succeeded":
            skipped.append(Skipped(unit, "picking", "picked in PAC"))
            continue
        if not checks_current(run_folder / unit, attempt.finished_at):
            skipped.append(Skipped(unit, "picking", "changed in PAC after PACo picked it"))
            continue
        image = load_image(folder / unit)
        modes = pick_modes(image, PickingParameters.model_validate(attempt.parameters))
        if not modes or modes[0].curve is None:
            compared.append(Compared(unit, "picking", None, "no curve picked again"))
            continue
        save_pick(folder / unit, image, modes[0].curve)
        replayed = saved_m0(folder / unit / CURVES_FILE)
        if replayed is None or len(replayed.fs) != len(run_curve.fs):
            points = len(replayed.fs) if replayed is not None else 0
            said = f"{len(run_curve.fs)} points against {points}"
            compared.append(Compared(unit, "picking", None, said))
            continue
        difference = max(_relative(run_curve.fs, replayed.fs), _relative(run_curve.vs, replayed.vs))
        compared.append(Compared(unit, "picking", difference))
    return compared, skipped


def _not_replayed(attempts: Sequence[Attempt]) -> list[Skipped]:
    """The windows' inversions and soil columns, said, not made again."""
    why: dict[Stage, str] = {
        "inversion": "a random search without a seed: made again, it draws other models",
        "petro_inversion": "the soils come from the inversion's models",
    }
    said = {
        Skipped(one.unit, one.stage, why[one.stage])
        for one in attempts
        if one.stage in why and one.unit != LINE
    }
    return sorted(said, key=lambda one: (one.unit, one.stage))


def _stream(path: Path) -> Stream:
    (stream,) = Load(file_paths=[path], data_type="stream").transform([])
    if not isinstance(stream, Stream):
        raise TypeError(f"{path} did not load as a stream")
    return stream


def _relative(run: np.ndarray, replayed: np.ndarray) -> float:
    """The largest difference between the two, relative to the run's largest value."""
    largest = float(np.max(np.abs(run))) if run.size else 0.0
    difference = float(np.max(np.abs(run.astype(float) - replayed))) if run.size else 0.0
    return difference / largest if largest > 0 else difference


def _differs(one: Compared) -> str:
    return one.said if one.difference is None else f"differs by {one.difference:.1e}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="paco-replay", description=__doc__)
    parser.add_argument("run_id", help="the run to make again, e.g. 20261001-120000-abcd")
    parser.add_argument("--keep", action="store_true", help="keep the replay's outputs")
    arguments = parser.parse_args()
    try:
        replay = replay_run(arguments.run_id, get_settings(), keep=arguments.keep)
    except RunError as error:
        sys.exit(str(error))
    print("\n".join(replay.lines()))
    sys.exit(0 if replay.same else 1)


if __name__ == "__main__":
    main()
