"""grid: a search's variants scored in code, fast, as trials score them: a full grid of the line's
settings, for the model to choose from. Most of an image's cost is where
it begins, and many variants share that beginning, their base: the windows' length and the
records' preprocessing, on a passive line the segments' slicing, whitening and normalization too.
Each base is made once a window: an active line's dispersion image of each shot, a
passive-active line's correlation of each shot, a passive line's correlations of each segment
(straight and space-flipped) and their f-k ratios; then each variant takes its own from it (the
shots its distances keep, the segments its FK selection keeps, flipped where their energy runs
the other way), stacks them as sigpipe's pipeline does, and its image is judged as a trial's
(trials.judged): the scores a trial gives it. Each variant's images at PLACES places along the line
are kept small (thumbnails), as PAC draws them, for the sheets the model judges by eye.

Its scratch: <study>/grid/, removed after."""

import json
import shutil
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sigpipe.algorithms.flipping.flipping import FlipAxis, flip
from sigpipe.algorithms.picking.dispersion.curve import (
    longest_reached_wavelength,
    min_resolvable_wavelength,
)
from sigpipe.algorithms.selection.stream.fk import fk_ratios
from sigpipe.algorithms.stacking.stream.phase_weighted import (
    instantaneous_phase,
    stack_phase_weighted_given,
)
from sigpipe.base import Stream
from sigpipe.dataio.selection_plotting import selection_scores
from sigpipe.masw.picks import CURVES_FILE
from sigpipe.masw.pipelines import PREPROCESSED
from sigpipe.masw.pipelines.common import load_preprocessed, stage_kwargs
from sigpipe.masw.profiles import Profile
from sigpipe.masw.runs import WindowOutcome, load_image
from sigpipe.masw.runs.processing import GEOMETRIES, RECORDS_FOLDER
from sigpipe.masw.runs.stopping import finished
from sigpipe.masw.runs.writing import write_atomic
from sigpipe.masw.windows import Exclusions, MASWWindow, apply_exclusions
from sigpipe.transformers import (
    ActiveShotCorrelation,
    Apodize,
    Correlate,
    Dispersion,
    Load,
    Normalize,
    Save,
    Slice,
    Stack,
    Whiten,
)
from sigpipe.workers import one_thread_each

from paco import stopping
from paco.autonomous.tools import trials
from paco.qc.coherence import TrialJudge
from paco.qc.config import QCConfig
from paco.qc.judging import saved_m0
from paco.settings import Settings

GRID = "grid"  # the scratch, in the study's folder
# A variant's thumbnails: its windows nearest PLACES places along the line, each image THUMB
# frequencies by THUMB velocities.
PLACES = 4
THUMB = 150
# A passive window's selection, as sigpipe's PlotSelection saves it for G2.
SELECTION = "Selection_0000.json"
# The stage a passive base holds besides the length and the preprocessing: its segments (their
# whitening and normalization made within it, the segments and their f-k ratios shared).
SEGMENTED = ("slicing",)


@dataclass(frozen=True)
class Entry:
    """A variant on one window of its base: its label, its trial, and its window (None when it
    has none to image, `error` saying why); `sheet`: its image kept as a thumbnail."""

    label: str
    trial: trials.Trial
    window: MASWWindow | None
    error: str | None = None
    sheet: bool = False


@dataclass(frozen=True)
class Thumb:
    """A variant's window on a sheet: where it is, its image as PAC draws it (each frequency's
    row scaled from its least value to its largest, 0 to 255; rows the velocities from the
    lowest, columns the frequencies), the frequencies and velocities it spans, its frequency
    step, its automatic pick, and the wavelengths of the lines λ = 2·dx and 3·L (0: none)."""

    xmid: float
    image: np.ndarray
    extent: tuple[float, float, float, float]  # fmin, fmax, vmin, vmax
    step_hz: float
    pick: tuple[np.ndarray, np.ndarray]  # frequencies, velocities
    lines_m: tuple[float, float]


def write_thumbs(path: Path, thumbs: Sequence[Thumb]) -> None:
    """A variant's thumbnails into `path` (.npz), in order along the line."""
    arrays: dict[str, np.ndarray] = {
        "xmids": np.array([one.xmid for one in thumbs], dtype=np.float32),
        "images": np.stack([one.image for one in thumbs]),
        "extents": np.array([one.extent for one in thumbs], dtype=np.float32),
        "steps": np.array([one.step_hz for one in thumbs], dtype=np.float32),
        "lines": np.array([one.lines_m for one in thumbs], dtype=np.float32),
    }
    for i, one in enumerate(thumbs):
        arrays[f"pick_f{i}"], arrays[f"pick_v{i}"] = one.pick
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]  # every key an array


def read_thumbs(path: Path) -> list[Thumb]:
    """A variant's thumbnails, as write_thumbs keeps them."""
    with np.load(path) as found:
        return [
            Thumb(
                xmid=float(found["xmids"][i]),
                image=found["images"][i],
                extent=tuple(float(one) for one in found["extents"][i]),  # pyright: ignore[reportArgumentType]
                step_hz=float(found["steps"][i]),
                pick=(found[f"pick_f{i}"], found[f"pick_v{i}"]),
                lines_m=(float(found["lines"][i][0]), float(found["lines"][i][1])),
            )
            for i in range(len(found["xmids"]))
        ]


def thumbnail(folder: Path, xmid: float) -> Thumb:
    """The thumbnail of the window in `folder`: each column its nearest frequency (a coarse step
    stays blocky, as PAC draws it), each row the largest of its velocities (a thin ridge stays)."""
    image = load_image(folder)
    fv = np.asarray(image.fv_map, dtype=float)  # a row a frequency
    low, high = fv.min(axis=1, keepdims=True), fv.max(axis=1, keepdims=True)
    scaled = (fv - low) / np.where(high > low, high - low, 1.0)
    fs, vs = np.asarray(image.fs, dtype=float), np.asarray(image.vs, dtype=float)
    columns = np.abs(fs[None, :] - np.linspace(fs[0], fs[-1], THUMB)[:, None]).argmin(axis=1)
    rows = np.unique(np.linspace(0, len(vs), THUMB + 1).astype(int)[:-1])
    pooled = np.maximum.reduceat(scaled[columns], rows, axis=1)
    curve = saved_m0(folder / CURVES_FILE)
    pick = (
        (np.asarray(curve.fs, dtype=np.float32), np.asarray(curve.vs, dtype=np.float32))
        if curve is not None
        else (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
    )
    return Thumb(
        xmid=xmid,
        image=np.round(pooled.T * 255).astype(np.uint8),
        extent=(float(fs[0]), float(fs[-1]), float(vs[0]), float(vs[-1])),
        step_hz=float(np.median(np.diff(fs))) if len(fs) > 1 else 0.0,
        pick=pick,
        lines_m=(
            float(min_resolvable_wavelength(image.acquisition) or 0.0),
            float(longest_reached_wavelength(image.acquisition) or 0.0),
        ),
    )


def nearest(xmids: Sequence[float], places: Sequence[float]) -> set[float]:
    """Of a variant's windows (by xmid), those nearest each of `places`."""
    return {min(xmids, key=lambda xmid: abs(xmid - place)) for place in places} if xmids else set()


@dataclass(frozen=True)
class Task:
    """One window of a base, every variant of it on that window: what a worker makes."""

    xmid: float
    records: Path  # the base's preprocessed records
    folder: Path  # its scratch: the window's folder made there, variant after variant
    usable: dict[str, tuple[float, float] | None]
    config: QCConfig
    entries: tuple[Entry, ...]


def base_of(trial: trials.Trial) -> str:
    """What `trial`'s images share with the trials of the same base: its mode, the windows'
    length, the dispersion, the records' preprocessing, and on a passive line the segments'."""
    preset = trial.preset.model_dump(mode="json")
    mode = preset[trials.MODE]
    held = {
        trials.MODE: mode,
        "length": preset["masw"]["length"],
        "dispersion": preset["dispersion"],
        **{stage: preset.get(stage) for stage in trials.PREPROCESSING},
        **({stage: preset[stage] for stage in SEGMENTED} if mode == "passive" else {}),
    }
    return json.dumps(held, sort_keys=True)


def score(
    study: trials.Study,
    variants: Mapping[str, trials.Trial],
    windows: str,
    line: trials.LineState | None,
    profile: Profile,
    settings: Settings,
    config: QCConfig,
    sheets: Path | None = None,
    places: Sequence[float] = (),
) -> tuple[dict[str, trials.TrialRun], dict[str, str]]:
    """Each of `variants` (by label) on the trial set `windows`, judged as `tried` judges it:
    its run, and why each variant that could not be tried could not. With `sheets`, each
    variant's thumbnails at its windows nearest `places` written there, <label>.npz."""
    parts = trials.parse_windows(windows)
    last = trials.last_xmids(study)
    bases: dict[str, list[str]] = {}
    for label, trial in variants.items():
        bases.setdefault(base_of(trial), []).append(label)
    scratch = study.folder / GRID
    shutil.rmtree(scratch, ignore_errors=True)
    tasks: list[Task] = []
    made: dict[str, dict[str, Any]] = {}
    refused: dict[str, str] = {}
    # The windows a variant takes depend on its mode, masw and near field alone (and the records
    # G1 leaves out): each set built once.
    built: dict[str, list[MASWWindow] | str] = {}
    try:
        for number, labels in enumerate(bases.values()):
            stopping.check()
            folder = scratch / f"b{number}"
            first = variants[labels[0]].preset
            records, _ = trials.with_records(study, folder, first, profile, settings, config)
            cone = trials.fast_cone(profile, first, config)
            exclusions, usable, g1 = trials.settled(
                folder, profile, first, records, config, cone, study
            )
            failed = {r.name: r.error or "" for r in records.outcomes if r.status == "failed"}
            by_xmid: dict[float, list[Entry]] = {}
            for label in labels:
                trial = variants[label]
                held = json.dumps(
                    [
                        str(trial.preset.mode),
                        trial.preset.masw.model_dump(mode="json"),
                        trial.near_field_m,
                        sorted(exclusions.records),
                    ]
                )
                if held not in built:
                    built[held] = _windows(trial, profile, parts, line, last, exclusions)
                chosen = built[held]
                if isinstance(chosen, str):
                    refused[label] = chosen
                    continue
                xs = [receiver.x for receiver in chosen[0].acquisitions[0].receivers]
                made[label] = {
                    "key": trials.trial_key(study, trial, chosen, config),
                    "length_m": round(max(xs) - min(xs), 3),
                    "g1": g1,
                }
                shown = nearest([w.xmid for w in chosen], places) if sheets is not None else set()
                for window in chosen:
                    entry = _entry(label, trial, window, exclusions, failed, window.xmid in shown)
                    by_xmid.setdefault(window.xmid, []).append(entry)
            for i, (xmid, entries) in enumerate(sorted(by_xmid.items())):
                tasks.append(
                    Task(
                        xmid,
                        folder / RECORDS_FOLDER,
                        folder / f"t{i}",
                        usable,
                        config,
                        tuple(entries),
                    )
                )
        results, thumbs = _made(tasks, scratch, settings.workers)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if sheets is not None:
        sheets.mkdir(parents=True, exist_ok=True)
        for label, kept in thumbs.items():
            write_thumbs(sheets / f"{label}.npz", sorted(kept, key=lambda one: one.xmid))
    now = datetime.now(UTC)
    runs: dict[str, trials.TrialRun] = {}
    for label, held in made.items():
        trial = variants[label]
        windows_made = sorted(results.get(label, []), key=lambda one: one[0].xmid)
        runs[label] = trials.TrialRun(
            key=held["key"],
            mode=str(trial.preset.mode),
            settings=trial.settings,
            length=trial.preset.masw.length,
            length_m=held["length_m"],
            g1=held["g1"],
            windows=tuple(result for result, _ in windows_made),
            seconds={"images": round(sum(cost for _, cost in windows_made), 3)},
            made_at=now,
        )
    return runs, refused


def _windows(
    trial: trials.Trial,
    profile: Profile,
    parts: Sequence[trials.Part],
    line: trials.LineState | None,
    last: Sequence[float],
    exclusions: Exclusions,
) -> list[MASWWindow] | str:
    """`trial`'s trial windows, as tried takes them; why it has none, when it has none."""
    try:
        chosen = trials.trial_windows(trial, profile, parts, line, last)
    except ValueError as error:
        return str(error)
    if exclusions.records:
        # The near-field rule on the records G1 keeps, as tried applies it.
        kept = {w.xmid: w for w in trials.line_windows(profile, trial, exclusions.records)}
        chosen = [kept[window.xmid] for window in chosen]
    return chosen


def _entry(
    label: str,
    trial: trials.Trial,
    window: MASWWindow,
    exclusions: Exclusions,
    failed: Mapping[str, str],
    sheet: bool = False,
) -> Entry:
    """A variant's window without G1's exclusions, as process_windows takes it: none to image
    when G1 left it too little, or a record it reads failed its preprocessing."""
    applied = apply_exclusions(window, exclusions, GEOMETRIES[trial.preset.mode])
    if applied is None:
        return Entry(
            label,
            trial,
            None,
            "every record, or all but 2 of its receivers, excluded by the signal QC",
        )
    if missing := [path.name for path in applied.selected_files if path.name in failed]:
        return Entry(
            label, trial, None, f"record {missing[0]} failed preprocessing: {failed[missing[0]]}"
        )
    return Entry(label, trial, applied, sheet=sheet)


def _made(
    tasks: Sequence[Task], scratch: Path, workers: int
) -> tuple[dict[str, list[tuple[trials.WindowResult, float]]], dict[str, list[Thumb]]]:
    """Every task on `workers` processes: each variant's window results, with what each cost,
    and its thumbnails."""
    results: dict[str, list[tuple[trials.WindowResult, float]]] = {}
    thumbs: dict[str, list[Thumb]] = {}
    if not tasks:
        return results, thumbs
    one_thread_each()  # the workers are the cores the search takes
    scratch.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_start, initargs=(scratch,)
    ) as executor:
        futures = [executor.submit(made_window, task) for task in tasks]
        for future in finished(executor, futures, stopping.current()):
            for label, result, cost, thumb in future.result():
                results.setdefault(label, []).append((result, cost))
                if thumb is not None:
                    thumbs.setdefault(label, []).append(thumb)
    return results, thumbs


def _start(scratch: Path) -> None:
    """A worker: in the scratch, with no GUI backend."""
    import os

    os.chdir(scratch)
    matplotlib.use("Agg")


def made_window(task: Task) -> list[tuple[str, trials.WindowResult, float, Thumb | None]]:
    """Runs in a worker: the task's base made once on its window, then each variant's image from
    it, judged as a trial's; each with what it cost, seconds (the base's share included), and its
    thumbnail where its entry keeps one."""
    out: list[tuple[str, trials.WindowResult, float, Thumb | None]] = []
    products: _Products | None = None
    for entry in sorted(task.entries, key=_prepared):
        started = time.perf_counter()
        if entry.window is None:
            out.append(
                (
                    entry.label,
                    trials.WindowResult(xmid=task.xmid, cached=False, error=entry.error),
                    0.0,
                    None,
                )
            )
            continue
        folder = task.folder / f"xmid_{entry.window.xmid:.2f}"  # PAC's window folder name
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        try:
            write_atomic(folder / "window.json", entry.window.model_dump_json(indent=2))
            products = products or _PRODUCTS[str(entry.trial.preset.mode)](task)
            products.image(entry, folder)
        except Exception as error:  # a window's failure, as process_windows records it
            result = trials.WindowResult(
                xmid=entry.window.xmid, cached=False, error=f"{type(error).__name__}: {error}"
            )
        else:
            judge = TrialJudge(
                task.config.coherence, task.config.curve, entry.trial.picking, task.config.image
            )
            outcome = WindowOutcome(xmid=entry.window.xmid, folder=folder.name, status="succeeded")
            result = trials.judged(
                task.folder, outcome, entry.trial, judge, task.usable, task.config
            )
        thumb = None
        if entry.sheet and result.error is None:
            thumb = thumbnail(folder, entry.window.xmid)
        shutil.rmtree(folder, ignore_errors=True)
        out.append((entry.label, result, round(time.perf_counter() - started, 3), thumb))
    return out


class _Products:
    """What a base makes once on a window, and each variant's image made from it."""

    def __init__(self, task: Task) -> None:
        self.records = task.records

    def image(self, entry: Entry, folder: Path) -> None:
        raise NotImplementedError


class _Shots(_Products):
    """An active window's shots: each one's dispersion image, made once; a variant's, its shots'
    stacked (sigpipe's active pipeline)."""

    def __init__(self, task: Task) -> None:
        super().__init__(task)
        self.images: dict[str, Any] = {}

    def image(self, entry: Entry, folder: Path) -> None:
        assert entry.window is not None
        preset = entry.trial.preset
        window = entry.window
        missing = [
            i for i, path in enumerate(window.selected_files) if path.name not in self.images
        ]
        if missing:
            made = Dispersion(method="phase", **stage_kwargs(preset, "dispersion")).transform(
                _load(window, missing, self.records)
            )
            for i, image in zip(missing, made, strict=True):
                self.images[window.selected_files[i].name] = image
        stacked = Stack(**stage_kwargs(preset, "image_stacking")).transform(
            [self.images[path.name] for path in window.selected_files]
        )
        Save(folder_path=folder).transform(stacked)


class _Correlated(_Products):
    """A passive-active window's shots: each one's correlation, made once (none for a shot within
    the window), and its phase where a variant stacks them phase-weighted; a variant's, its
    shots' stacked, then its image (sigpipe's passive-active pipeline)."""

    def __init__(self, task: Task) -> None:
        super().__init__(task)
        self.correlations: dict[str, Stream | None] = {}
        self.phases: dict[str, np.ndarray] = {}

    def image(self, entry: Entry, folder: Path) -> None:
        assert entry.window is not None
        preset = entry.trial.preset
        window = entry.window
        missing = [
            i for i, path in enumerate(window.selected_files) if path.name not in self.correlations
        ]
        if missing:
            tapered = Apodize(method="hanning", frac=0.1).transform(
                _load(window, missing, self.records)
            )
            correlate = ActiveShotCorrelation(method="cross")
            for i, stream in zip(missing, tapered, strict=True):
                correlated = correlate.transform([stream])
                self.correlations[window.selected_files[i].name] = (
                    correlated[0] if correlated else None
                )
        held = [
            (path.name, one)
            for path in window.selected_files
            if (one := self.correlations[path.name]) is not None
        ]
        kept = [one for _, one in held]
        _imaged(_stacked(preset, kept, [name for name, _ in held], self.phases), preset, folder)


class _Segments(_Products):
    """A passive window's noise segments: their f-k ratios measured once, in every band the task's
    variants select in; for each whitening and normalization, each segment correlated the way a
    variant can keep it (space-flipped where its ratio is positive, straight where negative), and
    its phase where a variant stacks them phase-weighted; a variant's, the segments its FK
    selection keeps, stacked, then its image (sigpipe's passive pipeline: its per-segment steps,
    each the same on a segment kept or not, whichever way it is flipped)."""

    def __init__(self, task: Task) -> None:
        super().__init__(task)
        self.segments: list[Stream] = []
        self.whitened: tuple[str, list[Stream | None]] | None = None
        self.prepared = ""  # the whitening and normalization of the correlations held
        self.straight: dict[int, Stream] = {}
        self.flipped: dict[int, Stream] = {}
        self.phases: dict[tuple[bool, int], np.ndarray] = {}
        selections = [stage_kwargs(entry.trial.preset, "selection") for entry in task.entries]
        self.everything = any(selection["method"] == "none" for selection in selections)
        chosen = [selection for selection in selections if selection["method"] != "none"]
        self.bands = list(dict.fromkeys((one.get("vmin"), one.get("vmax")) for one in chosen))
        self.lowest = min((float(one["threshold"]) for one in chosen), default=1.0)
        self.ratios: dict[tuple[float | None, float | None], list[float]] = {}
        self.needed: tuple[set[int], set[int]] = (set(), set())

    def image(self, entry: Entry, folder: Path) -> None:
        assert entry.window is not None
        preset = entry.trial.preset
        if not self.segments:
            self._sliced(entry.window, preset)
        if _prepared(entry) != self.prepared:
            self._correlated(preset, _prepared(entry))
        selection = stage_kwargs(preset, "selection")
        if selection["method"] == "none":
            chosen = [(False, j) for j in range(len(self.segments))]
        else:
            threshold = float(selection["threshold"])
            band = (selection.get("vmin"), selection.get("vmax"))
            scores = [(ratio, abs(ratio) > threshold) for ratio in self.ratios[band]]
            (folder / SELECTION).write_text(
                selection_scores(scores, threshold, True).model_dump_json()
            )
            chosen = [(ratio > 0, j) for j, (ratio, keep) in enumerate(scores) if keep]
            if not chosen:
                raise ValueError(
                    f"The fk selection kept none of the {len(scores)} segments (|f-k ratio| > "
                    f"{selection['threshold']}): lower its threshold"
                )
        kept = [(self.flipped if turned else self.straight)[j] for turned, j in chosen]
        _imaged(_stacked(preset, kept, chosen, self.phases), preset, folder)

    def _sliced(self, window: MASWWindow, preset: trials.Preset) -> None:
        """The window's segments, their f-k ratios, and which a variant may keep, each way."""
        loaded = load_preprocessed(window, self.records).transform()
        streams = [one for one in loaded if isinstance(one, Stream)]
        self.segments = list(Slice(**stage_kwargs(preset, "slicing")).transform(streams))
        measured = [fk_ratios(one, self.bands) for one in self.segments] if self.bands else []
        self.ratios = {
            band: [ratios[i] for ratios in measured] for i, band in enumerate(self.bands)
        }
        flipped = {
            j
            for ratios in self.ratios.values()
            for j, ratio in enumerate(ratios)
            if ratio > 0 and ratio > self.lowest
        }
        straight = {
            j
            for ratios in self.ratios.values()
            for j, ratio in enumerate(ratios)
            if ratio < 0 and -ratio > self.lowest
        }
        self.needed = (set(range(len(self.segments))) if self.everything else straight, flipped)

    def _correlated(self, preset: trials.Preset, prepared: str) -> None:
        """The segments a variant may keep, whitened (once for every normalization), normalized,
        tapered and correlated each way it may keep them; the correlations held before let go."""
        straight, flipped = self.needed
        needed = sorted(straight | flipped)
        whitening = json.dumps(stage_kwargs(preset, "whitening"), sort_keys=True)
        if self.whitened is None or self.whitened[0] != whitening:
            made = (
                Whiten(**stage_kwargs(preset, "whitening")).transform(
                    [self.segments[j] for j in needed]
                )
                if needed
                else []
            )
            whitened: list[Stream | None] = [None] * len(self.segments)
            for j, one in zip(needed, made, strict=True):
                whitened[j] = one
            self.whitened = (whitening, whitened)
        ready: Sequence[Stream] = [one for j in needed if (one := self.whitened[1][j]) is not None]
        for step in (
            Normalize(**stage_kwargs(preset, "normalization")),
            Apodize(method="hanning", frac=0.1),
        ):
            ready = step.transform(ready) if ready else []
        correlate = Correlate(method="cross", virtual_source_index=0, part="causal")
        made_ready = dict(zip(needed, ready, strict=True))
        self.straight, self.flipped, self.phases = {}, {}, {}
        if keep := [j for j in needed if j in straight]:
            self.straight = dict(
                zip(keep, correlate.transform([made_ready[j] for j in keep]), strict=True)
            )
        if keep := [j for j in needed if j in flipped]:
            turned = [
                flip(stream=made_ready[j], axis=FlipAxis.SPACE, flip_acquisition=False)
                for j in keep
            ]
            self.flipped = dict(zip(keep, correlate.transform(turned), strict=True))
        self.prepared = prepared


def _prepared(entry: Entry) -> str:
    """What a passive variant's correlations share beyond its base: its whitening and
    normalization (none on another line)."""
    preset = entry.trial.preset.model_dump(mode="json")
    return json.dumps([preset.get("whitening"), preset.get("normalization")], sort_keys=True)


def _stacked[K](
    preset: trials.Preset, kept: Sequence[Stream], keys: Sequence[K], phases: dict[K, np.ndarray]
) -> list[Stream]:
    """`kept` stacked as the preset's stacking says, by sigpipe's Stack; phase-weighted, by its
    phase-weighted stack given each one's instantaneous phase, made once by its key in
    `phases`."""
    stacking = stage_kwargs(preset, "stacking")
    if stacking["method"] != "phase_weighted":
        return Stack(**stacking).transform(kept)
    for key, one in zip(keys, kept, strict=True):
        if key not in phases:
            phases[key] = instantaneous_phase(one.xt)
    given = [phases[key] for key in keys]
    return [stack_phase_weighted_given(kept, given, nu=stacking["nu"])]


def _imaged(stacked: Sequence[Stream], preset: trials.Preset, folder: Path) -> None:
    """A passive or passive-active window's stacked correlations saved (PREPROCESSED, which G2
    measures), then their dispersion image."""
    Save(folder_path=folder).transform(stacked)
    assert (folder / PREPROCESSED).exists()
    image = Dispersion(method="phase", **stage_kwargs(preset, "dispersion")).transform(stacked)
    Save(folder_path=folder).transform(image)


def _load(window: MASWWindow, indices: Sequence[int], records: Path) -> list[Stream]:
    """Records `indices` of `window`, preprocessed, cut to the window's receivers, as
    load_preprocessed loads them."""
    receivers = (
        [window.record_receivers[i] for i in indices]
        if window.record_receivers
        else window.receiver_indices
    )
    loaded = Load(
        file_paths=[records / window.selected_files[i].stem / PREPROCESSED for i in indices],
        data_type="stream",
        acquisitions=[window.acquisitions[i] for i in indices],
        receivers_to_load=receivers,
    ).transform()
    return [one for one in loaded if isinstance(one, Stream)]


_PRODUCTS: dict[str, type[_Products]] = {
    "active": _Shots,
    "passive-active": _Correlated,
    "passive": _Segments,
}
