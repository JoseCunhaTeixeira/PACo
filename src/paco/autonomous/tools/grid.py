"""grid: a search's variants scored in code, fast, as trials score them (the user, 2026-10-20: a
full grid of the line's settings, for the model to choose from). Most of an image's cost is where
it begins, and many variants share that beginning, their base: the windows' length and the
records' preprocessing, on a passive line the segments' slicing, whitening and normalization too.
Each base is made once a window: an active line's dispersion image of each shot, a
passive-active line's correlation of each shot, a passive line's correlations of each segment
(straight and space-flipped) and their f-k ratios; then each variant takes its own from it (the
shots its distances keep, the segments its FK selection keeps, flipped where their energy runs
the other way), stacks them as sigpipe's pipeline does, and its image is judged as a trial's
(trials.judged): the scores a trial gives it. The images are not kept: the shortlist is tried
again as an experiment, whose images the model looks at.

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
from sigpipe.algorithms.selection.stream.fk import fk_ratios
from sigpipe.algorithms.stacking.stream.phase_weighted import (
    instantaneous_phase,
    stack_phase_weighted_given,
)
from sigpipe.base import Stream
from sigpipe.dataio.selection_plotting import selection_scores
from sigpipe.masw.pipelines import PREPROCESSED
from sigpipe.masw.pipelines.common import load_preprocessed, stage_kwargs
from sigpipe.masw.profiles import Profile
from sigpipe.masw.runs import WindowOutcome
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
from paco.settings import Settings

GRID = "grid"  # the scratch, in the study's folder
# A passive window's selection, as sigpipe's PlotSelection saves it for G2.
SELECTION = "Selection_0000.json"
# The stage a passive base holds besides the length and the preprocessing: its segments (their
# whitening and normalization made within it, the segments and their f-k ratios shared).
SEGMENTED = ("slicing",)


@dataclass(frozen=True)
class Entry:
    """A variant on one window of its base: its label, its trial, and its window (None when it
    has none to image, `error` saying why)."""

    label: str
    trial: trials.Trial
    window: MASWWindow | None
    error: str | None = None


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
) -> tuple[dict[str, trials.TrialRun], dict[str, str]]:
    """Each of `variants` (by label) on the trial set `windows`, judged as `tried` judges it:
    its run, and why each variant that could not be tried could not."""
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
                for window in chosen:
                    entry = _entry(label, trial, window, exclusions, failed)
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
        results = _made(tasks, scratch, settings.workers)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
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
    return Entry(label, trial, applied)


def _made(
    tasks: Sequence[Task], scratch: Path, workers: int
) -> dict[str, list[tuple[trials.WindowResult, float]]]:
    """Every task on `workers` processes: each variant's window results, with what each cost."""
    results: dict[str, list[tuple[trials.WindowResult, float]]] = {}
    if not tasks:
        return results
    one_thread_each()  # the workers are the cores the search takes
    scratch.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_start, initargs=(scratch,)
    ) as executor:
        futures = [executor.submit(made_window, task) for task in tasks]
        for future in finished(executor, futures, stopping.current()):
            for label, result, cost in future.result():
                results.setdefault(label, []).append((result, cost))
    return results


def _start(scratch: Path) -> None:
    """A worker: in the scratch, with no GUI backend."""
    import os

    os.chdir(scratch)
    matplotlib.use("Agg")


def made_window(task: Task) -> list[tuple[str, trials.WindowResult, float]]:
    """Runs in a worker: the task's base made once on its window, then each variant's image from
    it, judged as a trial's; each with what it cost, seconds (the base's share included)."""
    out: list[tuple[str, trials.WindowResult, float]] = []
    products: _Products | None = None
    for entry in sorted(task.entries, key=_prepared):
        started = time.perf_counter()
        if entry.window is None:
            out.append(
                (
                    entry.label,
                    trials.WindowResult(xmid=task.xmid, cached=False, error=entry.error),
                    0.0,
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
        shutil.rmtree(folder, ignore_errors=True)
        out.append((entry.label, result, round(time.perf_counter() - started, 3)))
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
