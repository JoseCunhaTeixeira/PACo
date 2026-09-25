import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from sigpipe.base import Coordinate, LinearAcquisition, Pipeline, Stream, Transformer
from sigpipe.transformers import (
    Apodize,
    Correlate,
    Detrend,
    Dispersion,
    Filter,
    Load,
    Mute,
    Normalize,
    Pad,
    Plot,
    Save,
    Selection,
    Slice,
    Stack,
    Whiten,
)

from paco.pipelines import (
    PIPELINE_BUILDERS,
    PREPROCESSED,
    SelectReceivers,
    build_active_pipeline,
    build_image_pipeline,
    build_passive_pipeline,
    build_preprocessing_pipeline,
    record_folder,
)
from paco.pipelines.common import stage_kwargs
from paco.presets import ActivePreset, PassivePreset, make_preset, resolve_preset
from paco.profiles import Profile, Record
from paco.transformers import ShiftTrigger
from paco.windows import MASWWindow, build_windows

# The steps of PAC's adapters/active.py and adapters/passive.py, cut at the window (the
# preprocessing works on whole records, the rest on the window's receivers), without PAC's
# Pad(n=1000, taper=25) before the phase shift: a frequency step finer than 1/T only interpolates
# (the user's decision, 2026-09-24).
PREPROCESSING_CHAIN = {
    # The trigger correction (t0 = 0 by default) is PACo's, for active records only.
    "active": ["Load", "ShiftTrigger", "Detrend", "Detrend", "Mute", "Filter", "Plot", "Save"],
    "passive": ["Load", "Detrend", "Detrend", "Mute", "Filter", "Plot", "Save"],
}
ACTIVE_CHAIN = ["Load", "SelectReceivers", "Plot", "Dispersion", "Stack", "Plot", "Save"]
PASSIVE_CHAIN = [
    "Load", "SelectReceivers", "Slice", "Selection", "Whiten", "Normalize", "Apodize",
    "Correlate", "Stack", "Plot", "Save", "Dispersion", "Plot", "Save",
]  # fmt: skip

# Every tunable stage switched on, with values moved away from the defaults.
ACTIVE_ON = {
    "muting": {"method": "mute", "vmax": 800},
    "filtering": {"method": "iir"},
    "dispersion": {"fmin": 5, "fmax": 60, "vmin": 50, "vmax": 500, "nv": 200},
}
PASSIVE_ON = {
    "muting": {"method": "mute"},
    "filtering": {"method": "iir", "fmin": 5},
    "slicing": {"segment_duration": 0.2, "segment_step": 0.1},
    "selection": {"method": "fk", "threshold": 0.0},
    "whitening": {"method": "onebit_apod"},
    "normalization": {"method": "onebit"},
    "stacking": {"method": "phase_weighted", "nu": 3},
    "dispersion": {"fmax": 60},
}

BOTH_MODES = pytest.mark.parametrize(
    ("profile", "name"), [("active_p1", "active"), ("passive_p1", "passive")]
)
# Windows of 24 receivers, as in test_runs.py: see test_default_windows_agree_within_rounding
# for the default of 5.
SMALL_WINDOWS = {"masw": {"length": 24, "step": 24}}
DEMO_CASES = pytest.mark.parametrize(
    ("profile", "name", "overrides"),
    [
        pytest.param("active_p1", "active", SMALL_WINDOWS, id="active-defaults"),
        pytest.param("active_p1", "active", {**SMALL_WINDOWS, **ACTIVE_ON}, id="active-stages-on"),
        pytest.param("passive_p1", "passive", SMALL_WINDOWS, id="passive-defaults"),
        pytest.param(
            "passive_p1", "passive", {**SMALL_WINDOWS, **PASSIVE_ON}, id="passive-stages-on"
        ),
        pytest.param(
            "passive_p1",
            "passive",
            {
                **SMALL_WINDOWS,
                "whitening": {"method": "onebit"},
                "stacking": {"method": "root"},
            },
            id="passive-other-methods",
        ),
    ],
)


@dataclass(frozen=True)
class Built:
    """A profile's first window and its pipelines, writing under one folder."""

    profile: Profile
    preset: ActivePreset | PassivePreset
    window: MASWWindow
    preprocessing: Pipeline  # of the window's first record
    image: Pipeline  # of the window
    records_folder: Path  # <folder>/records, the preprocessed records
    window_folder: Path  # <folder>/window, the image


def _record(profile: Profile, path: Path) -> Record:
    (record,) = [record for record in profile.records if record.path == path]
    return record


def _build(profile: Profile, name: str, overrides: dict[str, Any], folder: Path) -> Built:
    preset = resolve_preset(make_preset(name, overrides), profile)
    window = build_windows(profile, preset.masw)[0]
    records_folder = folder / "records"
    record = _record(profile, window.selected_files[0])
    preprocessing = build_preprocessing_pipeline(
        preset, record, profile, record_folder(records_folder, record)
    )
    image = build_image_pipeline(preset, window, records_folder, folder / "window")
    return Built(profile, preset, window, preprocessing, image, records_folder, folder / "window")


def _run(built: Built) -> None:
    """Preprocess the window's records, then compute its image."""
    for path in built.window.selected_files:
        record = _record(built.profile, path)
        folder = record_folder(built.records_folder, record)
        folder.mkdir(parents=True)
        build_preprocessing_pipeline(built.preset, record, built.profile, folder).run(
            show_log=False
        )
    built.window_folder.mkdir()
    built.image.run(show_log=False)


def _only[T: Transformer[Any, Any]](pipeline: Pipeline, kind: type[T]) -> T:
    """The single step of `pipeline` of type `kind`."""
    (step,) = [step for step in pipeline.steps if isinstance(step, kind)]
    return step


def _todays_pipeline(
    preset: ActivePreset | PassivePreset, window: MASWWindow, output_folder: Path
) -> Pipeline:
    """PAC's single pipeline per window, as PACo built it before the split (2026-09-24), from the
    raw records with the window's receivers, minus the padding: what the two-stage path must
    match bit for bit."""
    head = (
        Load(
            file_paths=window.selected_files,
            acquisitions=window.acquisitions,
            data_type="seismic",
            receivers_to_load=window.receiver_indices,
        )
        >> Detrend(method="constant")
        >> Detrend(method="linear")
        >> Mute(**stage_kwargs(preset, "muting"))
        >> Filter(**stage_kwargs(preset, "filtering"))
    )
    if isinstance(preset, ActivePreset):
        return (
            head
            >> Plot(folder_path=output_folder)
            >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
            >> Stack(method="linear")
            >> Plot(folder_path=output_folder, normalize=True)
            >> Save(folder_path=output_folder)
        )
    return (
        head
        >> Slice(**stage_kwargs(preset, "slicing"))
        >> Selection(**stage_kwargs(preset, "selection"), flip_negatives=True)
        >> Whiten(**stage_kwargs(preset, "whitening"))
        >> Normalize(**stage_kwargs(preset, "normalization"))
        >> Apodize(method="hanning", frac=0.1)
        >> Correlate(method="cross", virtual_source_index=0, part="causal")
        >> Stack(**stage_kwargs(preset, "stacking"))
        >> Plot(folder_path=output_folder)
        >> Save(folder_path=output_folder)
        >> Dispersion(method="phase", **stage_kwargs(preset, "dispersion"))
        >> Plot(folder_path=output_folder, normalize=True)
        >> Save(folder_path=output_folder)
    )


def _datasets(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as file:
        return {str(name): np.asarray(file[name]) for name in file}


# ---------------------------------------------------------------- structure, as in PAC


@pytest.mark.parametrize(
    ("profile", "name", "chain"),
    [("active_p1", "active", ACTIVE_CHAIN), ("passive_p1", "passive", PASSIVE_CHAIN)],
)
def test_steps_are_pacs(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str, chain: list[str]
) -> None:
    built = _build(profiles[profile], name, {}, tmp_path)

    assert [step.name for step in built.preprocessing.steps] == PREPROCESSING_CHAIN[name]
    assert [step.name for step in built.image.steps] == chain


def test_each_preset_mode_has_its_builder() -> None:
    expected = {"active": build_active_pipeline, "passive": build_passive_pipeline}

    assert expected == PIPELINE_BUILDERS


@BOTH_MODES
def test_preprocessing_loads_the_whole_record(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str
) -> None:
    built = _build(profiles[profile], name, {}, tmp_path)
    record = _record(built.profile, built.window.selected_files[0])
    load = _only(built.preprocessing, Load)

    assert load.file_paths == [record.path]
    assert load.data_type == "seismic"
    (acquisition,) = load.params["acquisitions"]  # pyright: ignore[reportGeneralTypeIssues]
    assert isinstance(acquisition, LinearAcquisition)
    assert acquisition.receivers == built.profile.receivers
    # A passive record has no source: the first receiver stands in, as in build_windows.
    assert acquisition.source == (record.source or built.profile.receivers[0])


@BOTH_MODES
def test_the_image_pipeline_reads_the_preprocessed_records(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str
) -> None:
    built = _build(profiles[profile], name, {}, tmp_path)
    load = _only(built.image, Load)
    select = _only(built.image, SelectReceivers)

    assert load.file_paths == [
        built.records_folder / path.stem / PREPROCESSED for path in built.window.selected_files
    ]
    assert load.data_type == "stream"
    assert load.params == {}
    # The window's receivers, as Load's receivers_to_load took them from the raw records.
    assert select.indices == built.window.receiver_indices
    assert select.acquisitions == built.window.acquisitions


@BOTH_MODES
def test_figures_and_results_go_to_their_folders(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str
) -> None:
    built = _build(profiles[profile], name, {}, tmp_path)

    def folders(pipeline: Pipeline) -> set[Path]:
        return {step.folder_path for step in pipeline.steps if isinstance(step, Plot | Save)}

    record = _record(built.profile, built.window.selected_files[0])
    assert folders(built.preprocessing) == {record_folder(built.records_folder, record)}
    assert folders(built.image) == {built.window_folder}
    # Records or correlations are plotted as they are, the dispersion image normalized.
    plots = [step for step in built.image.steps if isinstance(step, Plot)]
    assert [plot.params for plot in plots] == [{}, {"normalize": True}]


@BOTH_MODES
def test_pacs_fixed_steps(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str
) -> None:
    built = _build(profiles[profile], name, {}, tmp_path)
    detrends = [vars(step) for step in built.preprocessing.steps if isinstance(step, Detrend)]

    assert detrends == [{"method": "constant", "params": {}}, {"method": "linear", "params": {}}]
    # No padding before the phase shift: a frequency step finer than 1/T only interpolates.
    assert not [step for step in built.image.steps if isinstance(step, Pad)]
    assert _only(built.image, Dispersion).method == "phase"


def test_pacs_fixed_steps_of_the_active_pipeline(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    built = _build(profiles["active_p1"], "active", {}, tmp_path)

    # One dispersion image per shot, stacked linearly.
    assert vars(_only(built.image, Stack)) == {"method": "linear", "params": {}}


def test_pacs_fixed_steps_of_the_passive_pipeline(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    built = _build(profiles["passive_p1"], "passive", {}, tmp_path)

    assert vars(_only(built.image, Apodize)) == {"method": "hanning", "params": {"frac": 0.1}}
    assert vars(_only(built.image, Correlate)) == {
        "method": "cross",
        "virtual_source_index": 0,
        "params": {"part": "causal"},
    }
    assert _only(built.image, Selection).params == {"flip_negatives": True}


def test_select_receivers_keeps_the_rows_and_sets_the_acquisition() -> None:
    receivers = tuple(Coordinate(float(x), 0.0, 0.0) for x in range(4))
    whole = LinearAcquisition(source=Coordinate(-1.0, 0.0, 0.0), receivers=receivers)
    stream = Stream(
        xt=np.arange(12, dtype=np.float32).reshape(4, 3),
        ts=np.arange(3, dtype=np.float32),
        sampling_freq=1.0,
        acquisition=whole,
    )
    part = LinearAcquisition(source=whole.source, receivers=receivers[1:3])

    (selected,) = SelectReceivers([1, 2], [part]).transform([stream])

    assert np.array_equal(selected.xt, stream.xt[1:3])
    assert selected.xt.dtype == np.float32
    assert (selected.ts, selected.sampling_freq) == (stream.ts, 1.0)
    assert selected.acquisition == part
    with pytest.raises(ValueError, match="2 streams for 1 acquisition"):
        SelectReceivers([1, 2], [part]).transform([stream, stream])


# ---------------------------------------------------------------- preset values


def test_active_preset_values_reach_the_transformers(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    built = _build(profiles["active_p1"], "active", ACTIVE_ON, tmp_path)

    assert vars(_only(built.preprocessing, Mute)) == {
        "method": "mute",
        "tmin": 0.0,
        "tmax": 2.0,
        "vmin": 0.0,
        "vmax": 800.0,
        "taper": 0,
    }
    assert vars(_only(built.preprocessing, Filter)) == {
        "method": "iir",
        "params": {"fmin": 0.0, "fmax": 950.0, "order": 4},
    }
    assert vars(_only(built.image, Dispersion)) == {
        "method": "phase",
        "params": {"fmin": 5.0, "fmax": 60.0, "vmin": 50.0, "vmax": 500.0, "nv": 200},
    }


def test_passive_preset_values_reach_the_transformers(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    built = _build(profiles["passive_p1"], "passive", PASSIVE_ON, tmp_path)

    assert vars(_only(built.preprocessing, Mute)) == {
        "method": "mute",
        "tmin": 0.0,
        "tmax": 130.0,
        "vmin": 0.0,
        "vmax": 100_000.0,
        "taper": 0,
    }
    assert vars(_only(built.preprocessing, Filter)) == {
        "method": "iir",
        "params": {"fmin": 5.0, "fmax": 237.5, "order": 4},
    }
    assert vars(_only(built.image, Slice)) == {
        "segment_duration": 0.2,
        "segment_step": 0.1,
        "params": {},
    }
    assert vars(_only(built.image, Selection)) == {
        "method": "fk",
        "params": {"threshold": 0.0, "vmin": 0.0, "vmax": 100_000.0, "flip_negatives": True},
    }
    assert vars(_only(built.image, Whiten)) == {
        "method": "onebit_apod",
        "params": {"fmin": 0.0, "fmax": 250.0, "taper_width_Hz": 5.0},
    }
    assert vars(_only(built.image, Normalize)) == {"method": "onebit", "params": {}}
    assert vars(_only(built.image, Stack)) == {"method": "phase_weighted", "params": {"nu": 3}}
    assert vars(_only(built.image, Dispersion)) == {
        "method": "phase",
        "params": {"fmin": 0.0, "fmax": 60.0, "vmin": 1.0, "vmax": 1_000.0, "nv": 1_000},
    }


@pytest.mark.parametrize(
    ("profile", "name", "overrides", "missing"),
    [
        ("active_p1", "active", {"muting": {"method": "mute"}}, "muting.tmax"),
        ("passive_p1", "passive", {"filtering": {"method": "iir"}}, "filtering.fmax"),
        ("passive_p1", "passive", {"whitening": {"method": "onebit_apod"}}, "whitening.fmax"),
    ],
)
def test_an_unresolved_preset_is_refused(
    profiles: dict[str, Profile],
    tmp_path: Path,
    profile: str,
    name: str,
    overrides: dict[str, Any],
    missing: str,
) -> None:
    preset = make_preset(name, overrides)
    window = build_windows(profiles[profile], preset.masw)[0]
    record = _record(profiles[profile], window.selected_files[0])

    with pytest.raises(ValueError, match=rf"{re.escape(missing)} is not set: resolve the preset"):
        build_preprocessing_pipeline(preset, record, profiles[profile], tmp_path)
        build_image_pipeline(preset, window, tmp_path, tmp_path)


# ---------------------------------------------------------------- real runs on a demo window


EXPECTED_FILES = {
    # Each shot plotted, then the stacked dispersion image plotted and saved.
    "active": {
        "Stream_0000.png",
        "Stream_0001.png",
        "DispersionImage_0000.png",
        "DispersionImage_0000.hdf5",
    },
    # The stacked correlation plotted and saved, then its dispersion image.
    "passive": {
        "Stream_0000.png",
        "Stream_0000.hdf5",
        "DispersionImage_0000.png",
        "DispersionImage_0000.hdf5",
    },
}


@DEMO_CASES
def test_the_split_runs_and_gives_todays_results(
    profiles: dict[str, Profile],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    name: str,
    overrides: dict[str, Any],
) -> None:
    # sigpipe's Pipeline.run creates a logs/ folder in the working directory.
    monkeypatch.chdir(tmp_path)
    built = _build(profiles[profile], name, overrides, tmp_path / "split")
    _run(built)
    reference = tmp_path / "reference"
    reference.mkdir()
    _todays_pipeline(built.preset, built.window, reference).run(show_log=False)

    assert {path.name for path in built.window_folder.iterdir()} == EXPECTED_FILES[name]
    for path in built.window.selected_files:
        folder = record_folder(built.records_folder, _record(built.profile, path))
        assert {path.name for path in folder.iterdir()} == {"Stream_0000.hdf5", "Stream_0000.png"}
    # The pipeline works in float32 from the loader on, and the stream files keep it: bit for bit
    # on 12 cores. On 4 (CI's runners) the passive correlations round some values differently
    # in the last float32 bit, at most 1.3e-7 of the largest, as LAPACK's batches do below.
    for file in EXPECTED_FILES[name]:
        if file.endswith(".hdf5"):
            split, todays = _datasets(built.window_folder / file), _datasets(reference / file)
            assert split.keys() == todays.keys()
            for key in todays:
                assert split[key].dtype == todays[key].dtype, key
                if todays[key].dtype.kind == "f":
                    largest = float(np.max(np.abs(todays[key]), initial=0.0))
                    np.testing.assert_allclose(
                        split[key], todays[key], rtol=0, atol=1e-6 * largest, err_msg=key
                    )
                else:
                    assert np.array_equal(split[key], todays[key]), key


def test_default_windows_agree_within_rounding(
    profiles: dict[str, Profile], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """scipy's linear detrend solves all traces at once, and LAPACK rounds a 5-trace batch
    differently from the 96-trace record (the last float32 bit; multiples of 4 traces come out
    the same). With the default windows of 5 receivers, the images agree to 1e-6, except at
    0 Hz: a sum of the signs of near-zero DC terms after detrending, which flips with that bit."""
    monkeypatch.chdir(tmp_path)
    built = _build(profiles["active_p1"], "active", {}, tmp_path / "split")
    _run(built)
    reference = tmp_path / "reference"
    reference.mkdir()
    _todays_pipeline(built.preset, built.window, reference).run(show_log=False)

    split = _datasets(built.window_folder / "DispersionImage_0000.hdf5")
    todays = _datasets(reference / "DispersionImage_0000.hdf5")
    assert len(built.window.receiver_indices) == 5
    assert np.array_equal(split["fs"], todays["fs"])
    assert np.array_equal(split["vs"], todays["vs"])
    assert np.allclose(split["fv_map"][1:], todays["fv_map"][1:], rtol=1e-5, atol=1e-6)


def test_shift_trigger_moves_the_time_origin_and_keeps_the_length() -> None:
    receivers = tuple(Coordinate(float(x), 0.0, 0.0) for x in range(2))
    acquisition = LinearAcquisition(source=Coordinate(-1.0, 0.0, 0.0), receivers=receivers)
    xt = np.arange(20, dtype=np.float32).reshape(2, 10)
    stream = Stream(xt=xt, ts=np.arange(10) / 100.0, sampling_freq=100.0, acquisition=acquisition)

    (late,) = ShiftTrigger(0.03).transform([stream])  # triggered 30 ms late: 3 samples
    (early,) = ShiftTrigger(-0.02).transform([stream])
    (same,) = ShiftTrigger(0.0).transform([stream])

    assert late.xt.shape == xt.shape and late.xt.dtype == np.float32
    assert np.array_equal(late.xt[0], [3, 4, 5, 6, 7, 8, 9, 0, 0, 0])
    assert np.array_equal(early.xt[0], [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    assert same is stream
    assert late.acquisition == stream.acquisition and np.array_equal(late.ts, stream.ts)
