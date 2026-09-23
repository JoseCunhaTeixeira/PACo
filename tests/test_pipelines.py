import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from paco.pipelines import (
    PIPELINE_BUILDERS,
    build_active_pipeline,
    build_passive_pipeline,
    build_pipeline,
)
from paco.presets import ActivePreset, PassivePreset, make_preset, resolve_preset
from paco.profiles import Profile
from paco.windows import MASWWindow, build_windows
from sigpipe.base import DispersionImage, Pipeline, Transformer
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

# The steps of PAC's adapters/active.py and adapters/passive.py.
ACTIVE_CHAIN = [
    "Load", "Detrend", "Detrend", "Mute", "Filter", "Plot", "Pad", "Dispersion", "Stack", "Plot",
    "Save",
]  # fmt: skip
PASSIVE_CHAIN = [
    "Load", "Detrend", "Detrend", "Mute", "Filter", "Slice", "Selection", "Whiten", "Normalize",
    "Apodize", "Correlate", "Stack", "Plot", "Save", "Pad", "Dispersion", "Plot", "Save",
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


def _build(
    profile: Profile, name: str, overrides: dict[str, Any], output_folder: Path
) -> tuple[ActivePreset | PassivePreset, MASWWindow, Pipeline]:
    """Resolved preset, first window and pipeline of `profile`."""
    preset = resolve_preset(make_preset(name, overrides), profile)
    window = build_windows(profile, preset.masw)[0]
    return preset, window, build_pipeline(preset, window, output_folder)


def _only[T: Transformer[Any, Any]](pipeline: Pipeline, kind: type[T]) -> T:
    """The single step of `pipeline` of type `kind`."""
    (step,) = [step for step in pipeline.steps if isinstance(step, kind)]
    return step


# ---------------------------------------------------------------- structure, as in PAC


@pytest.mark.parametrize(
    ("profile", "name", "chain"),
    [("active_p1", "active", ACTIVE_CHAIN), ("passive_p1", "passive", PASSIVE_CHAIN)],
)
def test_steps_are_pacs(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str, chain: list[str]
) -> None:
    _, _, pipeline = _build(profiles[profile], name, {}, tmp_path)

    assert [step.name for step in pipeline.steps] == chain


def test_each_preset_mode_has_its_builder() -> None:
    expected = {"active": build_active_pipeline, "passive": build_passive_pipeline}

    assert expected == PIPELINE_BUILDERS


@BOTH_MODES
def test_load_reads_the_window_records_and_receivers(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str
) -> None:
    _, window, pipeline = _build(profiles[profile], name, {}, tmp_path)
    load = _only(pipeline, Load)

    assert load.file_paths == window.selected_files
    assert load.data_type == "seismic"
    assert load.params == {
        "acquisitions": window.acquisitions,
        "receivers_to_load": window.receiver_indices,
    }


@BOTH_MODES
def test_figures_and_results_go_to_the_output_folder(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str
) -> None:
    output_folder = tmp_path / "window"
    _, _, pipeline = _build(profiles[profile], name, {}, output_folder)
    plots = [step for step in pipeline.steps if isinstance(step, Plot)]
    saves = [step for step in pipeline.steps if isinstance(step, Save)]

    assert {step.folder_path for step in [*plots, *saves]} == {output_folder}
    # Records or correlations are plotted as they are, the dispersion image normalized.
    assert [plot.params for plot in plots] == [{}, {"normalize": True}]


@BOTH_MODES
def test_pacs_fixed_steps(
    profiles: dict[str, Profile], tmp_path: Path, profile: str, name: str
) -> None:
    _, _, pipeline = _build(profiles[profile], name, {}, tmp_path)
    detrends = [vars(step) for step in pipeline.steps if isinstance(step, Detrend)]

    assert detrends == [{"method": "constant", "params": {}}, {"method": "linear", "params": {}}]
    assert vars(_only(pipeline, Pad)) == {"n": 1_000, "taper": 25}
    assert _only(pipeline, Dispersion).method == "phase"


def test_pacs_fixed_steps_of_the_active_pipeline(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    _, _, pipeline = _build(profiles["active_p1"], "active", {}, tmp_path)

    # One dispersion image per shot, stacked linearly.
    assert vars(_only(pipeline, Stack)) == {"method": "linear", "params": {}}


def test_pacs_fixed_steps_of_the_passive_pipeline(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    _, _, pipeline = _build(profiles["passive_p1"], "passive", {}, tmp_path)

    assert vars(_only(pipeline, Apodize)) == {"method": "hanning", "params": {"frac": 0.1}}
    assert vars(_only(pipeline, Correlate)) == {
        "method": "cross",
        "virtual_source_index": 0,
        "params": {"part": "causal"},
    }
    assert _only(pipeline, Selection).params == {"flip_negatives": True}


# ---------------------------------------------------------------- preset values


def test_active_preset_values_reach_the_transformers(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    _, _, pipeline = _build(profiles["active_p1"], "active", ACTIVE_ON, tmp_path)

    assert vars(_only(pipeline, Mute)) == {
        "method": "mute",
        "tmin": 0.0,
        "tmax": 2.0,
        "vmin": 0.0,
        "vmax": 800.0,
        "taper": 0,
    }
    assert vars(_only(pipeline, Filter)) == {
        "method": "iir",
        "params": {"fmin": 0.0, "fmax": 950.0, "order": 4},
    }
    assert vars(_only(pipeline, Dispersion)) == {
        "method": "phase",
        "params": {"fmin": 5.0, "fmax": 60.0, "vmin": 50.0, "vmax": 500.0, "nv": 200},
    }


def test_passive_preset_values_reach_the_transformers(
    profiles: dict[str, Profile], tmp_path: Path
) -> None:
    _, _, pipeline = _build(profiles["passive_p1"], "passive", PASSIVE_ON, tmp_path)

    assert vars(_only(pipeline, Mute)) == {
        "method": "mute",
        "tmin": 0.0,
        "tmax": 130.0,
        "vmin": 0.0,
        "vmax": 100_000.0,
        "taper": 0,
    }
    assert vars(_only(pipeline, Filter)) == {
        "method": "iir",
        "params": {"fmin": 5.0, "fmax": 237.5, "order": 4},
    }
    assert vars(_only(pipeline, Slice)) == {
        "segment_duration": 0.2,
        "segment_step": 0.1,
        "params": {},
    }
    assert vars(_only(pipeline, Selection)) == {
        "method": "fk",
        "params": {"threshold": 0.0, "vmin": 0.0, "vmax": 100_000.0, "flip_negatives": True},
    }
    assert vars(_only(pipeline, Whiten)) == {
        "method": "onebit_apod",
        "params": {"fmin": 0.0, "fmax": 250.0, "taper_width_Hz": 5.0},
    }
    assert vars(_only(pipeline, Normalize)) == {"method": "onebit", "params": {}}
    assert vars(_only(pipeline, Stack)) == {"method": "phase_weighted", "params": {"nu": 3}}
    assert vars(_only(pipeline, Dispersion)) == {
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

    with pytest.raises(ValueError, match=rf"{re.escape(missing)} is not set: resolve the preset"):
        build_pipeline(preset, window, tmp_path)


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


@pytest.mark.parametrize(
    ("profile", "name", "overrides"),
    [
        pytest.param("active_p1", "active", {}, id="active-defaults"),
        pytest.param("active_p1", "active", ACTIVE_ON, id="active-stages-on"),
        pytest.param("passive_p1", "passive", {}, id="passive-defaults"),
        pytest.param("passive_p1", "passive", PASSIVE_ON, id="passive-stages-on"),
        pytest.param(
            "passive_p1",
            "passive",
            {"whitening": {"method": "onebit"}, "stacking": {"method": "root"}},
            id="passive-other-methods",
        ),
    ],
)
def test_pipeline_runs_on_a_demo_window(
    profiles: dict[str, Profile],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    name: str,
    overrides: dict[str, Any],
) -> None:
    # sigpipe's Pipeline.run creates a logs/ folder in the working directory.
    monkeypatch.chdir(tmp_path)
    output_folder = tmp_path / "window"
    preset, _, pipeline = _build(profiles[profile], name, overrides, output_folder)

    pipeline.run(show_log=False)

    assert {path.name for path in output_folder.iterdir()} == EXPECTED_FILES[name]
    (image,) = Load(
        file_paths=[output_folder / "DispersionImage_0000.hdf5"], data_type="dispersion_image"
    ).transform()
    assert isinstance(image, DispersionImage)
    dispersion = preset.model_dump()["dispersion"]
    assert image.vs.size == dispersion["nv"]
    assert image.vs[[0, -1]] == pytest.approx([dispersion["vmin"], dispersion["vmax"]])
    assert dispersion["fmin"] <= image.fs[0] and image.fs[-1] <= dispersion["fmax"]
    assert np.all(np.isfinite(image.fv_map))
