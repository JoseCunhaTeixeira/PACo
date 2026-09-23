from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from paco.presets import (
    ActivePreset,
    PassivePreset,
    Preset,
    PresetError,
    make_preset,
    resolve_preset,
)
from paco.profiles import Profile, load_profile
from paco.settings import Settings
from paco.windows import MASWParameters
from sigpipe.base import LinearAcquisition, Stream
from sigpipe.transformers import Filter, Load

# PAC's form defaults (ActiveConfigForm.tsx and PassiveConfigForm.tsx), except distance_max:
# 1000 m in PACo, 100 m in PAC.
MASW_DEFAULTS = {"length": 3, "step": 1, "distance_min": 0.0, "distance_max": 1000.0}
DISPERSION_DEFAULTS = {"fmin": 0.0, "fmax": 100.0, "vmin": 1.0, "vmax": 1000.0, "nv": 1000}

ACTIVE_DEFAULTS = {
    "mode": "active",
    "masw": MASW_DEFAULTS,
    "muting": {"method": "none"},
    "filtering": {"method": "none"},
    "dispersion": DISPERSION_DEFAULTS,
}

PASSIVE_DEFAULTS = {
    "mode": "passive",
    "masw": MASW_DEFAULTS,
    "muting": {"method": "none"},
    "filtering": {"method": "none"},
    "slicing": {"segment_duration": 0.1, "segment_step": 0.1},
    "selection": {"method": "none"},
    "whitening": {"method": "none"},
    "normalization": {"method": "none"},
    "stacking": {"method": "linear"},
    "dispersion": DISPERSION_DEFAULTS,
}


@pytest.fixture(scope="module")
def profiles(demo_input_dir: Path) -> dict[str, Profile]:
    settings = Settings(input_dir=demo_input_dir)
    return {name: load_profile(name, settings) for name in ("active_p1", "passive_p1")}


# ---------------------------------------------------------------- defaults


def test_active_preset_holds_pacs_defaults() -> None:
    assert make_preset("active").model_dump(mode="json") == ACTIVE_DEFAULTS


def test_passive_preset_holds_pacs_defaults() -> None:
    assert make_preset("passive").model_dump(mode="json") == PASSIVE_DEFAULTS


def test_unknown_preset_lists_the_available_ones() -> None:
    with pytest.raises(
        PresetError, match=r"Unknown preset 'activ'\. Available presets: active, passive\."
    ):
        make_preset("activ")


def test_presets_are_frozen() -> None:
    preset = make_preset("active")

    with pytest.raises(ValidationError, match="frozen"):
        preset.masw = MASWParameters(length=24)


@pytest.mark.parametrize("name", ["active", "passive"])
def test_presets_round_trip_through_json(name: str) -> None:
    preset = make_preset(name, {"filtering": {"method": "iir", "fmin": 5}})

    assert TypeAdapter(Preset).validate_json(preset.model_dump_json()) == preset


# ---------------------------------------------------------------- overrides


def test_overriding_a_field_keeps_the_other_values() -> None:
    preset = make_preset("passive", {"dispersion": {"fmax": 60}, "masw": {"length": 24}})

    assert preset.dispersion.model_dump() == DISPERSION_DEFAULTS | {"fmax": 60.0}
    assert preset.masw.model_dump() == MASW_DEFAULTS | {"length": 24}
    assert preset.model_dump(exclude={"dispersion", "masw"}) == make_preset("passive").model_dump(
        exclude={"dispersion", "masw"}
    )


@pytest.mark.parametrize(
    ("name", "stage", "override", "expected"),
    [
        (
            "active",
            "muting",
            {"method": "mute"},
            {
                "method": "mute",
                "tmin": 0.0,
                "tmax": None,
                "vmin": 0.0,
                "vmax": 100000.0,
                "taper": 0,
            },
        ),
        (
            "active",
            "filtering",
            {"method": "iir"},
            {"method": "iir", "fmin": 0.0, "fmax": None, "order": 4},
        ),
        (
            "passive",
            "selection",
            {"method": "fk"},
            {"method": "fk", "threshold": 0.1, "vmin": 0.0, "vmax": 100000.0},
        ),
        (
            "passive",
            "whitening",
            {"method": "onebit_apod"},
            {"method": "onebit_apod", "fmin": 0.0, "fmax": None, "taper_width_Hz": 5.0},
        ),
        ("passive", "whitening", {"method": "onebit"}, {"method": "onebit"}),
        ("passive", "normalization", {"method": "onebit"}, {"method": "onebit"}),
        (
            "passive",
            "stacking",
            {"method": "phase_weighted"},
            {"method": "phase_weighted", "nu": 2},
        ),
        ("passive", "stacking", {"method": "root"}, {"method": "root", "n": 2}),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_switching_a_stage_on_uses_pacs_form_values(
    name: str, stage: str, override: dict[str, object], expected: dict[str, object]
) -> None:
    preset = make_preset(name, {stage: override})

    # None marks a value derived from the profile later, by resolve_preset.
    assert getattr(preset, stage).model_dump() == expected
    assert preset.model_dump(exclude={stage}) == make_preset(name).model_dump(exclude={stage})


@pytest.mark.parametrize(
    ("name", "overrides", "location"),
    [
        pytest.param(
            "active", {"whitening": {"method": "onebit"}}, "whitening", id="passive-only stage"
        ),
        pytest.param("active", {"mode": "passive"}, "mode", id="other mode"),
        pytest.param("active", {"masw": {"lenght": 24}}, r"masw\.lenght", id="masw typo"),
        pytest.param("active", {"masw": {"length": 2}}, r"masw\.length", id="window too short"),
        pytest.param("active", {"filtering": {"fmin": 5}}, "filtering", id="stage without method"),
        pytest.param(
            "active",
            {"filtering": {"method": "iir", "fmaxx": 3}},
            r"filtering\.iir\.fmaxx",
            id="field typo",
        ),
        pytest.param(
            "active",
            {"filtering": {"method": "none", "fmin": 5}},
            r"filtering\.none\.fmin",
            id="value on none",
        ),
        pytest.param(
            "active",
            {"filtering": {"method": "iir", "fmin": 50, "fmax": 20}},
            r"filtering\.iir",
            id="inverted band",
        ),
        pytest.param(
            "active",
            {"muting": {"method": "mute", "tmin": 1, "tmax": 0.5}},
            r"muting\.mute",
            id="inverted times",
        ),
        pytest.param(
            "active",
            {"dispersion": {"fmin": 50, "fmax": 20}},
            "dispersion",
            id="inverted dispersion",
        ),
        pytest.param(
            "passive", {"whitening": {"method": "savgol"}}, "whitening", id="method not in PAC"
        ),
        pytest.param(
            "passive",
            {"selection": {"method": "fk", "threshold": 1.5}},
            r"selection\.fk\.threshold",
            id="threshold",
        ),
        pytest.param(
            "passive",
            {"stacking": {"method": "root", "n": -1}},
            r"stacking\.root\.n",
            id="negative root",
        ),
        pytest.param(
            "passive",
            {"slicing": {"segment_duration": 0}},
            r"slicing\.segment_duration",
            id="empty segment",
        ),
    ],
)
def test_invalid_overrides_point_at_the_faulty_field(
    name: str, overrides: dict[str, object], location: str
) -> None:
    with pytest.raises(ValidationError, match=location):
        make_preset(name, overrides)


# ---------------------------------------------------------------- resolution against a profile


@pytest.mark.parametrize(("profile", "name"), [("active_p1", "active"), ("passive_p1", "passive")])
def test_resolving_the_defaults_changes_nothing(
    profiles: dict[str, Profile], profile: str, name: str
) -> None:
    # Every derived value belongs to a stage that is off by default.
    preset = make_preset(name)

    assert resolve_preset(preset, profiles[profile]) == preset


@pytest.mark.parametrize(
    ("profile", "name", "stage", "override", "field", "expected"),
    [
        # The longest record, rounded to 10 ms: 1.9995 s and 129.998 s.
        ("active_p1", "active", "muting", {"method": "mute"}, "tmax", 2.0),
        ("passive_p1", "passive", "muting", {"method": "mute"}, "tmax", 130.0),
        # 0.95 x Nyquist, just below sigpipe's limit.
        ("active_p1", "active", "filtering", {"method": "iir"}, "fmax", 950.0),
        ("passive_p1", "passive", "filtering", {"method": "iir"}, "fmax", 237.5),
        # Nyquist, as in PAC.
        ("passive_p1", "passive", "whitening", {"method": "onebit_apod"}, "fmax", 250.0),
    ],
)
def test_values_derived_from_the_profile(
    profiles: dict[str, Profile],
    profile: str,
    name: str,
    stage: str,
    override: dict[str, object],
    field: str,
    expected: float,
) -> None:
    resolved = resolve_preset(make_preset(name, {stage: override}), profiles[profile])

    assert getattr(getattr(resolved, stage), field) == pytest.approx(expected)


def test_explicit_values_are_kept(profiles: dict[str, Profile]) -> None:
    passive = make_preset("passive", {"filtering": {"method": "iir", "fmin": 5, "fmax": 100}})
    active = make_preset("active", {"muting": {"method": "mute", "tmax": 1.0}})

    assert resolve_preset(passive, profiles["passive_p1"]).filtering == passive.filtering
    assert resolve_preset(active, profiles["active_p1"]).muting == active.muting


def test_resolving_twice_changes_nothing(profiles: dict[str, Profile]) -> None:
    preset = make_preset(
        "passive",
        {
            "muting": {"method": "mute"},
            "filtering": {"method": "iir"},
            "whitening": {"method": "onebit_apod"},
        },
    )
    resolved = resolve_preset(preset, profiles["passive_p1"])

    assert resolve_preset(resolved, profiles["passive_p1"]) == resolved


def test_derived_filter_band_passes_sigpipe_where_pacs_default_fails(
    profiles: dict[str, Profile],
) -> None:
    profile = profiles["active_p1"]
    record = profile.records[0]
    assert record.source is not None
    acquisition = LinearAcquisition(source=record.source, receivers=profile.receivers)
    stream = Load(
        file_paths=[record.path], data_type="seismic", acquisitions=[acquisition]
    ).transform()[0]
    assert isinstance(stream, Stream)

    preset = resolve_preset(make_preset("active", {"filtering": {"method": "iir"}}), profile)
    Filter(**preset.filtering.model_dump()).transform([stream])

    # PAC's form default, fmax at Nyquist, is what sigpipe's filter rejects.
    with pytest.raises(ValueError, match=r"fmax < sampling_freq/2"):
        Filter(method="iir", fmin=0.0, fmax=profile.nyquist_hz, order=4).transform([stream])


# ---------------------------------------------------------------- errors against a profile


@pytest.mark.parametrize("fmax", [250.0, 300.0])
def test_filter_fmax_at_or_above_nyquist_is_rejected(
    profiles: dict[str, Profile], fmax: float
) -> None:
    preset = make_preset("passive", {"filtering": {"method": "iir", "fmax": fmax}})

    with pytest.raises(
        PresetError, match=r"must be below the Nyquist frequency of profile 'passive_p1' \(250 Hz\)"
    ):
        resolve_preset(preset, profiles["passive_p1"])


@pytest.mark.parametrize(
    ("profile", "name", "stage", "override", "message"),
    [
        (
            "passive_p1",
            "passive",
            "filtering",
            {"method": "iir", "fmin": 240},
            r"filtering\.fmin \(240 Hz\) must be below filtering\.fmax, which defaults to 237\.5 Hz",
        ),
        (
            "passive_p1",
            "passive",
            "whitening",
            {"method": "onebit_apod", "fmin": 260},
            r"whitening\.fmin \(260 Hz\) must be below whitening\.fmax, which defaults to 250 Hz",
        ),
        (
            "active_p1",
            "active",
            "muting",
            {"method": "mute", "tmin": 3},
            r"muting\.tmin \(3 s\) must be below muting\.tmax, which defaults to 2 s",
        ),
    ],
)
def test_a_derived_value_clashing_with_an_override_is_explained(
    profiles: dict[str, Profile],
    profile: str,
    name: str,
    stage: str,
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PresetError, match=message):
        resolve_preset(make_preset(name, {stage: override}), profiles[profile])


@pytest.mark.parametrize(
    ("profile", "name", "message"),
    [
        (
            "passive_p1",
            "active",
            r"Preset 'active' only fits active profiles, but 'passive_p1' is passive\. Use preset 'passive'\.",
        ),
        (
            "active_p1",
            "passive",
            r"Preset 'passive' only fits passive profiles, but 'active_p1' is active\. Use preset 'active'\.",
        ),
    ],
)
def test_preset_must_match_the_profile_kind(
    profiles: dict[str, Profile], profile: str, name: str, message: str
) -> None:
    with pytest.raises(PresetError, match=message):
        resolve_preset(make_preset(name), profiles[profile])


def test_resolution_keeps_the_preset_type(profiles: dict[str, Profile]) -> None:
    assert isinstance(resolve_preset(make_preset("active"), profiles["active_p1"]), ActivePreset)
    assert isinstance(resolve_preset(make_preset("passive"), profiles["passive_p1"]), PassivePreset)
