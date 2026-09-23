import json

import pytest
from pydantic import TypeAdapter, ValidationError
from sigpipe.algorithms import WHITENING_METHODS
from sigpipe.base import LinearAcquisition, Stream
from sigpipe.transformers import Filter, Load, Slice

from paco.presets import (
    ActivePreset,
    PassivePreset,
    Preset,
    PresetError,
    make_preset,
    override_schema,
    resolve_preset,
    schema_size,
)
from paco.profiles import Profile
from paco.windows import MASWParameters

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

    assert preset.model_dump()["dispersion"] == DISPERSION_DEFAULTS | {"fmax": 60.0}
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


# One case per kind of mistake: the line the agent reads, and what it says to send instead.
INVALID_OVERRIDES = [
    pytest.param(
        "active",
        {"whitening": {"method": "onebit"}},
        "whitening: not a stage of preset 'active', only of passive. "
        "Stages: masw, muting, filtering, dispersion.",
        id="stage of the other preset",
    ),
    pytest.param(
        "active",
        {"filterng": {"method": "iir"}},
        "filterng: unknown stage. Allowed: masw, muting, filtering, dispersion. "
        "Did you mean filtering?",
        id="stage typo",
    ),
    pytest.param(
        "active",
        {"mode": "passive"},
        "mode: set by the preset name ('active'); leave it out.",
        id="mode",
    ),
    pytest.param(
        "active",
        {"masw": {"lenght": 24}},
        "masw.lenght: unknown parameter. Allowed: length, step, distance_min, distance_max. "
        "Did you mean length?",
        id="parameter typo",
    ),
    pytest.param(
        "active",
        {"filtering": {"method": "iir", "fmaxx": 3}},
        "filtering.fmaxx: unknown parameter. Allowed: fmin, fmax, order. Did you mean fmax?",
        id="parameter typo in a method",
    ),
    pytest.param(
        "active",
        {"filtering": {"method": "none", "fmin": 5}},
        "filtering.fmin: unknown parameter; method 'none' takes no parameters.",
        id="value on none",
    ),
    pytest.param(
        "passive",
        {"whitening": {"method": "onebit_apd"}},
        "whitening.method: unknown method 'onebit_apd'. Allowed: none, onebit, onebit_apod. "
        "Did you mean onebit_apod?",
        id="method typo",
    ),
    pytest.param(
        "passive",
        {"whitening": {"method": "savgol"}},
        "whitening.method: unknown method 'savgol'. Allowed: none, onebit, onebit_apod.",
        id="method not in PAC",
    ),
    pytest.param(
        "active",
        {"filtering": {"fmin": 5}},
        'filtering: missing "method". Methods: none, iir.',
        id="stage without method",
    ),
    pytest.param(
        "active", {"masw": {"length": 2}}, "masw.length: must be >= 3 (got 2).", id="below bound"
    ),
    pytest.param(
        "passive",
        {"selection": {"method": "fk", "threshold": 1.5}},
        "selection.threshold: must be <= 1 (got 1.5).",
        id="above bound",
    ),
    pytest.param(
        "passive",
        {"slicing": {"segment_duration": 0}},
        "slicing.segment_duration: must be > 0 (got 0).",
        id="empty segment",
    ),
    pytest.param(
        "passive",
        {"stacking": {"method": "root", "n": -1}},
        "stacking.n: must be >= 1 (got -1).",
        id="negative root",
    ),
    pytest.param(
        "passive",
        {"stacking": {"method": "root", "n": 2.5}},
        "stacking.n: must be an integer (got 2.5).",
        id="fractional integer",
    ),
    pytest.param(
        "passive",
        {"slicing": {"segment_duration": "long"}},
        "slicing.segment_duration: must be a number (got 'long').",
        id="text for a number",
    ),
    pytest.param(
        "passive",
        {"filtering": "iir"},
        "filtering: must be an object (got 'iir'). Methods: none, iir.",
        id="method name for a stage",
    ),
    pytest.param(
        "active",
        {"masw": 24},
        "masw: must be an object (got 24). Parameters: length, step, distance_min, distance_max.",
        id="number for an object",
    ),
    pytest.param(
        "active",
        {"filtering": {"method": "iir", "fmin": 50, "fmax": 20}},
        "filtering: fmax (20) must be greater than fmin (50).",
        id="inverted band",
    ),
    pytest.param(
        "active",
        {"muting": {"method": "mute", "tmin": 1, "tmax": 0.5}},
        "muting: tmax (0.5) must be greater than tmin (1).",
        id="inverted times",
    ),
    pytest.param(
        "active",
        {"dispersion": {"fmin": 50, "fmax": 20}},
        "dispersion: fmax (20) must be greater than fmin (50).",
        id="inverted dispersion",
    ),
    pytest.param(
        "active",
        {"masw": {"distance_min": 50, "distance_max": 10}},
        "masw: distance_max (10) must be greater than distance_min (50).",
        id="inverted distances",
    ),
]


@pytest.mark.parametrize(("name", "overrides", "line"), INVALID_OVERRIDES)
def test_invalid_overrides_are_explained_for_the_agent(
    name: str, overrides: dict[str, object], line: str
) -> None:
    with pytest.raises(PresetError) as caught:
        make_preset(name, overrides)

    assert str(caught.value) == f"Invalid overrides for preset '{name}':\n- {line}"
    # pydantic's own error stays attached, for debugging.
    assert isinstance(caught.value.__cause__, ValidationError)


def test_every_problem_gets_its_own_line() -> None:
    overrides = {"whitening": {"method": "onebit_apd"}, "masw": {"lenght": 24}}

    with pytest.raises(PresetError) as caught:
        make_preset("passive", overrides)

    assert str(caught.value).splitlines() == [
        "Invalid overrides for preset 'passive':",
        "- masw.lenght: unknown parameter. Allowed: length, step, distance_min, distance_max. "
        "Did you mean length?",
        "- whitening.method: unknown method 'onebit_apd'. Allowed: none, onebit, onebit_apod. "
        "Did you mean onebit_apod?",
    ]


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

    resolved_passive = resolve_preset(passive, profiles["passive_p1"]).model_dump()
    resolved_active = resolve_preset(active, profiles["active_p1"]).model_dump()

    assert resolved_passive["filtering"] == passive.model_dump()["filtering"]
    assert resolved_active["muting"] == active.model_dump()["muting"]


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
    Filter(**preset.model_dump()["filtering"]).transform([stream])

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


# ---------------------------------------------------------------- sigpipe's rules, before any work

PASSIVE_RULES = [
    pytest.param(
        {"slicing": {"segment_duration": 200, "segment_step": 200}},
        "slicing.segment_duration (200 s) must not exceed the shortest record of profile "
        "'passive_p1' (89.998 s).",
        id="segments longer than a record",
    ),
    pytest.param(
        {"slicing": {"segment_duration": 0.1, "segment_step": 0.2}},
        "slicing.segment_step (0.2 s) must not exceed slicing.segment_duration (0.1 s).",
        id="step longer than segments",
    ),
    pytest.param(
        {"whitening": {"method": "onebit_apod", "fmin": 10, "fmax": 40, "taper_width_Hz": 30}},
        "whitening.taper_width_Hz (30 Hz) must be smaller than the band fmax - fmin (30 Hz).",
        id="taper wider than the band",
    ),
    pytest.param(
        {"whitening": {"method": "onebit_apod", "fmin": 10, "fmax": 25, "taper_width_Hz": 1}},
        "whitening: the band fmax - fmin (15 Hz) must span at least 2 frequency steps of the "
        "0.1 s segments (19.6 Hz). Widen the band or lengthen slicing.segment_duration.",
        id="band narrower than two frequency steps",
    ),
    pytest.param(
        {"dispersion": {"fmin": 300, "fmax": 400}},
        "dispersion.fmin (300 Hz) must be below the Nyquist frequency of profile 'passive_p1' "
        "(250 Hz).",
        id="dispersion above Nyquist",
    ),
]


@pytest.mark.parametrize(("overrides", "line"), PASSIVE_RULES)
def test_sigpipes_rules_are_checked_before_any_work(
    profiles: dict[str, Profile], overrides: dict[str, object], line: str
) -> None:
    with pytest.raises(PresetError) as caught:
        resolve_preset(make_preset("passive", overrides), profiles["passive_p1"])

    assert str(caught.value) == f"Preset 'passive' does not fit profile 'passive_p1':\n- {line}"


def test_every_problem_with_the_profile_is_listed(profiles: dict[str, Profile]) -> None:
    overrides = {
        "filtering": {"method": "iir", "fmax": 260},
        "slicing": {"segment_duration": 95, "segment_step": 100},
        "dispersion": {"fmin": 251, "fmax": 300},
    }

    with pytest.raises(PresetError) as caught:
        resolve_preset(make_preset("passive", overrides), profiles["passive_p1"])

    lines = str(caught.value).splitlines()
    assert lines[0] == "Preset 'passive' does not fit profile 'passive_p1':"
    assert [line.split(" (")[0] for line in lines[1:]] == [
        "- filtering.fmax",
        "- slicing.segment_step",
        "- slicing.segment_duration",
        "- dispersion.fmin",
    ]


def test_dispersion_fmax_above_nyquist_is_left_to_sigpipe(profiles: dict[str, Profile]) -> None:
    # sigpipe lowers it to Nyquist, with a warning, as PAC does.
    preset = make_preset("passive", {"dispersion": {"fmax": 300}})

    resolved = resolve_preset(preset, profiles["passive_p1"])

    assert resolved.model_dump()["dispersion"]["fmax"] == 300.0


@pytest.mark.parametrize("duration", [0.1, 0.37, 1.0])
def test_whitening_band_rule_agrees_with_sigpipe(
    profiles: dict[str, Profile], duration: float
) -> None:
    # PACo recomputes sigpipe's frequency step for a segment: compare on real segments, around
    # the two-step limit, so a change in sigpipe's arithmetic shows up here.
    profile = profiles["passive_p1"]
    record = profile.records[1]
    acquisition = LinearAcquisition(source=profile.receivers[0], receivers=profile.receivers)
    stream = Load(
        file_paths=[record.path], data_type="seismic", acquisitions=[acquisition]
    ).transform()[0]
    assert isinstance(stream, Stream)
    segment = Slice(segment_duration=duration, segment_step=duration).transform([stream])[0]
    df = profile.sampling_rate_hz / (round(duration * profile.sampling_rate_hz) + 1)

    for band in [k * df / 4 for k in range(6, 11)]:
        whitening = {
            "method": "onebit_apod",
            "fmin": 0.0,
            "fmax": band,
            "taper_width_Hz": band / 10,
        }
        overrides = {"slicing": {"segment_duration": duration, "segment_step": duration}}
        try:
            WHITENING_METHODS["onebit_apod"](
                segment, **{k: v for k, v in whitening.items() if k != "method"}
            )
            sigpipe_accepts = True
        except ValueError:
            sigpipe_accepts = False
        try:
            resolve_preset(make_preset("passive", overrides | {"whitening": whitening}), profile)
            paco_accepts = True
        except PresetError:
            paco_accepts = False

        assert paco_accepts == sigpipe_accepts, f"band {band:.2f} Hz"


# ---------------------------------------------------------------- the schema the agent reads

# Characters of each override schema, written compactly: the lean sizes plus a small margin.
# The schema travels with every request to a model with an 8-16k context, so growing it has to
# be a deliberate choice: raise the budget here if it is worth it.
SCHEMA_BUDGET = {"active": 3_000, "passive": 6_600}


@pytest.mark.parametrize("name", ["active", "passive"])
def test_override_schema_stays_within_budget(name: str) -> None:
    assert schema_size(override_schema(name)) <= SCHEMA_BUDGET[name]


@pytest.mark.parametrize("name", ["active", "passive"])
def test_override_schema_keeps_what_the_agent_needs(name: str) -> None:
    schema = override_schema(name)
    full = make_preset(name).model_json_schema()

    # Only the noise is gone: pydantic's titles, and the mode the preset name sets.
    assert '"title"' not in json.dumps(schema)
    assert set(schema["properties"]) == set(full["properties"]) - {"mode"}
    assert set(schema["$defs"]) == set(full["$defs"])
    assert schema["properties"]["dispersion"]["default"] == DISPERSION_DEFAULTS
    # A value the profile fills says so, instead of showing a bare null.
    fmax = schema["$defs"]["FilteringIir"]["properties"]["fmax"]
    assert fmax["default"] is None
    assert fmax["description"] == "Hz; null: from the profile"


def test_override_schema_of_an_unknown_preset() -> None:
    with pytest.raises(
        PresetError, match=r"Unknown preset 'activ'\. Available presets: active, passive\."
    ):
        override_schema("activ")
