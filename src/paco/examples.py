"""An example call and result for each of PACo's tools (T11 of PACo's agent guidelines): what a
call looks like and what it returns, for people reading the tools and for the tests, which
check each call against the tool's arguments and each result against what it returns. They stay
out of the model's prompts: a model copies the values of examples."""

from dataclasses import dataclass
from typing import Any

RUN = "20261001-120000-abcd"
JOB = "inv-20261001-121500-abcd"


@dataclass(frozen=True)
class Example:
    """A call's arguments, and its result: an object as the tool returns it, or its text."""

    arguments: dict[str, Any]
    result: dict[str, Any] | str


EXAMPLES: dict[str, Example] = {
    "inspect": Example(
        {"what": "profiles"},
        "active_p1: a profile to process, no run yet\npassive_p1: a profile to process, 2 runs",
    ),
    "preset_settings": Example(
        {"profile": "active_p1"},
        '{"masw":{"length":{"type":"integer","description":"Receivers in a window"}}}',
    ),
    "run_processing": Example(
        {"profile": "active_p1", "overrides": {"masw": {"length": 24, "step": 24}}},
        {
            "run_id": RUN,
            "status": "ok",
            "did": f"Processed run {RUN}, 4 windows: G1 3 pass; G2 4 pass.",
            "summary": "G1: 3 pass. G2: 4 pass.",
            "next": f"pick comes next for run_id {RUN}, if the user asked for curves or models.",
        },
    ),
    "compare": Example(
        {
            "profile": "active_p1",
            "variants": [{"masw": {"length_m": 3}}, {"masw": {"length_m": 6}}],
            "metric": "depth",
        },
        {
            "profile": "active_p1",
            "metric": "depth",
            "did": "Compared 2 variants of active_p1 on the depth of investigation, half the "
            "longest wavelength, 9 trial windows each: variant 2 best, 17 m against 7.8.",
            "variants": [
                {
                    "label": "variant 1",
                    "settings": {"masw": {"length_m": 3}},
                    "length": 13,
                    "passed": 8,
                    "windows": 9,
                    "depth_m": 7.8,
                    "band_hz": [13.0, 36.1],
                    "wavelengths_m": [5.0, 15.5],
                    "value": 7.8,
                },
                {
                    "label": "variant 2",
                    "settings": {"masw": {"length_m": 6}},
                    "length": 25,
                    "passed": 9,
                    "windows": 9,
                    "depth_m": 17.0,
                    "band_hz": [9.6, 39.1],
                    "wavelengths_m": [5.0, 34.0],
                    "value": 17.0,
                },
            ],
            "best": "variant 2",
            "table": [
                "variant 1 {'masw': {'length_m': 3}}: windows of 13 receivers, 8 of 9 trial "
                "windows passed G3; depth of investigation 7.8 m",
                "variant 2 (best) {'masw': {'length_m': 6}}: windows of 25 receivers, 9 of 9 "
                "trial windows passed G3; depth of investigation 17 m",
            ],
            "next": "Say which is best and why. Processing the line with variant 2's settings is "
            "the user's to ask.",
        },
    ),
    "pick": Example(
        {"run_id": RUN},
        {
            "run_id": RUN,
            "status": "partial",
            "did": f"Picked run {RUN}, 4 windows: G3 3 pass, 1 reject; G4 3 pass.",
            "left": ["xmid 2.88 (1): G3 narrow_span"],
            "summary": "G3: 3 pass, 1 reject.",
            "next": f"invert comes next for run_id {RUN}, if the user asked for models.",
        },
    ),
    "judge": Example(
        {"run_id": RUN, "positions": [9.0]},
        {
            "run_id": RUN,
            "status": "ok",
            "did": f"Judged run {RUN}, 1 window: G3 1 pass; G4 4 pass.",
            "summary": "G3: 1 pass.",
            "next": "invert takes the curves G3 and G4 passed.",
        },
    ),
    "inversion_settings": Example(
        {},
        '{"properties":{"n_iterations":{"type":"integer"},"vs_layers":{"type":"array"}}}',
    ),
    "invert": Example(
        {"run_id": RUN, "parameters": {"n_iterations": 20000}},
        {
            "job_id": JOB,
            "run_id": RUN,
            "state": "succeeded",
            "done": 3,
            "total": 3,
            "n_failed": 0,
            "elapsed_s": 412.0,
            "depths_m": [1.0, 3.0, 6.0],
            "vs_m_s": [[150.0, 170.0], [190.0, 230.0], [260.0, 300.0]],
            "depth_informed_m": [7.5, 10.5],
            "misfit": [0.15, 0.23],
            "errors": [],
            "error": None,
        },
    ),
    "job_status": Example(
        {"job_id": JOB},
        {
            "job_id": JOB,
            "run_id": RUN,
            "state": "running",
            "done": 1,
            "total": 3,
            "n_failed": 0,
            "elapsed_s": 140.0,
            "depths_m": [1.0, 3.0, 6.0],
            "vs_m_s": [[160.0, 160.0], [210.0, 210.0], [280.0, 280.0]],
            "depth_informed_m": [9.0, 9.0],
            "misfit": [0.18, 0.18],
            "errors": [],
            "error": None,
        },
    ),
    "petro_models": Example(
        {"run_id": RUN},
        {
            "models": [
                {
                    "name": "silex",
                    "trained_on": "15 to 50 Hz",
                    "covers": "curves reaching 43 Hz",
                    "n_covered": 3,
                }
            ],
            "next": f'invert_petro(run_id="{RUN}", model="silex") for the 3 curves it covers.',
        },
    ),
    "invert_petro": Example(
        {"run_id": RUN, "model": "silex"},
        {
            "run_id": RUN,
            "status": "ok",
            "did": f"Inverted to soils run {RUN}, 3 windows: G7 3 pass; G8 3 pass.",
            "summary": "G7: 3 pass. G8: 3 pass.",
            "next": "The soils and the water table in PAC's Visualization page.",
        },
    ),
    "redo": Example(
        {"run_id": RUN, "stage": "phase_shift", "flag": "ridge_at_vmax"},
        {
            "run_id": RUN,
            "status": "ok",
            "did": f"Redid the phase shift of run {RUN}, 1 window: G2 1 pass.",
            "summary": "G2: 1 pass.",
            "next": f"pick comes next for run_id {RUN}.",
        },
    ),
}
