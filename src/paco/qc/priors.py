"""The checks before S4 (docs/qc_workflow.md): the inversion's parameters coherent with the curve
it inverts. Bounds come from the window's own curve (a decision of milestone 13): Vs brackets
the curve's velocities with a margin (Vs is about 1.09 Vr at PAC's Vp/Vs), no layer is thinner
than the shortest wavelength resolves, and the half-space starts no deeper than the longest one
reaches. Values given by the user or the loop are kept when they pass, and changed with a note
when they do not."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sigpipe.base import DispersionCurve

from paco.inversion import InversionError, InversionParameters

# PAC's steps against its ranges: 20 m/s for Vs over 100-1,000 m/s, 1 m for thicknesses over
# 1-10 m. Derived bounds keep the same proportions.
VS_STEP_SHARE = 20 / 900
THICKNESS_STEP_SHARE = 1 / 9
# Vs over Vr for a homogeneous half-space at Vp/Vs 1.77: the least a bound must allow above the
# curve's fastest point.
VS_OVER_VR = 1.09


class PriorRules(BaseModel):
    """The rules the bounds follow: provisional, measured on the demo profiles (rule 9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vs_low: float = Field(
        default=0.8, gt=0, description="Lowest Vs, as a share of the curve's slowest velocity."
    )
    vs_high: float = Field(
        default=1.5, gt=0, description="Highest Vs, as a multiple of the curve's fastest velocity."
    )
    min_thickness: float = Field(
        default=1 / 3,
        gt=0,
        description="Thinnest layer, as a share of the shortest wavelength: thinner is not "
        "resolved.",
    )
    max_depth: float = Field(
        default=0.5,
        gt=0,
        description="Deepest top of the half-space, as a share of the longest wavelength: deeper "
        "is not resolved.",
    )
    n_layers: int = Field(default=2, ge=2, description="Layers to start with, PAC's.")


@dataclass(frozen=True)
class Derived:
    """The parameters an inversion runs with, and what the checks changed in the values given."""

    parameters: InversionParameters
    notes: tuple[str, ...]
    reach_m: float  # the deepest the half-space's top may be: the curve's reach


def derive_inversion(
    curve: DispersionCurve, rules: PriorRules, given: Mapping[str, Any] | None = None
) -> Derived:
    """The parameters to invert `curve` with: `given` (InversionParameters' fields, from the
    user or the loop) where they pass the checks, the rest derived from the curve."""
    given = broadcast_layers(given or {}, rules.n_layers)
    velocities = np.asarray(curve.vs, dtype=float)
    wavelengths = velocities / np.asarray(curve.fs, dtype=float)
    vr_min, vr_max = float(velocities.min()), float(velocities.max())
    thinnest = rules.min_thickness * float(wavelengths.min())
    deepest = rules.max_depth * float(wavelengths.max())
    notes: list[str] = []

    n_layers = int(given.get("n_layers", rules.n_layers))
    # Layers above the half-space, each at least `thinnest`, all within `deepest`.
    resolved = max(2, math.ceil(deepest / thinnest))
    if n_layers > resolved:
        notes.append(
            f"n_layers {n_layers}: the curve resolves {resolved} (layers of at least "
            f"{thinnest:.2f} m down to {deepest:.2f} m); set to {resolved}."
        )
        n_layers = resolved
        given.pop("vs_layers", None)
        given.pop("thickness_layers", None)

    low, high = round(rules.vs_low * vr_min), round(rules.vs_high * vr_max)
    derived_vs = {
        "vs_min": float(low),
        "vs_max": float(high),
        "vs_perturb_std": round((high - low) * VS_STEP_SHARE, 1),
    }
    vs_layers = [dict(derived_vs) for _ in range(n_layers)]
    if "vs_layers" in given:
        vs_layers = [{**derived_vs, **dict(layer)} for layer in given["vs_layers"]]
        slow = [i for i, layer in enumerate(vs_layers) if layer["vs_min"] > vr_min]
        fast = [i for i, layer in enumerate(vs_layers) if layer["vs_max"] < VS_OVER_VR * vr_max]
        for index in slow:
            vs_layers[index]["vs_min"] = derived_vs["vs_min"]
        for index in fast:
            vs_layers[index]["vs_max"] = derived_vs["vs_max"]
        if slow:
            notes.append(
                f"vs_min above the curve's slowest velocity ({vr_min:.0f} m/s) in "
                f"{_layers(slow)}: set to {low} m/s."
            )
        if fast:
            notes.append(
                f"vs_max below {VS_OVER_VR} times the curve's fastest velocity ({vr_max:.0f} m/s) "
                f"in {_layers(fast)}: set to {high} m/s."
            )

    top, bottom = round(thinnest, 2), round(deepest / (n_layers - 1), 2)
    derived_thickness = {
        "thickness_min": top,
        "thickness_max": bottom,
        "thickness_perturb_std": round((bottom - top) * THICKNESS_STEP_SHARE, 2),
    }
    thickness_layers = [dict(derived_thickness) for _ in range(n_layers - 1)]
    if "thickness_layers" in given:
        thickness_layers = [
            {**derived_thickness, **dict(layer)} for layer in given["thickness_layers"]
        ]
        thin = [i for i, layer in enumerate(thickness_layers) if layer["thickness_min"] < top]
        for index in thin:
            thickness_layers[index]["thickness_min"] = top
        if thin:
            notes.append(
                f"thickness_min thinner than the curve resolves ({top:g} m) in {_layers(thin)}: "
                f"set to {top:g} m."
            )
        total = sum(float(layer["thickness_max"]) for layer in thickness_layers)
        if round(total, 2) > round(deepest, 2):
            scale = deepest / total
            for layer in thickness_layers:
                layer["thickness_max"] = round(float(layer["thickness_max"]) * scale, 2)
            notes.append(
                f"thickness_max puts the half-space as deep as {total:g} m, below the "
                f"{deepest:.2f} m the curve reaches: scaled by {scale:.2f}."
            )

    values = {
        **{
            key: value
            for key, value in given.items()
            if key not in ("vs_layers", "thickness_layers")
        },
        "n_layers": n_layers,
        "vs_layers": vs_layers,
        "thickness_layers": thickness_layers,
    }
    try:
        return Derived(InversionParameters.model_validate(values), tuple(notes), round(deepest, 2))
    except ValidationError as error:
        problems = "; ".join(str(problem["msg"]) for problem in error.errors())
        raise InversionError(f"The inversion's parameters do not hold: {problems}") from error


def broadcast_layers(given: Mapping[str, Any], default_layers: int = 2) -> dict[str, Any]:
    """`given` with a single Vs range (or thickness range) standing for every layer: "Vs between
    100 and 180 m/s" is one range, where the parameters want one per layer (Qwen3-8B sent one,
    and needed six calls to find the form)."""
    values = dict(given)
    n_layers = int(values.get("n_layers", default_layers))
    vs_layers = values.get("vs_layers")
    if isinstance(vs_layers, list | tuple) and len(cast(Sequence[Any], vs_layers)) == 1:
        values["vs_layers"] = list(cast(Sequence[Any], vs_layers)) * n_layers
    thicknesses = values.get("thickness_layers")
    if (
        isinstance(thicknesses, list | tuple)
        and len(cast(Sequence[Any], thicknesses)) == 1
        and n_layers > 2
    ):
        values["thickness_layers"] = list(cast(Sequence[Any], thicknesses)) * (n_layers - 1)
    return values


def _layers(indices: list[int]) -> str:
    """Layers named from the top, from 1, as the notes give them."""
    names = [str(index + 1) for index in indices]
    return f"layer {names[0]}" if len(names) == 1 else f"layers {', '.join(names)}"
