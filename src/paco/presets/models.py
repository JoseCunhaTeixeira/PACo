"""Preset models: PAC's processing configuration, one field per tunable stage.

Default values are PAC's form defaults, so a preset is simply a model built without arguments,
and overrides are a partial dict validated by the same model. Fields left to None are derived
from the profile by resolve_preset, as PAC's forms derive them from the acquisition.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paco.profiles import ProfileKind
from paco.windows import MASWParameters


class PresetError(ValueError):
    """A preset is unknown or does not fit a profile. Messages are written to be read by the agent."""


class _StrictModel(BaseModel):
    # Unknown keys are errors, not silently dropped: a typo in an override must reach the agent.
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------- muting


class NoneMuting(_StrictModel):
    method: Literal["none"] = "none"


class Muting(_StrictModel):
    method: Literal["mute"] = "mute"
    tmin: float = Field(default=0.0, ge=0)
    tmax: float | None = Field(default=None, ge=0)  # None: the record length
    vmin: float = Field(default=0.0, ge=0)
    vmax: float = Field(default=100_000.0, ge=0)
    taper: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if self.tmax is not None and self.tmax <= self.tmin:
            raise ValueError("tmax must be greater than tmin")
        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")
        return self


type MutingParameters = Annotated[NoneMuting | Muting, Field(discriminator="method")]


# ---------------------------------------------------------------- filtering


class NoneFiltering(_StrictModel):
    method: Literal["none"] = "none"


class IIRFiltering(_StrictModel):
    method: Literal["iir"] = "iir"
    fmin: float = Field(default=0.0, ge=0)
    fmax: float | None = Field(default=None, gt=0)  # None: just below the Nyquist frequency
    order: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _check_band(self) -> Self:
        if self.fmax is not None and self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")
        return self


type FilteringParameters = Annotated[NoneFiltering | IIRFiltering, Field(discriminator="method")]


# ---------------------------------------------------------------- slicing, passive only


class SlicingParameters(_StrictModel):
    segment_duration: float = Field(default=0.1, gt=0)
    segment_step: float = Field(default=0.1, gt=0)


# ---------------------------------------------------------------- selection, passive only


class NoneSelection(_StrictModel):
    method: Literal["none"] = "none"


class FKSelection(_StrictModel):
    method: Literal["fk"] = "fk"
    threshold: float = Field(default=0.1, ge=0, le=1)
    vmin: float = Field(default=0.0, ge=0)
    vmax: float = Field(default=100_000.0, gt=0)

    @model_validator(mode="after")
    def _check_velocities(self) -> Self:
        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")
        return self


type SelectionParameters = Annotated[NoneSelection | FKSelection, Field(discriminator="method")]


# ---------------------------------------------------------------- whitening, passive only


class NoneWhitening(_StrictModel):
    method: Literal["none"] = "none"


class OnebitWhitening(_StrictModel):
    method: Literal["onebit"] = "onebit"


class OnebitApodWhitening(_StrictModel):
    method: Literal["onebit_apod"] = "onebit_apod"
    fmin: float = Field(default=0.0, ge=0)
    fmax: float | None = Field(default=None, gt=0)  # None: the Nyquist frequency
    taper_width_Hz: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _check_band(self) -> Self:
        if self.fmax is not None and self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")
        return self


type WhiteningParameters = Annotated[
    NoneWhitening | OnebitWhitening | OnebitApodWhitening, Field(discriminator="method")
]


# ---------------------------------------------------------------- normalization, passive only


class NoneNormalization(_StrictModel):
    method: Literal["none"] = "none"


class OneBitNormalization(_StrictModel):
    method: Literal["onebit"] = "onebit"


type NormalizationParameters = Annotated[
    NoneNormalization | OneBitNormalization, Field(discriminator="method")
]


# ---------------------------------------------------------------- stacking, passive only


class LinearStacking(_StrictModel):
    method: Literal["linear"] = "linear"


class PhaseWeightedStacking(_StrictModel):
    method: Literal["phase_weighted"] = "phase_weighted"
    nu: int = Field(default=2, ge=0)


class RootStacking(_StrictModel):
    method: Literal["root"] = "root"
    n: int = Field(default=2, ge=0)


type StackingParameters = Annotated[
    LinearStacking | PhaseWeightedStacking | RootStacking, Field(discriminator="method")
]


# ---------------------------------------------------------------- dispersion


class DispersionParameters(_StrictModel):
    fmin: float = Field(default=0.0, ge=0)
    fmax: float = Field(default=100.0, gt=0)
    vmin: float = Field(default=1.0, gt=0)
    vmax: float = Field(default=1_000.0, gt=0)
    nv: int = Field(default=1_000, gt=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if self.fmax <= self.fmin:
            raise ValueError("fmax must be greater than fmin")
        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")
        return self


# ---------------------------------------------------------------- presets


class ActivePreset(_StrictModel):
    """PAC's active processing. Stages in pipeline order."""

    mode: Literal[ProfileKind.ACTIVE] = ProfileKind.ACTIVE
    masw: MASWParameters = MASWParameters()
    muting: MutingParameters = NoneMuting()
    filtering: FilteringParameters = NoneFiltering()
    dispersion: DispersionParameters = DispersionParameters()


class PassivePreset(_StrictModel):
    """PAC's passive processing. Stages in pipeline order."""

    mode: Literal[ProfileKind.PASSIVE] = ProfileKind.PASSIVE
    masw: MASWParameters = MASWParameters()
    muting: MutingParameters = NoneMuting()
    filtering: FilteringParameters = NoneFiltering()
    slicing: SlicingParameters = SlicingParameters()
    selection: SelectionParameters = NoneSelection()
    whitening: WhiteningParameters = NoneWhitening()
    normalization: NormalizationParameters = NoneNormalization()
    stacking: StackingParameters = LinearStacking()
    dispersion: DispersionParameters = DispersionParameters()


type Preset = Annotated[ActivePreset | PassivePreset, Field(discriminator="mode")]
