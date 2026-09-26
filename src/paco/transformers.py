"""PACo's own sigpipe transformers, for the steps PAC's pipelines did not have: the trigger
correction of a record (G1 measures the delay), the window's receivers taken from a
preprocessed record, and the geometry of passive-active correlation gathers. Nothing here
imports the presets: the presets import this."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from sigpipe.base import LinearAcquisition, Stream, Transformer


class ShiftTrigger(Transformer[Stream, Stream]):
    """Move each record's time origin to `t0`: a trigger `t0` seconds late (G1 measures it from
    the first breaks) is corrected by dropping the first `t0` seconds and padding zeros at the
    end, so that the length stays; an early trigger pads zeros at the start. With t0 = 0 the
    records pass through unchanged."""

    def __init__(self, t0: float = 0.0) -> None:
        self.t0 = t0

    def transform(self, data: Sequence[Stream]) -> list[Stream]:
        self.validate_sequence(data, Stream)
        if self.t0 == 0.0:
            return list(data)
        shifted: list[Stream] = []
        for stream in data:
            samples = round(self.t0 * stream.sampling_freq)
            xt = np.zeros_like(stream.xt)
            if samples >= 0:
                xt[:, : xt.shape[1] - samples] = stream.xt[:, samples:]
            else:
                xt[:, -samples:] = stream.xt[:, : xt.shape[1] + samples]
            shifted.append(replace(stream, xt=xt))
        return shifted


class SelectReceivers(Transformer[Stream, Stream]):
    """Keep some receivers of each stream, with the acquisition of that subset: the same ones
    for every stream, or each stream its own (`per_stream`)."""

    def __init__(
        self,
        indices: Sequence[int],
        acquisitions: Sequence[LinearAcquisition],
        per_stream: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.indices = list(indices)
        self.acquisitions = list(acquisitions)
        self.per_stream = [list(own) for own in per_stream] if per_stream is not None else None

    def transform(self, data: Sequence[Stream]) -> list[Stream]:
        self.validate_sequence(data, Stream)
        if len(data) != len(self.acquisitions):
            raise ValueError(f"{len(data)} streams for {len(self.acquisitions)} acquisitions")
        indices = self.per_stream or [self.indices] * len(data)
        return [
            replace(stream, xt=stream.xt[own, :], acquisition=acquisition)
            for stream, acquisition, own in zip(data, self.acquisitions, indices, strict=True)
        ]


class FromFirstReceiver(Transformer[Stream, Stream]):
    """Each correlation gather seen from a virtual source at its first receiver, where its
    zero-lag trace is: sigpipe's ActiveShotCorrelation flips the gather of a shot past the
    window's far end without its acquisition (the source stays at the last receiver), so the
    phase shift would read its offsets backwards, and a stack takes the first gather's
    acquisition, whichever side its shot was on."""

    def transform(self, data: Sequence[Stream]) -> list[Stream]:
        self.validate_sequence(data, Stream)
        aligned: list[Stream] = []
        for stream in data:
            receivers = stream.acquisition.receivers
            acquisition = LinearAcquisition(source=receivers[0], receivers=receivers)
            aligned.append(replace(stream, acquisition=acquisition))
        return aligned


class SurfaceWaveWindow(Transformer[Stream, Stream]):
    """Each trace kept between the arrivals at `vmax` and `vmin` from its source, with cosine
    ramps `pad` seconds wide on both sides (G1's surface-wave window): before a shot is
    correlated, the rest of its record adds only noise, and noise common to every trace sits at
    zero moveout (the demo's passive-active images peaked at the grid's top velocity)."""

    def __init__(self, vmin: float = 80.0, vmax: float = 1500.0, pad: float = 0.05) -> None:
        self.vmin, self.vmax, self.pad = vmin, vmax, pad

    def transform(self, data: Sequence[Stream]) -> list[Stream]:
        self.validate_sequence(data, Stream)
        windowed: list[Stream] = []
        for stream in data:
            ts = np.asarray(stream.ts, dtype=float)[None, :]
            offsets = np.asarray(stream.acquisition.offsets, dtype=float)[:, None]
            start, stop = offsets / self.vmax, offsets / self.vmin
            weight = ((ts >= start) & (ts <= stop)).astype(float)
            if self.pad > 0:
                before = (ts < start) & (ts >= start - self.pad)
                after = (ts > stop) & (ts <= stop + self.pad)
                ramp_in = 0.5 * (1 + np.cos(np.pi * (start - ts) / self.pad))
                ramp_out = 0.5 * (1 + np.cos(np.pi * (ts - stop) / self.pad))
                weight = np.where(before, ramp_in, np.where(after, ramp_out, weight))
            windowed.append(replace(stream, xt=(stream.xt * weight).astype(stream.xt.dtype)))
        return windowed
