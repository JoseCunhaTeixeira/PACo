"""PACo's own sigpipe transformers, for the steps PAC's pipelines did not have: the trigger
correction of a record (G1 measures the delay), and the window's receivers taken from a
preprocessed record. Nothing here imports the presets: the presets import this."""

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
    """Keep some receivers of each stream, with the acquisition of that subset."""

    def __init__(self, indices: Sequence[int], acquisitions: Sequence[LinearAcquisition]) -> None:
        self.indices = list(indices)
        self.acquisitions = list(acquisitions)

    def transform(self, data: Sequence[Stream]) -> list[Stream]:
        self.validate_sequence(data, Stream)
        if len(data) != len(self.acquisitions):
            raise ValueError(f"{len(data)} streams for {len(self.acquisitions)} acquisitions")
        return [
            replace(stream, xt=stream.xt[self.indices, :], acquisition=acquisition)
            for stream, acquisition in zip(data, self.acquisitions, strict=True)
        ]
