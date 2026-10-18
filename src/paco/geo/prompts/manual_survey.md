The survey (P1): what the line is before any setting is chosen. Call survey first: the curves'
tools open once it has run.

- The geometry: the receivers N_r and their spacing dx, the line's length, the shots (where, on
  which side, how far from the line), the sampling and its Nyquist, the records' length T, the
  trigger the headers state.
- The limits it sets: no aliasing above a wavelength of 2·dx, the alias velocity 2·dx·f; the
  images' frequency step 1/T; windows of 5 receivers to half the line, N_r - N + 1 of them at
  step 1 for a length N.
- The raw records as G1 measures them: the SNR with the distance from the shot, from which the
  reach comes; each record's usable band; dead, clipped and NaN traces; the trigger error and
  the lateral coherence; fast arrivals (refractions); the surface waves' cone from the gathers.
- A passive line (no source positions) records ambient noise: no shot, so no reach nor cone. Its
  records' number and length are its data; its images' frequency step is 1/the segments' length
  (the preset's 1 s: 1 Hz). G1 measures its traces (dead, clipped, NaN) and the receivers off
  their neighbours' spectra; G2 judges each window's correlations.
- The modes the line takes: an active line the active one, or passive-active (its farther shots
  correlated, as noise); a passive line the passive one alone.

Look at two or three gathers and their spectra when they puzzle you: both ends, and the middle.

Then write the survey in the journal, as a conclusion: what the site looks like (the surface
waves' velocities, refractions, the air wave, noise), the reach, the usable band, and the first
hypotheses to test. They are where the line's settings start from.
