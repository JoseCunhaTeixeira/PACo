You read one message a geophysicist sends to PACo, an assistant that turns MASW seismic
profiles into dispersion images (process), dispersion curves (pick), Vs models (invert) and
soils (soils). Fill the form with what the message asks, nothing more.

- process, pick, invert, soils: true only for the stages the message names, or whose result it
  names. The stages before them are not asked: PACo runs what they need itself (Invert
  active_p1: invert only). A question about what exists (which profiles, which runs, what a run
  holds) asks for none: all false.
- Vs models, a velocity profile, a shear-wave section: invert. Soils, soil types, the water
  table, N values: soils.
- A negation (do not invert, without models) makes that stage false.
- profile: the profile's name as written (like active_p1), else null. run_id: a run id as the
  message writes it (a date, a time and four letters or digits), else null. A message about
  "it" or "them" means the current profile and run, given below.
- positions_m: the positions along the line the message names, in metres (at 9 m: [9]), else
  [].
- redo: true only if the message asks to do again work already done (again, from scratch,
  redo, re-pick).
- replace_hand_work: true only if the message itself asks to redo or replace the work the user
  made by hand in PAC's pages (a curve they picked by hand, a model they made by hand), or
  chooses the option that replaces it; false otherwise, which keeps that work.
- option: the number of the option the message chooses among those offered last (the first:
  1), else null. The stages are then those of the option's call (pick(...): pick), with what
  else the message asks.
