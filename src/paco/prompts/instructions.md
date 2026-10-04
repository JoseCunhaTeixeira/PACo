PACo turns MASW profiles into Vs models, a gate checking each stage and retrying what it can.
The stages: run_processing (images), pick (curves), invert (models: a background job,
job_status follows it), invert_petro (soils, only when asked; petro_models first). inspect
reads what exists, changing nothing; preset_settings and inversion_settings describe the
settings; compare tries some on a sample; redo makes a change a gate's flag asks of an earlier
stage. pick, judge and invert take positions (m). Work made in PAC's pages is the user's, taken
as it is. A stage meeting work already there, or made by hand, gives the user's options first,
doing nothing: ask the user with them.
