"""The settings a call gives that its tool does not have, set aside and said, as pydantic
ignores the fields a model does not expect; a wrong value of a setting the tool has, kept for the
tool to refuse."""

from sigpipe.masw.presets import make_preset

from paco.ignoring import (
    inversion_parameters,
    picking_changes,
    preset_overrides,
    undeclared,
)


def test_the_picking_drops_the_names_it_does_not_have() -> None:
    cleaned = picking_changes({"mode": "M1", "threshold": 0.4, "corridor": -1})

    # The wrong value stays, for the picking to refuse with why.
    assert cleaned.values == {"threshold": 0.4, "corridor": -1}
    (mode,) = cleaned.ignored
    # Said, with where the higher modes are picked.
    assert mode.startswith('mode "M1": not a picking setting')
    assert "PAC's Dispersion picking page" in mode
    assert picking_changes({"threshold": 0.4}).ignored == ()
    assert picking_changes(None).values is None


def test_the_inversion_drops_the_names_it_does_not_have_the_workers_said() -> None:
    cleaned = inversion_parameters({"n_workers": 10, "n_iterations": 1500, "n_chains": "many"})

    assert cleaned.values == {"n_iterations": 1500, "n_chains": "many"}
    assert cleaned.ignored == (
        "n_workers 10: not an inversion setting: the workers are your message's, which PACo "
        "gives every stage",
    )
    # Nothing left: no parameters.
    assert inversion_parameters({"n_workers": 10}).values is None


def test_the_processing_drops_the_stages_and_settings_it_does_not_have() -> None:
    cleaned = preset_overrides(
        "active",
        {
            "mode": "active",
            "dispersion": {"vmax": "fast", "speed": 2},
            "fk": {"threshold": 0.1},
            "masw": {"length_m": 6.0},
        },
    )

    # The mode and the windows in metres are the tool's to read; a wrong value its to refuse.
    assert cleaned.values == {
        "mode": "active",
        "dispersion": {"vmax": "fast"},
        "masw": {"length_m": 6.0},
    }
    assert cleaned.ignored == (
        "dispersion speed 2: not a dispersion setting",
        'fk {"threshold": 0.1}: not a processing stage',
    )


def test_a_redo_reads_its_changes_over_the_runs_own_preset() -> None:
    # A stage's setting alone, over the run's own preset (its method there already).
    run = make_preset(
        "passive", {"selection": {"method": "fk", "threshold": 0.2, "vmin": 100, "vmax": 1000}}
    )

    cleaned = preset_overrides(run, {"selection": {"threshold": 0.1, "fk": 3}})

    assert cleaned.values == {"selection": {"threshold": 0.1}}
    assert cleaned.ignored == ("selection fk 3: not a selection setting",)


def test_the_arguments_a_tool_does_not_declare_are_left_out_said() -> None:
    declared = {"run_id": {}, "parameters": {}}

    cleaned = undeclared({"run_id": "r", "n_workers": 10}, declared, "invert")

    assert cleaned.values == {"run_id": "r"}
    assert cleaned.ignored == (
        "n_workers 10: not an argument of invert: the workers are your message's, which PACo "
        "gives every stage",
    )


def test_a_name_misspelling_a_setting_is_refused_not_ignored() -> None:
    # Ignored, the user's setting would be lost: the tool refuses it, saying which it is.
    picking = picking_changes({"threshhold": 0.3, "mode": "M1"})
    assert picking.typos == ("threshhold: not a picking setting. Did you mean threshold?",)
    assert picking.values is None  # refused before it runs

    # Kept for the inversion's and the preset's own checks, which say which setting it is.
    inversion = inversion_parameters({"iterations": 2000, "n_workers": 10})
    assert inversion.values == {"iterations": 2000}
    assert inversion.typos == ("iterations: not an inversion setting. Did you mean n_iterations?",)
    processing = preset_overrides("active", {"masw": {"lenght": 24}, "fk": {"threshold": 0.1}})
    assert processing.values == {"masw": {"lenght": 24}}
    assert processing.ignored == ('fk {"threshold": 0.1}: not a processing stage',)
    # An abbreviation of one stage only is its misspelling; of several, none.
    stacking = {"stack": {"method": "phase_weighted", "nu": 1}, "disp": {}, "mut": {}}
    abbreviated = preset_overrides("passive-active", stacking)
    assert abbreviated.values == {"stack": {"method": "phase_weighted", "nu": 1}, "disp": {}}
    assert abbreviated.ignored == ("mut {}: not a processing stage",)
