import pytest

from refua_clinical.modality import apply_modality_preset
from refua_clinical.models import default_simulation_config


def test_apply_modality_preset_biologic_sc_updates_config() -> None:
    config = default_simulation_config()
    updated = apply_modality_preset(
        config,
        preset="biologic-sc",
        dosing_interval_hours=336.0,
        tmdd_strength=0.42,
    )

    assert updated.pk_model.modality == "biologic"
    assert updated.pk_model.route == "sc"
    assert float(updated.pk_model.tmdd_strength) == 0.42
    assert int(updated.endpoint.assessment_day) >= 112
    assert all(
        float(arm.dosing_interval_hours or 0.0) == 336.0
        for arm in updated.arms
        if not arm.is_control
    )


def test_apply_modality_preset_rejects_unknown_name() -> None:
    config = default_simulation_config()
    with pytest.raises(ValueError, match="Unknown modality preset"):
        apply_modality_preset(config, preset="cell-therapy-v1")
