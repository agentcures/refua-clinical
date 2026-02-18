import pytest

from refua_clinical.io import config_from_mapping, config_to_mapping
from refua_clinical.models import default_simulation_config


def test_config_from_mapping_parses_biologics_fields() -> None:
    payload = config_to_mapping(default_simulation_config())
    payload["pk_model"]["modality"] = "biologic"
    payload["pk_model"]["route"] = "sc"
    payload["pk_model"]["tmdd_strength"] = 0.42
    payload["pk_model"]["tmdd_cavg_ref"] = 18.0
    payload["arms"][1]["dosing_interval_hours"] = 336.0

    parsed = config_from_mapping(payload)
    assert parsed.pk_model.modality == "biologic"
    assert parsed.pk_model.route == "sc"
    assert float(parsed.pk_model.tmdd_strength) == 0.42
    assert float(parsed.pk_model.tmdd_cavg_ref) == 18.0
    assert float(parsed.arms[1].dosing_interval_hours or 0.0) == 336.0


def test_config_from_mapping_rejects_invalid_pk_modality() -> None:
    payload = config_to_mapping(default_simulation_config())
    payload["pk_model"]["modality"] = "cell_therapy"

    with pytest.raises(ValueError, match="pk_model.modality"):
        config_from_mapping(payload)


def test_config_from_mapping_rejects_invalid_pk_route() -> None:
    payload = config_to_mapping(default_simulation_config())
    payload["pk_model"]["route"] = "intramuscular"

    with pytest.raises(ValueError, match="pk_model.route"):
        config_from_mapping(payload)
