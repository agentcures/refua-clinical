import pytest

from refua_clinical.models import default_simulation_config
from refua_clinical.refua_bridge import (
    RefuaIntegrationPolicy,
    apply_refua_adjustments,
    extract_admet_profile_from_refua_payload,
    summarize_refua_payload,
)


def _payload() -> dict:
    return {
        "ligands": [
            {
                "ligand_id": "lead_a",
                "smiles": "CCO",
                "rdkit": {
                    "mol_wt": 320.0,
                    "mol_log_p": 2.2,
                    "tpsa": 78.0,
                    "num_h_donors": 1.0,
                    "num_h_acceptors": 5.0,
                    "num_rotatable_bonds": 4.0,
                    "qed": 0.76,
                    "medchem_alert_count": 0.0,
                },
                "admet": {
                    "smiles": "CCO",
                    "admet_score": 0.74,
                    "adme_score": 0.72,
                    "safety_score": 0.71,
                    "red_flags": [],
                    "yellow_flags": ["CYP3A4_Veith"],
                    "num_predictions": 44,
                    "scores": {
                        "score_Bioavailability_Ma": 0.78,
                        "score_HIA_Hou": 0.75,
                        "score_Caco2_Wang": 0.72,
                        "score_PAMPA_NCATS": 0.70,
                        "score_Solubility_AqSolDB": 0.66,
                        "score_BBB_Martins": 0.60,
                        "score_VDss_Lombardo": 0.65,
                        "score_PPBR_AZ": 0.69,
                        "score_Pgp_Broccatelli": 0.62,
                        "score_CYP2D6_Veith": 0.66,
                        "score_CYP3A4_Veith": 0.59,
                        "score_CYP2C9_Veith": 0.71,
                        "score_CYP2C19_Veith": 0.70,
                        "score_CYP1A2_Veith": 0.67,
                        "score_CYP2D6_Substrate_CarbonMangels": 0.60,
                        "score_CYP3A4_Substrate_CarbonMangels": 0.61,
                        "score_CYP2C9_Substrate_CarbonMangels": 0.63,
                        "score_Clearance_Hepatocyte_AZ": 0.66,
                        "score_Clearance_Microsome_AZ": 0.64,
                        "score_Half_Life_Obach": 0.63,
                        "score_hERG": 0.68,
                        "score_AMES": 0.74,
                        "score_DILI": 0.69,
                        "score_ClinTox": 0.70,
                        "score_Carcinogens_Lagunin": 0.73,
                        "score_Tox21_SR_MMP": 0.67,
                        "score_Tox21_SR_p53": 0.71,
                        "score_admet": 0.74,
                    },
                },
                "affinity": {"ic50": 35.0, "binding_probability": 0.82},
                "structure": {"confidence_score": 0.79},
            },
            {
                "ligand_id": "backup_b",
                "smiles": "CCCN",
                "rdkit": {
                    "mol_wt": 530.0,
                    "mol_log_p": 4.1,
                    "tpsa": 138.0,
                    "num_h_donors": 4.0,
                    "num_h_acceptors": 11.0,
                    "num_rotatable_bonds": 10.0,
                    "qed": 0.38,
                    "medchem_alert_count": 3.0,
                },
                "admet": {
                    "smiles": "CCCN",
                    "admet_score": 0.43,
                    "adme_score": 0.41,
                    "safety_score": 0.39,
                    "red_flags": ["hERG", "DILI"],
                    "yellow_flags": [],
                    "num_predictions": 44,
                    "scores": {
                        "score_Bioavailability_Ma": 0.35,
                        "score_hERG": 0.31,
                        "score_DILI": 0.28,
                        "score_admet": 0.43,
                    },
                },
                "affinity": {"ic50": 180.0, "binding_probability": 0.45},
                "structure": {"confidence_score": 0.55},
            },
        ],
        "target_properties": {
            "length": 980.0,
            "instability_index": 46.0,
            "gravy": -0.22,
            "antibody_liability_score": 3.1,
        },
    }


def test_summarize_refua_payload_selects_best_candidate() -> None:
    summary = summarize_refua_payload(_payload())
    assert summary["candidate_count"] == 2
    assert summary["selected_ligand_id"] == "lead_a"

    selected = summary["selected_candidate"]
    assert selected["signal_summary"]["binding_probability"] > 0.7
    assert selected["signal_summary"]["red_flag_count"] == 0


def test_extract_admet_profile_from_payload_honors_preferred_ligand() -> None:
    extracted = extract_admet_profile_from_refua_payload(
        _payload(),
        preferred_ligand_id="backup_b",
    )
    assert extracted is not None
    assert float(extracted["admet_score"]) == 0.43


def test_summarize_refua_payload_reports_legacy_contract_keys() -> None:
    payload = {
        "ligand_admet": {
            "legacy_ligand": {"admet_score": 0.55, "safety_score": 0.52},
        },
        "protein_properties": {"length": 1000.0},
    }

    summary = summarize_refua_payload(payload)
    contract = summary["contract"]
    assert contract["is_canonical"] is False
    assert "ligand_admet" in contract["legacy_root_keys"]
    assert "protein_properties" in contract["legacy_root_keys"]


def test_summarize_refua_payload_strict_contract_rejects_legacy_payload() -> None:
    payload = {
        "ligand_admet": {
            "legacy_ligand": {"admet_score": 0.55, "safety_score": 0.52},
        }
    }

    with pytest.raises(ValueError, match="strict contract validation"):
        summarize_refua_payload(payload, strict_contract=True)


def test_apply_refua_adjustments_changes_multiple_model_dimensions() -> None:
    config = default_simulation_config()
    adjusted, report = apply_refua_adjustments(
        config,
        _payload(),
        policy=RefuaIntegrationPolicy(max_candidate_arms=2),
    )

    assert adjusted.pk_model.bioavailability != config.pk_model.bioavailability
    assert adjusted.pd_model.ec50_auc != config.pd_model.ec50_auc
    assert adjusted.pd_model.residual_sd != config.pd_model.residual_sd
    assert adjusted.endpoint.assessment_day != config.endpoint.assessment_day
    assert len(adjusted.arms) == 3

    management = report["management"]
    assert 0.0 <= float(management["readiness_index"]) <= 100.0
    assert management["recommended_actions"]


def test_apply_refua_adjustments_in_biologic_mode_skips_small_molecule_adjustments() -> None:
    config = default_simulation_config()
    config.pk_model.modality = "biologic"
    config.pk_model.route = "sc"

    adjusted, report = apply_refua_adjustments(
        config,
        _payload(),
        policy=RefuaIntegrationPolicy(max_candidate_arms=2),
    )

    assert adjusted.pk_model.modality == "biologic"
    assert "admet_base" not in report["adjustments"]
    assert "admet_endpoints" not in report["adjustments"]
    assert "rdkit" not in report["adjustments"]
    assert report["adjustments"]["biologic_mode"]["admet_adjustments_skipped"] is True
