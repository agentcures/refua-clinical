import json
from pathlib import Path

from refua_clinical.object_api import ClinicalStudy


def _refua_payload() -> dict:
    return {
        "ligands": [
            {
                "ligand_id": "lead_a",
                "rdkit": {
                    "mol_wt": 340.0,
                    "mol_log_p": 2.1,
                    "tpsa": 82.0,
                    "num_h_donors": 1.0,
                    "num_h_acceptors": 5.0,
                    "num_rotatable_bonds": 4.0,
                    "qed": 0.73,
                    "medchem_alert_count": 0.0,
                },
                "admet": {
                    "smiles": "CCO",
                    "admet_score": 0.68,
                    "adme_score": 0.70,
                    "safety_score": 0.64,
                    "red_flags": [],
                    "yellow_flags": [],
                    "scores": {
                        "score_Bioavailability_Ma": 0.72,
                        "score_hERG": 0.66,
                        "score_DILI": 0.63,
                        "score_admet": 0.68,
                    },
                },
                "affinity": {"ic50": 42.0, "binding_probability": 0.79},
                "structure": {"confidence_score": 0.77},
            }
        ],
        "target_properties": {"length": 850.0, "instability_index": 45.0, "gravy": -0.2},
    }


def test_object_api_end_to_end(tmp_path: Path) -> None:
    study = (
        ClinicalStudy.default()
        .trial(
            trial_id="oo-api-test",
            indication="Oncology",
            phase="Phase II",
            replicates=20,
            seed=5,
        )
        .set("enrollment.total_n", 120)
        .refua_payload(_refua_payload(), apply=True, max_candidate_arms=2)
    )

    run = study.simulate()
    assert run.run_id
    assert 0.0 <= float(run.summary["power"]) <= 1.0
    assert "refua" in run.payload

    protocol = run.recommend_protocol(
        replicates_per_candidate=20,
        candidate_total_n=[90, 120],
        candidate_interims=[20],
    )
    assert protocol.protocol["protocol_id"]

    optimization = run.optimize(
        replicates_per_candidate=20,
        candidate_total_n=[90, 120],
        candidate_interims=[20],
    )
    assert optimization.payload["best_candidate"]

    voi = run.value_of_information(extra_n=[0, 20], replicates_per_scenario=20)
    assert voi.payload["best_scenario"]

    advice = run.advise(
        protocol=protocol,
        optimization=optimization,
        voi=voi,
        include_sensitivity=False,
    )
    assert advice.report["recommendations"]

    workup = run.workup(
        replicates_per_candidate=20,
        candidate_total_n=[90, 120],
        candidate_interims=[20],
        voi_extra_n=[0, 20],
        voi_replicates_per_scenario=20,
        include_sensitivity=False,
    )
    manifest = workup.save(tmp_path / "workup")

    assert manifest["run"] is not None
    assert manifest["protocol"] is not None
    assert manifest["optimization"] is not None
    assert manifest["voi"] is not None
    assert manifest["advice"] is not None

    manifest_path = tmp_path / "workup" / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "run" in payload


def test_object_api_biologics_mode_applies_pk_preset() -> None:
    study = (
        ClinicalStudy.default()
        .trial(
            trial_id="oo-biologic",
            indication="Immunology",
            phase="Phase II",
            replicates=12,
            seed=3,
        )
        .biologics_mode(route="sc", dosing_interval_hours=336.0, tmdd_strength=0.4)
    )

    config = study.config
    assert config.pk_model.modality == "biologic"
    assert config.pk_model.route == "sc"
    assert float(config.pk_model.tmdd_strength) == 0.4
    assert int(config.endpoint.assessment_day) >= 112

    treatment_arms = [arm for arm in config.arms if not arm.is_control]
    assert treatment_arms
    assert all(float(arm.dosing_interval_hours or 0.0) == 336.0 for arm in treatment_arms)

    run = study.simulate()
    assert 0.0 <= float(run.summary["power"]) <= 1.0


def test_object_api_modality_preset_applies_shared_profile() -> None:
    study = (
        ClinicalStudy.default()
        .trial(replicates=10, seed=4)
        .modality_preset(
            preset="biologic-iv",
            dosing_interval_hours=336.0,
            tmdd_strength=0.3,
        )
    )
    config = study.config
    assert config.pk_model.modality == "biologic"
    assert config.pk_model.route == "iv"
    assert float(config.pk_model.tmdd_strength) == 0.3
