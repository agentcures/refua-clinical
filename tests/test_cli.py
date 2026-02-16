import json
from pathlib import Path

from refua_clinical.cli import main


def test_cli_init_simulate_and_rerun(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    run_path = tmp_path / "run.json"
    run_admet_path = tmp_path / "run_admet.json"
    rerun_path = tmp_path / "run_rerun.json"
    optimize_path = tmp_path / "optimization.json"
    voi_path = tmp_path / "voi.json"
    transport_path = tmp_path / "transportability.json"
    workup_dir = tmp_path / "workup"
    advice_json_path = tmp_path / "advice.json"
    advice_md_path = tmp_path / "advice.md"
    admet_json_path = tmp_path / "admet_profile.json"

    rc = main(["init-config", "--output", str(config_path)])
    assert rc == 0
    assert config_path.exists()

    payload = config_path.read_text(encoding="utf-8")
    assert "trial_id" in payload
    config_path.write_text(payload.replace("replicates: 250", "replicates: 24"), encoding="utf-8")

    rc = main(["simulate", "--config", str(config_path), "--output", str(run_path)])
    assert rc == 0
    assert run_path.exists()

    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert "summary" in run_payload

    admet_json_path.write_text(
        json.dumps(
            {
                "smiles": "CCO",
                "admet_score": 0.51,
                "adme_score": 0.55,
                "safety_score": 0.45,
                "red_flags": ["hERG"],
                "yellow_flags": [],
                "num_predictions": 44,
                "scores": {
                    "score_Bioavailability_Ma": 0.4,
                    "score_hERG": 0.3,
                    "score_admet": 0.51,
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "simulate",
            "--config",
            str(config_path),
            "--admet-json",
            str(admet_json_path),
            "--admet-adjustments",
            "--output",
            str(run_admet_path),
        ]
    )
    assert rc == 0
    run_admet_payload = json.loads(run_admet_path.read_text(encoding="utf-8"))
    assert "admet" in run_admet_payload
    assert run_admet_payload["admet"]["adjustments"] is not None

    rc = main(
        [
            "rerun",
            "--run",
            str(run_path),
            "--set",
            "enrollment.total_n=90",
            "--set",
            "replicates=12",
            "--output",
            str(rerun_path),
        ]
    )
    assert rc == 0
    rerun_payload = json.loads(rerun_path.read_text(encoding="utf-8"))
    assert rerun_payload["config"]["enrollment"]["total_n"] == 90
    assert rerun_payload["config"]["replicates"] == 12

    rc = main(
        [
            "optimize",
            "--run",
            str(run_path),
            "--candidate-total-n",
            "90",
            "--candidate-interims",
            "20",
            "--replicates-per-candidate",
            "20",
            "--output",
            str(optimize_path),
        ]
    )
    assert rc == 0
    optimize_payload = json.loads(optimize_path.read_text(encoding="utf-8"))
    assert "best_candidate" in optimize_payload

    rc = main(
        [
            "voi",
            "--run",
            str(run_path),
            "--extra-n",
            "0",
            "20",
            "--replicates-per-scenario",
            "20",
            "--output",
            str(voi_path),
        ]
    )
    assert rc == 0
    voi_payload = json.loads(voi_path.read_text(encoding="utf-8"))
    assert voi_payload["scenarios"]

    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    reference_path.write_text(
        "age,weight,egfr\n60,75,80\n55,72,84\n68,81,78\n",
        encoding="utf-8",
    )
    target_path.write_text(
        "age,weight,egfr\n63,77,76\n61,74,79\n70,83,73\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "transportability",
            "--reference",
            str(reference_path),
            "--target",
            str(target_path),
            "--output",
            str(transport_path),
        ]
    )
    assert rc == 0
    transport_payload = json.loads(transport_path.read_text(encoding="utf-8"))
    assert "risk_level" in transport_payload

    rc = main(
        [
            "workup",
            "--config",
            str(config_path),
            "--output-dir",
            str(workup_dir),
            "--candidate-total-n",
            "90",
            "--candidate-interims",
            "20",
            "--replicates-per-candidate",
            "20",
            "--voi-extra-n",
            "0",
            "20",
            "--voi-replicates-per-scenario",
            "20",
        ]
    )
    assert rc == 0
    assert (workup_dir / "manifest.json").exists()
    assert (workup_dir / "run.json").exists()
    assert (workup_dir / "advice.json").exists()

    rc = main(
        [
            "advise",
            "--run",
            str(run_admet_path),
            "--output-json",
            str(advice_json_path),
            "--output-markdown",
            str(advice_md_path),
        ]
    )
    assert rc == 0
    assert advice_json_path.exists()
    assert advice_md_path.exists()

    advice_payload = json.loads(advice_json_path.read_text(encoding="utf-8"))
    assert advice_payload["recommendations"]
