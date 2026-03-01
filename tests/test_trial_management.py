from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime, timedelta

from refua_clinical.trial_management import ClinicalTrialManager


def test_trial_manager_crud_flow(tmp_path: Path) -> None:
    manager = ClinicalTrialManager(tmp_path / "trial_store.json")

    created = manager.create_trial(
        trial_id="mgmt-demo",
        indication="Oncology",
        phase="Phase II",
        objective="Manage and adapt trial operations",
        status="planned",
    )
    assert created["trial"] is not None
    assert created["trial"]["trial_id"] == "mgmt-demo"
    assert created["trial"]["status"] == "planned"

    listing = manager.list_trials()
    assert len(listing) == 1
    assert listing[0]["trial_id"] == "mgmt-demo"

    updated = manager.update_trial(
        "mgmt-demo",
        updates={
            "status": "active",
            "config": {
                "replicates": 8,
                "enrollment": {"total_n": 80},
                "adaptive": {"interim_every": 20},
            },
        },
    )
    assert updated["trial"] is not None
    assert updated["trial"]["status"] == "active"
    assert updated["trial"]["config"]["replicates"] == 8
    assert updated["trial"]["config"]["enrollment"]["total_n"] == 80

    removed = manager.remove_trial("mgmt-demo")
    assert removed["removed"] is True
    assert manager.get_trial("mgmt-demo") is None


def test_trial_manager_simulation_blends_observed_data(tmp_path: Path) -> None:
    manager = ClinicalTrialManager(tmp_path / "trial_store.json")

    manager.create_trial(trial_id="blend-demo", status="enrolling")
    manager.update_trial(
        "blend-demo",
        updates={
            "config": {
                "replicates": 6,
                "enrollment": {"total_n": 60},
                "adaptive": {"burn_in_n": 20, "interim_every": 20},
            }
        },
    )

    manager.enroll_patient(
        "blend-demo",
        patient_id="human-001",
        source="human",
        arm_id="control",
        demographics={"age": 63, "weight": 78},
    )
    manager.enroll_patient(
        "blend-demo",
        patient_id="human-002",
        source="human",
        arm_id="low",
        demographics={"age": 57, "weight": 74},
    )
    simulated = manager.enroll_simulated_patients("blend-demo", count=4, seed=11)
    assert simulated["count"] == 4

    manager.record_result(
        "blend-demo",
        patient_id="human-001",
        values={
            "arm_id": "control",
            "change": 4.2,
            "responder": False,
            "safety_event": False,
        },
    )
    manager.record_result(
        "blend-demo",
        patient_id="human-002",
        values={
            "arm_id": "low",
            "change": 10.1,
            "responder": True,
            "safety_event": False,
        },
    )

    sim = manager.simulate_trial("blend-demo", replicates=4, seed=5)

    summary = sim["simulation"]["summary"]
    management = sim["management"]

    assert management["patient_count_human"] == 2
    assert management["patient_count_simulated"] == 4
    assert management["result_count"] == 2
    assert management["observed_effect_estimate"] is not None

    assert summary["observed_effect_estimate"] is not None
    assert summary["blended_effect_estimate"] is not None
    assert 0.0 <= float(summary["observed_data_weight"]) <= 1.0

    trial = manager.get_trial("blend-demo")
    assert trial is not None
    assert len(trial["simulations"]) >= 1


def test_trial_manager_clinops_operations(tmp_path: Path) -> None:
    manager = ClinicalTrialManager(tmp_path / "trial_store.json")
    manager.create_trial(trial_id="ops-demo", status="enrolling")

    site = manager.upsert_site(
        "ops-demo",
        site_id="site-001",
        name="Boston General",
        country_id="US",
        status="active",
        target_enrollment=24,
        metadata={"region": "NA"},
    )
    assert site["site"]["site_id"] == "site-001"

    manager.record_screening(
        "ops-demo",
        site_id="site-001",
        patient_id="pt-001",
        status="screen_failed",
        failure_reason="did_not_meet_inclusion",
    )
    manager.record_screening(
        "ops-demo",
        site_id="site-001",
        patient_id="pt-002",
        status="enrolled",
        arm_id="control",
        auto_enroll=True,
        demographics={"age": 54},
    )

    overdue_due_at = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    opened = manager.add_query(
        "ops-demo",
        patient_id="pt-002",
        site_id="site-001",
        description="Missing week-4 lab panel",
        status="open",
        due_at=overdue_due_at,
    )
    assert opened["query"]["status"] == "open"

    manager.record_monitoring_visit(
        "ops-demo",
        site_id="site-001",
        visit_type="interim",
        findings=["Source notes incomplete for one visit"],
        action_items=["Retrain coordinator on source documentation"],
        risk_score=0.82,
    )
    manager.record_deviation(
        "ops-demo",
        description="Visit window outside protocol limit",
        site_id="site-001",
        patient_id="pt-002",
        severity="major",
        status="open",
    )
    manager.record_safety_event(
        "ops-demo",
        patient_id="pt-002",
        site_id="site-001",
        event_term="grade_3_neutropenia",
        seriousness="serious",
        expected=False,
    )
    manager.upsert_milestone(
        "ops-demo",
        milestone_id="ms-lpi",
        name="Last Patient In",
        target_date=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
        status="at_risk",
    )

    snapshot = manager.operations_snapshot("ops-demo")
    clinops = snapshot["clinops"]
    assert clinops["site_count"] == 1
    assert clinops["active_site_count"] == 1
    assert clinops["screened_count"] == 2
    assert clinops["screen_fail_count"] == 1
    assert clinops["open_queries"] == 1
    assert clinops["overdue_queries"] == 1
    assert clinops["major_deviations"] == 1
    assert clinops["serious_safety_events"] == 1
    assert clinops["overdue_milestones"] == 1
    assert clinops["at_risk_sites"][0]["site_id"] == "site-001"

    sites = manager.list_sites("ops-demo")
    assert sites["count"] == 1
