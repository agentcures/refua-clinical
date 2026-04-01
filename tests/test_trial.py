from refua_clinical.models import ArmSpec, default_simulation_config
from refua_clinical.trial import simulate_trials


def test_simulate_trials_returns_summary_and_replicates() -> None:
    config = default_simulation_config()
    config.replicates = 16
    config.enrollment.total_n = 120
    config.adaptive.burn_in_n = 40
    config.adaptive.interim_every = 20

    result = simulate_trials(config)

    assert len(result.replicates) == 16
    assert 0.0 <= float(result.summary["power"]) <= 1.0
    assert result.summary["mean_effect"] > -50.0

    traces = [len(rep.allocation_trace) for rep in result.replicates]
    assert any(count > 0 for count in traces)


def test_simulate_trials_supports_time_to_event_and_arm_lifecycle() -> None:
    config = default_simulation_config()
    config.replicates = 8
    config.enrollment.total_n = 90
    config.endpoint.kind = "time_to_event"
    config.endpoint.event_horizon_day = 120
    config.adaptive.burn_in_n = 30
    config.adaptive.interim_every = 15
    config.adaptive.arm_drop_threshold = 0.10
    config.arms[1].opens_at_enrollment = 0
    config.arms[2].opens_at_enrollment = 30
    config.arms[2].closes_at_interim = 2

    result = simulate_trials(config)

    assert result.summary["endpoint_kind"] == "time_to_event"
    assert "event_rate" in result.summary
    assert any(
        "high" not in rep.active_arm_ids or rep.dropped_arm_ids
        for rep in result.replicates
    )


def test_simulate_trials_supports_live_arm_addition_and_backfill() -> None:
    config = default_simulation_config()
    config.replicates = 6
    config.enrollment.total_n = 120
    config.endpoint.kind = "longitudinal"
    config.endpoint.visit_days = [28, 56, 84]
    config.adaptive.burn_in_n = 30
    config.adaptive.interim_every = 15
    config.arms.append(
        ArmSpec(
            arm_id="late_combo",
            label="Late Combo Arm",
            dose_mg=220.0,
            opens_at_interim=2,
            backfill_enabled=True,
            backfill_target_n=18,
            backfill_allocation_multiplier=3.0,
            concurrent_control_only=True,
        )
    )

    result = simulate_trials(config)

    assert result.summary["endpoint_kind"] == "longitudinal"
    assert result.summary["analysis_method"] == "mmrm_cluster_ols"
    assert any(rep.arm_enrollment_counts.get("late_combo", 0) > 0 for rep in result.replicates)
    assert any(
        "late_combo" in card.get("active_arms", [])
        for rep in result.replicates
        for card in rep.decision_cards
    )
