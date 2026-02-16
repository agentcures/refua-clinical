from refua_clinical.models import default_simulation_config
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
