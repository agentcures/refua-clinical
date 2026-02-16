from refua_clinical.models import default_simulation_config
from refua_clinical.optimization import optimization_to_markdown, optimize_design_space


def test_optimize_design_space_returns_best_and_pareto() -> None:
    config = default_simulation_config()
    config.replicates = 12
    config.enrollment.total_n = 120

    payload = optimize_design_space(
        config,
        candidate_total_n=[90, 120],
        candidate_interims=[20],
        replicates_per_candidate=20,
    )

    assert "best_candidate" in payload
    assert payload["candidates"]
    assert payload["pareto_front"]

    best = payload["best_candidate"]
    assert float(best["utility_score"]) >= 0.0

    markdown = optimization_to_markdown(payload)
    assert "Design Optimization" in markdown
