from refua_clinical.models import default_simulation_config
from refua_clinical.protocol import recommend_protocol, render_protocol_markdown


def test_recommend_protocol_returns_ranked_candidates() -> None:
    config = default_simulation_config()
    config.replicates = 20
    config.enrollment.total_n = 120

    recommendation = recommend_protocol(
        config,
        replicates_per_candidate=24,
        candidate_total_n=[90, 120],
        candidate_interims=[20, 30],
    )

    assert recommendation.candidates
    assert recommendation.candidates[0].utility >= recommendation.candidates[-1].utility

    protocol = recommendation.protocol
    assert protocol["design"]["planned_enrollment"] in {90, 120}
    assert "research_basis" in protocol

    markdown = render_protocol_markdown(protocol)
    assert protocol["title"] in markdown
