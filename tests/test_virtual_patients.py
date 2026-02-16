import numpy as np

from refua_clinical.models import CovariateSpec, VirtualPopulationSpec
from refua_clinical.virtual_patients import generate_virtual_population


def test_generate_virtual_population_respects_shape_and_correlation() -> None:
    spec = VirtualPopulationSpec(
        size=3000,
        covariates=[
            CovariateSpec("age", "normal", {"mean": 60.0, "sd": 10.0}),
            CovariateSpec("weight", "normal", {"mean": 75.0, "sd": 12.0}),
            CovariateSpec("egfr", "normal", {"mean": 85.0, "sd": 18.0}),
        ],
        correlation=[
            [1.0, 0.30, -0.20],
            [0.30, 1.0, 0.10],
            [-0.20, 0.10, 1.0],
        ],
    )

    pop = generate_virtual_population(spec, seed=13)

    assert len(pop.table) == 3000
    assert {"patient_id", "age", "weight", "egfr"}.issubset(pop.table.columns)

    corr = np.corrcoef(pop.table[["age", "weight", "egfr"]].to_numpy().T)
    assert corr[0, 1] > 0.20
    assert corr[0, 2] < -0.10
