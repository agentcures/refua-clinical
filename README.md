# refua-clinical

`refua-clinical` is a simulation package for PK/PD-driven clinical trial design in the Refua ecosystem.
It generates copula-based virtual patients, runs adaptive multi-arm trial simulations, and produces a tailored protocol that can be re-simulated with modified inputs.

## What it provides

- Population PK/PD simulation with inter-individual variability and covariate effects.
- Virtual patient generation using Gaussian copulas and configurable marginals.
- Estimand-aware analysis (`treatment_policy`, `hypothetical`, `composite`, `while_on_treatment`) for intercurrent-event handling.
- Clinical trial simulation with multi-arm randomization, Bayesian response-adaptive allocation, early stopping (success/futility), enrollment drift, and optional site/country heterogeneity.
- Dynamic external-control borrowing with commensurability-weighted discounting.
- Protocol recommendation engine that scores candidate designs (sample size + interim cadence) on simulated operating characteristics and expected cost.
- Multi-objective design optimization with Pareto front outputs.
- Value-of-information (VOI) analysis for expansion decisions.
- Transportability diagnostics (covariate shift, overlap score, risk level) for target-population extrapolation.
- Explainability/advice engine that outputs a narrative, interim decision-card summaries, and prioritized actionable recommendations.
- Re-run workflow from prior run artifacts with YAML/JSON or inline overrides.
- ADMET-aware simulation path with optional parameter adjustments from Refua ADMET profiles.
- Biologics mode with IV/SC route support, interval dosing, and optional TMDD-like clearance scaling.
- One-shot `workup` CLI to generate all major artifacts in one run.
- Optional integrations:
  - `refua-data` for covariate inference from materialized datasets.
  - `refua-regulatory` for audit/evidence bundles.

## Install

```bash
cd refua-clinical
pip install -e .
```

With integrations:

```bash
pip install -e .[integrations]
```

With Refua ADMET support from SMILES:

```bash
pip install -e .[admet]
```

Check installed CLI version:

```bash
refua-clinical --version
```

## CLI quickstart

Create a starter config:

```bash
refua-clinical init-config --output examples/default_config.yaml
```

Run simulation:

```bash
refua-clinical simulate \
  --config examples/default_config.yaml \
  --output artifacts/run.json
```

Run with ADMET-informed adjustments:

```bash
refua-clinical simulate \
  --config examples/default_config.yaml \
  --admet-json artifacts/admet_profile.json \
  --admet-adjustments \
  --output artifacts/run_admet.json
```

Run with richer Refua payload integration (affinity + confidence + properties + ADMET):

```bash
refua-clinical simulate \
  --config examples/default_config.yaml \
  --refua-json examples/refua_integration_payload.json \
  --refua-apply \
  --output artifacts/run_refua.json
```

Generate an adjusted config from Refua payload without running simulation:

```bash
refua-clinical integrate-refua \
  --config examples/default_config.yaml \
  --refua-json examples/refua_integration_payload.json \
  --output-config artifacts/config_refua.yaml \
  --output-summary artifacts/refua_integration_summary.json
```

Generate tailored protocol:

```bash
refua-clinical protocol \
  --run artifacts/run.json \
  --output artifacts/protocol.json \
  --markdown artifacts/protocol.md
```

Run multi-objective optimization:

```bash
refua-clinical optimize \
  --run artifacts/run.json \
  --output artifacts/optimization.json \
  --markdown artifacts/optimization.md
```

Run value-of-information scenarios:

```bash
refua-clinical voi \
  --run artifacts/run.json \
  --extra-n 0 30 60 90 \
  --output artifacts/voi.json \
  --markdown artifacts/voi.md
```

Assess transportability between reference and target populations:

```bash
refua-clinical transportability \
  --reference data/reference_population.csv \
  --target data/target_population.csv \
  --output artifacts/transportability.json \
  --markdown artifacts/transportability.md
```

Generate explainable narrative and actionable advice:

```bash
refua-clinical advise \
  --run artifacts/run.json \
  --protocol artifacts/protocol.json \
  --include-sensitivity \
  --output-json artifacts/advice.json \
  --output-markdown artifacts/advice.md
```

ADMET-aware advice from SMILES (requires `.[admet]`):

```bash
refua-clinical advise \
  --run artifacts/run.json \
  --admet-smiles "CCO" \
  --output-json artifacts/advice_with_admet.json
```

Run everything all at once (simulate + protocol + optimize + VOI + advice):

```bash
refua-clinical workup \
  --config examples/default_config.yaml \
  --output-dir artifacts/full_workup \
  --include-sensitivity
```

Re-run with overrides:

```bash
refua-clinical rerun \
  --run artifacts/run.json \
  --set enrollment.total_n=240 \
  --set adaptive.interim_every=20 \
  --output artifacts/run_rerun.json
```

Infer virtual-patient covariates from `refua-data`:

```bash
refua-clinical from-data \
  --dataset-id chembl_activity_ki_human \
  --output examples/from_data.yaml
```

Create an evidence bundle with `refua-regulatory`:

```bash
refua-clinical evidence \
  --run artifacts/run.json \
  --output-dir artifacts/evidence/clinical_run_001
```

## Python API (Object-Oriented)

The primary API is the fluent object model centered on `ClinicalStudy` and `ClinicalRun`.

```python
from refua_clinical import ClinicalStudy

study = (
    ClinicalStudy.default()
    .trial(
        trial_id="oncology-refua-oo-demo",
        indication="Oncology",
        phase="Phase II",
        replicates=96,
    )
    .set("enrollment.total_n", 220)
)

run = study.simulate()
protocol = run.recommend_protocol(replicates_per_candidate=30)
print(run.summary["power"], protocol.protocol["protocol_id"])
```

Biologics preset example:

```python
from refua_clinical import ClinicalStudy

study = (
    ClinicalStudy.default()
    .trial(trial_id="mab-phase2", indication="Immunology", phase="Phase II", replicates=80)
    .biologics_mode(route="sc", dosing_interval_hours=336.0, tmdd_strength=0.35)  # q14d
)
run = study.simulate()
print(run.summary["power"], run.config.pk_model.modality, run.config.pk_model.route)
```

Refua payload-to-clinical mapping API:

```python
from refua_clinical import (
    ClinicalStudy,
)

payload = {
    "ligands": [
        {
            "ligand_id": "lead_a",
            "affinity": {"ic50": 42.0, "binding_probability": 0.79},
            "structure": {"confidence_score": 0.77},
            "admet": {"admet_score": 0.68, "adme_score": 0.70, "safety_score": 0.64},
            "rdkit": {"mol_wt": 340.0, "mol_log_p": 2.1, "qed": 0.73},
        }
    ],
    "target_properties": {"length": 850.0, "instability_index": 45.0},
}

study = (
    ClinicalStudy.default()
    .trial(trial_id="refua-bridge-demo", replicates=80)
    .refua_payload(payload, apply=True, max_candidate_arms=3)
)
run = study.simulate()
workup = run.workup(
    replicates_per_candidate=30,
    voi_extra_n=[0, 30, 60],
    include_sensitivity=True,
)
print(run.summary["power"], workup.advice.report["summary"]["safety_event_rate"])
```

## End-to-end notebook (Refua + refua-clinical)

Use `examples/refua_api_to_clinical_e2e.ipynb` for a full workflow that starts with a
small-molecule SMILES and protein target in `refua`, computes molecule/target properties, and
feeds those into `refua-clinical` simulation and protocol recommendation outputs.

From a monorepo checkout:

```bash
cd refua
pip install -e .[admet]
cd ../refua-clinical
pip install -e .[admet]
jupyter notebook examples/refua_api_to_clinical_e2e.ipynb
```

Minimal combined API sketch:

```python
from refua import Protein, SM
from refua_clinical import (
    ClinicalStudy,
)

sm = SM("Cn1cnc2n(C)c(=O)n(C)c(=O)c12")  # small molecule
target = Protein("MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQQLAGKYTPEEIRNVLSTLQKAD", ids="A")

study = (
    ClinicalStudy.default()
    .trial(
        trial_id="refua-object-api-demo",
        indication="Target-program demo",
        replicates=72,
    )
    .admet_profile(sm.admet_profile(include_scoring=True), apply=True)
)

run = study.simulate()
protocol = run.recommend_protocol(replicates_per_candidate=30)
print(target.length(), run.summary["power"], protocol.protocol["protocol_id"])
```

## Research-informed design choices

This package maps current literature and regulatory guidance into practical simulation defaults:

1. ICH M15 (Step 5, adopted November 24, 2025): https://www.ema.europa.eu/en/ich-m15-general-principles-model-informed-drug-development-step-5
2. ICH E9 (R1) estimands addendum: https://www.ema.europa.eu/en/ich-e9-statistical-principles-clinical-trials-scientific-guideline
3. FDA adaptive design guidance (2019): https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry
4. FDA Bayesian methods guidance page (updated January 2026): https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-bayesian-methodology-clinical-trials-drug-and-biological-products
5. Copula virtual patients (CPT Pharmacometrics Syst Pharmacol, 2024): https://pubmed.ncbi.nlm.nih.gov/38853786/
6. Copula adult populations (J Pharmacokinet Pharmacodyn, 2024): https://pubmed.ncbi.nlm.nih.gov/38661766/
7. Dynamic borrowing framework (BMC Med Res Methodol, 2025): https://link.springer.com/article/10.1186/s12874-025-02691-2
8. Non-concurrent control time-trend methods (Biometrical Journal, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12458466/
9. Backfilling in adaptive platform trials (BMJ, 2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11698248/
10. Trial-to-target transportability with synthetic populations (BMJ Open, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12306361/
11. High-throughput Bayesian simulation tooling (BATSS, 2024): https://arxiv.org/abs/2410.02050

## Test

```bash
cd refua-clinical
python -m pytest -q
```

## Notes

- Intended for **research design support** and internal decision analysis.
- Not a substitute for clinical, biostatistical, ethics, or regulatory review.

## Project docs

- Changelog: `CHANGELOG.md`
- Contributing guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
