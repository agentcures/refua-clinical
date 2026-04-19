# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [Unreleased]

### Added
- No changes yet.

## [0.8.0] - 2026-04-19

### Changed
- Reduced import overhead by switching the top-level package exports to lazy loading.
- Reworked the trial simulator hot path to use array-backed enrollment state instead of per-patient dict/DataFrame churn.
- Reused generated virtual populations across protocol, optimization, and VOI candidate evaluation loops.
- Replaced repeated config serialize/parse cycles in internal hot paths with direct config cloning.
- Vectorized more of the longitudinal PD generation path and applied operational shifts during PD simulation.

## [0.2.0] - 2026-02-16

### Added
- ADMET integration helpers for loading/computing Refua ADMET profiles and applying risk-aware PK/PD simulation adjustments.
- `advise` CLI command for explainable narrative reports with prioritized, actionable recommendations.
- One-way sensitivity analysis workflow for transparent decision-driver ranking.
- Optional `admet` extra (`pip install -e .[admet]`) for SMILES-driven ADMET scoring via Refua.
- Estimand/intercurrent-event handling strategies (`treatment_policy`, `hypothetical`, `composite`, `while_on_treatment`).
- Interim stopping framework with alpha-spending, posterior superiority, and predictive-success criteria.
- Dynamic external-control borrowing with commensurability-weighted discounting.
- Site/country heterogeneity simulation and operational cost modeling.
- `optimize` CLI command for multi-objective candidate search and Pareto fronts.
- `voi` CLI command for value-of-information sample-size expansion scenarios.
- `transportability` CLI command for covariate-shift diagnostics between trial and target populations.
- `workup` CLI command for one-shot generation of run/protocol/optimization/VOI/advice artifacts.
- Expanded explainability narrative to include early-stop rates, expected sample size, and interim decision-card summaries.
- Updated research basis with 2025-2026 Bayesian/adaptive/transportability references.

## [0.1.0] - 2026-02-16

### Added
- Initial `refua-clinical` package release.
- Copula-based virtual patient generation with configurable marginals and covariance structure.
- PK/PD simulation engine with population variability, exposure-response, and safety event modeling.
- Multi-arm trial simulator with adaptive randomization, enrollment-block drift effects, and optional external-control borrowing.
- Tailored clinical protocol recommendation workflow with design candidate scoring.
- Re-run workflow that applies YAML/JSON or inline overrides on prior run artifacts.
- Optional integration with `refua-data` for covariate inference from materialized datasets.
- Optional integration with `refua-regulatory` for audit/evidence bundle generation.
- CLI commands: `init-config`, `simulate`, `rerun`, `protocol`, `from-data`, `evidence`, and `research`.
- Release quality gates for linting, typing, tests, and distribution validation.
- Release documentation (`CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`).

### Notes
- Intended for research planning and simulation support only.
- Not a substitute for clinical, statistical, ethics, or regulatory review.
