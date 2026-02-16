"""Research references used by refua-clinical design choices."""

from __future__ import annotations

from typing import Any

RESEARCH_REFERENCES: list[dict[str, Any]] = [
    {
        "topic": "MIDD policy",
        "title": "ICH M15: General Principles for Model-Informed Drug Development",
        "year": 2025,
        "url": "https://www.ema.europa.eu/en/ich-m15-general-principles-model-informed-drug-development-step-5",
        "key_takeaway": (
            "Regulators expect model context-of-use definition, verification, and uncertainty "
            "quantification when simulation informs clinical decisions."
        ),
    },
    {
        "topic": "Estimands",
        "title": (
            "ICH E9 (R1) addendum on estimands and sensitivity analysis in clinical "
            "trials to the guideline on statistical principles for clinical trials"
        ),
        "year": 2020,
        "url": "https://www.ema.europa.eu/en/ich-e9-statistical-principles-clinical-trials-scientific-guideline",
        "key_takeaway": (
            "Define treatment effect targets and intercurrent-event handling explicitly, "
            "and align sensitivity analyses to the estimand."
        ),
    },
    {
        "topic": "Adaptive design guidance",
        "title": "Adaptive Design Clinical Trials for Drugs and Biologics Guidance for Industry",
        "year": 2019,
        "url": (
            "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/"
            "adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry"
        ),
        "key_takeaway": (
            "Prospective simulation of adaptation rules and operating characteristics is "
            "expected for adaptive designs."
        ),
    },
    {
        "topic": "Bayesian guidance update",
        "title": "Use of Bayesian Methodology in Clinical Trials of Drug and Biological Products",
        "year": 2026,
        "url": (
            "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/"
            "use-bayesian-methodology-clinical-trials-drug-and-biological-products"
        ),
        "key_takeaway": (
            "Bayesian methods are increasingly formalized for primary inference, interim "
            "adaptation, and trial-planning decisions."
        ),
    },
    {
        "topic": "Virtual patients",
        "title": "Virtual patient simulation using copula modeling",
        "journal": "CPT Pharmacometrics Syst Pharmacol",
        "year": 2024,
        "url": "https://pubmed.ncbi.nlm.nih.gov/38853786/",
        "key_takeaway": (
            "Gaussian copulas preserve observed covariate dependencies and improve realism "
            "of generated virtual populations for trial simulation."
        ),
    },
    {
        "topic": "Virtual populations",
        "title": "Generating realistic virtual adult populations using a copula approach",
        "journal": "J Pharmacokinet Pharmacodyn",
        "year": 2024,
        "url": "https://pubmed.ncbi.nlm.nih.gov/38661766/",
        "key_takeaway": (
            "Correlated covariate generation with copulas is practical for high-dimensional "
            "clinical trial virtual populations."
        ),
    },
    {
        "topic": "QSP virtual trials",
        "title": (
            "Simulation of oral treprostinil clinical trials with virtual populations "
            "and weighted plausible populations"
        ),
        "journal": "npj Syst Biol Appl",
        "year": 2024,
        "url": "https://www.nature.com/articles/s41540-024-00379-y",
        "key_takeaway": (
            "Weighted plausible populations and mechanistic disease modeling improve "
            "interpretability of dose-response and subgroup effects."
        ),
    },
    {
        "topic": "Dynamic borrowing",
        "title": (
            "A bayesian dynamic borrowing framework for improving the efficiency "
            "of clinical trials"
        ),
        "journal": "BMC Medical Research Methodology",
        "year": 2025,
        "url": "https://link.springer.com/article/10.1186/s12874-025-02691-2",
        "key_takeaway": (
            "Multi-source dynamic borrowing with consistency-weighted discounting can "
            "improve efficiency while controlling type I error under heterogeneity."
        ),
    },
    {
        "topic": "Non-concurrent controls",
        "title": "Modeling time trends in platform trials including non-concurrent controls",
        "journal": "Biometrical Journal",
        "year": 2025,
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12458466/",
        "key_takeaway": (
            "Adjusting for time trends can reduce bias from non-concurrent controls, with "
            "spline or logistic time models outperforming unadjusted analyses."
        ),
    },
    {
        "topic": "Adaptive platform operations",
        "title": "Backfilling in adaptive platform trials",
        "journal": "BMJ",
        "year": 2024,
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11698248/",
        "key_takeaway": (
            "Adaptive platform operations require simulation to evaluate randomization, "
            "backfilling, and potential bias before deployment."
        ),
    },
    {
        "topic": "Transportability",
        "title": (
            "Transporting trial results to synthetic real-world populations in order to estimate "
            "real-world effectiveness of newly marketed medicines"
        ),
        "journal": "BMJ Open",
        "year": 2025,
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12306361/",
        "key_takeaway": (
            "Generalizability/transportability analyses plus synthetic populations provide "
            "actionable external-validity diagnostics before mature RWD accrues."
        ),
    },
    {
        "topic": "Bayesian simulation tooling",
        "title": (
            "A fast, flexible simulation framework for Bayesian adaptive designs "
            "(BATSS package)"
        ),
        "year": 2024,
        "url": "https://arxiv.org/abs/2410.02050",
        "key_takeaway": (
            "Bayesian pre-trial simulation at scale enables stress-testing adaptive rules and "
            "operational constraints before trial launch."
        ),
    },
    {
        "topic": "Adaptive stopping + BRAR",
        "title": (
            "Bayesian Optimal Phase II design with optimised stopping boundaries and "
            "response-adaptive randomisation"
        ),
        "year": 2025,
        "url": "https://arxiv.org/abs/2511.14714",
        "key_takeaway": (
            "Joint optimization of interim timing, stopping rules, and response-adaptive "
            "randomization can improve allocation efficiency with limited power loss."
        ),
    },
]


def list_references() -> list[dict[str, Any]]:
    return [dict(item) for item in RESEARCH_REFERENCES]
