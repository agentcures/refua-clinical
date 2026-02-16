"""ADMET integration helpers for trial simulation and recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import config_from_mapping, config_to_mapping
from .models import SimulationConfig

_CRITICAL_SAFETY_FLAGS = {
    "hERG",
    "AMES",
    "DILI",
    "Carcinogens_Lagunin",
    "ClinTox",
    "Tox21_SR_p53",
    "Tox21_SR_MMP",
}


def load_admet_profile(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ADMET profile file must contain a JSON object")
    return dict(payload)


def compute_admet_profile(
    smiles: str,
    *,
    model_variant: str = "9b-chat",
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    try:
        from refua.admet import admet_profile as _admet_profile  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "refua is not installed. Install refua to compute ADMET from SMILES."
        ) from exc

    profile = _admet_profile(
        smiles,
        model_variant=model_variant,
        max_new_tokens=max_new_tokens,
        include_scoring=True,
    )
    if not isinstance(profile, dict):
        raise ValueError("Unexpected ADMET profile result")
    return profile


def summarize_admet_profile(profile: dict[str, Any]) -> dict[str, Any]:
    scores = profile.get("scores")
    score_map = dict(scores) if isinstance(scores, dict) else {}
    predictions = profile.get("predictions")
    prediction_map = dict(predictions) if isinstance(predictions, dict) else {}

    top_risks = sorted(
        (
            {
                "endpoint": key.replace("score_", ""),
                "score": float(value),
            }
            for key, value in score_map.items()
            if key.startswith("score_")
            and key not in {"score_admet"}
            and isinstance(value, int | float)
        ),
        key=lambda item: item["score"],
    )[:8]

    red_flags = [str(item) for item in profile.get("red_flags", []) if isinstance(item, str)]
    yellow_flags = [str(item) for item in profile.get("yellow_flags", []) if isinstance(item, str)]

    return {
        "smiles": profile.get("smiles"),
        "admet_score": float(profile.get("admet_score", 0.5)),
        "adme_score": float(profile.get("adme_score", 0.5)),
        "safety_score": float(profile.get("safety_score", 0.5)),
        "red_flags": red_flags,
        "yellow_flags": yellow_flags,
        "top_low_scores": top_risks,
        "num_predictions": int(profile.get("num_predictions", len(prediction_map))),
        "score_keys": sorted(score_map.keys()),
    }


def apply_admet_adjustments(
    config: SimulationConfig,
    profile: dict[str, Any],
) -> tuple[SimulationConfig, dict[str, Any]]:
    summary = summarize_admet_profile(profile)

    admet_score = float(summary["admet_score"])
    adme_score = float(summary["adme_score"])
    safety_score = float(summary["safety_score"])
    red_flags = {str(item) for item in summary["red_flags"]}

    scores = profile.get("scores")
    score_map = dict(scores) if isinstance(scores, dict) else {}
    bio_score = float(score_map.get("score_Bioavailability_Ma", adme_score))

    critical_count = len(red_flags.intersection(_CRITICAL_SAFETY_FLAGS))
    general_risk = max(0.0, 0.70 - admet_score)
    safety_risk = max(0.0, 0.70 - safety_score)
    exposure_risk = max(0.0, 0.70 - adme_score)

    bioavailability_factor = float(np.clip(0.55 + 0.90 * bio_score, 0.40, 1.20))
    ec50_multiplier = float(np.clip(1.0 + 0.8 * exposure_risk, 1.0, 1.6))
    safety_intercept_shift = float(
        np.clip(1.8 * safety_risk + 0.18 * critical_count + 0.12 * general_risk, 0.0, 1.8)
    )
    dropout_multiplier = float(
        np.clip(1.0 + 1.2 * safety_risk + 0.10 * critical_count + 0.25 * general_risk, 1.0, 1.9)
    )

    payload = config_to_mapping(config)
    payload["pk_model"]["bioavailability"] = float(
        payload["pk_model"]["bioavailability"] * bioavailability_factor
    )
    payload["pd_model"]["ec50_auc"] = float(payload["pd_model"]["ec50_auc"] * ec50_multiplier)
    payload["pd_model"]["safety_intercept"] = float(
        payload["pd_model"]["safety_intercept"] + safety_intercept_shift
    )
    payload["enrollment"]["dropout_base"] = float(
        np.clip(payload["enrollment"]["dropout_base"] * dropout_multiplier, 0.01, 0.95)
    )

    adjusted = config_from_mapping(payload)

    adjustments = {
        "bioavailability_factor": bioavailability_factor,
        "ec50_auc_multiplier": ec50_multiplier,
        "safety_intercept_shift": safety_intercept_shift,
        "dropout_base_multiplier": dropout_multiplier,
        "critical_flag_count": critical_count,
        "assumptions": {
            "low_adme_reduces_exposure": True,
            "low_safety_score_increases_ae_and_dropout": True,
        },
    }
    return adjusted, adjustments
