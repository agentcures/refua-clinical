"""Estimand and intercurrent-event handling utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import EstimandSpec


def apply_estimand(
    observed: pd.DataFrame,
    *,
    control_arm_id: str,
    estimand: EstimandSpec,
) -> pd.DataFrame:
    if observed.empty:
        return observed.copy()

    frame = observed.copy()
    if "rescue_use" not in frame.columns:
        frame["rescue_use"] = False

    frame["analysis_value"] = frame["endpoint_value"].astype(float)
    frame["analysis_responder"] = frame["responder"].astype(bool)

    strategy = estimand.strategy

    if strategy == "treatment_policy":
        return frame

    if strategy == "while_on_treatment":
        return frame.loc[~frame["dropped_out"].astype(bool)].copy()

    if strategy == "hypothetical":
        return _apply_hypothetical(frame, control_arm_id=control_arm_id, estimand=estimand)

    if strategy == "composite":
        return _apply_composite(frame, estimand=estimand)

    return frame


def estimand_summary(analysis_frame: pd.DataFrame) -> dict[str, Any]:
    if analysis_frame.empty:
        return {
            "n": 0,
            "mean_endpoint": float("nan"),
            "responder_rate": float("nan"),
            "dropout_rate": float("nan"),
            "rescue_rate": float("nan"),
        }

    return {
        "n": int(len(analysis_frame)),
        "mean_endpoint": float(analysis_frame["analysis_value"].mean()),
        "responder_rate": float(analysis_frame["analysis_responder"].mean()),
        "dropout_rate": float(analysis_frame["dropped_out"].mean()),
        "rescue_rate": float(analysis_frame["rescue_use"].mean()),
    }


def _apply_hypothetical(
    frame: pd.DataFrame,
    *,
    control_arm_id: str,
    estimand: EstimandSpec,
) -> pd.DataFrame:
    out = frame.copy()
    dropped = out["dropped_out"].astype(bool)

    control = out[out["arm_id"] == control_arm_id]
    if control.empty:
        control_mean = float(out["endpoint_value"].mean())
    else:
        control_mean = float(control["endpoint_value"].mean())

    imputed_value = control_mean + float(estimand.control_imputation_shift)
    out.loc[dropped, "analysis_value"] = imputed_value
    out.loc[dropped, "analysis_responder"] = (
        out.loc[dropped, "analysis_value"] >= out.loc[dropped, "analysis_value"].median()
    )
    return out


def _apply_composite(frame: pd.DataFrame, *, estimand: EstimandSpec) -> pd.DataFrame:
    out = frame.copy()
    intercurrent = out["dropped_out"].astype(bool) | out["rescue_use"].astype(bool)
    out.loc[intercurrent, "analysis_value"] = (
        out.loc[intercurrent, "analysis_value"].astype(float) - float(estimand.rescue_penalty)
    )
    out.loc[intercurrent, "analysis_responder"] = False
    return out


def arm_level_analysis(
    analysis_frame: pd.DataFrame,
    *,
    value_col: str = "analysis_value",
) -> dict[str, dict[str, float]]:
    by_arm: dict[str, dict[str, float]] = {}
    if analysis_frame.empty:
        return by_arm

    for arm_id, group in analysis_frame.groupby("arm_id"):
        values = group[value_col].to_numpy(dtype=float)
        by_arm[str(arm_id)] = {
            "n": float(len(values)),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "responder_rate": float(group["analysis_responder"].mean()),
        }
    return by_arm
