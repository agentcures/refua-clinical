"""Clinical trial registry and management framework.

This module extends refua-clinical from simulation-only workflows to a practical
trial-management layer that can:

- create/update/remove managed trials,
- enroll human and simulated patients,
- ingest observed outcomes,
- rerun simulations and blend observed data into model summaries.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .io import config_from_mapping, config_to_mapping, merge_mappings
from .models import SimulationConfig, default_simulation_config
from .trial import simulate_trials, trial_result_to_mapping
from .virtual_patients import generate_virtual_population

_STORE_VERSION = 1
_ALLOWED_PATIENT_SOURCES = {"human", "simulated"}
_ALLOWED_TRIAL_STATUSES = {
    "draft",
    "planned",
    "enrolling",
    "active",
    "completed",
    "paused",
    "terminated",
}
_ALLOWED_SITE_STATUSES = {
    "planned",
    "initiated",
    "active",
    "paused",
    "closed",
}
_ALLOWED_SCREENING_STATUSES = {
    "screened",
    "eligible",
    "screen_failed",
    "randomized",
    "enrolled",
}
_ALLOWED_MONITORING_VISIT_TYPES = {
    "selection",
    "initiation",
    "interim",
    "for_cause",
    "closeout",
    "remote",
}
_ALLOWED_QUERY_STATUSES = {
    "open",
    "answered",
    "resolved",
    "cancelled",
}
_ALLOWED_DEVIATION_SEVERITIES = {"minor", "major", "critical"}
_ALLOWED_DEVIATION_STATUSES = {"open", "in_review", "resolved", "closed"}
_ALLOWED_SAFETY_SERIOUSNESS = {"non_serious", "serious"}
_ALLOWED_MILESTONE_STATUSES = {"planned", "at_risk", "completed", "missed"}


def default_trial_store_path(base_dir: str | Path | None = None) -> Path:
    """Resolve the default trial-store path.

    Environment override:
    - ``REFUA_CLINICAL_TRIAL_STORE``
    """
    env_path = os.getenv("REFUA_CLINICAL_TRIAL_STORE")
    if env_path:
        return Path(env_path).expanduser().resolve()

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root.resolve() / ".refua-clinical" / "trial_registry.json"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True))


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
        return None
    return None


def _extract_nested_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_status(status: str | None) -> str:
    if status is None:
        return "draft"
    normalized = str(status).strip().lower()
    if not normalized:
        return "draft"
    if normalized not in _ALLOWED_TRIAL_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_TRIAL_STATUSES))
        raise ValueError(
            f"Unsupported trial status '{status}'. Allowed values: {allowed}"
        )
    return normalized


def _normalize_source(source: str | None) -> str:
    normalized = str(source or "").strip().lower()
    if not normalized:
        normalized = "human"
    if normalized not in _ALLOWED_PATIENT_SOURCES:
        allowed = ", ".join(sorted(_ALLOWED_PATIENT_SOURCES))
        raise ValueError(
            f"Unsupported patient source '{source}'. Allowed values: {allowed}"
        )
    return normalized


def _normalize_choice(
    value: str | None,
    *,
    default: str,
    allowed: set[str],
    name: str,
) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        normalized = default
    if normalized not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported {name} '{value}'. Allowed values: {allowed_values}"
        )
    return normalized


def _normalize_site_status(status: str | None) -> str:
    return _normalize_choice(
        status,
        default="planned",
        allowed=_ALLOWED_SITE_STATUSES,
        name="site status",
    )


def _normalize_screening_status(status: str | None) -> str:
    return _normalize_choice(
        status,
        default="screened",
        allowed=_ALLOWED_SCREENING_STATUSES,
        name="screening status",
    )


def _normalize_monitoring_visit_type(visit_type: str | None) -> str:
    return _normalize_choice(
        visit_type,
        default="interim",
        allowed=_ALLOWED_MONITORING_VISIT_TYPES,
        name="monitoring visit type",
    )


def _normalize_query_status(status: str | None) -> str:
    return _normalize_choice(
        status,
        default="open",
        allowed=_ALLOWED_QUERY_STATUSES,
        name="query status",
    )


def _normalize_deviation_severity(severity: str | None) -> str:
    return _normalize_choice(
        severity,
        default="minor",
        allowed=_ALLOWED_DEVIATION_SEVERITIES,
        name="deviation severity",
    )


def _normalize_deviation_status(status: str | None) -> str:
    return _normalize_choice(
        status,
        default="open",
        allowed=_ALLOWED_DEVIATION_STATUSES,
        name="deviation status",
    )


def _normalize_safety_seriousness(seriousness: str | None) -> str:
    return _normalize_choice(
        seriousness,
        default="non_serious",
        allowed=_ALLOWED_SAFETY_SERIOUSNESS,
        name="safety seriousness",
    )


def _normalize_milestone_status(status: str | None) -> str:
    return _normalize_choice(
        status,
        default="planned",
        allowed=_ALLOWED_MILESTONE_STATUSES,
        name="milestone status",
    )


def _normalize_required_id(value: str | None, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                output.append(text)
    return output


def _ensure_trial_clinops_collections(trial: dict[str, Any]) -> None:
    list_fields = (
        "sites",
        "screenings",
        "monitoring_visits",
        "queries",
        "deviations",
        "safety_events",
        "milestones",
    )
    for key in list_fields:
        if not isinstance(trial.get(key), list):
            trial[key] = []


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _patient_site_id(patient: dict[str, Any]) -> str | None:
    direct = _safe_text(patient.get("site_id"))
    if direct is not None:
        return direct
    metadata = _extract_nested_mapping(patient.get("metadata"))
    return _safe_text(metadata.get("site_id"))


def _site_enrollment_counts(trial: dict[str, Any]) -> dict[str, int]:
    patients = trial.get("patients")
    if not isinstance(patients, list):
        return {}
    counts: dict[str, int] = {}
    for item in patients:
        if not isinstance(item, dict):
            continue
        if str(item.get("source", "human")).lower() != "human":
            continue
        site_id = _patient_site_id(item)
        if site_id is None:
            continue
        counts[site_id] = counts.get(site_id, 0) + 1
    return counts


def _build_clinops_summary(trial: dict[str, Any]) -> dict[str, Any]:
    _ensure_trial_clinops_collections(trial)

    now = datetime.now(UTC)
    sites = trial.get("sites", [])
    screenings = trial.get("screenings", [])
    monitoring_visits = trial.get("monitoring_visits", [])
    queries = trial.get("queries", [])
    deviations = trial.get("deviations", [])
    safety_events = trial.get("safety_events", [])
    milestones = trial.get("milestones", [])
    management = _build_management_summary(trial)
    config = _config_from_trial(trial)

    active_site_statuses = {"initiated", "active"}
    active_sites = [
        site
        for site in sites
        if isinstance(site, dict)
        and _normalize_site_status(str(site.get("status") or "planned"))
        in active_site_statuses
    ]

    enrollment_counts = _site_enrollment_counts(trial)
    screened_count = 0
    screen_fail_count = 0
    randomized_count = 0
    for item in screenings:
        if not isinstance(item, dict):
            continue
        status = _normalize_screening_status(str(item.get("status") or "screened"))
        screened_count += 1
        if status == "screen_failed":
            screen_fail_count += 1
        if status in {"randomized", "enrolled"}:
            randomized_count += 1

    open_queries = 0
    overdue_queries = 0
    for query in queries:
        if not isinstance(query, dict):
            continue
        status = _normalize_query_status(str(query.get("status") or "open"))
        if status in {"resolved", "cancelled"}:
            continue
        open_queries += 1
        due_at = _parse_iso_datetime(query.get("due_at"))
        if due_at is not None and due_at < now:
            overdue_queries += 1

    major_deviations = 0
    unresolved_deviations = 0
    for deviation in deviations:
        if not isinstance(deviation, dict):
            continue
        severity = _normalize_deviation_severity(
            str(deviation.get("severity") or "minor")
        )
        status = _normalize_deviation_status(str(deviation.get("status") or "open"))
        if severity in {"major", "critical"}:
            major_deviations += 1
        if status not in {"resolved", "closed"}:
            unresolved_deviations += 1

    serious_safety_events = 0
    for event in safety_events:
        if not isinstance(event, dict):
            continue
        seriousness = _normalize_safety_seriousness(
            str(event.get("seriousness") or "non_serious")
        )
        if seriousness == "serious" or bool(event.get("is_sae")):
            serious_safety_events += 1

    open_action_items = 0
    latest_risk_by_site: dict[str, float] = {}
    for visit in monitoring_visits:
        if not isinstance(visit, dict):
            continue
        site_id = _safe_text(visit.get("site_id"))
        if site_id is None:
            continue
        risk_score = _coerce_float(visit.get("risk_score"))
        if risk_score is not None:
            latest_risk_by_site[site_id] = risk_score

        action_items = visit.get("action_items")
        if isinstance(action_items, list):
            for item in action_items:
                if isinstance(item, dict):
                    if not bool(item.get("completed")):
                        open_action_items += 1
                elif isinstance(item, str) and item.strip():
                    open_action_items += 1

    overdue_milestones = 0
    completed_milestones = 0
    for milestone in milestones:
        if not isinstance(milestone, dict):
            continue
        status = _normalize_milestone_status(str(milestone.get("status") or "planned"))
        if status == "completed":
            completed_milestones += 1
            continue
        target_date = _parse_iso_datetime(milestone.get("target_date"))
        if target_date is not None and target_date < now:
            overdue_milestones += 1

    planned_n = int(config.enrollment.total_n)
    enrolled_n = int(management.get("patient_count_human") or 0)
    enrollment_progress = float(enrolled_n / planned_n) if planned_n > 0 else 0.0

    human_patients = [
        item
        for item in (trial.get("patients") or [])
        if isinstance(item, dict)
        and str(item.get("source", "human")).lower() == "human"
    ]
    first_enrolled_at: datetime | None = None
    for patient in human_patients:
        enrolled_at = _parse_iso_datetime(patient.get("enrolled_at"))
        if enrolled_at is None:
            continue
        if first_enrolled_at is None or enrolled_at < first_enrolled_at:
            first_enrolled_at = enrolled_at

    enrollment_rate_per_day: float | None = None
    projected_completion_at: str | None = None
    if first_enrolled_at is not None and enrolled_n > 0:
        duration_days = max((now - first_enrolled_at).total_seconds() / 86400.0, 1.0)
        enrollment_rate_per_day = float(enrolled_n / duration_days)
        remaining_n = max(planned_n - enrolled_n, 0)
        if enrollment_rate_per_day > 0.0:
            projected_completion = now + timedelta(
                seconds=float(remaining_n / enrollment_rate_per_day * 86400.0)
            )
            projected_completion_at = projected_completion.isoformat()

    at_risk_sites: list[dict[str, Any]] = []
    for site in sites:
        if not isinstance(site, dict):
            continue
        site_id = _safe_text(site.get("site_id"))
        if site_id is None:
            continue
        risk = latest_risk_by_site.get(site_id, 0.0)
        if risk >= 0.7:
            at_risk_sites.append(
                {
                    "site_id": site_id,
                    "risk_score": float(risk),
                    "status": _normalize_site_status(
                        str(site.get("status") or "planned")
                    ),
                    "enrolled_human": int(enrollment_counts.get(site_id, 0)),
                }
            )
    at_risk_sites.sort(key=lambda item: item["risk_score"], reverse=True)

    return {
        "site_count": len(sites),
        "active_site_count": len(active_sites),
        "screened_count": int(screened_count),
        "screen_fail_count": int(screen_fail_count),
        "screen_fail_rate": (
            float(screen_fail_count / screened_count) if screened_count > 0 else None
        ),
        "randomized_count": int(randomized_count),
        "open_queries": int(open_queries),
        "overdue_queries": int(overdue_queries),
        "major_deviations": int(major_deviations),
        "unresolved_deviations": int(unresolved_deviations),
        "serious_safety_events": int(serious_safety_events),
        "open_monitoring_actions": int(open_action_items),
        "milestone_count": len(milestones),
        "completed_milestones": int(completed_milestones),
        "overdue_milestones": int(overdue_milestones),
        "planned_enrollment_human": int(planned_n),
        "enrolled_human": int(enrolled_n),
        "enrollment_progress_human": float(enrollment_progress),
        "enrollment_rate_human_per_day": enrollment_rate_per_day,
        "projected_completion_at": projected_completion_at,
        "site_enrollment_human": enrollment_counts,
        "at_risk_sites": at_risk_sites,
    }


def _trial_summary(trial: dict[str, Any]) -> dict[str, Any]:
    management = _build_management_summary(trial)
    clinops = _build_clinops_summary(trial)
    latest_simulation = _latest_simulation_summary(trial)
    return {
        "trial_id": trial["trial_id"],
        "indication": trial.get("indication"),
        "phase": trial.get("phase"),
        "objective": trial.get("objective"),
        "status": trial.get("status", "draft"),
        "created_at": trial.get("created_at"),
        "updated_at": trial.get("updated_at"),
        "patient_count": management["patient_count"],
        "patient_count_human": management["patient_count_human"],
        "patient_count_simulated": management["patient_count_simulated"],
        "result_count": management["result_count"],
        "site_count": clinops["site_count"],
        "active_site_count": clinops["active_site_count"],
        "open_query_count": clinops["open_queries"],
        "serious_safety_event_count": clinops["serious_safety_events"],
        "screen_fail_rate": clinops["screen_fail_rate"],
        "milestone_overdue_count": clinops["overdue_milestones"],
        "latest_simulation": latest_simulation,
    }


def _latest_simulation_summary(trial: dict[str, Any]) -> dict[str, Any] | None:
    simulations = trial.get("simulations")
    if not isinstance(simulations, list) or not simulations:
        return None

    latest = simulations[-1]
    if not isinstance(latest, dict):
        return None

    summary = _extract_nested_mapping(latest.get("summary"))
    return {
        "run_id": latest.get("run_id"),
        "created_at": latest.get("created_at"),
        "power": summary.get("power"),
        "mean_effect": summary.get("mean_effect"),
        "blended_effect_estimate": summary.get("blended_effect_estimate"),
    }


def _extract_result_bool(result: dict[str, Any], key: str) -> bool | None:
    direct = result.get(key)
    if isinstance(direct, bool):
        return direct
    values = _extract_nested_mapping(result.get("values"))
    nested = values.get(key)
    if isinstance(nested, bool):
        return nested
    return None


def _extract_result_numeric(result: dict[str, Any]) -> float | None:
    for key in ("change", "endpoint_value", "analysis_value", "value", "effect"):
        direct = _coerce_float(result.get(key))
        if direct is not None:
            return direct
        values = _extract_nested_mapping(result.get("values"))
        nested = _coerce_float(values.get(key))
        if nested is not None:
            return nested
    return None


def _extract_result_arm_id(result: dict[str, Any]) -> str | None:
    arm_id = result.get("arm_id")
    if isinstance(arm_id, str) and arm_id.strip():
        return arm_id.strip()
    values = _extract_nested_mapping(result.get("values"))
    nested = values.get("arm_id")
    if isinstance(nested, str) and nested.strip():
        return nested.strip()
    return None


def _config_from_trial(trial: dict[str, Any]) -> SimulationConfig:
    config_mapping = _extract_nested_mapping(trial.get("config"))
    return config_from_mapping(config_mapping)


def _build_management_summary(trial: dict[str, Any]) -> dict[str, Any]:
    patients = trial.get("patients")
    if not isinstance(patients, list):
        patients = []
    results = trial.get("results")
    if not isinstance(results, list):
        results = []

    patient_count_human = 0
    patient_count_simulated = 0
    for item in patients:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "human")).lower()
        if source == "simulated":
            patient_count_simulated += 1
        else:
            patient_count_human += 1

    numeric_values: list[float] = []
    responder_values: list[float] = []
    safety_values: list[float] = []
    by_arm: dict[str, list[float]] = {}

    for item in results:
        if not isinstance(item, dict):
            continue

        numeric = _extract_result_numeric(item)
        if numeric is not None:
            numeric_values.append(numeric)
            arm_id = _extract_result_arm_id(item)
            if arm_id is not None:
                by_arm.setdefault(arm_id, []).append(numeric)

        responder = _extract_result_bool(item, "responder")
        if responder is not None:
            responder_values.append(1.0 if responder else 0.0)

        safety = _extract_result_bool(item, "safety_event")
        if safety is not None:
            safety_values.append(1.0 if safety else 0.0)

    control_arm_id: str | None = None
    try:
        config = _config_from_trial(trial)
        for arm in config.arms:
            if arm.is_control:
                control_arm_id = arm.arm_id
                break
    except Exception:
        control_arm_id = None

    arm_means = {
        arm_id: float(np.mean(values)) for arm_id, values in by_arm.items() if values
    }

    observed_effect: float | None = None
    if control_arm_id is not None and control_arm_id in arm_means:
        treatment_means = [
            mean for arm_id, mean in arm_means.items() if arm_id != control_arm_id
        ]
        if treatment_means:
            observed_effect = float(max(treatment_means) - arm_means[control_arm_id])

    return {
        "patient_count": int(patient_count_human + patient_count_simulated),
        "patient_count_human": int(patient_count_human),
        "patient_count_simulated": int(patient_count_simulated),
        "result_count": len(results),
        "result_count_numeric": len(numeric_values),
        "observed_endpoint_mean": (
            float(np.mean(numeric_values)) if numeric_values else None
        ),
        "observed_response_rate": (
            float(np.mean(responder_values)) if responder_values else None
        ),
        "observed_safety_event_rate": (
            float(np.mean(safety_values)) if safety_values else None
        ),
        "control_arm_id": control_arm_id,
        "arm_mean_outcomes": arm_means,
        "observed_effect_estimate": observed_effect,
    }


def _blend_simulation_summary(
    summary: dict[str, Any],
    management_summary: dict[str, Any],
) -> dict[str, Any]:
    blended = dict(summary)

    simulated_effect = _coerce_float(summary.get("mean_effect"))
    observed_effect = _coerce_float(management_summary.get("observed_effect_estimate"))
    expected_sample_size = _coerce_float(summary.get("expected_sample_size"))
    observed_n = _coerce_float(management_summary.get("result_count_numeric"))

    weight = 0.0
    if (
        expected_sample_size is not None
        and expected_sample_size > 0
        and observed_n is not None
    ):
        weight = float(np.clip(observed_n / expected_sample_size, 0.0, 1.0))

    blended_effect: float | None = None
    if simulated_effect is not None and observed_effect is not None:
        blended_effect = float(
            (1.0 - weight) * simulated_effect + weight * observed_effect
        )
    elif simulated_effect is not None:
        blended_effect = simulated_effect
    elif observed_effect is not None:
        blended_effect = observed_effect

    blended["observed_effect_estimate"] = observed_effect
    blended["observed_data_weight"] = weight
    blended["blended_effect_estimate"] = blended_effect
    blended["patient_count_human"] = management_summary.get("patient_count_human")
    blended["patient_count_simulated"] = management_summary.get(
        "patient_count_simulated"
    )
    blended["observed_response_rate"] = management_summary.get("observed_response_rate")
    blended["observed_safety_event_rate"] = management_summary.get(
        "observed_safety_event_rate"
    )

    return blended


def _empty_store_payload() -> dict[str, Any]:
    return {
        "version": _STORE_VERSION,
        "trials": [],
        "updated_at": _utc_now_iso(),
    }


class ClinicalTrialManager:
    """Persistent manager for clinical trial operations."""

    def __init__(self, store_path: str | Path) -> None:
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_store_file()

    @property
    def store_path(self) -> Path:
        return self._store_path

    def list_trials(self) -> list[dict[str, Any]]:
        with self._lock:
            store = self._load_store_unlocked()
            trials = store.get("trials")
            if not isinstance(trials, list):
                return []
            summaries = [
                _trial_summary(_extract_nested_mapping(item)) for item in trials
            ]
        summaries.sort(
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )
        return summaries

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        needle = str(trial_id).strip()
        if not needle:
            raise ValueError("trial_id must be non-empty")

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, needle)
            if trial is None:
                return None
            _ensure_trial_clinops_collections(trial)
            payload = _json_clone(trial)
            payload["management_summary"] = _build_management_summary(trial)
            payload["clinops_summary"] = _build_clinops_summary(trial)
            payload["latest_simulation"] = _latest_simulation_summary(trial)
            return payload

    def create_trial(
        self,
        *,
        trial_id: str | None = None,
        config: dict[str, Any] | None = None,
        indication: str | None = None,
        phase: str | None = None,
        objective: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config_obj = self._resolve_config(
            trial_id=trial_id,
            config=config,
            indication=indication,
            phase=phase,
            objective=objective,
        )
        status_value = _normalize_status(status)
        metadata_payload = _extract_nested_mapping(metadata)
        now = _utc_now_iso()

        trial_payload = {
            "trial_id": config_obj.trial_id,
            "indication": config_obj.indication,
            "phase": config_obj.phase,
            "objective": config_obj.objective,
            "status": status_value,
            "metadata": metadata_payload,
            "config": config_to_mapping(config_obj),
            "patients": [],
            "results": [],
            "simulations": [],
            "sites": [],
            "screenings": [],
            "monitoring_visits": [],
            "queries": [],
            "deviations": [],
            "safety_events": [],
            "milestones": [],
            "created_at": now,
            "updated_at": now,
        }

        with self._lock:
            store = self._load_store_unlocked()
            existing = self._find_trial_unlocked(store, config_obj.trial_id)
            if existing is not None:
                raise ValueError(f"Trial '{config_obj.trial_id}' already exists")

            trials = store.setdefault("trials", [])
            if not isinstance(trials, list):
                raise ValueError("Store payload is corrupted: 'trials' must be a list")
            trials.append(trial_payload)
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(config_obj.trial_id),
            "store_path": str(self._store_path),
        }

    def update_trial(self, trial_id: str, *, updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict):
            raise ValueError("updates must be a JSON object")

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)
            _ensure_trial_clinops_collections(trial)
            _ensure_trial_clinops_collections(trial)

            if "status" in updates:
                trial["status"] = _normalize_status(str(updates.get("status")))

            for field in ("indication", "phase", "objective"):
                if field in updates and isinstance(updates[field], str):
                    trial[field] = str(updates[field]).strip()

            if "metadata" in updates:
                metadata_update = _extract_nested_mapping(updates.get("metadata"))
                existing_metadata = _extract_nested_mapping(trial.get("metadata"))
                trial["metadata"] = merge_mappings(existing_metadata, metadata_update)

            if "config" in updates:
                config_patch = _extract_nested_mapping(updates.get("config"))
                existing_config = _extract_nested_mapping(trial.get("config"))
                merged = merge_mappings(existing_config, config_patch)
                validated = config_from_mapping(merged)
                trial["config"] = config_to_mapping(validated)

            config_mapping = _extract_nested_mapping(trial.get("config"))
            config_obj = config_from_mapping(config_mapping)
            config_obj.indication = str(
                trial.get("indication") or config_obj.indication
            )
            config_obj.phase = str(trial.get("phase") or config_obj.phase)
            config_obj.objective = str(trial.get("objective") or config_obj.objective)
            trial["config"] = config_to_mapping(config_obj)
            trial["indication"] = config_obj.indication
            trial["phase"] = config_obj.phase
            trial["objective"] = config_obj.objective

            trial["updated_at"] = _utc_now_iso()
            store["updated_at"] = trial["updated_at"]
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "store_path": str(self._store_path),
        }

    def remove_trial(self, trial_id: str) -> dict[str, Any]:
        needle = str(trial_id).strip()
        if not needle:
            raise ValueError("trial_id must be non-empty")

        removed = False
        with self._lock:
            store = self._load_store_unlocked()
            trials = store.get("trials")
            if not isinstance(trials, list):
                raise ValueError("Store payload is corrupted: 'trials' must be a list")

            kept: list[dict[str, Any]] = []
            for item in trials:
                trial = _extract_nested_mapping(item)
                if str(trial.get("trial_id")) == needle:
                    removed = True
                    continue
                kept.append(trial)

            if removed:
                store["trials"] = kept
                store["updated_at"] = _utc_now_iso()
                self._save_store_unlocked(store)

        return {
            "trial_id": needle,
            "removed": removed,
            "store_path": str(self._store_path),
        }

    def enroll_patient(
        self,
        trial_id: str,
        *,
        patient_id: str | None = None,
        source: str | None = None,
        arm_id: str | None = None,
        site_id: str | None = None,
        demographics: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_source = _normalize_source(source)
        resolved_patient_id = str(patient_id).strip() if patient_id else ""
        if not resolved_patient_id:
            resolved_patient_id = f"pt-{uuid.uuid4().hex[:10]}"
        resolved_site_id = _safe_text(site_id)

        now = _utc_now_iso()

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            known_sites = {
                str(site.get("site_id"))
                for site in trial.get("sites", [])
                if isinstance(site, dict) and site.get("site_id")
            }
            if resolved_site_id is not None and resolved_site_id not in known_sites:
                raise ValueError(
                    f"Unknown site_id '{resolved_site_id}'. Add the site before enrolling patients."
                )

            patients = trial.setdefault("patients", [])
            if not isinstance(patients, list):
                raise ValueError(
                    "Store payload is corrupted: trial.patients must be a list"
                )

            existing: dict[str, Any] | None = None
            for item in patients:
                if (
                    isinstance(item, dict)
                    and str(item.get("patient_id")) == resolved_patient_id
                ):
                    existing = item
                    break

            merged_metadata = _extract_nested_mapping(metadata)
            if resolved_site_id is not None:
                merged_metadata["site_id"] = resolved_site_id
            payload = {
                "patient_id": resolved_patient_id,
                "source": normalized_source,
                "arm_id": arm_id,
                "site_id": resolved_site_id,
                "demographics": _extract_nested_mapping(demographics),
                "baseline": _extract_nested_mapping(baseline),
                "metadata": merged_metadata,
                "enrolled_at": now,
                "updated_at": now,
            }

            created = False
            if existing is None:
                patients.append(payload)
                created = True
            else:
                existing.update(payload)
                payload = existing

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "patient": _json_clone(payload),
            "created": created,
            "store_path": str(self._store_path),
        }

    def enroll_simulated_patients(
        self,
        trial_id: str,
        *,
        count: int,
        seed: int | None = None,
    ) -> dict[str, Any]:
        total = int(count)
        if total <= 0:
            raise ValueError("count must be >= 1")

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)

            config = _config_from_trial(trial)
            population = generate_virtual_population(
                config.population,
                seed=int(seed) if seed is not None else int(config.seed),
            )
            sample = population.table.head(total)

            patients = trial.setdefault("patients", [])
            if not isinstance(patients, list):
                raise ValueError(
                    "Store payload is corrupted: trial.patients must be a list"
                )

            existing_ids = {
                str(item.get("patient_id"))
                for item in patients
                if isinstance(item, dict) and item.get("patient_id")
            }

            now = _utc_now_iso()
            enrolled: list[str] = []
            next_counter = len(existing_ids) + 1

            for _, row in sample.iterrows():
                patient_id = f"sim-{trial_id}-{next_counter:05d}"
                while patient_id in existing_ids:
                    next_counter += 1
                    patient_id = f"sim-{trial_id}-{next_counter:05d}"

                existing_ids.add(patient_id)
                next_counter += 1

                row_map = {
                    str(column): _coerce_numpy_scalar(value)
                    for column, value in row.items()
                    if column != "patient_id"
                }

                patients.append(
                    {
                        "patient_id": patient_id,
                        "source": "simulated",
                        "arm_id": None,
                        "demographics": row_map,
                        "baseline": {},
                        "metadata": {"origin": "virtual_population"},
                        "enrolled_at": now,
                        "updated_at": now,
                    }
                )
                enrolled.append(patient_id)

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "enrolled_patient_ids": enrolled,
            "count": len(enrolled),
            "store_path": str(self._store_path),
        }

    def record_result(
        self,
        trial_id: str,
        *,
        patient_id: str,
        values: dict[str, Any],
        result_type: str = "endpoint",
        visit: str | None = None,
        source: str | None = None,
        site_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_patient_id = str(patient_id).strip()
        if not resolved_patient_id:
            raise ValueError("patient_id must be non-empty")
        if not isinstance(values, dict):
            raise ValueError("values must be a JSON object")

        now = _utc_now_iso()

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            known_sites = {
                str(site.get("site_id"))
                for site in trial.get("sites", [])
                if isinstance(site, dict) and site.get("site_id")
            }
            resolved_site_id = _safe_text(site_id)
            if resolved_site_id is not None and resolved_site_id not in known_sites:
                raise ValueError(
                    f"Unknown site_id '{resolved_site_id}'. Add the site before recording results."
                )

            patients = trial.setdefault("patients", [])
            if not isinstance(patients, list):
                raise ValueError(
                    "Store payload is corrupted: trial.patients must be a list"
                )

            patient_payload: dict[str, Any] | None = None
            for item in patients:
                if (
                    isinstance(item, dict)
                    and str(item.get("patient_id")) == resolved_patient_id
                ):
                    patient_payload = item
                    break

            if patient_payload is None:
                inferred_source = _normalize_source(source)
                auto_metadata: dict[str, Any] = {"auto_enrolled": True}
                if resolved_site_id is not None:
                    auto_metadata["site_id"] = resolved_site_id
                patient_payload = {
                    "patient_id": resolved_patient_id,
                    "source": inferred_source,
                    "arm_id": values.get("arm_id"),
                    "site_id": resolved_site_id,
                    "demographics": {},
                    "baseline": {},
                    "metadata": auto_metadata,
                    "enrolled_at": now,
                    "updated_at": now,
                }
                patients.append(patient_payload)

            resolved_source = _normalize_source(
                source or str(patient_payload.get("source", "human"))
            )
            arm_id = values.get("arm_id") or patient_payload.get("arm_id")
            resolved_site_id = (
                resolved_site_id
                or _safe_text(patient_payload.get("site_id"))
                or _patient_site_id(patient_payload)
            )

            result_payload = {
                "result_id": f"res-{uuid.uuid4().hex[:12]}",
                "patient_id": resolved_patient_id,
                "source": resolved_source,
                "arm_id": arm_id,
                "site_id": resolved_site_id,
                "result_type": str(result_type or "endpoint"),
                "visit": visit,
                "values": _json_clone(values),
                "recorded_at": now,
            }

            for key in ("change", "endpoint_value", "responder", "safety_event"):
                if key in values:
                    result_payload[key] = values[key]

            results = trial.setdefault("results", [])
            if not isinstance(results, list):
                raise ValueError(
                    "Store payload is corrupted: trial.results must be a list"
                )
            results.append(result_payload)

            patient_payload["updated_at"] = now
            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "result": _json_clone(result_payload),
            "store_path": str(self._store_path),
        }

    def simulate_trial(
        self,
        trial_id: str,
        *,
        replicates: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)

            config = _config_from_trial(trial)
            if replicates is not None:
                config.replicates = max(1, int(replicates))
            if seed is not None:
                config.seed = int(seed)

            result = simulate_trials(config)
            run_payload = trial_result_to_mapping(result)

            management_summary = _build_management_summary(trial)
            clinops_summary = _build_clinops_summary(trial)
            run_payload["summary"] = _blend_simulation_summary(
                _extract_nested_mapping(run_payload.get("summary")),
                management_summary,
            )
            run_payload["management"] = management_summary
            run_payload["clinops"] = clinops_summary

            simulations = trial.setdefault("simulations", [])
            if not isinstance(simulations, list):
                raise ValueError(
                    "Store payload is corrupted: trial.simulations must be a list"
                )

            simulations.append(
                {
                    "run_id": run_payload.get("run_id"),
                    "created_at": run_payload.get("created_at") or _utc_now_iso(),
                    "summary": _extract_nested_mapping(run_payload.get("summary")),
                    "management": management_summary,
                    "clinops": clinops_summary,
                    "result": run_payload,
                }
            )

            trial["status"] = "active"
            trial["updated_at"] = _utc_now_iso()
            store["updated_at"] = trial["updated_at"]
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "simulation": run_payload,
            "management": management_summary,
            "clinops": clinops_summary,
            "store_path": str(self._store_path),
        }

    def list_sites(self, trial_id: str) -> dict[str, Any]:
        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)
            trial_copy = _json_clone(trial)
            sites = [
                _json_clone(item)
                for item in trial_copy.get("sites", [])
                if isinstance(item, dict)
            ]
            sites.sort(key=lambda item: str(item.get("site_id") or ""))

        return {
            "trial_id": str(trial_id),
            "count": len(sites),
            "sites": sites,
            "clinops_summary": _build_clinops_summary(trial_copy),
            "store_path": str(self._store_path),
        }

    def upsert_site(
        self,
        trial_id: str,
        *,
        site_id: str,
        name: str | None = None,
        country_id: str | None = None,
        status: str | None = None,
        principal_investigator: str | None = None,
        target_enrollment: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_site_id = _normalize_required_id(site_id, field="site_id")
        now = _utc_now_iso()

        if target_enrollment is not None and int(target_enrollment) < 0:
            raise ValueError("target_enrollment must be >= 0 when provided")

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            sites = trial.setdefault("sites", [])
            if not isinstance(sites, list):
                raise ValueError(
                    "Store payload is corrupted: trial.sites must be a list"
                )

            existing = self._find_site_unlocked(trial, resolved_site_id)
            resolved_status = _normalize_site_status(
                str(existing.get("status") or "planned")
                if (status is None and existing is not None)
                else status
            )
            payload = {
                "site_id": resolved_site_id,
                "name": _safe_text(name),
                "country_id": _safe_text(country_id),
                "status": resolved_status,
                "principal_investigator": _safe_text(principal_investigator),
                "target_enrollment": (
                    int(target_enrollment) if target_enrollment is not None else None
                ),
                "metadata": _extract_nested_mapping(metadata),
                "updated_at": now,
            }
            if resolved_status in {"initiated", "active"}:
                payload["activated_at"] = now
            if resolved_status == "closed":
                payload["closed_at"] = now

            created = False
            if existing is None:
                payload["created_at"] = now
                sites.append(payload)
                created = True
            else:
                existing_meta = _extract_nested_mapping(existing.get("metadata"))
                payload["metadata"] = merge_mappings(
                    existing_meta,
                    _extract_nested_mapping(payload.get("metadata")),
                )
                if "created_at" in existing:
                    payload["created_at"] = existing.get("created_at")
                if "activated_at" not in payload and existing.get("activated_at"):
                    payload["activated_at"] = existing.get("activated_at")
                if "closed_at" not in payload and existing.get("closed_at"):
                    payload["closed_at"] = existing.get("closed_at")
                existing.update(payload)
                payload = existing

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "site": _json_clone(payload),
            "created": created,
            "store_path": str(self._store_path),
        }

    def record_screening(
        self,
        trial_id: str,
        *,
        site_id: str,
        patient_id: str | None = None,
        status: str | None = None,
        arm_id: str | None = None,
        source: str | None = None,
        failure_reason: str | None = None,
        demographics: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        auto_enroll: bool = False,
    ) -> dict[str, Any]:
        resolved_site_id = _normalize_required_id(site_id, field="site_id")
        resolved_status = _normalize_screening_status(status)
        resolved_source = _normalize_source(source)
        resolved_patient_id = (
            _normalize_required_id(patient_id, field="patient_id")
            if patient_id is not None
            else f"scr-{uuid.uuid4().hex[:10]}"
        )
        now = _utc_now_iso()

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            if self._find_site_unlocked(trial, resolved_site_id) is None:
                raise ValueError(
                    f"Unknown site_id '{resolved_site_id}'. Add the site before screening patients."
                )

            screenings = trial.setdefault("screenings", [])
            if not isinstance(screenings, list):
                raise ValueError(
                    "Store payload is corrupted: trial.screenings must be a list"
                )

            screening_payload = {
                "screening_id": f"scrn-{uuid.uuid4().hex[:12]}",
                "patient_id": resolved_patient_id,
                "site_id": resolved_site_id,
                "source": resolved_source,
                "status": resolved_status,
                "arm_id": _safe_text(arm_id),
                "failure_reason": _safe_text(failure_reason),
                "demographics": _extract_nested_mapping(demographics),
                "baseline": _extract_nested_mapping(baseline),
                "metadata": _extract_nested_mapping(metadata),
                "recorded_at": now,
            }
            screenings.append(screening_payload)

            patient_payload: dict[str, Any] | None = None
            should_enroll = auto_enroll or resolved_status == "enrolled"
            if should_enroll:
                patients = trial.setdefault("patients", [])
                if not isinstance(patients, list):
                    raise ValueError(
                        "Store payload is corrupted: trial.patients must be a list"
                    )

                existing: dict[str, Any] | None = None
                for item in patients:
                    if (
                        isinstance(item, dict)
                        and str(item.get("patient_id")) == resolved_patient_id
                    ):
                        existing = item
                        break

                patient_metadata = merge_mappings(
                    _extract_nested_mapping(metadata),
                    {
                        "screening_id": screening_payload["screening_id"],
                        "site_id": resolved_site_id,
                    },
                )
                patient_payload = {
                    "patient_id": resolved_patient_id,
                    "source": resolved_source,
                    "arm_id": _safe_text(arm_id),
                    "site_id": resolved_site_id,
                    "demographics": _extract_nested_mapping(demographics),
                    "baseline": _extract_nested_mapping(baseline),
                    "metadata": patient_metadata,
                    "enrolled_at": now,
                    "updated_at": now,
                }
                if existing is None:
                    patients.append(patient_payload)
                else:
                    existing.update(patient_payload)
                    patient_payload = existing

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "screening": _json_clone(screening_payload),
            "patient": (
                _json_clone(patient_payload) if patient_payload is not None else None
            ),
            "store_path": str(self._store_path),
        }

    def record_monitoring_visit(
        self,
        trial_id: str,
        *,
        site_id: str,
        visit_type: str | None = None,
        findings: list[str] | None = None,
        action_items: list[Any] | None = None,
        risk_score: float | None = None,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_site_id = _normalize_required_id(site_id, field="site_id")
        resolved_visit_type = _normalize_monitoring_visit_type(visit_type)
        risk_value = _coerce_float(risk_score)
        if risk_value is not None:
            risk_value = float(np.clip(risk_value, 0.0, 1.0))
        now = _utc_now_iso()

        normalized_actions: list[dict[str, Any]] = []
        if isinstance(action_items, list):
            for item in action_items:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        normalized_actions.append(
                            {"text": text, "completed": False, "owner": None}
                        )
                elif isinstance(item, dict):
                    item_text = _safe_text(item.get("text"))
                    if item_text is None:
                        continue
                    normalized_actions.append(
                        {
                            "text": item_text,
                            "completed": bool(item.get("completed", False)),
                            "owner": _safe_text(item.get("owner")),
                        }
                    )

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            if self._find_site_unlocked(trial, resolved_site_id) is None:
                raise ValueError(
                    "Unknown site_id "
                    f"'{resolved_site_id}'. Add the site before recording "
                    "monitoring visits."
                )

            visits = trial.setdefault("monitoring_visits", [])
            if not isinstance(visits, list):
                raise ValueError(
                    "Store payload is corrupted: trial.monitoring_visits must be a list"
                )

            visit_payload = {
                "visit_id": f"mv-{uuid.uuid4().hex[:12]}",
                "site_id": resolved_site_id,
                "visit_type": resolved_visit_type,
                "findings": _coerce_string_list(findings),
                "action_items": normalized_actions,
                "risk_score": risk_value,
                "outcome": _safe_text(outcome),
                "metadata": _extract_nested_mapping(metadata),
                "recorded_at": now,
            }
            visits.append(visit_payload)

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "monitoring_visit": _json_clone(visit_payload),
            "store_path": str(self._store_path),
        }

    def add_query(
        self,
        trial_id: str,
        *,
        patient_id: str | None = None,
        site_id: str | None = None,
        field_name: str | None = None,
        description: str,
        status: str | None = None,
        severity: str | None = None,
        assignee: str | None = None,
        due_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_description = _normalize_required_id(description, field="description")
        resolved_status = _normalize_query_status(status)
        resolved_patient_id = _safe_text(patient_id)
        resolved_site_id = _safe_text(site_id)
        resolved_due_at = _safe_text(due_at)
        now = _utc_now_iso()

        if resolved_patient_id is None and resolved_site_id is None:
            raise ValueError(
                "Either patient_id or site_id must be provided for a query"
            )

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            if (
                resolved_site_id is not None
                and self._find_site_unlocked(trial, resolved_site_id) is None
            ):
                raise ValueError(
                    f"Unknown site_id '{resolved_site_id}'. Add the site before opening queries."
                )

            queries = trial.setdefault("queries", [])
            if not isinstance(queries, list):
                raise ValueError(
                    "Store payload is corrupted: trial.queries must be a list"
                )

            query_payload = {
                "query_id": f"qry-{uuid.uuid4().hex[:12]}",
                "patient_id": resolved_patient_id,
                "site_id": resolved_site_id,
                "field_name": _safe_text(field_name),
                "description": resolved_description,
                "status": resolved_status,
                "severity": _safe_text(severity),
                "assignee": _safe_text(assignee),
                "due_at": resolved_due_at,
                "metadata": _extract_nested_mapping(metadata),
                "opened_at": now,
                "updated_at": now,
            }
            if resolved_status in {"resolved", "cancelled"}:
                query_payload["resolved_at"] = now
            queries.append(query_payload)

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "query": _json_clone(query_payload),
            "store_path": str(self._store_path),
        }

    def update_query(
        self,
        trial_id: str,
        *,
        query_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_query_id = _normalize_required_id(query_id, field="query_id")
        if not isinstance(updates, dict):
            raise ValueError("updates must be a JSON object")
        now = _utc_now_iso()

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            query = self._find_query_unlocked(trial, resolved_query_id)
            if query is None:
                raise KeyError(resolved_query_id)

            if "status" in updates:
                query["status"] = _normalize_query_status(str(updates.get("status")))
            for field in ("assignee", "resolution", "field_name", "description"):
                if field in updates:
                    query[field] = _safe_text(updates.get(field))
            if "due_at" in updates:
                query["due_at"] = _safe_text(updates.get("due_at"))
            if "severity" in updates:
                query["severity"] = _safe_text(updates.get("severity"))
            if "metadata" in updates:
                existing_meta = _extract_nested_mapping(query.get("metadata"))
                query["metadata"] = merge_mappings(
                    existing_meta,
                    _extract_nested_mapping(updates.get("metadata")),
                )
            if query.get("status") in {"resolved", "cancelled"} and not query.get(
                "resolved_at"
            ):
                query["resolved_at"] = now
            query["updated_at"] = now

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "query": _json_clone(query),
            "store_path": str(self._store_path),
        }

    def record_deviation(
        self,
        trial_id: str,
        *,
        description: str,
        site_id: str | None = None,
        patient_id: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        corrective_action: str | None = None,
        preventive_action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_description = _normalize_required_id(description, field="description")
        resolved_site_id = _safe_text(site_id)
        resolved_patient_id = _safe_text(patient_id)
        resolved_severity = _normalize_deviation_severity(severity)
        resolved_status = _normalize_deviation_status(status)
        now = _utc_now_iso()

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            if (
                resolved_site_id is not None
                and self._find_site_unlocked(trial, resolved_site_id) is None
            ):
                raise ValueError(
                    "Unknown site_id "
                    f"'{resolved_site_id}'. Add the site before recording "
                    "deviations."
                )

            deviations = trial.setdefault("deviations", [])
            if not isinstance(deviations, list):
                raise ValueError(
                    "Store payload is corrupted: trial.deviations must be a list"
                )

            deviation_payload = {
                "deviation_id": f"dev-{uuid.uuid4().hex[:12]}",
                "description": resolved_description,
                "site_id": resolved_site_id,
                "patient_id": resolved_patient_id,
                "category": _safe_text(category) or "protocol",
                "severity": resolved_severity,
                "status": resolved_status,
                "corrective_action": _safe_text(corrective_action),
                "preventive_action": _safe_text(preventive_action),
                "metadata": _extract_nested_mapping(metadata),
                "recorded_at": now,
                "updated_at": now,
            }
            if resolved_status in {"resolved", "closed"}:
                deviation_payload["closed_at"] = now
            deviations.append(deviation_payload)

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "deviation": _json_clone(deviation_payload),
            "store_path": str(self._store_path),
        }

    def record_safety_event(
        self,
        trial_id: str,
        *,
        patient_id: str,
        event_term: str,
        site_id: str | None = None,
        seriousness: str | None = None,
        expected: bool | None = None,
        relatedness: str | None = None,
        outcome: str | None = None,
        action_taken: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_patient_id = _normalize_required_id(patient_id, field="patient_id")
        resolved_event_term = _normalize_required_id(event_term, field="event_term")
        resolved_site_id = _safe_text(site_id)
        resolved_seriousness = _normalize_safety_seriousness(seriousness)
        expected_flag = bool(expected) if expected is not None else True
        now = _utc_now_iso()

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            patients = trial.setdefault("patients", [])
            if not isinstance(patients, list):
                raise ValueError(
                    "Store payload is corrupted: trial.patients must be a list"
                )
            patient_payload: dict[str, Any] | None = None
            for item in patients:
                if (
                    isinstance(item, dict)
                    and str(item.get("patient_id")) == resolved_patient_id
                ):
                    patient_payload = item
                    break
            if patient_payload is not None and resolved_site_id is None:
                resolved_site_id = _safe_text(
                    patient_payload.get("site_id")
                ) or _patient_site_id(patient_payload)

            if (
                resolved_site_id is not None
                and self._find_site_unlocked(trial, resolved_site_id) is None
            ):
                raise ValueError(
                    "Unknown site_id "
                    f"'{resolved_site_id}'. Add the site before recording "
                    "safety events."
                )

            events = trial.setdefault("safety_events", [])
            if not isinstance(events, list):
                raise ValueError(
                    "Store payload is corrupted: trial.safety_events must be a list"
                )

            is_sae = resolved_seriousness == "serious"
            event_payload = {
                "event_id": f"sae-{uuid.uuid4().hex[:12]}",
                "patient_id": resolved_patient_id,
                "site_id": resolved_site_id,
                "event_term": resolved_event_term,
                "seriousness": resolved_seriousness,
                "is_sae": is_sae,
                "expected": expected_flag,
                "relatedness": _safe_text(relatedness),
                "outcome": _safe_text(outcome),
                "action_taken": _safe_text(action_taken),
                "requires_expedited_report": bool(is_sae and not expected_flag),
                "metadata": _extract_nested_mapping(metadata),
                "recorded_at": now,
            }
            events.append(event_payload)

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "safety_event": _json_clone(event_payload),
            "store_path": str(self._store_path),
        }

    def upsert_milestone(
        self,
        trial_id: str,
        *,
        milestone_id: str | None = None,
        name: str | None = None,
        target_date: str | None = None,
        status: str | None = None,
        owner: str | None = None,
        actual_date: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_milestone_id = (
            _normalize_required_id(milestone_id, field="milestone_id")
            if milestone_id is not None
            else f"ms-{uuid.uuid4().hex[:10]}"
        )
        resolved_name = _safe_text(name)
        now = _utc_now_iso()
        resolved_actual_date = _safe_text(actual_date)

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)

            milestones = trial.setdefault("milestones", [])
            if not isinstance(milestones, list):
                raise ValueError(
                    "Store payload is corrupted: trial.milestones must be a list"
                )

            existing = self._find_milestone_unlocked(trial, resolved_milestone_id)
            if existing is None and resolved_name is None:
                raise ValueError("name is required when creating a milestone")
            resolved_status = _normalize_milestone_status(
                str(existing.get("status") or "planned")
                if (status is None and existing is not None)
                else status
            )
            if resolved_actual_date and status is None:
                resolved_status = "completed"

            payload = {
                "milestone_id": resolved_milestone_id,
                "name": resolved_name
                or _safe_text(existing.get("name") if existing else None),
                "target_date": _safe_text(target_date),
                "status": resolved_status,
                "owner": _safe_text(owner),
                "actual_date": resolved_actual_date,
                "metadata": _extract_nested_mapping(metadata),
                "updated_at": now,
            }
            if resolved_status == "completed" and payload.get("actual_date") is None:
                payload["actual_date"] = now
            if existing is None:
                payload["created_at"] = now
                milestones.append(payload)
                created = True
            else:
                existing_meta = _extract_nested_mapping(existing.get("metadata"))
                payload["metadata"] = merge_mappings(
                    existing_meta,
                    _extract_nested_mapping(payload.get("metadata")),
                )
                if existing.get("created_at"):
                    payload["created_at"] = existing.get("created_at")
                if payload.get("target_date") is None and existing.get("target_date"):
                    payload["target_date"] = existing.get("target_date")
                existing.update(payload)
                payload = existing
                created = False

            trial["updated_at"] = now
            store["updated_at"] = now
            self._save_store_unlocked(store)

        return {
            "trial": self.get_trial(trial_id),
            "milestone": _json_clone(payload),
            "created": created,
            "store_path": str(self._store_path),
        }

    def operations_snapshot(self, trial_id: str) -> dict[str, Any]:
        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)
            _ensure_trial_clinops_collections(trial)
            trial_copy = _json_clone(trial)

        return {
            "trial_id": str(trial_id),
            "updated_at": trial_copy.get("updated_at"),
            "management": _build_management_summary(trial_copy),
            "clinops": _build_clinops_summary(trial_copy),
            "latest_simulation": _latest_simulation_summary(trial_copy),
            "store_path": str(self._store_path),
        }

    def _resolve_config(
        self,
        *,
        trial_id: str | None,
        config: dict[str, Any] | None,
        indication: str | None,
        phase: str | None,
        objective: str | None,
    ) -> SimulationConfig:
        if config is None:
            cfg = default_simulation_config()
        else:
            cfg = config_from_mapping(config)

        if trial_id is not None and str(trial_id).strip():
            cfg.trial_id = str(trial_id).strip()
        if indication is not None and str(indication).strip():
            cfg.indication = str(indication).strip()
        if phase is not None and str(phase).strip():
            cfg.phase = str(phase).strip()
        if objective is not None and str(objective).strip():
            cfg.objective = str(objective).strip()

        cfg.trial_id = str(cfg.trial_id).strip()
        if not cfg.trial_id:
            raise ValueError("trial_id must be non-empty")

        return cfg

    def _ensure_store_file(self) -> None:
        with self._lock:
            if self._store_path.exists():
                _ = self._load_store_unlocked()
                return
            self._save_store_unlocked(_empty_store_payload())

    def _load_store_unlocked(self) -> dict[str, Any]:
        raw = self._store_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Trial store root payload must be an object")

        trials = payload.get("trials")
        if trials is None:
            payload["trials"] = []
        elif not isinstance(trials, list):
            raise ValueError("Trial store 'trials' field must be a list")

        version = payload.get("version")
        if version is None:
            payload["version"] = _STORE_VERSION
        return payload

    def _save_store_unlocked(self, payload: dict[str, Any]) -> None:
        payload = _json_clone(payload)
        payload["version"] = _STORE_VERSION
        if "updated_at" not in payload:
            payload["updated_at"] = _utc_now_iso()

        tmp_path = self._store_path.with_name(
            f"{self._store_path.name}.{uuid.uuid4().hex}.tmp"
        )
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self._store_path)

    @staticmethod
    def _find_trial_unlocked(
        store: dict[str, Any], trial_id: str
    ) -> dict[str, Any] | None:
        trials = store.get("trials")
        if not isinstance(trials, list):
            return None
        for item in trials:
            if isinstance(item, dict) and str(item.get("trial_id")) == str(trial_id):
                return item
        return None

    @staticmethod
    def _find_site_unlocked(
        trial: dict[str, Any], site_id: str
    ) -> dict[str, Any] | None:
        sites = trial.get("sites")
        if not isinstance(sites, list):
            return None
        for item in sites:
            if isinstance(item, dict) and str(item.get("site_id")) == str(site_id):
                return item
        return None

    @staticmethod
    def _find_query_unlocked(
        trial: dict[str, Any], query_id: str
    ) -> dict[str, Any] | None:
        queries = trial.get("queries")
        if not isinstance(queries, list):
            return None
        for item in queries:
            if isinstance(item, dict) and str(item.get("query_id")) == str(query_id):
                return item
        return None

    @staticmethod
    def _find_milestone_unlocked(
        trial: dict[str, Any],
        milestone_id: str,
    ) -> dict[str, Any] | None:
        milestones = trial.get("milestones")
        if not isinstance(milestones, list):
            return None
        for item in milestones:
            if isinstance(item, dict) and str(item.get("milestone_id")) == str(
                milestone_id
            ):
                return item
        return None


def _coerce_numpy_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value
