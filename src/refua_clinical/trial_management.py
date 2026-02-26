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
from datetime import UTC, datetime
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
        raise ValueError(f"Unsupported trial status '{status}'. Allowed values: {allowed}")
    return normalized


def _normalize_source(source: str | None) -> str:
    normalized = str(source or "").strip().lower()
    if not normalized:
        normalized = "human"
    if normalized not in _ALLOWED_PATIENT_SOURCES:
        allowed = ", ".join(sorted(_ALLOWED_PATIENT_SOURCES))
        raise ValueError(f"Unsupported patient source '{source}'. Allowed values: {allowed}")
    return normalized


def _trial_summary(trial: dict[str, Any]) -> dict[str, Any]:
    management = _build_management_summary(trial)
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
        arm_id: float(np.mean(values))
        for arm_id, values in by_arm.items()
        if values
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
        "result_count": int(len(results)),
        "result_count_numeric": int(len(numeric_values)),
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
    if expected_sample_size is not None and expected_sample_size > 0 and observed_n is not None:
        weight = float(np.clip(observed_n / expected_sample_size, 0.0, 1.0))

    blended_effect: float | None = None
    if simulated_effect is not None and observed_effect is not None:
        blended_effect = float((1.0 - weight) * simulated_effect + weight * observed_effect)
    elif simulated_effect is not None:
        blended_effect = simulated_effect
    elif observed_effect is not None:
        blended_effect = observed_effect

    blended["observed_effect_estimate"] = observed_effect
    blended["observed_data_weight"] = weight
    blended["blended_effect_estimate"] = blended_effect
    blended["patient_count_human"] = management_summary.get("patient_count_human")
    blended["patient_count_simulated"] = management_summary.get("patient_count_simulated")
    blended["observed_response_rate"] = management_summary.get("observed_response_rate")
    blended["observed_safety_event_rate"] = management_summary.get("observed_safety_event_rate")

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
            summaries = [_trial_summary(_extract_nested_mapping(item)) for item in trials]
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
            payload = _json_clone(trial)
            payload["management_summary"] = _build_management_summary(trial)
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
            config_obj.indication = str(trial.get("indication") or config_obj.indication)
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
        demographics: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_source = _normalize_source(source)
        resolved_patient_id = str(patient_id).strip() if patient_id else ""
        if not resolved_patient_id:
            resolved_patient_id = f"pt-{uuid.uuid4().hex[:10]}"

        now = _utc_now_iso()

        with self._lock:
            store = self._load_store_unlocked()
            trial = self._find_trial_unlocked(store, trial_id)
            if trial is None:
                raise KeyError(trial_id)

            patients = trial.setdefault("patients", [])
            if not isinstance(patients, list):
                raise ValueError("Store payload is corrupted: trial.patients must be a list")

            existing: dict[str, Any] | None = None
            for item in patients:
                if isinstance(item, dict) and str(item.get("patient_id")) == resolved_patient_id:
                    existing = item
                    break

            payload = {
                "patient_id": resolved_patient_id,
                "source": normalized_source,
                "arm_id": arm_id,
                "demographics": _extract_nested_mapping(demographics),
                "baseline": _extract_nested_mapping(baseline),
                "metadata": _extract_nested_mapping(metadata),
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
                raise ValueError("Store payload is corrupted: trial.patients must be a list")

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

            patients = trial.setdefault("patients", [])
            if not isinstance(patients, list):
                raise ValueError("Store payload is corrupted: trial.patients must be a list")

            patient_payload: dict[str, Any] | None = None
            for item in patients:
                if isinstance(item, dict) and str(item.get("patient_id")) == resolved_patient_id:
                    patient_payload = item
                    break

            if patient_payload is None:
                inferred_source = _normalize_source(source)
                patient_payload = {
                    "patient_id": resolved_patient_id,
                    "source": inferred_source,
                    "arm_id": values.get("arm_id"),
                    "demographics": {},
                    "baseline": {},
                    "metadata": {"auto_enrolled": True},
                    "enrolled_at": now,
                    "updated_at": now,
                }
                patients.append(patient_payload)

            resolved_source = _normalize_source(source or str(patient_payload.get("source", "human")))
            arm_id = values.get("arm_id") or patient_payload.get("arm_id")

            result_payload = {
                "result_id": f"res-{uuid.uuid4().hex[:12]}",
                "patient_id": resolved_patient_id,
                "source": resolved_source,
                "arm_id": arm_id,
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
                raise ValueError("Store payload is corrupted: trial.results must be a list")
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
            run_payload["summary"] = _blend_simulation_summary(
                _extract_nested_mapping(run_payload.get("summary")),
                management_summary,
            )
            run_payload["management"] = management_summary

            simulations = trial.setdefault("simulations", [])
            if not isinstance(simulations, list):
                raise ValueError("Store payload is corrupted: trial.simulations must be a list")

            simulations.append(
                {
                    "run_id": run_payload.get("run_id"),
                    "created_at": run_payload.get("created_at") or _utc_now_iso(),
                    "summary": _extract_nested_mapping(run_payload.get("summary")),
                    "management": management_summary,
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
    def _find_trial_unlocked(store: dict[str, Any], trial_id: str) -> dict[str, Any] | None:
        trials = store.get("trials")
        if not isinstance(trials, list):
            return None
        for item in trials:
            if isinstance(item, dict) and str(item.get("trial_id")) == str(trial_id):
                return item
        return None


def _coerce_numpy_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value
