"""Optional integrations with other Refua packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import VirtualPopulationSpec
from .virtual_patients import infer_population_spec_from_dataframe


def infer_population_from_refua_data(
    *,
    dataset_id: str,
    size: int,
    max_rows: int = 25_000,
    columns: list[str] | None = None,
    cache_root: Path | None = None,
) -> VirtualPopulationSpec:
    """Infer a virtual population specification from a refua-data dataset."""
    try:
        from refua_data import DataCache, DatasetManager  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "refua-data is not installed. "
            "Install refua-clinical[integrations] to enable this feature."
        ) from exc

    if cache_root is not None:
        manager = DatasetManager(cache=DataCache(cache_root))
    else:
        manager = DatasetManager()

    materialized = manager.materialize(dataset_id)
    parts = sorted(materialized.parts)
    if not parts:
        raise ValueError(f"Dataset '{dataset_id}' materialized with no parquet parts")

    chunks: list[pd.DataFrame] = []
    total = 0
    for part in parts:
        chunk = pd.read_parquet(part)
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_rows:
            break

    frame = pd.concat(chunks, ignore_index=True)
    if len(frame) > max_rows:
        frame = frame.iloc[:max_rows].copy()

    return infer_population_spec_from_dataframe(frame, size=size, columns=columns)


def build_refua_regulatory_bundle(
    *,
    run_artifact_path: Path,
    output_dir: Path,
    data_manifest_paths: list[Path] | None = None,
    model_name: str = "refua-clinical",
    model_version: str | None = None,
) -> dict[str, Any]:
    """Wrap a simulation run artifact into a refua-regulatory bundle."""
    try:
        from refua_regulatory.bundle import build_evidence_bundle  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "refua-regulatory is not installed. "
            "Install refua-clinical[integrations] to enable this feature."
        ) from exc

    payload = json.loads(Path(run_artifact_path).read_text(encoding="utf-8"))
    objective = str(
        payload.get("config", {}).get("objective", "Clinical trial simulation")
    )
    run_id = str(payload.get("run_id", "refua-clinical-run"))
    summary = payload.get("summary", {})

    campaign_payload = {
        "campaign_run_id": run_id,
        "objective": objective,
        "plan": {
            "calls": [
                {"tool": "refua_clinical_simulate", "args": payload.get("config", {})},
                {
                    "tool": "refua_clinical_protocol",
                    "args": {
                        "target_power": summary.get("power"),
                        "mean_effect": summary.get("mean_effect"),
                    },
                },
            ]
        },
        "results": [
            {
                "tool": "refua_clinical_simulate",
                "args": payload.get("config", {}),
                "output": {
                    "model_version": model_version,
                    "summary": summary,
                },
            }
        ],
        "warnings": payload.get("warnings", []),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_run_path = output_dir / "_clinical_campaign_run.json"
    campaign_run_path.write_text(
        json.dumps(campaign_payload, indent=2) + "\n", encoding="utf-8"
    )

    manifests = list(data_manifest_paths or [])
    return build_evidence_bundle(
        campaign_run_path=campaign_run_path,
        output_dir=output_dir,
        source_kind="refua-clinical",
        data_manifest_paths=manifests,
        extra_artifacts=[Path(run_artifact_path)],
        model_name=model_name,
        model_version=model_version,
        overwrite=True,
    )
