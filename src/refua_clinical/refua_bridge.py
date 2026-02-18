"""Bridge utilities to map Refua discovery outputs into clinical simulation inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np

from .admet_integration import apply_admet_adjustments, summarize_admet_profile
from .io import config_from_mapping, config_to_mapping, load_mapping
from .models import SimulationConfig

_ABSORPTION_ENDPOINTS = (
    "Bioavailability_Ma",
    "HIA_Hou",
    "Caco2_Wang",
    "PAMPA_NCATS",
    "Solubility_AqSolDB",
)
_DISTRIBUTION_ENDPOINTS = (
    "BBB_Martins",
    "VDss_Lombardo",
    "PPBR_AZ",
    "Pgp_Broccatelli",
)
_METABOLISM_ENDPOINTS = (
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP2C9_Veith",
    "CYP2C19_Veith",
    "CYP1A2_Veith",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "CYP2C9_Substrate_CarbonMangels",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "Half_Life_Obach",
)
_SAFETY_ENDPOINTS = (
    "hERG",
    "AMES",
    "DILI",
    "ClinTox",
    "Carcinogens_Lagunin",
    "Tox21_SR_MMP",
    "Tox21_SR_p53",
)

_LEGACY_ROOT_KEY_HINTS: dict[str, str] = {
    "ligand_rdkit": "Use 'ligands[].rdkit' for per-ligand chemistry descriptors.",
    "ligand_admet": "Use 'ligands[].admet' for per-ligand ADMET profiles.",
    "ligand_affinity": "Use 'ligands[].affinity' for per-ligand affinity signals.",
    "ligand_structure": "Use 'ligands[].structure' for per-ligand structure metrics.",
    "affinity": "Use 'ligands[].affinity' instead of root-level fallback affinity.",
    "structure": "Use 'ligands[].structure' instead of root-level fallback structure.",
    "admet": "Use 'ligands[].admet' instead of root-level fallback ADMET.",
    "rdkit": "Use 'ligands[].rdkit' instead of root-level fallback RDKit properties.",
    "protein_properties": "Use 'target_properties' for target-level property maps.",
}
_LEGACY_LIGAND_KEY_HINTS: dict[str, str] = {
    "id": "Use 'ligand_id' for ligand identifiers.",
    "chain_id": "Use 'ligand_id' for ligand identifiers.",
    "ligand_rdkit": "Use 'rdkit' for ligand descriptor blocks.",
    "properties": "Use 'rdkit' for ligand descriptor blocks.",
    "ligand_admet": "Use 'admet' for ligand ADMET payloads.",
    "admet_profile": "Use 'admet' for ligand ADMET payloads.",
}


@dataclass(slots=True)
class RefuaIntegrationPolicy:
    """Policy controls for translating Refua outputs into trial assumptions."""

    preferred_ligand_id: str | None = None
    include_admet_adjustments: bool = True
    include_endpoint_admet_adjustments: bool = True
    include_affinity_adjustments: bool = True
    include_structure_confidence_adjustments: bool = True
    include_rdkit_adjustments: bool = True
    include_target_adjustments: bool = True
    include_candidate_arms: bool = True
    max_candidate_arms: int = 4
    strict_contract: bool = False


def load_refua_payload(path: str | Path) -> dict[str, Any]:
    """Load a Refua payload from JSON/YAML."""
    payload = load_mapping(path)
    return dict(payload)


def assess_refua_payload_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Assess whether a payload matches the canonical Refua handoff contract."""
    legacy_root_keys = sorted(
        key for key in _LEGACY_ROOT_KEY_HINTS if key in payload
    )

    legacy_ligand_keys: set[str] = set()
    missing_ligand_id_count = 0
    invalid_ligand_entries = 0

    ligands_raw = payload.get("ligands")
    if isinstance(ligands_raw, list):
        ligand_count = len(ligands_raw)
        for raw in ligands_raw:
            if not isinstance(raw, dict):
                invalid_ligand_entries += 1
                continue
            if _read_str(raw, "ligand_id") is None:
                missing_ligand_id_count += 1
            for key in _LEGACY_LIGAND_KEY_HINTS:
                if key in raw:
                    legacy_ligand_keys.add(key)
    else:
        ligand_count = 0

    warnings: list[str] = []
    if not isinstance(ligands_raw, list):
        warnings.append("Missing canonical 'ligands' array.")

    for key in legacy_root_keys:
        warnings.append(f"Legacy root key '{key}' detected. {_LEGACY_ROOT_KEY_HINTS[key]}")
    for key in sorted(legacy_ligand_keys):
        warnings.append(f"Legacy ligand key '{key}' detected. {_LEGACY_LIGAND_KEY_HINTS[key]}")

    if missing_ligand_id_count:
        warnings.append(
            f"{missing_ligand_id_count} ligand entries are missing required 'ligand_id'."
        )
    if invalid_ligand_entries:
        warnings.append(
            f"{invalid_ligand_entries} ligand entries are not objects."
        )

    is_canonical = (
        isinstance(ligands_raw, list)
        and not legacy_root_keys
        and not legacy_ligand_keys
        and missing_ligand_id_count == 0
        and invalid_ligand_entries == 0
    )

    return {
        "schema": "refua_payload.v1",
        "is_canonical": is_canonical,
        "ligand_count": ligand_count,
        "legacy_root_keys": legacy_root_keys,
        "legacy_ligand_keys": sorted(legacy_ligand_keys),
        "missing_ligand_id_count": missing_ligand_id_count,
        "invalid_ligand_entries": invalid_ligand_entries,
        "warnings": warnings,
    }


def _enforce_contract(
    payload: dict[str, Any],
    *,
    strict_contract: bool,
) -> dict[str, Any]:
    contract = assess_refua_payload_contract(payload)
    if strict_contract and not bool(contract.get("is_canonical", False)):
        warnings = contract.get("warnings")
        reason = "; ".join(warnings[:4]) if isinstance(warnings, list) and warnings else ""
        suffix = f" {reason}" if reason else ""
        raise ValueError(f"Refua payload failed strict contract validation.{suffix}")
    return contract


def extract_admet_profile_from_refua_payload(
    payload: dict[str, Any],
    *,
    preferred_ligand_id: str | None = None,
    strict_contract: bool = False,
) -> dict[str, Any] | None:
    """Return the selected ligand ADMET profile from a Refua payload."""
    summary = summarize_refua_payload(
        payload,
        preferred_ligand_id=preferred_ligand_id,
        strict_contract=strict_contract,
    )
    selected = summary.get("selected_candidate")
    if not isinstance(selected, dict):
        return None
    admet_profile = selected.get("admet_profile")
    if not isinstance(admet_profile, dict):
        return None
    return dict(admet_profile)


def summarize_refua_payload(
    payload: dict[str, Any],
    *,
    preferred_ligand_id: str | None = None,
    strict_contract: bool = False,
) -> dict[str, Any]:
    """Summarize candidate ligands and select a default candidate."""
    contract = _enforce_contract(payload, strict_contract=strict_contract)
    candidates = _collect_ligand_candidates(payload)
    ranked = sorted(
        (_candidate_summary(candidate) for candidate in candidates),
        key=lambda item: float(item["candidate_score"]),
        reverse=True,
    )

    selected: dict[str, Any] | None = None
    preferred = preferred_ligand_id or _read_str(payload, "selected_ligand_id")
    if preferred is not None:
        selected = next(
            (
                item
                for item in ranked
                if str(item.get("ligand_id", "")).strip().lower() == preferred.strip().lower()
            ),
            None,
        )
    if selected is None and ranked:
        selected = ranked[0]

    target_properties = _mapping(payload.get("target_properties"))
    if not target_properties:
        target_properties = _mapping(payload.get("protein_properties"))

    return {
        "contract": contract,
        "candidate_count": len(ranked),
        "candidate_rankings": ranked,
        "selected_ligand_id": selected.get("ligand_id") if selected else None,
        "selected_candidate": selected,
        "target_properties": target_properties,
    }


def apply_refua_adjustments(
    config: SimulationConfig,
    payload: dict[str, Any],
    *,
    policy: RefuaIntegrationPolicy | None = None,
) -> tuple[SimulationConfig, dict[str, Any]]:
    """Apply broad Refua-informed adjustments to simulation assumptions."""
    integration_policy = policy or RefuaIntegrationPolicy()
    summary = summarize_refua_payload(
        payload,
        preferred_ligand_id=integration_policy.preferred_ligand_id,
        strict_contract=integration_policy.strict_contract,
    )
    selected = summary.get("selected_candidate")
    selected_candidate = _mapping(selected)

    working = config
    module_adjustments: dict[str, Any] = {}
    is_biologic_mode = str(working.pk_model.modality).strip().lower() == "biologic"
    allow_small_molecule_adjustments = not is_biologic_mode
    selected_admet = _mapping(selected_candidate.get("admet_profile"))
    if (
        integration_policy.include_admet_adjustments
        and selected_admet
        and allow_small_molecule_adjustments
    ):
        working, admet_adjustments = apply_admet_adjustments(working, selected_admet)
        module_adjustments["admet_base"] = admet_adjustments

    payload_map = config_to_mapping(working)
    if (
        integration_policy.include_endpoint_admet_adjustments
        and selected_admet
        and allow_small_molecule_adjustments
    ):
        endpoint_update = _apply_endpoint_admet_adjustments(payload_map, selected_admet)
        if endpoint_update:
            module_adjustments["admet_endpoints"] = endpoint_update

    selected_affinity = _mapping(selected_candidate.get("affinity"))
    if integration_policy.include_affinity_adjustments and selected_affinity:
        affinity_update = _apply_affinity_adjustments(payload_map, selected_affinity)
        if affinity_update:
            module_adjustments["affinity"] = affinity_update

    selected_structure = _mapping(selected_candidate.get("structure"))
    if integration_policy.include_structure_confidence_adjustments and selected_structure:
        structure_update = _apply_structure_adjustments(payload_map, selected_structure)
        if structure_update:
            module_adjustments["structure_confidence"] = structure_update

    selected_rdkit = _mapping(selected_candidate.get("rdkit"))
    if (
        integration_policy.include_rdkit_adjustments
        and selected_rdkit
        and allow_small_molecule_adjustments
    ):
        rdkit_update = _apply_rdkit_adjustments(payload_map, selected_rdkit)
        if rdkit_update:
            module_adjustments["rdkit"] = rdkit_update

    target_properties = _mapping(summary.get("target_properties"))
    if integration_policy.include_target_adjustments and target_properties:
        target_update = _apply_target_adjustments(payload_map, target_properties)
        if target_update:
            module_adjustments["target_properties"] = target_update

    if integration_policy.include_candidate_arms:
        arm_update = _apply_candidate_arm_overrides(
            payload_map,
            candidate_rankings=summary.get("candidate_rankings", []),
            max_candidate_arms=max(1, int(integration_policy.max_candidate_arms)),
        )
        if arm_update:
            module_adjustments["candidate_arms"] = arm_update

    if is_biologic_mode:
        module_adjustments["biologic_mode"] = {
            "admet_adjustments_skipped": True,
            "rdkit_adjustments_skipped": True,
        }

    adjusted = config_from_mapping(payload_map)
    management = _build_management_summary(summary, module_adjustments)
    report = {
        "policy": asdict(integration_policy),
        "summary": summary,
        "adjustments": module_adjustments,
        "management": management,
    }
    return adjusted, report


def _collect_ligand_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}

    def ensure(ligand_id: str) -> dict[str, Any]:
        key = _normalize_ligand_id(ligand_id)
        if key not in by_id:
            by_id[key] = {
                "ligand_id": key,
                "rdkit": {},
                "admet_profile": {},
                "affinity": {},
                "structure": {},
            }
        return by_id[key]

    ligands_raw = payload.get("ligands")
    if isinstance(ligands_raw, list):
        for index, raw in enumerate(ligands_raw):
            item = _mapping(raw)
            ligand_id = _normalize_ligand_id(
                _read_str(item, "ligand_id")
                or _read_str(item, "id")
                or _read_str(item, "chain_id")
                or f"ligand_{index + 1:03d}",
            )
            candidate = ensure(ligand_id)
            candidate["rdkit"] = _mapping(
                item.get("rdkit")
                or item.get("ligand_rdkit")
                or item.get("properties")
                or candidate["rdkit"]
            )
            candidate["admet_profile"] = _mapping(
                item.get("admet")
                or item.get("ligand_admet")
                or item.get("admet_profile")
                or candidate["admet_profile"]
            )
            candidate["affinity"] = _mapping(item.get("affinity") or candidate["affinity"])
            candidate["structure"] = _mapping(item.get("structure") or candidate["structure"])
            smiles = _read_str(item, "smiles")
            if smiles is not None:
                candidate["smiles"] = smiles

    rdkit_map = _mapping(payload.get("ligand_rdkit"))
    for ligand_id, raw in rdkit_map.items():
        candidate = ensure(str(ligand_id))
        candidate["rdkit"] = _mapping(raw)

    admet_map = _mapping(payload.get("ligand_admet"))
    for ligand_id, raw in admet_map.items():
        candidate = ensure(str(ligand_id))
        candidate["admet_profile"] = _mapping(raw)

    affinity_map = _mapping(payload.get("ligand_affinity"))
    for ligand_id, raw in affinity_map.items():
        candidate = ensure(str(ligand_id))
        candidate["affinity"] = _mapping(raw)

    structure_map = _mapping(payload.get("ligand_structure"))
    for ligand_id, raw in structure_map.items():
        candidate = ensure(str(ligand_id))
        candidate["structure"] = _mapping(raw)

    fallback_affinity = _mapping(payload.get("affinity"))
    fallback_structure = _mapping(payload.get("structure"))
    fallback_admet = _mapping(payload.get("admet"))
    fallback_rdkit = _mapping(payload.get("rdkit"))
    fallback_smiles = _read_str(payload, "smiles")

    if not by_id and (fallback_affinity or fallback_structure or fallback_admet or fallback_rdkit):
        default_id = _normalize_ligand_id(
            _read_str(payload, "ligand_id")
            or _read_str(payload, "selected_ligand_id")
            or "ligand_001"
        )
        by_id[default_id] = {
            "ligand_id": default_id,
            "rdkit": fallback_rdkit,
            "admet_profile": fallback_admet,
            "affinity": fallback_affinity,
            "structure": fallback_structure,
            "smiles": fallback_smiles,
        }
    elif len(by_id) == 1:
        only = next(iter(by_id.values()))
        if fallback_affinity and not only["affinity"]:
            only["affinity"] = fallback_affinity
        if fallback_structure and not only["structure"]:
            only["structure"] = fallback_structure
        if fallback_admet and not only["admet_profile"]:
            only["admet_profile"] = fallback_admet
        if fallback_rdkit and not only["rdkit"]:
            only["rdkit"] = fallback_rdkit
        if fallback_smiles and "smiles" not in only:
            only["smiles"] = fallback_smiles

    return list(by_id.values())


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    admet_profile = _mapping(candidate.get("admet_profile"))
    affinity = _mapping(candidate.get("affinity"))
    structure = _mapping(candidate.get("structure"))
    rdkit = _mapping(candidate.get("rdkit"))

    admet_summary = summarize_admet_profile(admet_profile) if admet_profile else None
    admet_score = (
        float(admet_summary["admet_score"])
        if isinstance(admet_summary, dict) and "admet_score" in admet_summary
        else 0.5
    )
    safety_score = (
        float(admet_summary["safety_score"])
        if isinstance(admet_summary, dict) and "safety_score" in admet_summary
        else 0.5
    )
    adme_score = (
        float(admet_summary["adme_score"])
        if isinstance(admet_summary, dict) and "adme_score" in admet_summary
        else 0.5
    )
    red_flags = admet_summary.get("red_flags", []) if isinstance(admet_summary, dict) else []
    red_flag_count = len(red_flags) if isinstance(red_flags, list) else 0

    binding_probability = _read_float(affinity, "binding_probability", default=0.5)
    ic50 = _read_float(affinity, "ic50")
    potency = _potency_score_from_ic50(ic50)
    confidence = _extract_structure_confidence(structure)
    quality = _read_float(rdkit, "qed", default=0.55)

    score = (
        0.26 * admet_score
        + 0.14 * safety_score
        + 0.08 * adme_score
        + 0.18 * binding_probability
        + 0.18 * potency
        + 0.10 * confidence
        + 0.06 * quality
        - 0.04 * float(min(red_flag_count, 5))
    )
    score = float(np.clip(score, 0.0, 1.0))
    return {
        "ligand_id": str(candidate.get("ligand_id", "ligand_001")),
        "smiles": candidate.get("smiles"),
        "candidate_score": score,
        "admet_profile": admet_profile if admet_profile else None,
        "admet_summary": admet_summary,
        "affinity": affinity if affinity else None,
        "structure": structure if structure else None,
        "rdkit": rdkit if rdkit else None,
        "signal_summary": {
            "admet_score": admet_score,
            "safety_score": safety_score,
            "adme_score": adme_score,
            "binding_probability": binding_probability,
            "potency_score": potency,
            "structure_confidence": confidence,
            "rdkit_qed": quality,
            "red_flag_count": red_flag_count,
        },
    }


def _apply_endpoint_admet_adjustments(
    payload: dict[str, Any],
    admet_profile: dict[str, Any],
) -> dict[str, Any]:
    score_map = _score_map(admet_profile)
    absorption = _mean_score(score_map, _ABSORPTION_ENDPOINTS, fallback=0.5)
    distribution = _mean_score(score_map, _DISTRIBUTION_ENDPOINTS, fallback=0.5)
    metabolism = _mean_score(score_map, _METABOLISM_ENDPOINTS, fallback=0.5)
    safety = _mean_score(score_map, _SAFETY_ENDPOINTS, fallback=0.5)

    pk_raw = payload.get("pk_model")
    pd_raw = payload.get("pd_model")
    stopping_raw = payload.get("stopping")
    external_control_raw = payload.get("external_control")
    if not isinstance(pk_raw, dict):
        return {}
    if not isinstance(pd_raw, dict):
        return {}
    if not isinstance(stopping_raw, dict):
        return {}
    if not isinstance(external_control_raw, dict):
        return {}

    pk = pk_raw
    pd = pd_raw
    stopping = stopping_raw
    external_control = external_control_raw

    ka_factor = float(np.clip(0.70 + 0.85 * absorption, 0.45, 1.45))
    cl_factor = float(np.clip(0.65 + 0.90 * metabolism, 0.45, 1.60))
    borrowing_factor = float(np.clip(0.70 + 0.45 * distribution, 0.25, 1.15))

    pk["ka_per_hour"] = float(pk["ka_per_hour"] * ka_factor)
    pk["cl_l_per_hour"] = float(pk["cl_l_per_hour"] * cl_factor)
    pk["omega_cl"] = float(np.clip(pk["omega_cl"] * (1.0 + 0.55 * (1.0 - metabolism)), 0.08, 1.20))
    pk["omega_v"] = float(np.clip(pk["omega_v"] * (1.0 + 0.25 * (1.0 - distribution)), 0.08, 1.20))

    safety_shift = float(np.clip((0.65 - safety) * 1.2, -0.35, 1.25))
    pd["safety_intercept"] = float(pd["safety_intercept"] + safety_shift)

    stopping_success_shift = float(np.clip((0.70 - safety) * 0.08, -0.01, 0.07))
    stopping_futility_shift = float(np.clip((0.70 - safety) * 0.15, 0.0, 0.20))
    stopping["success_posterior_threshold"] = float(
        np.clip(stopping["success_posterior_threshold"] + stopping_success_shift, 0.90, 0.999)
    )
    stopping["futility_posterior_threshold"] = float(
        np.clip(stopping["futility_posterior_threshold"] + stopping_futility_shift, 0.10, 0.70)
    )

    external_control["weight"] = float(
        np.clip(external_control["weight"] * borrowing_factor, 0.0, 1.0)
    )
    return {
        "category_scores": {
            "absorption": absorption,
            "distribution": distribution,
            "metabolism": metabolism,
            "safety": safety,
        },
        "ka_factor": ka_factor,
        "cl_factor": cl_factor,
        "borrowing_factor": borrowing_factor,
        "safety_intercept_shift": safety_shift,
        "stopping_success_shift": stopping_success_shift,
        "stopping_futility_shift": stopping_futility_shift,
    }


def _apply_affinity_adjustments(
    payload: dict[str, Any], affinity: dict[str, Any]
) -> dict[str, Any]:
    pd_raw = payload.get("pd_model")
    endpoint_raw = payload.get("endpoint")
    if not isinstance(pd_raw, dict):
        return {}
    if not isinstance(endpoint_raw, dict):
        return {}

    pd = pd_raw
    endpoint = endpoint_raw
    arms_raw = payload.get("arms")
    if not isinstance(arms_raw, list):
        return {}

    ic50 = _read_float(affinity, "ic50")
    binding_probability = _read_float(affinity, "binding_probability", default=0.5)
    potency = _potency_score_from_ic50(ic50)
    combined = float(np.clip(0.6 * potency + 0.4 * binding_probability, 0.05, 0.98))

    ec50_factor = float(np.clip(1.35 - 0.85 * combined, 0.55, 1.45))
    emax_factor = float(np.clip(0.65 + 0.90 * combined, 0.70, 1.60))
    endpoint_factor = float(np.clip(0.80 + 0.45 * combined, 0.65, 1.40))

    pd["ec50_auc"] = float(pd["ec50_auc"] * ec50_factor)
    pd["emax"] = float(pd["emax"] * emax_factor)
    endpoint["target_difference"] = float(endpoint["target_difference"] * endpoint_factor)
    endpoint["responder_threshold"] = float(
        max(endpoint["responder_threshold"], endpoint["target_difference"] + 2.0)
    )

    non_control_indices = [
        idx for idx, arm in enumerate(arms_raw) if not bool(arm.get("is_control"))
    ]
    base_dose = float(np.clip(160.0 - 95.0 * combined, 30.0, 180.0))
    dose_step = float(np.clip(30.0 + 45.0 * (1.0 - combined), 20.0, 70.0))
    dose_changes: list[dict[str, Any]] = []
    for rank, idx in enumerate(non_control_indices):
        old = float(arms_raw[idx].get("dose_mg", 0.0))
        new = float(np.clip(base_dose + rank * dose_step, 5.0, 250.0))
        arms_raw[idx]["dose_mg"] = new
        dose_changes.append(
            {
                "arm_id": str(arms_raw[idx].get("arm_id", f"arm_{idx + 1}")),
                "old_dose_mg": old,
                "new_dose_mg": new,
            }
        )

    return {
        "ic50": ic50,
        "binding_probability": binding_probability,
        "potency_score": potency,
        "combined_affinity_score": combined,
        "ec50_factor": ec50_factor,
        "emax_factor": emax_factor,
        "endpoint_factor": endpoint_factor,
        "dose_updates": dose_changes,
    }


def _apply_structure_adjustments(
    payload: dict[str, Any],
    structure: dict[str, Any],
) -> dict[str, Any]:
    confidence = _extract_structure_confidence(structure)
    uncertainty = float(np.clip(1.0 - confidence, 0.0, 1.0))

    pk_raw = payload.get("pk_model")
    pd_raw = payload.get("pd_model")
    enrollment_raw = payload.get("enrollment")
    adaptive_raw = payload.get("adaptive")
    stopping_raw = payload.get("stopping")
    external_control_raw = payload.get("external_control")
    heterogeneity_raw = payload.get("heterogeneity")
    if not isinstance(pk_raw, dict):
        return {}
    if not isinstance(pd_raw, dict):
        return {}
    if not isinstance(enrollment_raw, dict):
        return {}
    if not isinstance(adaptive_raw, dict):
        return {}
    if not isinstance(stopping_raw, dict):
        return {}
    if not isinstance(external_control_raw, dict):
        return {}
    if not isinstance(heterogeneity_raw, dict):
        return {}

    pk = pk_raw
    pd = pd_raw
    enrollment = enrollment_raw
    adaptive = adaptive_raw
    stopping = stopping_raw
    external_control = external_control_raw
    heterogeneity = heterogeneity_raw

    pk["residual_prop"] = float(
        np.clip(pk["residual_prop"] * (0.90 + 0.60 * uncertainty), 0.05, 0.60)
    )
    pd["residual_sd"] = float(np.clip(pd["residual_sd"] * (0.85 + 0.80 * uncertainty), 1.0, 20.0))
    heterogeneity["site_sd"] = float(
        np.clip(heterogeneity["site_sd"] * (0.85 + 0.80 * uncertainty), 0.3, 4.0)
    )
    heterogeneity["country_sd"] = float(
        np.clip(heterogeneity["country_sd"] * (0.85 + 0.70 * uncertainty), 0.2, 3.0)
    )
    external_control["weight"] = float(
        np.clip(external_control["weight"] * (0.70 + 0.60 * confidence), 0.0, 1.0)
    )

    enrollment["total_n"] = int(
        np.clip(round(enrollment["total_n"] * (0.95 + 0.30 * uncertainty)), 40, 2000)
    )
    adaptive["interim_every"] = int(
        np.clip(round(adaptive["interim_every"] * (1.0 - 0.25 * uncertainty)), 10, 120)
    )
    stopping["success_posterior_threshold"] = float(
        np.clip(stopping["success_posterior_threshold"] + 0.06 * uncertainty, 0.90, 0.999)
    )
    stopping["futility_posterior_threshold"] = float(
        np.clip(stopping["futility_posterior_threshold"] + 0.15 * uncertainty, 0.10, 0.70)
    )

    return {
        "confidence_score": confidence,
        "uncertainty": uncertainty,
    }


def _apply_rdkit_adjustments(payload: dict[str, Any], rdkit: dict[str, Any]) -> dict[str, Any]:
    pk_raw = payload.get("pk_model")
    pd_raw = payload.get("pd_model")
    enrollment_raw = payload.get("enrollment")
    costs_raw = payload.get("costs")
    if not isinstance(pk_raw, dict):
        return {}
    if not isinstance(pd_raw, dict):
        return {}
    if not isinstance(enrollment_raw, dict):
        return {}
    if not isinstance(costs_raw, dict):
        return {}

    pk = pk_raw
    pd = pd_raw
    enrollment = enrollment_raw
    costs = costs_raw

    mol_wt = _value_any_key(rdkit, ("mol_wt", "exact_mol_wt", "mw"))
    logp = _value_any_key(rdkit, ("mol_log_p", "logp", "clogp"))
    tpsa = _value_any_key(rdkit, ("tpsa",))
    hbd = _value_any_key(rdkit, ("num_h_donors", "hbd"))
    hba = _value_any_key(rdkit, ("num_h_acceptors", "hba"))
    rotb = _value_any_key(rdkit, ("num_rotatable_bonds", "rotatable_bonds"))
    qed = _value_any_key(rdkit, ("qed",), default=0.55)
    alerts = _value_any_key(rdkit, ("medchem_alert_count",), default=0.0)

    size_penalty = float(np.clip((mol_wt - 450.0) / 350.0, 0.0, 1.2))
    permeability_penalty = float(
        np.clip(
            (tpsa - 100.0) / 120.0 + max(hbd - 3.0, 0.0) / 5.0 + max(hba - 8.0, 0.0) / 10.0,
            0.0,
            1.4,
        )
    )
    lipophilicity_penalty = float(np.clip(max(logp - 3.0, 0.0) / 3.0, 0.0, 1.2))
    flexibility_penalty = float(np.clip(max(rotb - 8.0, 0.0) / 12.0, 0.0, 1.0))
    alert_penalty = float(np.clip(alerts / 6.0, 0.0, 1.5))
    quality_bonus = float(np.clip((qed - 0.5) / 0.5, -0.4, 0.8))

    bio_factor = float(
        np.clip(
            1.10
            - 0.35 * size_penalty
            - 0.35 * permeability_penalty
            - 0.20 * lipophilicity_penalty
            + 0.08 * quality_bonus,
            0.35,
            1.25,
        )
    )
    ka_factor = float(
        np.clip(1.0 - 0.30 * permeability_penalty - 0.10 * flexibility_penalty, 0.55, 1.15)
    )
    safety_shift = float(
        np.clip(
            0.50 * lipophilicity_penalty + 0.55 * alert_penalty - 0.18 * quality_bonus, -0.20, 1.20
        )
    )
    dropout_factor = float(
        np.clip(
            1.0 + 0.30 * alert_penalty + 0.20 * lipophilicity_penalty + 0.12 * flexibility_penalty,
            0.85,
            1.9,
        )
    )
    cost_factor = float(
        np.clip(1.0 + 0.20 * alert_penalty + 0.10 * lipophilicity_penalty, 0.90, 1.70)
    )

    pk["bioavailability"] = float(np.clip(pk["bioavailability"] * bio_factor, 0.1, 2.0))
    pk["ka_per_hour"] = float(np.clip(pk["ka_per_hour"] * ka_factor, 0.05, 4.0))
    pd["safety_intercept"] = float(pd["safety_intercept"] + safety_shift)
    enrollment["dropout_base"] = float(
        np.clip(enrollment["dropout_base"] * dropout_factor, 0.01, 0.95)
    )
    costs["cost_per_patient"] = float(
        np.clip(costs["cost_per_patient"] * cost_factor, 500.0, 500000.0)
    )

    return {
        "size_penalty": size_penalty,
        "permeability_penalty": permeability_penalty,
        "lipophilicity_penalty": lipophilicity_penalty,
        "flexibility_penalty": flexibility_penalty,
        "alert_penalty": alert_penalty,
        "quality_bonus": quality_bonus,
        "bioavailability_factor": bio_factor,
        "ka_factor": ka_factor,
        "safety_intercept_shift": safety_shift,
        "dropout_factor": dropout_factor,
    }


def _apply_target_adjustments(
    payload: dict[str, Any], target_properties: dict[str, Any]
) -> dict[str, Any]:
    pd_raw = payload.get("pd_model")
    endpoint_raw = payload.get("endpoint")
    enrollment_raw = payload.get("enrollment")
    stopping_raw = payload.get("stopping")
    if not isinstance(pd_raw, dict):
        return {}
    if not isinstance(endpoint_raw, dict):
        return {}
    if not isinstance(enrollment_raw, dict):
        return {}
    if not isinstance(stopping_raw, dict):
        return {}

    pd = pd_raw
    endpoint = endpoint_raw
    enrollment = enrollment_raw
    stopping = stopping_raw

    length = _value_any_key(target_properties, ("length",), default=450.0)
    instability = _value_any_key(target_properties, ("instability_index",), default=40.0)
    antibody_liability = _value_any_key(
        target_properties, ("antibody_liability_score",), default=0.0
    )
    peptide_liability = _value_any_key(
        target_properties, ("peptide_linear_liability_score",), default=0.0
    )
    gravy = _value_any_key(target_properties, ("gravy",), default=0.0)

    complexity = float(np.clip((length - 300.0) / 1200.0, 0.0, 1.2))
    instability_risk = float(np.clip((instability - 40.0) / 40.0, 0.0, 1.2))
    liability_risk = float(np.clip((antibody_liability + peptide_liability) / 10.0, 0.0, 1.2))
    hydrophilicity = float(np.clip(-gravy, -1.0, 1.0))

    assessment_factor = float(np.clip(0.95 + 0.35 * complexity, 0.8, 1.5))
    drift_factor = float(np.clip(1.0 + 0.25 * instability_risk, 0.8, 1.6))

    endpoint["assessment_day"] = int(
        np.clip(round(endpoint["assessment_day"] * assessment_factor), 14, 365)
    )
    enrollment["drift_per_block"] = float(
        np.clip(enrollment["drift_per_block"] * drift_factor, 0.0, 5.0)
    )
    pd["residual_sd"] = float(
        np.clip(pd["residual_sd"] * (1.0 + 0.20 * instability_risk), 1.0, 20.0)
    )
    pd["biomarker_effect"] = float(
        np.clip(pd["biomarker_effect"] * (1.0 + 0.15 * hydrophilicity), 0.1, 15.0)
    )
    pd["safety_intercept"] = float(pd["safety_intercept"] + 0.25 * liability_risk)
    stopping["min_interim_n"] = int(
        np.clip(round(stopping["min_interim_n"] * (1.0 + 0.20 * complexity)), 20, 600)
    )

    return {
        "complexity": complexity,
        "instability_risk": instability_risk,
        "liability_risk": liability_risk,
        "hydrophilicity": hydrophilicity,
        "assessment_factor": assessment_factor,
    }


def _apply_candidate_arm_overrides(
    payload: dict[str, Any],
    *,
    candidate_rankings: Any,
    max_candidate_arms: int,
) -> dict[str, Any]:
    if not isinstance(candidate_rankings, list):
        return {}

    ranked = [item for item in candidate_rankings if isinstance(item, dict)]
    if not ranked:
        return {}

    arms_raw = payload.get("arms")
    if not isinstance(arms_raw, list) or not arms_raw:
        return {}

    control = next((arm for arm in arms_raw if bool(_mapping(arm).get("is_control"))), None)
    control_arm = _mapping(control) if control is not None else _mapping(arms_raw[0])
    control_arm["is_control"] = True
    control_arm["dose_mg"] = float(control_arm.get("dose_mg", 0.0))
    control_arm["schedule_per_day"] = int(control_arm.get("schedule_per_day", 1))
    control_arm["arm_id"] = str(control_arm.get("arm_id", "control"))
    control_arm["label"] = str(control_arm.get("label", "Standard of Care"))

    new_arms: list[dict[str, Any]] = [control_arm]
    selected_rows: list[dict[str, Any]] = []
    used_arm_ids = {str(control_arm["arm_id"])}
    limit = max(1, max_candidate_arms)

    for rank, candidate in enumerate(ranked[:limit]):
        ligand_id = str(candidate.get("ligand_id", f"cand_{rank + 1}"))
        signal_summary = _mapping(candidate.get("signal_summary"))
        potency = _read_float(signal_summary, "potency_score", default=0.5)
        absorption = _read_float(signal_summary, "adme_score", default=0.5)
        dose_mg = float(np.clip(170.0 - 100.0 * potency, 20.0, 220.0))
        schedule = 1 if absorption >= 0.55 else 2
        arm_id = _next_arm_id(_slugify(ligand_id), used_arm_ids)
        used_arm_ids.add(arm_id)
        row = {
            "arm_id": arm_id,
            "label": f"Refua Candidate {ligand_id}",
            "dose_mg": dose_mg,
            "schedule_per_day": schedule,
            "is_control": False,
        }
        selected_rows.append(
            {
                "ligand_id": ligand_id,
                "arm_id": arm_id,
                "dose_mg": dose_mg,
                "candidate_score": float(candidate.get("candidate_score", 0.0)),
            }
        )
        new_arms.append(row)

    if len(new_arms) < 2:
        return {}

    payload["arms"] = new_arms
    adaptive_raw = payload.get("adaptive")
    if isinstance(adaptive_raw, dict):
        adaptive_raw["min_allocation"] = float(np.clip(0.60 / len(new_arms), 0.08, 0.25))
    return {
        "num_treatment_arms": len(new_arms) - 1,
        "arm_assignments": selected_rows,
    }


def _build_management_summary(
    summary: dict[str, Any], adjustments: dict[str, Any]
) -> dict[str, Any]:
    selected = _mapping(summary.get("selected_candidate"))
    signal_summary = _mapping(selected.get("signal_summary"))
    admet_summary = _mapping(selected.get("admet_summary"))

    admet_score = _read_float(signal_summary, "admet_score", default=0.5)
    potency = _read_float(signal_summary, "potency_score", default=0.5)
    confidence = _read_float(signal_summary, "structure_confidence", default=0.6)
    safety = _read_float(signal_summary, "safety_score", default=0.5)
    red_flags = admet_summary.get("red_flags")
    red_flag_count = len(red_flags) if isinstance(red_flags, list) else 0

    readiness = float(
        np.clip(
            100.0
            * (
                0.30 * admet_score
                + 0.20 * safety
                + 0.20 * potency
                + 0.20 * confidence
                + 0.10 * max(0.0, 1.0 - red_flag_count / 4.0)
            ),
            0.0,
            100.0,
        )
    )

    risks: list[dict[str, Any]] = []
    actions: list[str] = []
    if safety < 0.55 or red_flag_count > 0:
        risks.append(
            {
                "risk": "Safety signal uncertainty",
                "severity": "high" if safety < 0.45 or red_flag_count >= 2 else "medium",
            }
        )
        actions.append(
            "Tighten interim safety monitoring and predefine conservative stopping boundaries."
        )
    if potency < 0.45:
        risks.append({"risk": "Marginal potency signal", "severity": "medium"})
        actions.append(
            "Increase dose separation or enrich population for high-biomarker responders."
        )
    if confidence < 0.65:
        risks.append({"risk": "Low structural confidence", "severity": "medium"})
        actions.append("Run confirmatory structure/affinity replicates before pivotal design lock.")
    if float(summary.get("candidate_count", 0)) > 1:
        actions.append(
            "Treat the program as a multi-arm portfolio and monitor arm-level adaptation drift."
        )
    if not actions:
        actions.append(
            "Proceed with current assumptions and continue routine simulation-based "
            "sensitivity checks."
        )

    return {
        "readiness_index": readiness,
        "risk_register": risks,
        "recommended_actions": actions,
        "applied_modules": sorted(adjustments.keys()),
    }


def _score_map(admet_profile: dict[str, Any]) -> dict[str, float]:
    scores_raw = admet_profile.get("scores")
    if not isinstance(scores_raw, dict):
        return {}
    mapped: dict[str, float] = {}
    for key, value in scores_raw.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, int | float):
            continue
        mapped[key] = float(value)
    return mapped


def _mean_score(
    score_map: dict[str, float], endpoint_ids: tuple[str, ...], fallback: float
) -> float:
    collected: list[float] = []
    for endpoint_id in endpoint_ids:
        value = score_map.get(f"score_{endpoint_id}")
        if isinstance(value, int | float):
            collected.append(float(value))
    if not collected:
        return float(fallback)
    return float(np.clip(np.mean(collected), 0.0, 1.0))


def _potency_score_from_ic50(ic50: float | None) -> float:
    if ic50 is None or not np.isfinite(ic50) or ic50 <= 0.0:
        return 0.5
    log_ic50 = np.log10(np.clip(ic50, 1e-6, 1e9))
    score = 0.78 - 0.20 * log_ic50
    return float(np.clip(score, 0.05, 0.95))


def _extract_structure_confidence(structure: dict[str, Any]) -> float:
    confidence = _value_any_key(
        structure,
        ("confidence_score", "complex_plddt", "complex_iplddt"),
        default=0.60,
    )
    if confidence > 1.0:
        confidence = confidence / 100.0
    return float(np.clip(confidence, 0.0, 1.0))


def _value_any_key(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    *,
    default: float = 0.0,
) -> float:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int | float):
            return float(value)
    return float(default)


@overload
def _read_float(
    mapping: dict[str, Any],
    key: str,
    *,
    default: float,
) -> float: ...


@overload
def _read_float(
    mapping: dict[str, Any],
    key: str,
    *,
    default: None = None,
) -> float | None: ...


def _read_float(
    mapping: dict[str, Any],
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    value = mapping.get(key)
    if isinstance(value, int | float):
        return float(value)
    return default


def _read_str(mapping: dict[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_ligand_id(value: str) -> str:
    text = value.strip()
    return text or "ligand_001"


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    chars = [char if char.isalnum() else "_" for char in lowered]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "candidate"


def _next_arm_id(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    return candidate
