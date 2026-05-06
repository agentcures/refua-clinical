"""Object-oriented clinical API with fluent workflows similar to Refua."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .admet_integration import (
    apply_admet_adjustments,
    compute_admet_profile,
    load_admet_profile,
    summarize_admet_profile,
)
from .explainability import build_advice_report, render_advice_markdown
from .io import (
    apply_set_overrides,
    clinical_trial_to_mapping,
    clone_config,
    config_from_mapping,
    config_to_mapping,
    dump_json,
    dump_yaml,
    load_mapping,
)
from .modality import apply_modality_preset
from .models import (
    ClinicalTrial,
    ClinicalTrialArtifacts,
    ClinicalTrialContext,
    ProtocolRecommendation,
    SimulationConfig,
    default_simulation_config,
)
from .optimization import optimization_to_markdown, optimize_design_space
from .protocol import recommend_protocol, render_protocol_markdown
from .refua_bridge import (
    RefuaIntegrationPolicy,
    apply_refua_adjustments,
    load_refua_payload,
)
from .report import render_workup_html, write_workup_html
from .transportability import assess_transportability
from .trial import TrialSimulationResult, simulate_trials, trial_result_to_mapping
from .voi import estimate_value_of_information, voi_to_markdown


@dataclass(slots=True)
class ClinicalProtocol:
    """Protocol recommendation object with rendering helpers."""

    recommendation: ProtocolRecommendation

    @property
    def protocol(self) -> dict[str, Any]:
        return self.recommendation.protocol

    @property
    def candidates(self) -> list[Any]:
        return list(self.recommendation.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.recommendation.protocol,
            "candidates": [asdict(item) for item in self.recommendation.candidates],
        }

    def to_markdown(self) -> str:
        return render_protocol_markdown(self.recommendation.protocol)

    def save(
        self, path: str | Path, *, markdown: str | Path | None = None
    ) -> ClinicalProtocol:
        dump_json(path, self.to_dict())
        if markdown is not None:
            md_path = Path(markdown)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(self.to_markdown(), encoding="utf-8")
        return self


@dataclass(slots=True)
class ClinicalOptimization:
    """Optimization output with markdown and persistence helpers."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_markdown(self) -> str:
        return optimization_to_markdown(self.payload)

    def save(
        self, path: str | Path, *, markdown: str | Path | None = None
    ) -> ClinicalOptimization:
        dump_json(path, self.payload)
        if markdown is not None:
            md_path = Path(markdown)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(self.to_markdown(), encoding="utf-8")
        return self


@dataclass(slots=True)
class ClinicalVOI:
    """Value-of-information output object."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_markdown(self) -> str:
        return voi_to_markdown(self.payload)

    def save(
        self, path: str | Path, *, markdown: str | Path | None = None
    ) -> ClinicalVOI:
        dump_json(path, self.payload)
        if markdown is not None:
            md_path = Path(markdown)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(self.to_markdown(), encoding="utf-8")
        return self


@dataclass(slots=True)
class ClinicalAdvice:
    """Explainability/advice report object."""

    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.report)

    def to_markdown(self) -> str:
        return render_advice_markdown(self.report)

    def save(
        self, path: str | Path, *, markdown: str | Path | None = None
    ) -> ClinicalAdvice:
        dump_json(path, self.report)
        if markdown is not None:
            md_path = Path(markdown)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(self.to_markdown(), encoding="utf-8")
        return self


@dataclass(slots=True)
class ClinicalWorkup:
    """Bundle of run + protocol + optimization + VOI + advice outputs."""

    run: ClinicalRun
    protocol: ClinicalProtocol
    optimization: ClinicalOptimization
    voi: ClinicalVOI
    advice: ClinicalAdvice
    transportability: dict[str, Any] | None = None

    def to_trial(
        self,
        *,
        title: str | None = None,
        status: str = "simulated",
        sponsor: str | None = None,
        registry_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClinicalTrial:
        """Return this workup as a complete typed clinical trial aggregate."""
        return self.run.to_trial(
            title=title,
            status=status,
            sponsor=sponsor,
            registry_id=registry_id,
            protocol=self.protocol,
            optimization=self.optimization,
            voi=self.voi,
            advice=self.advice,
            transportability=self.transportability,
            metadata=metadata,
        )

    def save(self, output_dir: str | Path) -> dict[str, str | None]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        run_path = out / "run.json"
        protocol_json = out / "protocol.json"
        protocol_md = out / "protocol.md"
        optimization_json = out / "optimization.json"
        optimization_md = out / "optimization.md"
        voi_json = out / "voi.json"
        voi_md = out / "voi.md"
        advice_json = out / "advice.json"
        advice_md = out / "advice.md"
        report_html = out / "report.html"

        self.run.save(run_path)
        self.protocol.save(protocol_json, markdown=protocol_md)
        self.optimization.save(optimization_json, markdown=optimization_md)
        self.voi.save(voi_json, markdown=voi_md)
        self.advice.save(advice_json, markdown=advice_md)
        write_workup_html(
            report_html,
            html=render_workup_html(
                run_payload=self.run.to_dict(),
                protocol_payload=self.protocol.to_dict(),
                optimization_payload=self.optimization.to_dict(),
                voi_payload=self.voi.to_dict(),
                advice_payload=self.advice.to_dict(),
                transportability_payload=self.transportability,
            ),
        )

        transportability_path: str | None = None
        if self.transportability is not None:
            t_path = out / "transportability.json"
            dump_json(t_path, self.transportability)
            transportability_path = str(t_path)

        manifest = {
            "run": str(run_path),
            "protocol": str(protocol_json),
            "optimization": str(optimization_json),
            "voi": str(voi_json),
            "advice": str(advice_json),
            "report": str(report_html),
            "transportability": transportability_path,
        }
        dump_json(out / "manifest.json", manifest)
        return manifest


@dataclass(slots=True)
class ClinicalRun:
    """Simulation run object that exposes protocol/optimization/VOI/advice methods."""

    config: SimulationConfig
    result: TrialSimulationResult
    payload: dict[str, Any]
    admet_context: dict[str, Any] | None = None
    refua_context: dict[str, Any] | None = None

    @property
    def summary(self) -> dict[str, Any]:
        return self.result.summary

    @property
    def run_id(self) -> str:
        return self.result.run_id

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def save(self, path: str | Path) -> ClinicalRun:
        dump_json(path, self.payload)
        return self

    def to_trial(
        self,
        *,
        title: str | None = None,
        status: str = "simulated",
        sponsor: str | None = None,
        registry_id: str | None = None,
        protocol: ClinicalProtocol | None = None,
        optimization: ClinicalOptimization | None = None,
        voi: ClinicalVOI | None = None,
        advice: ClinicalAdvice | None = None,
        transportability: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClinicalTrial:
        """Return this run as a typed clinical trial aggregate."""
        trial = ClinicalTrial.from_result(
            self.result,
            title=title,
            status=status,
            sponsor=sponsor,
            registry_id=registry_id,
            artifacts=ClinicalTrialArtifacts(
                protocol=protocol.recommendation if protocol is not None else None,
                optimization=optimization.to_dict() if optimization is not None else None,
                value_of_information=voi.to_dict() if voi is not None else None,
                advice_report=advice.to_dict() if advice is not None else None,
                transportability=transportability,
            ),
            context=ClinicalTrialContext(
                admet=self.admet_context,
                refua=self.refua_context,
            ),
            metadata=metadata,
        )
        trial.validate_consistency()
        return trial

    def save_trial(self, path: str | Path, **kwargs: Any) -> ClinicalRun:
        """Persist the run as a complete clinical trial aggregate."""
        dump_json(path, clinical_trial_to_mapping(self.to_trial(**kwargs)))
        return self

    def recommend_protocol(
        self,
        *,
        replicates_per_candidate: int = 80,
        candidate_total_n: list[int] | None = None,
        candidate_interims: list[int] | None = None,
        candidate_burn_in_n: list[int] | None = None,
        candidate_min_allocations: list[float] | None = None,
        candidate_success_thresholds: list[float] | None = None,
    ) -> ClinicalProtocol:
        recommendation = recommend_protocol(
            self.config,
            replicates_per_candidate=max(20, int(replicates_per_candidate)),
            candidate_total_n=candidate_total_n,
            candidate_interims=candidate_interims,
            candidate_burn_in_n=candidate_burn_in_n,
            candidate_min_allocations=candidate_min_allocations,
            candidate_success_thresholds=candidate_success_thresholds,
        )
        return ClinicalProtocol(recommendation)

    def optimize(
        self,
        *,
        replicates_per_candidate: int = 60,
        candidate_total_n: list[int] | None = None,
        candidate_interims: list[int] | None = None,
        candidate_burn_in_n: list[int] | None = None,
        candidate_min_allocations: list[float] | None = None,
        candidate_success_thresholds: list[float] | None = None,
    ) -> ClinicalOptimization:
        payload = optimize_design_space(
            self.config,
            candidate_total_n=candidate_total_n,
            candidate_interims=candidate_interims,
            candidate_burn_in_n=candidate_burn_in_n,
            candidate_min_allocations=candidate_min_allocations,
            candidate_success_thresholds=candidate_success_thresholds,
            replicates_per_candidate=max(20, int(replicates_per_candidate)),
        )
        return ClinicalOptimization(payload)

    def value_of_information(
        self,
        *,
        extra_n: list[int] | None = None,
        candidate_success_thresholds: list[float] | None = None,
        candidate_min_allocations: list[float] | None = None,
        replicates_per_scenario: int = 60,
    ) -> ClinicalVOI:
        payload = estimate_value_of_information(
            self.config,
            candidate_extra_n=extra_n,
            candidate_success_thresholds=candidate_success_thresholds,
            candidate_min_allocations=candidate_min_allocations,
            replicates_per_scenario=max(20, int(replicates_per_scenario)),
        )
        return ClinicalVOI(payload)

    def advise(
        self,
        *,
        protocol: ClinicalProtocol | None = None,
        optimization: ClinicalOptimization | None = None,
        voi: ClinicalVOI | None = None,
        include_sensitivity: bool = False,
        sensitivity_replicates: int = 40,
        sensitivity_delta: float = 0.15,
    ) -> ClinicalAdvice:
        admet_payload = None
        if isinstance(self.admet_context, dict):
            summary = self.admet_context.get("summary")
            if isinstance(summary, dict):
                admet_payload = dict(summary)

        report = build_advice_report(
            self.payload,
            protocol_payload=protocol.to_dict() if protocol is not None else None,
            optimization_payload=(
                optimization.to_dict() if optimization is not None else None
            ),
            voi_payload=voi.to_dict() if voi is not None else None,
            admet_payload=admet_payload,
            include_sensitivity=bool(include_sensitivity),
            sensitivity_replicates=int(sensitivity_replicates),
            sensitivity_delta=float(sensitivity_delta),
        )
        return ClinicalAdvice(report)

    def assess_transportability(
        self,
        reference: Any,
        target: Any,
        *,
        columns: list[str] | None = None,
        method: str = "none",
    ) -> dict[str, Any]:
        return assess_transportability(reference, target, columns=columns, method=method)

    def workup(
        self,
        *,
        replicates_per_candidate: int = 60,
        candidate_total_n: list[int] | None = None,
        candidate_interims: list[int] | None = None,
        candidate_burn_in_n: list[int] | None = None,
        candidate_min_allocations: list[float] | None = None,
        candidate_success_thresholds: list[float] | None = None,
        voi_extra_n: list[int] | None = None,
        voi_success_thresholds: list[float] | None = None,
        voi_min_allocations: list[float] | None = None,
        voi_replicates_per_scenario: int = 50,
        include_sensitivity: bool = False,
        sensitivity_replicates: int = 40,
        sensitivity_delta: float = 0.15,
        transportability: dict[str, Any] | None = None,
    ) -> ClinicalWorkup:
        protocol = self.recommend_protocol(
            replicates_per_candidate=replicates_per_candidate,
            candidate_total_n=candidate_total_n,
            candidate_interims=candidate_interims,
            candidate_burn_in_n=candidate_burn_in_n,
            candidate_min_allocations=candidate_min_allocations,
            candidate_success_thresholds=candidate_success_thresholds,
        )
        optimization = self.optimize(
            replicates_per_candidate=replicates_per_candidate,
            candidate_total_n=candidate_total_n,
            candidate_interims=candidate_interims,
            candidate_burn_in_n=candidate_burn_in_n,
            candidate_min_allocations=candidate_min_allocations,
            candidate_success_thresholds=candidate_success_thresholds,
        )
        voi = self.value_of_information(
            extra_n=voi_extra_n,
            candidate_success_thresholds=voi_success_thresholds,
            candidate_min_allocations=voi_min_allocations,
            replicates_per_scenario=voi_replicates_per_scenario,
        )
        advice = self.advise(
            protocol=protocol,
            optimization=optimization,
            voi=voi,
            include_sensitivity=include_sensitivity,
            sensitivity_replicates=sensitivity_replicates,
            sensitivity_delta=sensitivity_delta,
        )
        return ClinicalWorkup(
            run=self,
            protocol=protocol,
            optimization=optimization,
            voi=voi,
            advice=advice,
            transportability=transportability,
        )


class ClinicalStudy:
    """Fluent OO builder for clinical simulation workflows."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self._config = config or default_simulation_config()
        self._admet_context: dict[str, Any] | None = None
        self._refua_context: dict[str, Any] | None = None

    @classmethod
    def default(cls) -> ClinicalStudy:
        return cls(default_simulation_config())

    @classmethod
    def from_config(cls, config: SimulationConfig | dict[str, Any]) -> ClinicalStudy:
        if isinstance(config, dict):
            return cls(config_from_mapping(config))
        return cls(config)

    @classmethod
    def from_file(cls, path: str | Path) -> ClinicalStudy:
        payload = load_mapping(path)
        return cls(config_from_mapping(payload))

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def admet_context(self) -> dict[str, Any] | None:
        return self._admet_context

    @property
    def refua_context(self) -> dict[str, Any] | None:
        return self._refua_context

    def copy(self) -> ClinicalStudy:
        clone = ClinicalStudy.from_config(clone_config(self._config))
        if self._admet_context is not None:
            clone._admet_context = dict(self._admet_context)
        if self._refua_context is not None:
            clone._refua_context = dict(self._refua_context)
        return clone

    def trial(
        self,
        *,
        trial_id: str | None = None,
        indication: str | None = None,
        phase: str | None = None,
        objective: str | None = None,
        seed: int | None = None,
        replicates: int | None = None,
    ) -> ClinicalStudy:
        if trial_id is not None:
            self._config.trial_id = str(trial_id)
        if indication is not None:
            self._config.indication = str(indication)
        if phase is not None:
            self._config.phase = str(phase)
        if objective is not None:
            self._config.objective = str(objective)
        if seed is not None:
            self._config.seed = int(seed)
        if replicates is not None:
            self._config.replicates = int(replicates)
        return self

    def biologics_mode(
        self,
        *,
        route: str = "sc",
        dosing_interval_hours: float = 336.0,
        tmdd_strength: float = 0.35,
    ) -> ClinicalStudy:
        """Apply a biologics-oriented preset and optional interval dosing."""
        normalized_route = str(route).strip().lower()
        self.modality_preset(
            preset=f"biologic-{normalized_route}",
            dosing_interval_hours=float(dosing_interval_hours),
            tmdd_strength=float(tmdd_strength),
        )
        return self

    def modality_preset(
        self,
        *,
        preset: str,
        dosing_interval_hours: float | None = None,
        tmdd_strength: float | None = None,
    ) -> ClinicalStudy:
        """Apply a shared modality preset by name."""
        apply_modality_preset(
            self._config,
            preset=str(preset),
            dosing_interval_hours=(
                float(dosing_interval_hours)
                if dosing_interval_hours is not None
                else None
            ),
            tmdd_strength=float(tmdd_strength) if tmdd_strength is not None else None,
        )
        return self

    def set(self, path: str, value: Any) -> ClinicalStudy:
        updated = apply_set_overrides(
            config_to_mapping(self._config), [f"{path}={value}"]
        )
        self._config = config_from_mapping(updated)
        return self

    def apply_overrides(self, values: list[str]) -> ClinicalStudy:
        updated = apply_set_overrides(config_to_mapping(self._config), values)
        self._config = config_from_mapping(updated)
        return self

    def admet_profile(
        self,
        profile_or_path: dict[str, Any] | str | Path,
        *,
        apply: bool = True,
    ) -> ClinicalStudy:
        if isinstance(profile_or_path, dict):
            profile = dict(profile_or_path)
        else:
            profile = load_admet_profile(profile_or_path)
        summary = summarize_admet_profile(profile)
        adjustments = None
        if apply:
            self._config, adjustments = apply_admet_adjustments(self._config, profile)
        self._admet_context = {
            "profile": profile,
            "summary": summary,
            "adjustments": adjustments,
        }
        return self

    def admet_smiles(
        self,
        smiles: str,
        *,
        model_variant: str = "9b-chat",
        apply: bool = True,
    ) -> ClinicalStudy:
        profile = compute_admet_profile(smiles, model_variant=model_variant)
        return self.admet_profile(profile, apply=apply)

    def refua_payload(
        self,
        payload_or_path: dict[str, Any] | str | Path,
        *,
        apply: bool = True,
        ligand_id: str | None = None,
        max_candidate_arms: int = 4,
        strict_contract: bool = False,
    ) -> ClinicalStudy:
        payload = (
            dict(payload_or_path)
            if isinstance(payload_or_path, dict)
            else load_refua_payload(payload_or_path)
        )
        policy = RefuaIntegrationPolicy(
            preferred_ligand_id=ligand_id,
            max_candidate_arms=max(1, int(max_candidate_arms)),
            strict_contract=bool(strict_contract),
        )
        adjustments = None
        summary = None
        if apply:
            self._config, adjustments = apply_refua_adjustments(
                self._config,
                payload,
                policy=policy,
            )
            summary = adjustments.get("summary")
        self._refua_context = {
            "payload": payload,
            "summary": summary,
            "adjustments": adjustments,
            "policy": asdict(policy),
        }
        if self._admet_context is None and isinstance(summary, dict):
            selected = summary.get("selected_candidate")
            if isinstance(selected, dict):
                admet_summary = selected.get("admet_summary")
                admet_profile = selected.get("admet_profile")
                if isinstance(admet_summary, dict):
                    self._admet_context = {
                        "profile": (
                            dict(admet_profile)
                            if isinstance(admet_profile, dict)
                            else None
                        ),
                        "summary": dict(admet_summary),
                        "adjustments": None,
                    }
        return self

    def simulate(
        self,
        *,
        replicates: int | None = None,
        seed: int | None = None,
    ) -> ClinicalRun:
        run_config = clone_config(self._config)
        if replicates is not None:
            run_config.replicates = int(replicates)
        if seed is not None:
            run_config.seed = int(seed)

        result = simulate_trials(run_config)
        payload = trial_result_to_mapping(result)
        if self._admet_context is not None:
            payload["admet"] = {
                "summary": self._admet_context.get("summary"),
                "adjustments": self._admet_context.get("adjustments"),
            }
        if self._refua_context is not None:
            payload["refua"] = {
                "summary": self._refua_context.get("summary"),
                "adjustments": self._refua_context.get("adjustments"),
                "policy": self._refua_context.get("policy"),
            }
        return ClinicalRun(
            config=run_config,
            result=result,
            payload=payload,
            admet_context=self._admet_context,
            refua_context=self._refua_context,
        )

    def save_config(self, path: str | Path) -> ClinicalStudy:
        out = Path(path)
        payload = config_to_mapping(self._config)
        if out.suffix.lower() == ".json":
            dump_json(out, payload)
        else:
            dump_yaml(out, payload)
        return self
