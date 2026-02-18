"""CLI for refua-clinical."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import __version__
from .admet_integration import (
    apply_admet_adjustments,
    compute_admet_profile,
    load_admet_profile,
    summarize_admet_profile,
)
from .explainability import (
    build_advice_report,
    load_optional_json,
    summarize_recommendations,
    write_advice_artifacts,
)
from .integrations import build_refua_regulatory_bundle, infer_population_from_refua_data
from .io import (
    apply_set_overrides,
    config_from_mapping,
    config_to_mapping,
    dump_json,
    dump_yaml,
    load_mapping,
    merge_mappings,
)
from .modality import apply_modality_preset, list_modality_presets
from .models import default_simulation_config
from .optimization import optimization_to_markdown, optimize_design_space
from .protocol import recommend_protocol, render_protocol_markdown
from .refua_bridge import (
    RefuaIntegrationPolicy,
    apply_refua_adjustments,
    extract_admet_profile_from_refua_payload,
    load_refua_payload,
    summarize_refua_payload,
)
from .research import list_references
from .transportability import assess_transportability, load_tabular, transportability_to_markdown
from .trial import simulate_trials, trial_result_to_mapping
from .voi import estimate_value_of_information, voi_to_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refua-clinical",
        description="PK/PD virtual patient and clinical trial simulation for Refua.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-config", help="Write a starter simulation config")
    init_parser.add_argument("--output", type=Path, required=True, help="Output YAML path")
    _add_modality_preset_arguments(init_parser)
    init_parser.set_defaults(handler=_cmd_init_config)

    simulate_parser = sub.add_parser("simulate", help="Run clinical trial simulations")
    simulate_parser.add_argument(
        "--config", type=Path, required=True, help="Simulation config YAML/JSON"
    )
    simulate_parser.add_argument(
        "--output", type=Path, required=True, help="Run artifact JSON path"
    )
    _add_modality_preset_arguments(simulate_parser)
    _add_admet_arguments(simulate_parser)
    simulate_parser.add_argument(
        "--admet-adjustments",
        action="store_true",
        help="Apply ADMET-informed parameter adjustments before simulation.",
    )
    _add_refua_arguments(simulate_parser)
    simulate_parser.set_defaults(handler=_cmd_simulate)

    rerun_parser = sub.add_parser("rerun", help="Rerun from previous run artifact with overrides")
    rerun_parser.add_argument("--run", type=Path, required=True, help="Previous run artifact JSON")
    rerun_parser.add_argument(
        "--overrides-file",
        type=Path,
        default=None,
        help="Optional YAML/JSON overrides applied to prior config",
    )
    rerun_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Inline override as dotted.path=value (repeatable)",
    )
    rerun_parser.add_argument("--output", type=Path, required=True, help="Run artifact JSON path")
    _add_modality_preset_arguments(rerun_parser)
    rerun_parser.set_defaults(handler=_cmd_rerun)

    protocol_parser = sub.add_parser(
        "protocol",
        help="Recommend a tailored protocol via simulation-based design optimization",
    )
    protocol_group = protocol_parser.add_mutually_exclusive_group(required=True)
    protocol_group.add_argument("--config", type=Path, help="Simulation config YAML/JSON")
    protocol_group.add_argument("--run", type=Path, help="Existing run artifact JSON")
    protocol_parser.add_argument("--output", type=Path, required=True, help="Protocol JSON output")
    protocol_parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown protocol output",
    )
    protocol_parser.add_argument(
        "--replicates-per-candidate",
        type=int,
        default=80,
        help="Number of replicates for each candidate design",
    )
    _add_modality_preset_arguments(protocol_parser)
    protocol_parser.set_defaults(handler=_cmd_protocol)

    optimize_parser = sub.add_parser(
        "optimize",
        help="Run multi-objective design optimization and Pareto-front analysis",
    )
    optimize_group = optimize_parser.add_mutually_exclusive_group(required=True)
    optimize_group.add_argument("--config", type=Path, help="Simulation config YAML/JSON")
    optimize_group.add_argument("--run", type=Path, help="Existing run artifact JSON")
    optimize_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Optimization result JSON output",
    )
    optimize_parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown optimization output",
    )
    optimize_parser.add_argument(
        "--replicates-per-candidate",
        type=int,
        default=60,
        help="Replicates for each optimization candidate",
    )
    optimize_parser.add_argument(
        "--candidate-total-n",
        type=int,
        nargs="*",
        default=None,
        help="Optional candidate total enrollments",
    )
    optimize_parser.add_argument(
        "--candidate-interims",
        type=int,
        nargs="*",
        default=None,
        help="Optional candidate interim cadences",
    )
    _add_modality_preset_arguments(optimize_parser)
    optimize_parser.set_defaults(handler=_cmd_optimize)

    voi_parser = sub.add_parser(
        "voi",
        help="Estimate value of information across sample-size expansion scenarios",
    )
    voi_group = voi_parser.add_mutually_exclusive_group(required=True)
    voi_group.add_argument("--config", type=Path, help="Simulation config YAML/JSON")
    voi_group.add_argument("--run", type=Path, help="Existing run artifact JSON")
    voi_parser.add_argument("--output", type=Path, required=True, help="VOI result JSON output")
    voi_parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown VOI output",
    )
    voi_parser.add_argument(
        "--replicates-per-scenario",
        type=int,
        default=60,
        help="Replicates per VOI scenario",
    )
    voi_parser.add_argument(
        "--extra-n",
        type=int,
        nargs="*",
        default=None,
        help="Optional patient expansions over baseline total_n",
    )
    _add_modality_preset_arguments(voi_parser)
    voi_parser.set_defaults(handler=_cmd_voi)

    transport_parser = sub.add_parser(
        "transportability",
        help="Assess transportability between reference and target populations",
    )
    transport_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference population table (.csv/.json/.parquet)",
    )
    transport_parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Target population table (.csv/.json/.parquet)",
    )
    transport_parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional covariate columns to assess",
    )
    transport_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Transportability result JSON output",
    )
    transport_parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown transportability output",
    )
    transport_parser.set_defaults(handler=_cmd_transportability)

    from_data_parser = sub.add_parser(
        "from-data",
        help="Infer virtual patient covariates from a refua-data dataset and write config",
    )
    from_data_parser.add_argument(
        "--dataset-id", required=True, help="Dataset id in refua-data catalog"
    )
    from_data_parser.add_argument("--output", type=Path, required=True, help="Output YAML path")
    from_data_parser.add_argument("--size", type=int, default=6000, help="Virtual population size")
    from_data_parser.add_argument(
        "--max-rows",
        type=int,
        default=25000,
        help="Maximum rows to load from materialized parquet",
    )
    from_data_parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional explicit column names to model",
    )
    from_data_parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache root override for refua-data",
    )
    _add_modality_preset_arguments(from_data_parser)
    from_data_parser.set_defaults(handler=_cmd_from_data)

    evidence_parser = sub.add_parser(
        "evidence",
        help="Wrap simulation output into a refua-regulatory bundle",
    )
    evidence_parser.add_argument("--run", type=Path, required=True, help="Run artifact JSON")
    evidence_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Evidence bundle directory"
    )
    evidence_parser.add_argument(
        "--data-manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional refua-data manifest path (repeatable)",
    )
    evidence_parser.add_argument(
        "--model-version", default=None, help="Optional model version override"
    )
    evidence_parser.set_defaults(handler=_cmd_evidence)

    advise_parser = sub.add_parser(
        "advise",
        help="Generate explainable narrative and actionable recommendations from a run artifact",
    )
    advise_parser.add_argument("--run", type=Path, required=True, help="Run artifact JSON")
    advise_parser.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="Optional protocol JSON from refua-clinical protocol command",
    )
    advise_parser.add_argument(
        "--optimization",
        type=Path,
        default=None,
        help="Optional optimization JSON from refua-clinical optimize command",
    )
    advise_parser.add_argument(
        "--voi",
        type=Path,
        default=None,
        help="Optional VOI JSON from refua-clinical voi command",
    )
    _add_admet_arguments(advise_parser)
    advise_parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Run one-way sensitivity analysis for explainability.",
    )
    advise_parser.add_argument(
        "--sensitivity-replicates",
        type=int,
        default=40,
        help="Replicates per sensitivity scenario.",
    )
    advise_parser.add_argument(
        "--sensitivity-delta",
        type=float,
        default=0.15,
        help="Relative parameter perturbation for sensitivity analysis.",
    )
    advise_parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write advice report JSON.",
    )
    advise_parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to write advice report markdown.",
    )
    advise_parser.set_defaults(handler=_cmd_advise)

    workup_parser = sub.add_parser(
        "workup",
        help=(
            "Run simulate + protocol + optimize + voi + advise in one pass and "
            "write a complete artifact bundle"
        ),
    )
    workup_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Simulation config YAML/JSON",
    )
    workup_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all generated artifacts",
    )
    _add_modality_preset_arguments(workup_parser)
    _add_admet_arguments(workup_parser)
    workup_parser.add_argument(
        "--admet-adjustments",
        action="store_true",
        help="Apply ADMET-informed parameter adjustments before simulation.",
    )
    _add_refua_arguments(workup_parser)
    workup_parser.add_argument(
        "--replicates-per-candidate",
        type=int,
        default=60,
        help="Replicates for optimization/protocol candidates",
    )
    workup_parser.add_argument(
        "--candidate-total-n",
        type=int,
        nargs="*",
        default=None,
        help="Optional candidate total enrollments",
    )
    workup_parser.add_argument(
        "--candidate-interims",
        type=int,
        nargs="*",
        default=None,
        help="Optional candidate interim cadences",
    )
    workup_parser.add_argument(
        "--voi-extra-n",
        type=int,
        nargs="*",
        default=None,
        help="Optional VOI sample-size expansions",
    )
    workup_parser.add_argument(
        "--voi-replicates-per-scenario",
        type=int,
        default=50,
        help="Replicates per VOI scenario",
    )
    workup_parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Run one-way sensitivity analysis in the advice report.",
    )
    workup_parser.add_argument(
        "--sensitivity-replicates",
        type=int,
        default=40,
        help="Replicates per sensitivity scenario.",
    )
    workup_parser.add_argument(
        "--sensitivity-delta",
        type=float,
        default=0.15,
        help="Relative parameter perturbation for sensitivity analysis.",
    )
    workup_parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference table for transportability diagnostics.",
    )
    workup_parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional target table for transportability diagnostics.",
    )
    workup_parser.add_argument(
        "--transport-columns",
        nargs="*",
        default=None,
        help="Optional transportability columns.",
    )
    workup_parser.set_defaults(handler=_cmd_workup)

    integrate_refua_parser = sub.add_parser(
        "integrate-refua",
        help="Apply Refua payload signals to a clinical config and write adjusted outputs",
    )
    integrate_refua_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Simulation config YAML/JSON",
    )
    integrate_refua_parser.add_argument(
        "--refua-json",
        type=Path,
        required=True,
        help="Refua payload JSON/YAML",
    )
    integrate_refua_parser.add_argument(
        "--output-config",
        type=Path,
        required=True,
        help="Output path for adjusted config (.yaml/.json)",
    )
    integrate_refua_parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional JSON output path for integration summary",
    )
    _add_modality_preset_arguments(integrate_refua_parser)
    integrate_refua_parser.add_argument(
        "--refua-ligand-id",
        default=None,
        help="Optional preferred ligand id from the Refua payload.",
    )
    integrate_refua_parser.add_argument(
        "--refua-max-candidate-arms",
        type=int,
        default=4,
        help="Maximum number of Refua candidate treatment arms to build.",
    )
    integrate_refua_parser.add_argument(
        "--refua-strict-contract",
        action="store_true",
        help="Require canonical Refua payload schema keys and reject legacy aliases.",
    )
    integrate_refua_parser.set_defaults(handler=_cmd_integrate_refua)

    refs_parser = sub.add_parser("research", help="Print research references used by this package")
    refs_parser.set_defaults(handler=_cmd_research)

    return parser


def _cmd_init_config(args: argparse.Namespace) -> int:
    config = default_simulation_config()
    config = _apply_modality_preset_for_args(config, args)
    dump_yaml(args.output, config_to_mapping(config))
    print(json.dumps({"output": str(args.output), "trial_id": config.trial_id}, indent=2))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    config = _apply_modality_preset_for_args(config, args)
    admet_info = _load_admet_for_args(args)
    refua_info = _load_refua_for_args(args)

    if admet_info is None and refua_info is not None and refua_info["selected_admet"] is not None:
        selected_admet = dict(refua_info["selected_admet"])
        admet_info = {
            "profile": selected_admet,
            "summary": summarize_admet_profile(selected_admet),
        }

    if refua_info is not None and bool(args.refua_apply):
        policy = _refua_policy_from_args(args)
        config, refua_adjustments = apply_refua_adjustments(
            config,
            refua_info["payload"],
            policy=policy,
        )
        refua_info["adjustments"] = refua_adjustments
    elif refua_info is not None:
        refua_info["adjustments"] = None

    if (
        admet_info is not None
        and bool(args.admet_adjustments)
        and not (refua_info is not None and bool(args.refua_apply))
    ):
        config, adjustments = apply_admet_adjustments(config, admet_info["profile"])
        admet_info["adjustments"] = adjustments
    elif admet_info is not None:
        admet_info["adjustments"] = None

    result = simulate_trials(config)
    payload = trial_result_to_mapping(result)
    if admet_info is not None:
        payload["admet"] = {
            "summary": admet_info["summary"],
            "adjustments": admet_info["adjustments"],
        }
    if refua_info is not None:
        payload["refua"] = {
            "summary": refua_info["summary"],
            "adjustments": refua_info["adjustments"],
        }
    dump_json(args.output, payload)
    output_payload: dict[str, Any] = {"run_id": payload["run_id"], "summary": payload["summary"]}
    if admet_info is not None:
        output_payload["admet"] = {
            "safety_score": admet_info["summary"]["safety_score"],
            "admet_score": admet_info["summary"]["admet_score"],
            "adjusted": bool(admet_info["adjustments"] is not None),
        }
    if refua_info is not None:
        refua_summary = _mapping(refua_info["summary"])
        management = _mapping(_mapping(refua_info.get("adjustments")).get("management"))
        output_payload["refua"] = {
            "selected_ligand_id": refua_summary.get("selected_ligand_id"),
            "candidate_count": refua_summary.get("candidate_count"),
            "adjusted": bool(refua_info.get("adjustments") is not None),
            "readiness_index": management.get("readiness_index"),
        }
    print(json.dumps(output_payload, indent=2))
    return 0


def _cmd_rerun(args: argparse.Namespace) -> int:
    base_payload = load_mapping(args.run)
    base_config = base_payload.get("config")
    if not isinstance(base_config, dict):
        raise ValueError("Run artifact missing config object")

    merged = dict(base_config)
    if args.overrides_file is not None:
        overrides = load_mapping(args.overrides_file)
        merged = merge_mappings(merged, overrides)

    if args.set:
        merged = apply_set_overrides(merged, list(args.set))

    config = config_from_mapping(merged)
    config = _apply_modality_preset_for_args(config, args)
    result = simulate_trials(config)
    payload = trial_result_to_mapping(result)
    dump_json(args.output, payload)
    print(json.dumps({"run_id": payload["run_id"], "summary": payload["summary"]}, indent=2))
    return 0


def _cmd_protocol(args: argparse.Namespace) -> int:
    config = _load_config_or_run(config_path=args.config, run_path=args.run)
    config = _apply_modality_preset_for_args(config, args)

    recommendation = recommend_protocol(
        config,
        replicates_per_candidate=max(args.replicates_per_candidate, 20),
    )

    payload = {
        "protocol": recommendation.protocol,
        "candidates": [asdict(candidate) for candidate in recommendation.candidates],
    }
    dump_json(args.output, payload)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_protocol_markdown(recommendation.protocol),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_id": recommendation.protocol["protocol_id"],
                "planned_enrollment": recommendation.protocol["design"]["planned_enrollment"],
                "power": recommendation.protocol["simulated_performance"]["power"],
            },
            indent=2,
        )
    )
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    config = _load_config_or_run(config_path=args.config, run_path=args.run)
    config = _apply_modality_preset_for_args(config, args)

    payload = optimize_design_space(
        config,
        candidate_total_n=list(args.candidate_total_n) if args.candidate_total_n else None,
        candidate_interims=list(args.candidate_interims) if args.candidate_interims else None,
        replicates_per_candidate=max(int(args.replicates_per_candidate), 20),
    )
    dump_json(args.output, payload)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(optimization_to_markdown(payload), encoding="utf-8")

    best = payload["best_candidate"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "best_total_n": int(best["total_n"]),
                "best_interim_every": int(best["interim_every"]),
                "utility_score": float(best["utility_score"]),
                "power": float(best["power"]),
            },
            indent=2,
        )
    )
    return 0


def _cmd_voi(args: argparse.Namespace) -> int:
    config = _load_config_or_run(config_path=args.config, run_path=args.run)
    config = _apply_modality_preset_for_args(config, args)

    payload = estimate_value_of_information(
        config,
        candidate_extra_n=list(args.extra_n) if args.extra_n else None,
        replicates_per_scenario=max(int(args.replicates_per_scenario), 20),
    )
    dump_json(args.output, payload)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(voi_to_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "recommendation": payload["recommendation"],
                "best_total_n": int(payload["best_scenario"]["total_n"]),
                "information_gain": float(payload["best_scenario"]["information_gain"]),
            },
            indent=2,
        )
    )
    return 0


def _cmd_transportability(args: argparse.Namespace) -> int:
    reference = load_tabular(str(args.reference))
    target = load_tabular(str(args.target))

    payload = assess_transportability(
        reference,
        target,
        columns=list(args.columns) if args.columns else None,
    )
    dump_json(args.output, payload)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(transportability_to_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "risk_level": payload["risk_level"],
                "overlap_score": float(payload["overlap_score"]),
                "max_abs_smd": float(payload["max_abs_smd"]),
            },
            indent=2,
        )
    )
    return 0


def _cmd_from_data(args: argparse.Namespace) -> int:
    pop_spec = infer_population_from_refua_data(
        dataset_id=str(args.dataset_id),
        size=int(args.size),
        max_rows=int(args.max_rows),
        columns=list(args.columns) if args.columns else None,
        cache_root=args.cache_root,
    )

    config = default_simulation_config()
    config.population = pop_spec
    config = _apply_modality_preset_for_args(config, args)
    dump_yaml(args.output, config_to_mapping(config))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dataset_id": args.dataset_id,
                "covariates": [cov.name for cov in pop_spec.covariates],
            },
            indent=2,
        )
    )
    return 0


def _cmd_evidence(args: argparse.Namespace) -> int:
    manifest = build_refua_regulatory_bundle(
        run_artifact_path=args.run,
        output_dir=args.output_dir,
        data_manifest_paths=list(args.data_manifest),
        model_version=args.model_version,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_research(_: argparse.Namespace) -> int:
    print(json.dumps({"references": list_references()}, indent=2))
    return 0


def _cmd_advise(args: argparse.Namespace) -> int:
    run_payload = load_mapping(args.run)
    protocol_payload = load_optional_json(args.protocol)
    optimization_payload = load_optional_json(args.optimization)
    voi_payload = load_optional_json(args.voi)

    admet_info = _load_admet_for_args(args)
    if admet_info is not None:
        admet_payload = admet_info["summary"]
    else:
        admet_payload = _admet_from_run_payload(run_payload)

    report = build_advice_report(
        run_payload,
        protocol_payload=protocol_payload,
        optimization_payload=optimization_payload,
        voi_payload=voi_payload,
        admet_payload=admet_payload,
        include_sensitivity=bool(args.include_sensitivity),
        sensitivity_replicates=int(args.sensitivity_replicates),
        sensitivity_delta=float(args.sensitivity_delta),
    )

    write_advice_artifacts(
        report,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )

    print(
        json.dumps(
            {
                "power": report["summary"]["power"],
                "safety_event_rate": report["summary"]["safety_event_rate"],
                "recommendations": summarize_recommendations(report["recommendations"])[:5],
                "output_json": str(args.output_json) if args.output_json else None,
                "output_markdown": str(args.output_markdown) if args.output_markdown else None,
            },
            indent=2,
        )
    )
    return 0


def _cmd_workup(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (args.reference is None) != (args.target is None):
        raise ValueError("Provide both --reference and --target for transportability diagnostics")

    config = _load_config(args.config)
    config = _apply_modality_preset_for_args(config, args)
    admet_info = _load_admet_for_args(args)
    refua_info = _load_refua_for_args(args)

    if admet_info is None and refua_info is not None and refua_info["selected_admet"] is not None:
        selected_admet = dict(refua_info["selected_admet"])
        admet_info = {
            "profile": selected_admet,
            "summary": summarize_admet_profile(selected_admet),
        }

    if refua_info is not None and bool(args.refua_apply):
        policy = _refua_policy_from_args(args)
        config, refua_adjustments = apply_refua_adjustments(
            config,
            refua_info["payload"],
            policy=policy,
        )
        refua_info["adjustments"] = refua_adjustments
    elif refua_info is not None:
        refua_info["adjustments"] = None

    if (
        admet_info is not None
        and bool(args.admet_adjustments)
        and not (refua_info is not None and bool(args.refua_apply))
    ):
        config, adjustments = apply_admet_adjustments(config, admet_info["profile"])
        admet_info["adjustments"] = adjustments
    elif admet_info is not None:
        admet_info["adjustments"] = None

    run_result = simulate_trials(config)
    run_payload = trial_result_to_mapping(run_result)
    if admet_info is not None:
        run_payload["admet"] = {
            "summary": admet_info["summary"],
            "adjustments": admet_info["adjustments"],
        }
    if refua_info is not None:
        run_payload["refua"] = {
            "summary": refua_info["summary"],
            "adjustments": refua_info["adjustments"],
        }
    run_path = output_dir / "run.json"
    dump_json(run_path, run_payload)

    recommendation = recommend_protocol(
        config,
        replicates_per_candidate=max(int(args.replicates_per_candidate), 20),
        candidate_total_n=list(args.candidate_total_n) if args.candidate_total_n else None,
        candidate_interims=list(args.candidate_interims) if args.candidate_interims else None,
    )
    protocol_payload = {
        "protocol": recommendation.protocol,
        "candidates": [asdict(candidate) for candidate in recommendation.candidates],
    }
    protocol_path = output_dir / "protocol.json"
    protocol_md_path = output_dir / "protocol.md"
    dump_json(protocol_path, protocol_payload)
    protocol_md_path.write_text(
        render_protocol_markdown(recommendation.protocol),
        encoding="utf-8",
    )

    optimization_payload = optimize_design_space(
        config,
        candidate_total_n=list(args.candidate_total_n) if args.candidate_total_n else None,
        candidate_interims=list(args.candidate_interims) if args.candidate_interims else None,
        replicates_per_candidate=max(int(args.replicates_per_candidate), 20),
    )
    optimization_path = output_dir / "optimization.json"
    optimization_md_path = output_dir / "optimization.md"
    dump_json(optimization_path, optimization_payload)
    optimization_md_path.write_text(
        optimization_to_markdown(optimization_payload),
        encoding="utf-8",
    )

    voi_payload = estimate_value_of_information(
        config,
        candidate_extra_n=list(args.voi_extra_n) if args.voi_extra_n else None,
        replicates_per_scenario=max(int(args.voi_replicates_per_scenario), 20),
    )
    voi_path = output_dir / "voi.json"
    voi_md_path = output_dir / "voi.md"
    dump_json(voi_path, voi_payload)
    voi_md_path.write_text(voi_to_markdown(voi_payload), encoding="utf-8")

    transportability_payload: dict[str, Any] | None = None
    transportability_path: Path | None = None
    transportability_md_path: Path | None = None
    if args.reference is not None and args.target is not None:
        reference = load_tabular(str(args.reference))
        target = load_tabular(str(args.target))
        transportability_payload = assess_transportability(
            reference,
            target,
            columns=list(args.transport_columns) if args.transport_columns else None,
        )
        transportability_path = output_dir / "transportability.json"
        transportability_md_path = output_dir / "transportability.md"
        dump_json(transportability_path, transportability_payload)
        transportability_md_path.write_text(
            transportability_to_markdown(transportability_payload),
            encoding="utf-8",
        )

    advice_payload = build_advice_report(
        run_payload,
        protocol_payload=protocol_payload,
        optimization_payload=optimization_payload,
        voi_payload=voi_payload,
        admet_payload=admet_info["summary"] if admet_info is not None else None,
        include_sensitivity=bool(args.include_sensitivity),
        sensitivity_replicates=int(args.sensitivity_replicates),
        sensitivity_delta=float(args.sensitivity_delta),
    )
    advice_path = output_dir / "advice.json"
    advice_md_path = output_dir / "advice.md"
    write_advice_artifacts(
        advice_payload,
        output_json=advice_path,
        output_markdown=advice_md_path,
    )

    manifest = {
        "run": str(run_path),
        "protocol": str(protocol_path),
        "optimization": str(optimization_path),
        "voi": str(voi_path),
        "advice": str(advice_path),
        "transportability": str(transportability_path) if transportability_path else None,
    }
    manifest_path = output_dir / "manifest.json"
    dump_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "summary": run_payload["summary"],
                "recommendations": summarize_recommendations(advice_payload["recommendations"])[:5],
                "transportability": transportability_payload["risk_level"]
                if transportability_payload is not None
                else None,
                "refua_selected_ligand": _mapping(
                    _mapping(run_payload.get("refua")).get("summary")
                ).get("selected_ligand_id"),
            },
            indent=2,
        )
    )
    return 0


def _cmd_integrate_refua(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    config = _apply_modality_preset_for_args(config, args)
    payload = load_refua_payload(args.refua_json)
    policy = _refua_policy_from_args(args)
    adjusted, report = apply_refua_adjustments(config, payload, policy=policy)
    _dump_config_like(args.output_config, config_to_mapping(adjusted))
    if args.output_summary is not None:
        dump_json(args.output_summary, report)

    management = _mapping(report.get("management"))
    summary = _mapping(report.get("summary"))
    print(
        json.dumps(
            {
                "output_config": str(args.output_config),
                "output_summary": str(args.output_summary) if args.output_summary else None,
                "selected_ligand_id": summary.get("selected_ligand_id"),
                "candidate_count": summary.get("candidate_count"),
                "readiness_index": management.get("readiness_index"),
            },
            indent=2,
        )
    )
    return 0


def _load_config(path: Path) -> Any:
    payload = load_mapping(path)
    return config_from_mapping(payload)


def _load_config_or_run(*, config_path: Path | None, run_path: Path | None) -> Any:
    if config_path is not None:
        return _load_config(config_path)
    if run_path is None:
        raise ValueError("Provide either --config or --run")

    run_payload = load_mapping(run_path)
    run_config = run_payload.get("config")
    if not isinstance(run_config, dict):
        raise ValueError("Run artifact missing config object")
    return config_from_mapping(run_config)


def _add_admet_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--admet-json",
        type=Path,
        default=None,
        help="Optional JSON file containing precomputed ADMET profile.",
    )
    parser.add_argument(
        "--admet-smiles",
        default=None,
        help="Optional SMILES string to compute ADMET profile via refua.",
    )
    parser.add_argument(
        "--admet-model-variant",
        default="9b-chat",
        help="TxGemma model variant used for ADMET when --admet-smiles is provided.",
    )


def _add_refua_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--refua-json",
        type=Path,
        default=None,
        help="Optional Refua payload JSON/YAML with affinity/ADMET/property signals.",
    )
    parser.add_argument(
        "--refua-apply",
        action="store_true",
        help="Apply Refua payload adjustments to trial assumptions.",
    )
    parser.add_argument(
        "--refua-ligand-id",
        default=None,
        help="Optional preferred ligand id when multiple Refua candidates are present.",
    )
    parser.add_argument(
        "--refua-max-candidate-arms",
        type=int,
        default=4,
        help="Maximum number of candidate treatment arms generated from Refua ligands.",
    )
    parser.add_argument(
        "--refua-strict-contract",
        action="store_true",
        help="Require canonical Refua payload schema keys and reject legacy aliases.",
    )


def _add_modality_preset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modality-preset",
        choices=list_modality_presets(),
        default=None,
        help="Optional shared modality preset (for example biologic-sc).",
    )
    parser.add_argument(
        "--preset-dosing-interval-hours",
        type=float,
        default=None,
        help="Optional dosing interval override (hours) when a modality preset is used.",
    )
    parser.add_argument(
        "--preset-tmdd-strength",
        type=float,
        default=None,
        help="Optional TMDD-strength override when a modality preset is used.",
    )


def _load_admet_for_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.admet_json is None and args.admet_smiles is None:
        return None
    if args.admet_json is not None and args.admet_smiles is not None:
        raise ValueError("Use either --admet-json or --admet-smiles, not both")

    if args.admet_json is not None:
        profile = load_admet_profile(args.admet_json)
    else:
        profile = compute_admet_profile(
            str(args.admet_smiles),
            model_variant=str(args.admet_model_variant),
        )
    return {
        "profile": profile,
        "summary": summarize_admet_profile(profile),
    }


def _load_refua_for_args(args: argparse.Namespace) -> dict[str, Any] | None:
    payload_path = getattr(args, "refua_json", None)
    if payload_path is None:
        return None

    payload = load_refua_payload(payload_path)
    preferred_ligand_id = getattr(args, "refua_ligand_id", None)
    strict_contract = bool(getattr(args, "refua_strict_contract", False))
    summary = summarize_refua_payload(
        payload,
        preferred_ligand_id=preferred_ligand_id,
        strict_contract=strict_contract,
    )
    selected_admet = extract_admet_profile_from_refua_payload(
        payload,
        preferred_ligand_id=preferred_ligand_id,
        strict_contract=strict_contract,
    )
    return {
        "payload": payload,
        "summary": summary,
        "selected_admet": selected_admet,
    }


def _refua_policy_from_args(args: argparse.Namespace) -> RefuaIntegrationPolicy:
    preferred_ligand_id = getattr(args, "refua_ligand_id", None)
    max_candidate_arms = max(int(getattr(args, "refua_max_candidate_arms", 4)), 1)
    return RefuaIntegrationPolicy(
        preferred_ligand_id=str(preferred_ligand_id) if preferred_ligand_id else None,
        max_candidate_arms=max_candidate_arms,
        strict_contract=bool(getattr(args, "refua_strict_contract", False)),
    )


def _apply_modality_preset_for_args(config: Any, args: argparse.Namespace) -> Any:
    preset = getattr(args, "modality_preset", None)
    if preset is None:
        return config
    dosing_interval_raw = getattr(args, "preset_dosing_interval_hours", None)
    tmdd_strength_raw = getattr(args, "preset_tmdd_strength", None)
    return apply_modality_preset(
        config,
        preset=str(preset),
        dosing_interval_hours=(
            float(dosing_interval_raw) if dosing_interval_raw is not None else None
        ),
        tmdd_strength=float(tmdd_strength_raw) if tmdd_strength_raw is not None else None,
    )


def _admet_from_run_payload(run_payload: dict[str, Any]) -> dict[str, Any] | None:
    admet = run_payload.get("admet")
    if not isinstance(admet, dict):
        refua = run_payload.get("refua")
        if isinstance(refua, dict):
            refua_summary = refua.get("summary")
            if isinstance(refua_summary, dict):
                selected = refua_summary.get("selected_candidate")
                if isinstance(selected, dict):
                    selected_admet = selected.get("admet_summary")
                    if isinstance(selected_admet, dict):
                        return dict(selected_admet)
        return None

    summary = admet.get("summary")
    if isinstance(summary, dict):
        return dict(summary)
    return None


def _dump_config_like(path: Path, payload: dict[str, Any]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        dump_json(path, payload)
    else:
        dump_yaml(path, payload)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
