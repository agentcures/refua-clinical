"""Shared modality presets for CLI and object API ergonomics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import RouteKind, SimulationConfig


class ModalityPreset(Protocol):
    def apply(
        self,
        config: SimulationConfig,
        *,
        dosing_interval_hours: float | None = None,
        tmdd_strength: float | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class BiologicsPreset:
    name: str
    route: RouteKind
    bioavailability: float
    ka_per_hour: float
    default_interval_hours: float = 336.0
    default_tmdd_strength: float = 0.35

    def apply(
        self,
        config: SimulationConfig,
        *,
        dosing_interval_hours: float | None = None,
        tmdd_strength: float | None = None,
    ) -> None:
        config.pk_model.modality = "biologic"
        config.pk_model.route = self.route
        config.pk_model.bioavailability = float(self.bioavailability)
        config.pk_model.ka_per_hour = float(self.ka_per_hour)

        config.pk_model.cl_l_per_hour = 0.25
        config.pk_model.v_l = 6.0
        config.pk_model.omega_cl = 0.20
        config.pk_model.omega_v = 0.18
        config.pk_model.covariate_effect_weight = 0.12
        config.pk_model.covariate_effect_egfr = 0.08
        config.pk_model.tmdd_strength = max(
            float(tmdd_strength if tmdd_strength is not None else self.default_tmdd_strength),
            0.0,
        )
        config.pk_model.tmdd_cavg_ref = 15.0

        # Biologics often need longer assessment windows than small molecules.
        config.endpoint.assessment_day = max(int(config.endpoint.assessment_day), 112)

        interval = max(
            float(
                dosing_interval_hours
                if dosing_interval_hours is not None
                else self.default_interval_hours
            ),
            12.0,
        )
        for arm in config.arms:
            if arm.is_control or arm.dose_mg <= 0.0:
                continue
            arm.schedule_per_day = 1
            arm.dosing_interval_hours = interval


_PRESETS: dict[str, ModalityPreset] = {
    "biologic-iv": BiologicsPreset(
        name="biologic-iv",
        route="iv",
        bioavailability=1.0,
        ka_per_hour=4.0,
    ),
    "biologic-sc": BiologicsPreset(
        name="biologic-sc",
        route="sc",
        bioavailability=0.65,
        ka_per_hour=0.03,
    ),
    "biologic-oral": BiologicsPreset(
        name="biologic-oral",
        route="oral",
        bioavailability=0.10,
        ka_per_hour=0.02,
    ),
}


def list_modality_presets() -> tuple[str, ...]:
    return tuple(_PRESETS.keys())


def apply_modality_preset(
    config: SimulationConfig,
    *,
    preset: str,
    dosing_interval_hours: float | None = None,
    tmdd_strength: float | None = None,
) -> SimulationConfig:
    key = str(preset).strip().lower()
    preset_impl = _PRESETS.get(key)
    if preset_impl is None:
        allowed = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown modality preset '{preset}'. Allowed presets: {allowed}")
    preset_impl.apply(
        config,
        dosing_interval_hours=dosing_interval_hours,
        tmdd_strength=tmdd_strength,
    )
    return config
