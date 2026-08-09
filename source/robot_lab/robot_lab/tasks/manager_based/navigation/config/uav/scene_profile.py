# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Validated scene profiles for tilting-UAV navigation tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


SCENE_PROFILE_PATH = Path(__file__).with_name("scene_profiles.yaml")


@dataclass(frozen=True)
class UavNavigationSceneProfile:
    """Immutable layout and sampling parameters for one static USD scene."""

    name: str
    layout: str
    usd_path: str
    scene_env_var: str
    rows: int
    columns: int
    cell_size: float
    scene_origin: tuple[float, float, float]
    bounds_half_size: float
    goal_distance_scale: float
    start_goal_sampler: str
    edge_offset: float
    lateral_range: tuple[float, float]
    episode_length_s: float
    default_num_envs: int
    curriculum: bool
    use_scene_identity: bool
    evaluation_mode: str
    lidar_horizontal_res: float
    lidar_ray_count: int

    @property
    def num_cells(self) -> int:
        return self.rows * self.columns


def _number_tuple(raw: Any, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise ValueError(f"Scene profile field '{name}' must contain exactly {length} numbers.")
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Scene profile field '{name}' must contain only numbers.") from error


def _profile_from_dict(name: str, raw: Mapping[str, Any]) -> UavNavigationSceneProfile:
    required_fields = {
        "layout",
        "usd_path",
        "scene_env_var",
        "rows",
        "columns",
        "cell_size",
        "scene_origin",
        "bounds_half_size",
        "goal_distance_scale",
        "start_goal_sampler",
        "edge_offset",
        "lateral_range",
        "episode_length_s",
        "default_num_envs",
        "curriculum",
        "use_scene_identity",
        "evaluation_mode",
        "lidar_horizontal_res",
        "lidar_ray_count",
    }
    missing_fields = sorted(required_fields - set(raw))
    if missing_fields:
        raise ValueError(f"Scene profile '{name}' is missing fields: {missing_fields}")

    profile = UavNavigationSceneProfile(
        name=name,
        layout=str(raw["layout"]),
        usd_path=str(raw["usd_path"]),
        scene_env_var=str(raw["scene_env_var"]),
        rows=int(raw["rows"]),
        columns=int(raw["columns"]),
        cell_size=float(raw["cell_size"]),
        scene_origin=_number_tuple(raw["scene_origin"], name="scene_origin", length=3),
        bounds_half_size=float(raw["bounds_half_size"]),
        goal_distance_scale=float(raw["goal_distance_scale"]),
        start_goal_sampler=str(raw["start_goal_sampler"]),
        edge_offset=float(raw["edge_offset"]),
        lateral_range=_number_tuple(raw["lateral_range"], name="lateral_range", length=2),
        episode_length_s=float(raw["episode_length_s"]),
        default_num_envs=int(raw["default_num_envs"]),
        curriculum=bool(raw["curriculum"]),
        use_scene_identity=bool(raw["use_scene_identity"]),
        evaluation_mode=str(raw["evaluation_mode"]),
        lidar_horizontal_res=float(raw["lidar_horizontal_res"]),
        lidar_ray_count=int(raw["lidar_ray_count"]),
    )

    if profile.layout not in {"grid", "single"}:
        raise ValueError(f"Scene profile '{name}' has unsupported layout '{profile.layout}'.")
    if profile.rows <= 0 or profile.columns <= 0 or profile.cell_size <= 0.0:
        raise ValueError(f"Scene profile '{name}' has non-positive grid dimensions.")
    if profile.layout == "single" and (profile.rows, profile.columns) != (1, 1):
        raise ValueError(f"Single-scene profile '{name}' must use rows=1 and columns=1.")
    if profile.bounds_half_size <= 0.0 or profile.bounds_half_size > profile.cell_size / 2.0:
        raise ValueError(f"Scene profile '{name}' has invalid bounds_half_size.")
    if not 0.0 < profile.edge_offset < profile.bounds_half_size:
        raise ValueError(f"Scene profile '{name}' has invalid edge_offset.")
    if profile.lateral_range[0] >= profile.lateral_range[1]:
        raise ValueError(f"Scene profile '{name}' has invalid lateral_range.")
    if max(abs(value) for value in profile.lateral_range) >= profile.bounds_half_size:
        raise ValueError(f"Scene profile '{name}' lateral_range exceeds its scene bounds.")
    if profile.start_goal_sampler != "scene_edges":
        raise ValueError(
            f"Scene profile '{name}' has unsupported sampler '{profile.start_goal_sampler}'."
        )
    if profile.evaluation_mode not in {"grid_cell", "single_scene"}:
        raise ValueError(f"Scene profile '{name}' has unsupported evaluation_mode.")
    expected_evaluation_mode = "grid_cell" if profile.layout == "grid" else "single_scene"
    if profile.evaluation_mode != expected_evaluation_mode:
        raise ValueError(
            f"Scene profile '{name}' layout '{profile.layout}' requires "
            f"evaluation_mode='{expected_evaluation_mode}'."
        )
    if profile.layout == "single" and profile.use_scene_identity:
        raise ValueError(f"Single-scene profile '{name}' cannot enable grid scene identity.")
    if profile.episode_length_s <= 0.0 or profile.default_num_envs <= 0:
        raise ValueError(f"Scene profile '{name}' has invalid runtime defaults.")
    if profile.lidar_horizontal_res <= 0.0 or profile.lidar_horizontal_res > 360.0:
        raise ValueError(f"Scene profile '{name}' has invalid lidar_horizontal_res.")
    expected_ray_count = int(round(360.0 / profile.lidar_horizontal_res))
    if expected_ray_count <= 0 or abs(expected_ray_count * profile.lidar_horizontal_res - 360.0) > 1.0e-4:
        raise ValueError(
            f"Scene profile '{name}' lidar_horizontal_res must divide 360 degrees."
        )
    if profile.lidar_ray_count != expected_ray_count:
        raise ValueError(
            f"Scene profile '{name}' lidar_ray_count={profile.lidar_ray_count} does not match "
            f"horizontal_res={profile.lidar_horizontal_res}."
        )
    return profile


def load_scene_profiles(path: Path = SCENE_PROFILE_PATH) -> Mapping[str, UavNavigationSceneProfile]:
    """Load and validate all scene profiles from YAML."""
    with path.open("r", encoding="utf-8") as profile_file:
        document = yaml.safe_load(profile_file)
    raw_profiles = document.get("profiles") if isinstance(document, dict) else None
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError(f"No scene profiles found in {path}.")
    profiles = {
        str(name): _profile_from_dict(str(name), raw)
        for name, raw in raw_profiles.items()
        if isinstance(raw, dict)
    }
    if len(profiles) != len(raw_profiles):
        raise ValueError(f"Every scene profile in {path} must be a mapping.")
    return MappingProxyType(profiles)


def resolve_scene_path(profile: UavNavigationSceneProfile) -> Path:
    """Resolve a profile USD, allowing profile-specific and legacy CLI overrides."""
    configured_path = (
        os.environ.get(profile.scene_env_var)
        or os.environ.get("NAVRL_UAV_USD_SCENE")
        or os.environ.get("NAVRL_USD_SCENE")
    )
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return (Path(ISAACLAB_ASSETS_DATA_DIR) / profile.usd_path).resolve()


SCENE_PROFILES = load_scene_profiles()
GRID_8X8_PROFILE = SCENE_PROFILES["grid_8x8"]
WAREHOUSE_100M_PROFILE = SCENE_PROFILES["warehouse_100m"]
