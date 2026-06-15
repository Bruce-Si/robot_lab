# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Debug utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_USD_CELL_SIZE = 50.0
_USD_NUM_LEVELS = 8
_USD_NUM_TYPES = 8
_USD_MAX_INIT_LEVEL = 3


class UsdCurriculumTerrain:
    """Terrain-like state holder for curriculum over a shared USD scene."""

    is_usd_curriculum = True
    has_debug_vis_implementation = False
    cfg = None

    def __init__(
        self,
        terrain_origins: torch.Tensor,
        env_origins: torch.Tensor,
        terrain_levels: torch.Tensor,
        terrain_types: torch.Tensor,
    ):
        self.terrain_origins = terrain_origins
        self.env_origins = env_origins
        self.terrain_levels = terrain_levels
        self.terrain_types = terrain_types
        self.max_terrain_level = _USD_NUM_LEVELS
        self.max_init_terrain_level = _USD_MAX_INIT_LEVEL
        self.flat_patches = {}
        self.debug_vis = False

    def set_debug_vis(self, debug_vis: bool) -> bool:
        self.debug_vis = debug_vis
        return False


def _ensure_usd_terrain_state(env: ManagerBasedEnv):
    """Create terrain-like curriculum buffers for a shared 8x8 USD scene."""
    terrain = env.scene.terrain
    if terrain is not None and getattr(terrain, "is_usd_curriculum", False):
        return terrain

    device = env.device
    terrain_origins = torch.zeros(_USD_NUM_LEVELS, _USD_NUM_TYPES, 3, device=device)
    levels_grid = torch.arange(_USD_NUM_LEVELS, device=device, dtype=torch.float32)
    types_grid = torch.arange(_USD_NUM_TYPES, device=device, dtype=torch.float32)
    # USD layout convention: x = obstacle-distribution type, y = curriculum level.
    terrain_origins[:, :, 0] = types_grid.view(1, _USD_NUM_TYPES) * _USD_CELL_SIZE + _USD_CELL_SIZE / 2
    terrain_origins[:, :, 1] = levels_grid.view(_USD_NUM_LEVELS, 1) * _USD_CELL_SIZE + _USD_CELL_SIZE / 2

    num_init_levels = _USD_MAX_INIT_LEVEL + 1
    num_init_cells = num_init_levels * _USD_NUM_TYPES
    env_idx = torch.arange(env.num_envs, device=device)
    cell_ids = env_idx % num_init_cells
    terrain_levels = torch.div(cell_ids, _USD_NUM_TYPES, rounding_mode="floor").to(torch.long)
    terrain_types = (cell_ids % _USD_NUM_TYPES).to(torch.long)
    env_origins = terrain_origins[terrain_levels, terrain_types].clone()

    terrain_state = UsdCurriculumTerrain(
        terrain_origins=terrain_origins,
        env_origins=env_origins,
        terrain_levels=terrain_levels,
        terrain_types=terrain_types,
    )
    env.scene._terrain = terrain_state

    if not hasattr(env, "_nav_usd_curriculum_logged"):
        per_cell = env.num_envs / num_init_cells
        print(
            "[INFO] USD curriculum: "
            f"{env.num_envs} envs over levels 0-{_USD_MAX_INIT_LEVEL} x {_USD_NUM_TYPES} variants "
            f"({per_cell:.2f} envs/cell)"
        )
        env._nav_usd_curriculum_logged = True

    return terrain_state


def setup_usd_scene(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Enable collision on geometry in the loaded USD scene."""
    if not os.environ.get("NAVRL_USD_SCENE"):
        return

    _ensure_usd_terrain_state(env)

    import isaaclab.sim as sim_utils
    from isaaclab.sim import schemas
    from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES
    from pxr import UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()
    scene_root = "/World/usd_scene"
    root_prim = stage.GetPrimAtPath(scene_root)
    if not root_prim.IsValid():
        print(f"[WARN] USD scene root not found at {scene_root}")
        return

    geom_types = set(PRIMITIVE_MESH_TYPES + ["Mesh"])
    geom_prims = sim_utils.get_all_matching_child_prims(
        scene_root,
        predicate=lambda prim: prim.GetTypeName() in geom_types,
        stage=stage,
    )
    collision_cfg = schemas.CollisionPropertiesCfg(collision_enabled=True)
    mesh_collision_cfg = schemas.ConvexHullPropertiesCfg()
    enabled_count = 0
    mesh_count = 0
    primitive_count = 0
    type_counts = {}
    non_triangle_mesh_count = 0
    non_triangle_face_count = 0
    for prim in geom_prims:
        prim_path = prim.GetPath().pathString
        prim_type = prim.GetTypeName()
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        if UsdPhysics.CollisionAPI(prim):
            schemas.modify_collision_properties(prim_path, collision_cfg, stage=stage)
        else:
            schemas.define_collision_properties(prim_path, collision_cfg, stage=stage)

        if prim_type == "Mesh":
            mesh_count += 1
            schemas.define_mesh_collision_properties(prim_path, mesh_collision_cfg, stage=stage)

            face_counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []
            bad_faces = sum(1 for count in face_counts if count != 3)
            if bad_faces:
                non_triangle_mesh_count += 1
                non_triangle_face_count += bad_faces
        else:
            primitive_count += 1
        enabled_count += 1

    print(
        "[INFO] USD collision setup: "
        f"enabled {enabled_count} geometry prims under {scene_root} "
        f"({primitive_count} primitives, {mesh_count} meshes), types={type_counts}."
    )
    if mesh_count:
        print(
            "[INFO] USD mesh collision setup: "
            f"applied convex-hull mesh collision to {mesh_count} Mesh prims; "
            f"{non_triangle_mesh_count} meshes contain {non_triangle_face_count} non-triangle faces."
        )


def fix_origins_for_usd(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Map env origins onto the active USD curriculum cells."""
    if not os.environ.get("NAVRL_USD_SCENE"):
        return

    terrain = _ensure_usd_terrain_state(env)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    print(
        "[INFO] Remapped env origins to USD cells: "
        f"levels 0-{_USD_MAX_INIT_LEVEL}, {_USD_NUM_TYPES} variants"
    )


def configure_viewer_visibility(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Optionally hide non-viewed robots to reduce viewport rendering cost."""
    if os.environ.get("NAVRL_VIEW_ONLY_ENV", "0").lower() not in ("1", "true", "on", "yes"):
        return
    if "robot" not in env.scene.articulations:
        return

    visible_env = int(os.environ.get("NAVRL_VIEW_ENV_INDEX", "0"))
    visible_env = max(0, min(visible_env, env.num_envs - 1))
    robot = env.scene["robot"]
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    hidden_env_ids = all_env_ids[all_env_ids != visible_env]
    robot.set_visibility(False, hidden_env_ids)
    robot.set_visibility(True, [visible_env])
    print(f"[INFO] Viewer visibility: showing env {visible_env} robot, hiding {hidden_env_ids.numel()} robots.")


def debug_print_prims(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Print prim paths under /World/usd_obs and /World/ground."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # USD obstacles
    usd_obs = stage.GetPrimAtPath("/World/usd_obs")
    if usd_obs:
        print("[DEBUG] === /World/usd_obs children ===")
        for child in usd_obs.GetChildren():
            print(f"  {child.GetPath().pathString} ({child.GetTypeName()})")
    else:
        print("[DEBUG] /World/usd_obs NOT FOUND - USD not loaded!")

    # Ground
    print("[DEBUG] === /World/ground tree (depth ≤ 4) ===")
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if p.startswith("/World/ground") and p.count('/') <= 4:
            print(f"  {p} ({prim.GetTypeName()})")
    print("[DEBUG] === End ===")
