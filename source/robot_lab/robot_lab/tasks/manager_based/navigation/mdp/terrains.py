# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Custom terrain with 5 obstacle types + bumpy ground + cell walls."""

from __future__ import annotations

import numpy as np
import os
import trimesh

from isaaclab.terrains.trimesh import mesh_terrains_cfg
from isaaclab.terrains.trimesh.mesh_terrains import make_box, make_cone, make_cylinder

_JSON_PATH = os.environ.get("NAVRL_OBSTACLE_JSON", "")


def _build_ground(cfg, difficulty, bump_amplitude, grid_spacing=0.5):
    """Generate bumpy ground mesh and a height lookup function."""
    nx = int(cfg.size[0] / grid_spacing) + 1
    ny = int(cfg.size[1] / grid_spacing) + 1
    xs = np.linspace(0, cfg.size[0], nx)
    ys = np.linspace(0, cfg.size[1], ny)
    xv, yv = np.meshgrid(xs, ys)

    rng = np.random.RandomState(int(difficulty * 1000 + cfg.size[0]))
    # Per-vertex random noise for sharp, locomotion-like bumps
    zv = np.random.uniform(-1, 1, xv.shape) * bump_amplitude

    vertices = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1)
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = i * nx + j
            faces.append([a, a + 1, a + nx + 1])
            faces.append([a, a + nx + 1, a + nx])
    ground = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))

    # Height lookup: bilinear interpolation on the grid
    def ground_z(x, y):
        """Return ground height at (x, y) via bilinear interpolation."""
        fx = np.clip((x - xs[0]) / grid_spacing, 0, nx - 2)
        fy = np.clip((y - ys[0]) / grid_spacing, 0, ny - 2)
        ix0 = np.floor(fx).astype(int)
        iy0 = np.floor(fy).astype(int)
        ix1, iy1 = ix0 + 1, iy0 + 1
        wx, wy = fx - ix0, fy - iy0
        return (zv[iy0, ix0] * (1 - wx) * (1 - wy) +
                zv[iy0, ix1] * wx * (1 - wy) +
                zv[iy1, ix0] * (1 - wx) * wy +
                zv[iy1, ix1] * wx * wy)

    return ground, ground_z


def mixed_objects_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshRepeatedObjectsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end

    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    platform_height = cfg.platform_height if cfg.platform_height >= 0.0 else height

    size_min = cp_0.size[0] if hasattr(cp_0, "size") else getattr(cp_0, "radius", 0.1)
    size_max = cp_1.size[0] if hasattr(cp_1, "size") else getattr(cp_1, "radius", 1.0)
    radius_min = max(0.25, size_min / 2.0)
    radius_max = max(0.5, size_max / 2.0)

    # Build bumpy ground FIRST, so we can place obstacles on top
    bump_amplitude = 0.05 + difficulty * 0.10  # 0.05m easy → 0.15m hard
    ground, ground_z = _build_ground(cfg, difficulty, bump_amplitude)

    meshes_list = []
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * platform_height))

    edge_margin = 5.0

    def _sample_centers(n):
        centers = np.zeros((n, 3))
        centers[:, 0] = np.random.uniform(edge_margin, cfg.size[0] - edge_margin, n)
        centers[:, 1] = np.random.uniform(edge_margin, cfg.size[1] - edge_margin, n)
        return centers

    def _rand_height():
        abs_h = np.random.uniform(*cfg.abs_height_noise)
        rel_h = np.random.uniform(*cfg.rel_height_noise)
        return max(0.01, height * rel_h + abs_h)

    # Load obstacles from JSON if specified (from USD)
    use_json = _JSON_PATH and os.path.exists(_JSON_PATH)
    if use_json:
        import json
        with open(_JSON_PATH) as f:
            all_cells = json.load(f)
        row = int(difficulty * 7 + 0.5)
        cell_names = [c for c in all_cells if c.startswith(f"Cell_{row}_")]
        col = np.random.randint(0, len(cell_names)) if cell_names else 0
        cell = cell_names[col] if cell_names else sorted(all_cells.keys())[0]
        n_boxes = n_cylinders = n_cones = n_smooth_cones = 0
        for obs in all_cells.get(cell, []):
            if not obs.get("enabled", True):
                continue
            cx, cy, cz = obs["center"][0], obs["center"][1], obs["center"][2]
            gz = ground_z(cx, cy)
            t, sz = obs["type"], obs["size"]
            if t == "box":
                w, d, h = sz[0], sz[1], sz[2]
                meshes_list.append(make_box(center=(cx, cy, gz + h / 2), height=h, length=w, width=d))
            elif t == "cylinder":
                r, h = sz[0], sz[1]
                meshes_list.append(make_cylinder(center=(cx, cy, gz + h / 2), height=h, radius=r))
            elif t == "cone":
                r, h = sz[0], sz[1]
                meshes_list.append(make_cone(center=(cx, cy, gz), height=h, radius=r))
                cone = trimesh.creation.cone(radius=r, height=h, sections=12)
                cone.apply_transform(trimesh.transformations.translation_matrix((cx, cy, gz)))
                meshes_list.append(cone)

    if not use_json:
        # Split evenly across 4 types (no spheres - collision issues)
        n_each = num_objects // 4
        n_boxes = n_each
        n_cylinders = n_each
        n_cones = n_each
        n_smooth_cones = num_objects - n_boxes - n_cylinders - n_cones

    # All obstacles placed on top of the bumpy ground surface
    for cx, cy, _ in _sample_centers(n_boxes):
        gz = ground_z(cx, cy)
        h = _rand_height()
        w = np.random.uniform(size_min, size_max)
        d = np.random.uniform(size_min, size_max)
        meshes_list.append(make_box(center=(cx, cy, gz + h / 2), height=h, length=w, width=d))

    for cx, cy, _ in _sample_centers(n_cylinders):
        gz = ground_z(cx, cy)
        h = _rand_height()
        r = np.random.uniform(radius_min, radius_max)
        cyl = trimesh.creation.cylinder(radius=r, height=h, sections=16)
        cyl.apply_transform(trimesh.transformations.translation_matrix((cx, cy, gz + h / 2)))
        meshes_list.append(cyl)

    # trimesh cone has base at z=0, tip at z=height; place base on ground
    for cx, cy, _ in _sample_centers(n_cones):
        gz = ground_z(cx, cy)
        h = _rand_height()
        r = np.random.uniform(radius_min, min(radius_max, h / 2.0))
        meshes_list.append(make_cone(center=(cx, cy, gz), height=h, radius=r))

    for cx, cy, _ in _sample_centers(n_smooth_cones):
        gz = ground_z(cx, cy)
        h = _rand_height()
        r = np.random.uniform(radius_min, min(radius_max, h / 2.0))
        cone = trimesh.creation.cone(radius=r, height=h, sections=12)
        cone.apply_transform(trimesh.transformations.translation_matrix((cx, cy, gz)))
        meshes_list.append(cone)

    # Spheres removed - collision mesh too coarse, dog clips through

    # Add ground after obstacles
    meshes_list.append(ground)

    # Perimeter walls
    wall_h, wall_t = 0.5, 0.1
    sx, sy = cfg.size[0], cfg.size[1]
    meshes_list.append(trimesh.creation.box((sx, wall_t, wall_h),
        trimesh.transformations.translation_matrix((sx / 2, wall_t / 2, wall_h / 2))))
    meshes_list.append(trimesh.creation.box((sx, wall_t, wall_h),
        trimesh.transformations.translation_matrix((sx / 2, sy - wall_t / 2, wall_h / 2))))
    meshes_list.append(trimesh.creation.box((wall_t, sy, wall_h),
        trimesh.transformations.translation_matrix((wall_t / 2, sy / 2, wall_h / 2))))
    meshes_list.append(trimesh.creation.box((wall_t, sy, wall_h),
        trimesh.transformations.translation_matrix((sx - wall_t / 2, sy / 2, wall_h / 2))))

    # Central platform
    dim = (cfg.platform_width, cfg.platform_width, 0.5 * platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.25 * platform_height)
    meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    return meshes_list, origin
