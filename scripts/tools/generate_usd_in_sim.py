#!/usr/bin/env python3
"""Generate obstacle USD from within Isaac Sim (guaranteed compatible)."""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="source/robot_lab/data/environments/navrl_sim.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import os
from pxr import Usd, UsdGeom, Sdf, Gf

# Config (same as generate_navrl_usd.py)
NUM_ROWS, NUM_COLS = 8, 8
CELL_SIZE = 50.0
EDGE_MARGIN = 5.0
DIFFICULTY = np.linspace(0, 1, NUM_ROWS)
NUM_OBJS = np.round(15 + DIFFICULTY * 75).astype(int)
SIZE_MIN, SIZE_MAX = 0.1, 2.0
H_MIN, H_MAX = 0.5, 4.0
CONE_SEGMENTS = 24
CYLINDER_SEGMENTS = 24

outpath = os.path.abspath(args.output)
os.makedirs(os.path.dirname(outpath), exist_ok=True)

stage = Usd.Stage.CreateNew(outpath)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(stage.DefinePrim("/World"))
root = stage.DefinePrim("/World", "Xform")

# Global ground plane (a simple mesh)
ground = UsdGeom.Cube.Define(stage, "/World/GlobalGround")
ground.AddScaleOp().Set(Gf.Vec3f(NUM_COLS * CELL_SIZE / 2, NUM_ROWS * CELL_SIZE / 2, 0.01))
ground.AddTranslateOp().Set(Gf.Vec3f(NUM_COLS * CELL_SIZE / 2, NUM_ROWS * CELL_SIZE / 2, 0))

obs_ctr = 0
total = 0


def define_cone_mesh(stage, prim_path, radius, height, segments=CONE_SEGMENTS):
    bottom_z = -height / 2.0
    top_z = height / 2.0
    points = [Gf.Vec3f(0.0, 0.0, top_z), Gf.Vec3f(0.0, 0.0, bottom_z)]
    for index in range(segments):
        theta = 2.0 * np.pi * index / segments
        points.append(Gf.Vec3f(radius * np.cos(theta), radius * np.sin(theta), bottom_z))

    counts = []
    indices = []
    for index in range(segments):
        a = 2 + index
        b = 2 + ((index + 1) % segments)
        counts.append(3)
        indices.extend([0, a, b])
    for index in range(segments):
        a = 2 + ((index + 1) % segments)
        b = 2 + index
        counts.append(3)
        indices.extend([1, a, b])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(points))
    return mesh


def define_cylinder_mesh(stage, prim_path, radius, height, segments=CYLINDER_SEGMENTS):
    bottom_z = -height / 2.0
    top_z = height / 2.0
    points = [Gf.Vec3f(0.0, 0.0, top_z), Gf.Vec3f(0.0, 0.0, bottom_z)]
    top_start = len(points)
    for index in range(segments):
        theta = 2.0 * np.pi * index / segments
        points.append(Gf.Vec3f(radius * np.cos(theta), radius * np.sin(theta), top_z))
    bottom_start = len(points)
    for index in range(segments):
        theta = 2.0 * np.pi * index / segments
        points.append(Gf.Vec3f(radius * np.cos(theta), radius * np.sin(theta), bottom_z))

    counts = []
    indices = []
    for index in range(segments):
        top_a = top_start + index
        top_b = top_start + ((index + 1) % segments)
        bottom_a = bottom_start + index
        bottom_b = bottom_start + ((index + 1) % segments)
        counts.extend([3, 3])
        indices.extend([top_a, bottom_a, bottom_b])
        indices.extend([top_a, bottom_b, top_b])
    for index in range(segments):
        a = top_start + index
        b = top_start + ((index + 1) % segments)
        counts.append(3)
        indices.extend([0, b, a])
    for index in range(segments):
        a = bottom_start + ((index + 1) % segments)
        b = bottom_start + index
        counts.append(3)
        indices.extend([1, a, b])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(points))
    return mesh

for row in range(NUM_ROWS):
    diff = DIFFICULTY[row]
    n_obj = NUM_OBJS[row]
    r_min, r_max = max(0.25, SIZE_MIN / 2), max(0.5, SIZE_MAX / 2)
    rng = np.random.RandomState(row * 100 + int(sum(NUM_OBJS)))

    for col in range(NUM_COLS):
        cx, cy = col * CELL_SIZE, row * CELL_SIZE
        cell = stage.DefinePrim(f"/World/Cell_{row}_{col}", "Xform")

        # Cell ground
        g = UsdGeom.Cube.Define(stage, f"/World/Cell_{row}_{col}/Ground")
        g.AddScaleOp().Set(Gf.Vec3f(CELL_SIZE / 2, CELL_SIZE / 2, 0.01))
        g.AddTranslateOp().Set(Gf.Vec3f(cx + CELL_SIZE / 2, cy + CELL_SIZE / 2, 0))

        # Walls (4 sides)
        wall_h, wall_t = 0.5, 0.1
        for wx, wy, sx, sy in [
            (cx + CELL_SIZE / 2, cy + wall_t / 2, CELL_SIZE, wall_t),
            (cx + CELL_SIZE / 2, cy + CELL_SIZE - wall_t / 2, CELL_SIZE, wall_t),
            (cx + wall_t / 2, cy + CELL_SIZE / 2, wall_t, CELL_SIZE),
            (cx + CELL_SIZE - wall_t / 2, cy + CELL_SIZE / 2, wall_t, CELL_SIZE),
        ]:
            w = UsdGeom.Cube.Define(stage, f"/World/Cell_{row}_{col}/Wall_{obs_ctr}")
            w.AddScaleOp().Set(Gf.Vec3f(sx / 2, sy / 2, wall_h / 2))
            w.AddTranslateOp().Set(Gf.Vec3f(wx, wy, wall_h / 2))
            obs_ctr += 1

        # Random centers
        centers = [(rng.uniform(cx + EDGE_MARGIN, cx + CELL_SIZE - EDGE_MARGIN),
                    rng.uniform(cy + EDGE_MARGIN, cy + CELL_SIZE - EDGE_MARGIN))
                   for _ in range(n_obj)]

        def rh():
            return max(0.5, rng.uniform(H_MIN, H_MAX))

        n_each = n_obj // 3
        n_boxes, n_cyls = n_each, n_each
        n_cones = n_obj - n_boxes - n_cyls
        idx = 0

        for _ in range(n_boxes):
            x, y = centers[idx]; idx += 1; h = rh()
            wb, db = rng.uniform(SIZE_MIN, SIZE_MAX), rng.uniform(SIZE_MIN, SIZE_MAX)
            b = UsdGeom.Cube.Define(stage, f"/World/Cell_{row}_{col}/Box_{obs_ctr}")
            b.AddScaleOp().Set(Gf.Vec3f(wb / 2, db / 2, h / 2))
            b.AddTranslateOp().Set(Gf.Vec3f(x, y, h / 2))
            obs_ctr += 1

        for _ in range(n_cyls):
            x, y = centers[idx]; idx += 1; h = rh()
            r = rng.uniform(r_min, r_max)
            c = define_cylinder_mesh(stage, f"/World/Cell_{row}_{col}/Cyl_{obs_ctr}", r, h)
            c.AddTranslateOp().Set(Gf.Vec3f(x, y, h / 2))
            obs_ctr += 1

        for _ in range(n_cones):
            x, y = centers[idx]; idx += 1; h = rh()
            r = rng.uniform(r_min, min(r_max, h / 2))
            c = define_cone_mesh(stage, f"/World/Cell_{row}_{col}/Cone_{obs_ctr}", r, h)
            c.AddTranslateOp().Set(Gf.Vec3f(x, y, h / 2))
            obs_ctr += 1

        total += n_obj

stage.Save()
print(f"DONE: {outpath}")
print(f"Cells: {NUM_ROWS}x{NUM_COLS}, obstacles: {total}, prims: {obs_ctr}")

simulation_app.close()
