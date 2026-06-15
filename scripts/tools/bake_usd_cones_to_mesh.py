#!/usr/bin/env python3
"""Bake scaled USD Cylinder/Cone primitives into Mesh prims.

This keeps round obstacles while avoiding PhysX warnings from non-uniform scale
on Cylinder/Cone primitives. The output USD is a copy of the input with only the
selected primitive types replaced.
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replace scaled USD Cylinder/Cone prims with baked Mesh prims.")
parser.add_argument("--input", required=True, help="Input USD/USDA path.")
parser.add_argument("--output", default=None, help="Output USD/USDA path. Defaults to *_baked_round_prims.usd.")
parser.add_argument("--segments", type=int, default=24, help="Number of radial segments for each round mesh.")
parser.add_argument(
    "--types",
    nargs="+",
    default=("Cylinder", "Cone"),
    choices=("Cylinder", "Cone"),
    help="Primitive types to bake.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom


def _op_value(xformable: UsdGeom.Xformable, op_name: str, default):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == op_name:
            value = op.Get()
            return value if value is not None else default
    return default


def _as_vec3(value, default):
    if value is None:
        return Gf.Vec3d(*default)
    return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))


def _as_quat(value):
    if value is None:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    imaginary = value.GetImaginary()
    return Gf.Quatf(
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _cone_mesh(radius_x: float, radius_y: float, height: float, segments: int):
    bottom_z = -height / 2.0
    top_z = height / 2.0

    points = [Gf.Vec3f(0.0, 0.0, top_z), Gf.Vec3f(0.0, 0.0, bottom_z)]
    for index in range(segments):
        theta = 2.0 * math.pi * index / segments
        points.append(Gf.Vec3f(radius_x * math.cos(theta), radius_y * math.sin(theta), bottom_z))

    counts = []
    indices = []

    # Side triangles.
    for index in range(segments):
        a = 2 + index
        b = 2 + ((index + 1) % segments)
        counts.append(3)
        indices.extend([0, a, b])

    # Bottom cap triangles.
    for index in range(segments):
        a = 2 + ((index + 1) % segments)
        b = 2 + index
        counts.append(3)
        indices.extend([1, a, b])

    return points, counts, indices


def _cylinder_mesh(radius_x: float, radius_y: float, height: float, segments: int):
    bottom_z = -height / 2.0
    top_z = height / 2.0

    points = [Gf.Vec3f(0.0, 0.0, top_z), Gf.Vec3f(0.0, 0.0, bottom_z)]
    top_start = len(points)
    for index in range(segments):
        theta = 2.0 * math.pi * index / segments
        points.append(Gf.Vec3f(radius_x * math.cos(theta), radius_y * math.sin(theta), top_z))
    bottom_start = len(points)
    for index in range(segments):
        theta = 2.0 * math.pi * index / segments
        points.append(Gf.Vec3f(radius_x * math.cos(theta), radius_y * math.sin(theta), bottom_z))

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

    return points, counts, indices


def _axis_rotation(axis):
    if axis == "X":
        return Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), 90.0).GetQuat()
    if axis == "Y":
        return Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), -90.0).GetQuat()
    return Gf.Quatf(1.0, 0.0, 0.0, 0.0)


def main():
    input_path = os.path.abspath(args.input)
    if args.output is None:
        root, _ = os.path.splitext(input_path)
        output_path = root + "_baked_round_prims.usd"
    else:
        output_path = os.path.abspath(args.output)

    if args.segments < 8:
        raise ValueError("--segments must be at least 8.")

    in_stage = Usd.Stage.Open(input_path)
    if in_stage is None:
        raise RuntimeError(f"Could not open USD: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    in_stage.Export(output_path)

    stage = Usd.Stage.Open(output_path)
    if stage is None:
        raise RuntimeError(f"Could not open exported USD: {output_path}")

    bake_types = set(args.types)
    prim_paths = [prim.GetPath() for prim in stage.Traverse() if prim.GetTypeName() in bake_types]
    baked_counts = {prim_type: 0 for prim_type in bake_types}
    for prim_path in prim_paths:
        src_prim = stage.GetPrimAtPath(prim_path)
        prim_type = src_prim.GetTypeName()
        xformable = UsdGeom.Xformable(src_prim)

        translate = _as_vec3(_op_value(xformable, "xformOp:translate", None), (0.0, 0.0, 0.0))
        scale = _as_vec3(_op_value(xformable, "xformOp:scale", None), (1.0, 1.0, 1.0))
        orient = _as_quat(_op_value(xformable, "xformOp:orient", None))
        material_targets = src_prim.GetRelationship("material:binding").GetTargets()

        if prim_type == "Cone":
            geom = UsdGeom.Cone(src_prim)
            radius = float(geom.GetRadiusAttr().Get() or 1.0)
            height = float(geom.GetHeightAttr().Get() or 2.0)
            axis = geom.GetAxisAttr().Get() or "Z"
            points, face_counts, face_indices = _cone_mesh(
                abs(radius * scale[0]), abs(radius * scale[1]), abs(height * scale[2]), args.segments
            )
        elif prim_type == "Cylinder":
            geom = UsdGeom.Cylinder(src_prim)
            radius = float(geom.GetRadiusAttr().Get() or 1.0)
            height = float(geom.GetHeightAttr().Get() or 2.0)
            axis = geom.GetAxisAttr().Get() or "Z"
            points, face_counts, face_indices = _cylinder_mesh(
                abs(radius * scale[0]), abs(radius * scale[1]), abs(height * scale[2]), args.segments
            )
        else:
            continue

        stage.RemovePrim(prim_path)
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(points))

        mesh_xform = UsdGeom.Xformable(mesh)
        mesh_xform.AddTranslateOp().Set(translate)
        axis_orient = _axis_rotation(axis)
        final_orient = orient * axis_orient
        if final_orient != Gf.Quatf(1.0, 0.0, 0.0, 0.0):
            mesh_xform.AddOrientOp().Set(final_orient)

        if material_targets:
            mesh.GetPrim().CreateRelationship("material:binding").SetTargets(material_targets)

        baked_counts[prim_type] += 1

    stage.Save()
    print(f"DONE: {output_path}")
    for prim_type in sorted(baked_counts):
        print(f"Baked {prim_type} prims: {baked_counts[prim_type]}")
    print(f"Mesh radial segments: {args.segments}")


try:
    main()
finally:
    simulation_app.close()
