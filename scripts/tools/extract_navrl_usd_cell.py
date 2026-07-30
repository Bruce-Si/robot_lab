#!/usr/bin/env python3
"""Extract one NavRL cell and its shared boundaries without moving geometry."""

from __future__ import annotations

import argparse
import hashlib
import uuid
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Extract one Cell_<row>_<column> subtree from a NavRL USD grid.")
parser.add_argument("--input", required=True, help="Source 8-by-8 USD/USDA scene.")
parser.add_argument("--output", required=True, help="Derived single-cell USD/USDA scene.")
parser.add_argument("--row", type=int, required=True, help="Cell row (curriculum difficulty level).")
parser.add_argument("--column", type=int, required=True, help="Cell column (obstacle-layout variant).")
parser.add_argument("--cell-size", type=float, default=50.0, help="Cell edge length in meters.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdGeom


GEOMETRY_TYPES = {"Capsule", "Cone", "Cube", "Cylinder", "Mesh", "Sphere"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input USD not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output USD paths must be different.")
    if args.row < 0 or args.column < 0:
        raise ValueError("Cell row and column must be non-negative.")
    if args.cell_size <= 0.0:
        raise ValueError("Cell size must be positive.")

    source_stage = Usd.Stage.Open(str(input_path))
    if source_stage is None:
        raise RuntimeError(f"Could not open source USD: {input_path}")
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError("Source USD must define a default prim.")

    cell_name = f"Cell_{args.row}_{args.column}"
    source_cell = default_prim.GetChild(cell_name)
    if not source_cell.IsValid():
        available = sorted(
            child.GetName() for child in default_prim.GetChildren() if child.GetName().startswith("Cell_")
        )
        raise RuntimeError(f"Cell {cell_name} not found. Available cell count: {len(available)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_stage.Export(str(output_path))
    stage = Usd.Stage.Open(str(output_path))
    if stage is None:
        raise RuntimeError(f"Could not reopen derived USD: {output_path}")

    derived_root = stage.GetDefaultPrim()
    derived_cell = derived_root.GetChild(cell_name)
    dependency_prims = []

    # The source grid authors each vertical boundary only once. Cells contain
    # their right wall and reuse the left neighbor's right wall as their left
    # boundary. Copy that shared wall into the extracted cell so LiDAR scans are
    # identical to the full grid while all unrelated neighbor geometry is removed.
    if args.column > 0:
        left_cell = derived_root.GetChild(f"Cell_{args.row}_{args.column - 1}")
        if not left_cell.IsValid():
            raise RuntimeError(f"Left neighbor for {cell_name} is missing.")
        boundary_x = args.column * args.cell_size
        cell_min_y = args.row * args.cell_size
        cell_max_y = (args.row + 1) * args.cell_size
        bounds_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        candidates = []
        for child in left_cell.GetChildren():
            if "Wall" not in child.GetName():
                continue
            bounds = bounds_cache.ComputeWorldBound(child).ComputeAlignedRange()
            minimum = bounds.GetMin()
            maximum = bounds.GetMax()
            if (
                abs(float(maximum[0]) - boundary_x) < 0.2
                and abs(float(minimum[1]) - cell_min_y) < 0.2
                and abs(float(maximum[1]) - cell_max_y) < 0.2
            ):
                candidates.append(child)
        if len(candidates) != 1:
            candidate_paths = [str(candidate.GetPath()) for candidate in candidates]
            raise RuntimeError(
                f"Expected one shared left boundary for {cell_name}, found {candidate_paths}."
            )
        source_boundary = candidates[0]
        destination_path = derived_cell.GetPath().AppendChild("SharedLeftBoundary")
        if not Sdf.CopySpec(
            stage.GetRootLayer(),
            source_boundary.GetPath(),
            stage.GetRootLayer(),
            destination_path,
        ):
            raise RuntimeError(f"Failed to copy shared boundary {source_boundary.GetPath()}.")
        dependency_prims.append(str(source_boundary.GetPath()))

    # The baked v1 scene uses one shared ground instead of per-cell ground.
    # Tilted body-frame LiDAR rays can hit it, so it is part of the observation.
    keep_names = {cell_name, "GlobalGround", "Materials"}
    remove_paths = [child.GetPath() for child in derived_root.GetChildren() if child.GetName() not in keep_names]
    for prim_path in remove_paths:
        stage.RemovePrim(prim_path)

    custom_data = dict(stage.GetRootLayer().customLayerData)
    custom_data.update(
        {
            "robotlab_source_cell": cell_name,
            "robotlab_source_dependencies": ",".join(dependency_prims),
            "robotlab_source_sha256": _sha256(input_path),
            "robotlab_source_usd": input_path.name,
        }
    )
    stage.GetRootLayer().customLayerData = custom_data
    compact_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}.compact{output_path.suffix}"
    )
    stage.Export(str(compact_path))
    compact_path.replace(output_path)

    verified_stage = Usd.Stage.Open(str(output_path))
    if verified_stage is None:
        raise RuntimeError(f"Could not verify derived USD: {output_path}")
    verified_root = verified_stage.GetDefaultPrim()
    retained_cells = [
        child.GetName() for child in verified_root.GetChildren() if child.GetName().startswith("Cell_")
    ]
    if retained_cells != [cell_name]:
        raise RuntimeError(f"Expected only {cell_name}, found cells: {retained_cells}")
    geometry_count = sum(prim.GetTypeName() in GEOMETRY_TYPES for prim in verified_stage.Traverse())

    print(f"source={input_path}")
    print(f"derived={output_path}")
    print(f"default_prim={verified_root.GetPath()}")
    print(f"retained_cell={cell_name}")
    print(f"copied_dependencies={dependency_prims}")
    print(f"geometry_count={geometry_count}")
    print(f"size_bytes={output_path.stat().st_size}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
