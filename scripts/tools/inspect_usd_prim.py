#!/usr/bin/env python3
"""Print all prim paths under /World/ground to debug USD loading."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str,
    default="/home/siwufei/work/code/isaaclab/robot_lab/source/robot_lab/data/environments/navrl_scene.usda")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd
import os

usd_path = os.path.abspath(args.usd)

stage = Usd.Stage.Open(usd_path)
if not stage:
    print(f"ERROR: Could not open {usd_path}")
    simulation_app.close()
    exit()

print(f"USD prims in {usd_path}:")
print(f"Root layer: {stage.GetRootLayer().realPath}")

for prim in stage.Traverse():
    depth = prim.GetPath().pathString.count('/') - 1
    print(f"  {'  ' * depth}{prim.GetPath().pathString} ({prim.GetTypeName()})")

print(f"\n--- Test: spawn USD at /World/ground ---")
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_from_usd, UsdFileCfg

cfg = UsdFileCfg(usd_path=usd_path)
spawn_from_usd("/World/ground", cfg)

# Force stage update
import omni.usd
stage2 = omni.usd.get_context().get_stage()

print("\nPrims under /World:")
for prim in stage2.Traverse():
    p = prim.GetPath().pathString
    if p.startswith("/World/ground") and p.count('/') <= 4:
        print(f"  {p} ({prim.GetTypeName()})")

simulation_app.close()
