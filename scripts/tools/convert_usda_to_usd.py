#!/usr/bin/env python3
"""Convert user-edited USDA to loadable USD with defaultPrim."""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Input USDA file")
parser.add_argument("--output", "-o", help="Output USD file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
from pxr import Usd, Sdf

inpath = os.path.abspath(args.input)
outpath = args.output or inpath.replace(".usda", ".usd")

stage = Usd.Stage.Open(inpath)
# Ensure defaultPrim is set
if not stage.GetDefaultPrim():
    # Find first Xform child as default
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim.GetTypeName() == "Xform":
            stage.SetDefaultPrim(prim)
            print(f"Set defaultPrim = {prim.GetName()}")
            break

stage.Export(outpath)
print(f"Converted: {inpath} -> {outpath}")

simulation_app.close()
