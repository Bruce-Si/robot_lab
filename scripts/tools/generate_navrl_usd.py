#!/usr/bin/env python3
"""Generate a USD file with 8x8 cells of individual obstacle prims (ASCII format).
No Isaac Sim needed - just python. Output: data/environments/navrl_scene.usda
"""

import numpy as np
import os

# ── Config ──
NUM_ROWS = 8
NUM_COLS = 8
CELL_SIZE = 50.0
WALL_H = 0.5
WALL_T = 0.1
EDGE_MARGIN = 5.0

DIFFICULTY = np.linspace(0, 1, NUM_ROWS)
NUM_OBJS = np.round(15 + DIFFICULTY * 75).astype(int)
SIZE_MIN = 0.1
SIZE_MAX = 2.0
H_MIN = 0.5
H_MAX = 4.0

script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "..", "..", "source", "robot_lab",
                         "data", "environments", "navrl_scene.usda")
out_path = os.path.normpath(out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

lines = []
def w(s):
    lines.append(s)

w('#usda 1.0')
w('(')
w('    defaultPrim = "GlobalGround"')
w('    metersPerUnit = 1')
w('    upAxis = "Z"')
w(')')
w('')
w('# Global flat plane for LiDAR raycasting')
w('def Cube "GlobalGround"')
w('{')
w(f'    double3 xformOp:scale = ({(NUM_COLS * CELL_SIZE) / 2}, {(NUM_ROWS * CELL_SIZE) / 2}, 0.01)')
w(f'    double3 xformOp:translate = ({(NUM_COLS * CELL_SIZE) / 2}, {(NUM_ROWS * CELL_SIZE) / 2}, 0)')
w('    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
w('    rel material:binding = </Materials/Ground>')
w('}')

obs_ctr = 0
total = 0

for row in range(NUM_ROWS):
    diff = DIFFICULTY[row]
    n_obj = NUM_OBJS[row]
    r_min = max(0.25, SIZE_MIN / 2)
    r_max = max(0.5, SIZE_MAX / 2)
    rng = np.random.RandomState(row * 100 + int(sum(NUM_OBJS)))

    for col in range(NUM_COLS):
        cx, cy = col * CELL_SIZE, row * CELL_SIZE
        w(f'def Xform "Cell_{row}_{col}"')
        w(f'{{')

        # Flat ground
        gx, gy = cx + CELL_SIZE / 2, cy + CELL_SIZE / 2
        w(f'    def Cube "Ground"')
        w(f'    {{')
        w(f'        double3 xformOp:scale = ({CELL_SIZE / 2}, {CELL_SIZE / 2}, 0.01)')
        w(f'        double3 xformOp:translate = ({gx}, {gy}, 0)')
        w(f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
        w(f'        rel material:binding = </Materials/Ground>')
        w(f'    }}')

        # Walls
        wall_specs = [
            (cx + CELL_SIZE / 2, cy + WALL_T / 2, CELL_SIZE, WALL_T, WALL_H),
            (cx + CELL_SIZE / 2, cy + CELL_SIZE - WALL_T / 2, CELL_SIZE, WALL_T, WALL_H),
            (cx + WALL_T / 2, cy + CELL_SIZE / 2, WALL_T, CELL_SIZE, WALL_H),
            (cx + CELL_SIZE - WALL_T / 2, cy + CELL_SIZE / 2, WALL_T, CELL_SIZE, WALL_H),
        ]
        for wx, wy, sx, sy, sh in wall_specs:
            w(f'    def Cube "Wall_{obs_ctr}"')
            w(f'    {{')
            w(f'        double3 xformOp:scale = ({sx / 2}, {sy / 2}, {sh / 2})')
            w(f'        double3 xformOp:translate = ({wx}, {wy}, {sh / 2})')
            w(f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            w(f'        rel material:binding = </Materials/Wall>')
            w(f'    }}')
            obs_ctr += 1

        # Obstacle centers
        centers = [(rng.uniform(cx + EDGE_MARGIN, cx + CELL_SIZE - EDGE_MARGIN),
                     rng.uniform(cy + EDGE_MARGIN, cy + CELL_SIZE - EDGE_MARGIN))
                    for _ in range(n_obj)]

        def rand_h():
            return max(0.5, rng.uniform(H_MIN, H_MAX))

        n_each = n_obj // 3
        n_boxes, n_cyls = n_each, n_each
        n_cones = n_obj - n_boxes - n_cyls
        idx = 0

        # Boxes
        for _ in range(n_boxes):
            x, y = centers[idx]; idx += 1; h = rand_h()
            w_box = rng.uniform(SIZE_MIN, SIZE_MAX)
            d_box = rng.uniform(SIZE_MIN, SIZE_MAX)
            w(f'    def Cube "Box_{obs_ctr}"')
            w(f'    {{')
            w(f'        double3 xformOp:scale = ({w_box / 2}, {d_box / 2}, {h / 2})')
            w(f'        double3 xformOp:translate = ({x}, {y}, {h / 2})')
            w(f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            w(f'        rel material:binding = </Materials/Box>')
            w(f'    }}')
            obs_ctr += 1

        # Cylinders
        for _ in range(n_cyls):
            x, y = centers[idx]; idx += 1; h = rand_h()
            r = rng.uniform(r_min, r_max)
            w(f'    def Cylinder "Cyl_{obs_ctr}"')
            w(f'    {{')
            w(f'        double3 xformOp:scale = ({r}, {r}, {h / 2})')
            w(f'        double3 xformOp:translate = ({x}, {y}, {h / 2})')
            w(f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            w(f'        rel material:binding = </Materials/Cyl>')
            w(f'    }}')
            obs_ctr += 1

        # Cones
        for _ in range(n_cones):
            x, y = centers[idx]; idx += 1; h = rand_h()
            r = rng.uniform(r_min, min(r_max, h / 2))
            w(f'    def Cone "Cone_{obs_ctr}"')
            w(f'    {{')
            w(f'        double3 xformOp:scale = ({r}, {r}, {h / 2})')
            w(f'        double3 xformOp:translate = ({x}, {y}, {h / 2})')
            w(f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            w(f'        rel material:binding = </Materials/Cone>')
            w(f'    }}')
            obs_ctr += 1

        w(f'}}')
        total += n_obj

w(f'def Scope "Materials"')
w(f'{{')
for name, color in [("Ground", (0.4, 0.4, 0.5)), ("Wall", (0.3, 0.3, 0.4)),
                     ("Box", (0.6, 0.5, 0.4)), ("Cyl", (0.5, 0.6, 0.5)),
                     ("Cone", (0.6, 0.4, 0.4))]:
    w(f'        def Material "{name}"')
    w(f'        {{')
    w(f'            token outputs:surface.connect = </Materials/{name}/Shader.outputs:surface>')
    w(f'            def Shader "Shader"')
    w(f'            {{')
    w(f'                uniform token info:id = "UsdPreviewSurface"')
    w(f'                color3f inputs:diffuseColor = ({color[0]}, {color[1]}, {color[2]})')
    w(f'            }}')
    w(f'        }}')
w(f'}}')

with open(out_path, 'w') as f:
    f.write('\n'.join(lines))

print(f"DONE: {out_path}")
print(f"Cells: {NUM_ROWS}x{NUM_COLS} = {NUM_ROWS * NUM_COLS}")
print(f"Obstacles: {total}")
print(f"Total prims: {obs_ctr}")
print(f"Obstacles per cell: {NUM_OBJS[0]} → {NUM_OBJS[-1]}")
