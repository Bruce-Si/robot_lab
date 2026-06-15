#!/usr/bin/env python3
"""Parse USDA file → extract obstacle position/type/size as JSON."""

import re, json, os, sys
from collections import defaultdict

def parse_usda(path):
    obstacles = defaultdict(list)
    with open(path) as f:
        content = f.read()

    cell_re = re.compile(r'def Xform "Cell_(\d+)_(\d+)"\s*\{')
    obs_re = re.compile(r'def (Cube|Cylinder|Cone) "(Box|Cyl|Cone)_(\d+)"\s*\{')

    for cm in cell_re.finditer(content):
        row, col = int(cm.group(1)), int(cm.group(2))
        cell = f"Cell_{row}_{col}"
        # Find cell body via brace matching
        i = cm.end(); d = 1
        while d > 0 and i < len(content):
            if content[i] == '{': d += 1
            elif content[i] == '}': d -= 1
            i += 1
        body = content[cm.end():i-1]

        for om in obs_re.finditer(body):
            ptype = om.group(1); oid = int(om.group(3))
            # Find obs body
            j = om.end(); d2 = 1
            while d2 > 0 and j < len(body):
                if body[j] == '{': d2 += 1
                elif body[j] == '}': d2 -= 1
                j += 1
            b = body[om.end():j-1]

            scale = trans = None
            for line in b.split('\n'):
                if 'xformOp:scale' in line:
                    m = re.search(r'\(([^)]+)\)', line)
                    if m: scale = [float(x) for x in m.group(1).split(',')]
                elif 'xformOp:translate' in line:
                    m = re.search(r'\(([^)]+)\)', line)
                    if m: trans = [float(x) for x in m.group(1).split(',')]
            if not scale or not trans:
                continue

            if ptype == 'Cube':
                obstacles[cell].append({"id": oid, "type": "box",
                    "center": trans, "size": [s*2 for s in scale], "enabled": True})
            elif ptype == 'Cylinder':
                obstacles[cell].append({"id": oid, "type": "cylinder",
                    "center": trans, "size": [scale[0], scale[2]*2], "enabled": True})
            elif ptype == 'Cone':
                obstacles[cell].append({"id": oid, "type": "cone",
                    "center": trans, "size": [scale[0], scale[2]*2], "enabled": True})

    return obstacles

if __name__ == "__main__":
    usd_path = sys.argv[1]
    usd_path = os.path.abspath(usd_path)
    data = parse_usda(usd_path)
    total = sum(len(v) for v in data.values())
    out = os.path.splitext(usd_path)[0] + "_obstacles.json"
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"OK: {total} obstacles from {len(data)} cells → {out}")
