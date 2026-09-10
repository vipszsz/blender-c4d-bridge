"""Checks the plugin's lossless key reduction without Cinema 4D."""
import ast
import math
import os
import random

PYP = os.path.join(os.path.dirname(__file__), "..", "cinema4d", "CameraBridge", "camera_bridge.pyp")
tree = ast.parse(open(PYP, encoding="utf-8").read())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reduce_keys")
ns = {}
exec(compile(ast.Module([fn], []), PYP, "exec"), ns)
reduce_keys = ns["reduce_keys"]


def check(frames, vals, eps):
    keys = reduce_keys(frames, vals, eps)
    worst = 0.0
    for a, b in zip(keys, keys[1:]):
        for i in range(a, b + 1):
            t = (frames[i] - frames[a]) / (frames[b] - frames[a])
            worst = max(worst, abs(vals[a] + t * (vals[b] - vals[a]) - vals[i]))
    return len(keys), worst


random.seed(1)
fr = list(range(1, 2001))
tests = {
    "static": [5.0] * 2000,
    "linear": [0.5 * f for f in fr],
    "slow ease": [1000 * (1 - math.cos(math.pi * f / 2000)) / 2 for f in fr],
    "noise": [random.uniform(-1, 1) for _ in fr],
    "steps": [float(f // 100) for f in fr],
}
bad = 0
for name, v in tests.items():
    n, w = check(fr, v, 1e-3)
    ok = w <= 1e-3 + 1e-12
    bad += not ok
    print(f"{name:10s} keys {n:5d}/2000  worst err {w:.2e}  {'OK' if ok else 'BAD'}")
print("PASS" if not bad else "FAIL")
