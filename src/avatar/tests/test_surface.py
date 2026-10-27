"""Surface chart tests: decal coverage classification is conservative (Codex review round 1: sparse probes could drop an
interior decal patch), and for every preset no base-only garment triangle carries decal artwork.
Results: tests/results/test_surface.json."""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import avatar_compose as AC
import avatar_surface as AS

results = []
THR = 0.02


def case(name, fn):
    try:
        d = fn()
        results.append((name, "pass", d or ""))
    except Exception as e:  # noqa: BLE001
        results.append((name, "FAIL", repr(e) + " " + traceback.format_exc().splitlines()[-2]))


def old_ten_probes(img, uvk, tris):
    """The superseded classifier (corners, edge midpoints, centroid, three quarter points), kept as the control."""
    a, b, c = uvk[tris[:, 0]], uvk[tris[:, 1]], uvk[tris[:, 2]]
    hit = np.zeros(len(tris), bool)
    for q in (a, b, c, (a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3, (2 * a + b + c) / 4, (a + 2 * b + c) / 4, (a + b + 2 * c) / 4):
        hit |= AC.sample(img, q)[:, 3] > THR
    return hit


def synthetic():
    img = np.zeros((64, 64, 4))
    img[13:15, 13:15] = (1, 1, 1, 1)                      # opaque 2x2 patch around UV (0.21, 0.21)
    uv = np.array([[0, 0], [1, 0], [0, 1], [0.8, 0.8], [0.9, 0.8], [0.8, 0.9], [0.205, 0.205], [0.207, 0.205], [0.205, 0.207]], float)
    tris = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    old = old_ten_probes(img, uv, tris)
    new = AS.tri_touches(img, uv, tris)
    assert not old[0], "control: the old classifier should miss the interior patch"
    assert new.tolist() == [True, False, True], new
    # Codex round 2 counterexample: long thin triangle crossing a single opaque texel between texel centres
    img2 = np.zeros((64, 64, 4))
    img2[12, 13] = (1, 1, 1, 1)
    uv2 = np.array([[0, .2], [1, .2], [1, .20001], [0, .2], [.5, .2], [1, .2], [0, .9], [1, .9], [1, .90001]], float)
    t2 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert AC.sample(img2, np.array([[13.5 / 64, .200002]]))[0, 3] > THR
    got = AS.tri_touches(img2, uv2, t2)
    assert got.tolist() == [True, True, False], got
    # property: no false negatives against a dense-sampling oracle (random sparse masks, many thin triangles)
    rng = np.random.default_rng(235)
    misses = checked = 0
    for _ in range(40):
        img3 = np.zeros((32, 48, 4))
        for _ in range(3):
            img3[rng.integers(0, 32), rng.integers(0, 48), 3] = 1.0
        c = rng.random((60, 2))
        d = rng.normal(0, 0.15, (60, 2, 2))
        d[:30, 1] = d[:30, 0] * (1 + rng.normal(0, 1e-4, (30, 1)))   # half nearly degenerate (thin) triangles
        uv3 = np.concatenate([c, c + d[:, 0], c + d[:, 1]])
        t3 = np.stack([np.arange(60), np.arange(60) + 60, np.arange(60) + 120], 1)
        truth = dense_alpha(img3, uv3, t3, n=64) > THR
        got3 = AS.tri_touches(img3, uv3, t3)
        misses += int((truth & ~got3).sum())
        checked += int(truth.sum())
    assert misses == 0 and checked > 50, (misses, checked)
    try:
        AS.tri_touches(img2, np.array([[0, 0], [np.nan, 0], [0, 1]], float), np.array([[0, 1, 2]]))
        raise AssertionError("non-finite UVs accepted")
    except AS.SurfaceError:
        pass
    return ("interior patch missed by the ten-probe control is found; far triangle excluded; sub-texel triangle found; "
            f"round-2 thin and collinear triangles across one opaque texel found; 0 misses vs dense oracle over {checked} covered random triangles")


def dense_alpha(img, uvk, tris, n=24):
    """Max decal alpha over a dense barycentric lattice of each triangle (clamp-to-border)."""
    a, b, c = uvk[tris[:, 0]], uvk[tris[:, 1]], uvk[tris[:, 2]]
    best = np.zeros(len(tris))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            q = (a * i + b * j + c * (n - i - j)) / n
            s = AC.sample(img, q)[:, 3]
            inside = (q[:, 0] >= 0) & (q[:, 0] <= 1) & (q[:, 1] >= 0) & (q[:, 1] <= 1)
            best = np.maximum(best, np.where(inside, s, 0))
    return best


def presets():
    rows = []
    for name in AC.PRESETS:
        parts, _ = AC.compose(name, head_res=64, garment_res=64)
        for p in parts:
            rec = p.get("record")
            if not rec or not p["role"].startswith(("shirt", "pants", "shoes")):
                continue
            decs = AS.decal_layers(rec)
            if not decs:
                continue
            charts = AS.charts_for_part(p, rec)
            base = [c for c in charts if c["kind"] == "base"]
            if not base:
                continue
            me = p["mesh"]
            worst = 0.0
            for img, k, _ in decs:
                uvk = me["texcoords"][:, 2 * k:2 * k + 2]
                worst = max(worst, float(dense_alpha(img, uvk, base[0]["tris"]).max()))
            assert worst <= THR, (name, p["role"], worst)
            dec = sum(len(c["tris"]) for c in charts if c["kind"] == "decal")
            rows.append(f"{name}/{p['role']}: decal chart {dec} tris, max decal alpha on base-only tris {worst:.3f}")
    assert rows
    return "; ".join(rows)


case("conservative decal classification (synthetic regression + control)", synthetic)
case("no base-only garment triangle carries decal artwork (all presets, dense sampling)", presets)

if __name__ == "__main__":
    fails = [r for r in results if r[1] != "pass"]
    for r in results:
        print(f"{r[1]:4}  {r[0]}  {str(r[2])[:600]}")
    (ROOT / "tests/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests/results/test_surface.json").write_text(json.dumps(dict(suite="test_surface", cases=len(results), failed=len(fails),
                                                                            results=[dict(name=a, status=b, detail=str(c)) for a, b, c in results]), indent=1) + "\n")
    sys.exit(1 if fails else 0)
