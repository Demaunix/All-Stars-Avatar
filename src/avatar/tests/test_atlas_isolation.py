"""Adversarial atlas-isolation test (C234 lesson): bilinear taps and mip filtering of a chart must not reach a
neighbouring chart. For every chart cell of the baked layout: repaint everything outside the cell red, rebuild the
box mip chain (same filter as bake_avatar.dds_dxt1), bilinear-sample every chart UV (corners, edge midpoints,
centroids) at each mip level with clamp addressing, and compare with the unmodified atlas. Reports the highest
level that is unchanged for every chart ("isolated levels"). A control rebuilds the v1-style layout (cell-edge UVs,
no gutters) and must show leaks at level 0 or 1. Results: tests/results/test_atlas_isolation.json."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import avatar_compose as AC
import avatar_surface as AS
import bake_avatar as BA

RESULTS = HERE / "results"


def sampled_uv(c, cell):
    """C253: what the engine samples for a chart's written vertices: the stored UV (native V = 1 - image V, rounded to half
    float as in the vertex stream) through the shader's inversion (sampled V = 1 - stored V)."""
    st = BA.chart_uv(c, cell).astype("<f2").astype(np.float64)
    st[:, 1] = 1.0 - st[:, 1]      # literal shader convention (o.tc.xy = float2(u, 1.0f - v)); not the producer helper BA.native_v
    return st


def mips(img):
    out, cur = [img.astype(np.float64)], img.astype(np.float64)
    while cur.shape[0] > 1:
        h = cur.shape[0] // 2
        cur = cur.reshape(h, 2, h, 2, 4).mean((1, 3))
        out.append(cur)
    return out


def bilinear(img, uv):
    h, w = img.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy


def sample_points(uv, tris):
    a, b, c = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    return np.concatenate([a, b, c, (a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3])


def isolation(atlas, charts, uv_of, cell_of):
    base = mips(atlas)
    res = {}
    for ch in charts:
        x0, y0, size = cell_of(ch)
        mod = atlas.copy()
        mask = np.ones(atlas.shape[:2], bool)
        mask[y0:y0 + size, x0:x0 + size] = False
        mod[mask] = (255, 0, 0, 255)
        mm = mips(mod)
        pts = sample_points(uv_of(ch), ch["tris"])
        levels = []
        for lv, (a, b) in enumerate(zip(base, mm)):
            d = np.abs(bilinear(a, pts) - bilinear(b, pts)).max()
            levels.append(float(d))
        ok = [d == 0.0 for d in levels]
        iso = -1
        for k, v in enumerate(ok):
            if v:
                iso = k
            else:
                break
        res[ch["name"]] = dict(cell=[x0, y0, size], isolated_levels=iso, max_diff_per_level=[round(x, 3) for x in levels[:8]])
    return res


def main():
    parts, _ = AC.compose("male_guest_1")
    for p in parts:
        p["charts"] = AS.charts_for_part(p, p.get("record"))
    charts = [c for p in parts for c in p["charts"]]
    atlas, layout = BA.build_atlas(charts)
    cand = isolation(atlas, charts, lambda c: sampled_uv(c, layout[c["name"]]), lambda c: layout[c["name"]]["cell"])
    # control: v1-style layout = same cells, content filling the whole cell, UVs reaching the cell edge
    ctl_atlas = np.zeros_like(atlas)
    ctl_atlas[..., 3] = 255
    from PIL import Image
    for c in charts:
        x0, y0, size = layout[c["name"]]["cell"]
        img = Image.fromarray((np.clip(c["image"], 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").resize((size, size), Image.BOX)
        ctl_atlas[y0:y0 + size, x0:x0 + size] = np.asarray(img)
    ctl = isolation(ctl_atlas, charts, lambda c: np.stack([(layout[c["name"]]["cell"][0] + np.clip(c["uv"][:, 0], 0, 1) * layout[c["name"]]["cell"][2]) / BA.ATLAS,
                                                          (layout[c["name"]]["cell"][1] + np.clip(c["uv"][:, 1], 0, 1) * layout[c["name"]]["cell"][2]) / BA.ATLAS], 1),
                    lambda c: layout[c["name"]]["cell"])
    checks = {
        "every_chart_isolated_at_mip0_and_mip1": all(v["isolated_levels"] >= 1 for v in cand.values()),
        "large_cells_isolated_to_level_4_or_more": all(v["isolated_levels"] >= 4 for v in cand.values() if v["cell"][2] >= 256),
        "small_cells_isolated_to_level_3_or_more": all(v["isolated_levels"] >= 3 for v in cand.values() if v["cell"][2] == 64),
        "control_leaks_somewhere_at_level_0_or_1": any(v["isolated_levels"] < 1 for v in ctl.values()),
    }
    out = dict(candidate=cand, control_v1_layout=ctl, checks=checks, passed=all(checks.values()))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test_atlas_isolation.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in cand.items():
        print("cand", k, v["cell"], "isolated levels", v["isolated_levels"])
    for k, v in ctl.items():
        print("ctl ", k, "isolated levels", v["isolated_levels"])
    print(checks, "PASS" if out["passed"] else "FAIL")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
