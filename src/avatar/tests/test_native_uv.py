"""C253 regression: the baked Avatar vertex V must follow the native engine convention (sampled V = 1 - stored V).

Primary-evidence basis (offline): the original engine's generated vertex shader emits
`o.tc%d.xy = float2(inputData._tc%d.%s, 1.0f - inputData._tc%d.%s)` (EXE VA 0x6e3220; case 1 of the generators at 0x66ffd0 / 0x671000), so
the vertex streams of the stock models hold bottom-origin V. Stock-engine convention control (not a key identification): the whole stock Eggman
scene (work/stockscene-eggman.png; its script includes EVERY eligible 32-byte-declaration material, not only EggHands, the material the Avatar
clones) and Amy/Sonic render correctly only through this rule. Which generator branch the Avatar's own runtime key selects is NOT established
offline. The baseline baked image-convention V and so sampled mirrored atlas rows (the user's screenshot).

What is tested, on the WRITTEN Avatar.zif/.zig pairs (decoded DDS mip 0 + half-float UVs read from the GPU payload):
  for every vertex, fetch the atlas colour at (u, 1 - stored_v) and compare it with the colour of the SOURCE chart texture at the source
  chart UV (independent of bake_avatar.chart_uv and of any viewer). Baseline files must FAIL, candidate files must PASS, and mutants of the
  candidate (U mirror, V flipped inside the chart only, V offset past the gutter, V not flipped) must FAIL.
A source-level pair repeats this without files: BA.gather (candidate rule) vs BA.gather with chart_uv replaced by the image-convention rule.

Oracle independence (C253 finalization, root note06): the consumer side (`native_fetch`, the mutants) uses the LITERAL shader convention
`native_sample_v` below and never calls the production helper BA.native_v, so a correlated change of the helper (e.g. 1.01 - v used
by producer and consumer alike) cannot cancel. tests/test_native_oracle.py proves that with known values, helper mutation and wrong conventions.

Usage: python tests/test_native_uv.py [--baseline-racer Z --baseline-select Z --candidate-racer Z --candidate-select Z]
(`.zig` is the sibling of each `.zif`). Result: tests/results/test_native_uv.json.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(MODULE / "tools"))
sys.dont_write_bytecode = True
import avatar_compose as AC  # noqa: E402
import avatar_surface as AS  # noqa: E402
import bake_avatar as BA  # noqa: E402
import bone_map  # noqa: E402
from zif import Zif  # noqa: E402

RESULTS = HERE / "results"
# 0..255 per-vertex max-channel error. Measured: candidate files mean 2.5-3.4 / p95 5.0-9.4 (DXT1 + chart resampling noise);
# weakest mutant (V flipped inside the chart only) mean 12.6-39.5 / p95 45-215; baseline files mean 131-156.
MEAN_LIMIT, P95_LIMIT = 6.0, 20.0


def bilinear(img, x, y):
    """Clamp-addressed bilinear fetch at texel coordinates (texel centres at integer + 0.5 in uv space)."""
    h, w = img.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy


# The original engine's generated vertex shader (EXE VA 0x6e3220, literal bytes bound by root in UV-LITERAL-EXE-BINDING.json):
SHADER_V_LITERAL = "o.tc%d.xy = float2(inputData._tc%d.%s,1.0f - inputData._tc%d.%s);"


def native_sample_v(v):
    """The consumer convention, written out literally from SHADER_V_LITERAL (second component: 1.0f - stored V).
    Deliberately NOT BA.native_v: the oracle must not share code with the producer it checks."""
    return 1.0 - np.asarray(v, np.float64)


def native_fetch(atlas, stored_uv, sample_v=native_sample_v):
    """What the original generated shader + sampler do with the stored coordinate: sample at (u, 1 - v).
    `sample_v` is injectable only so tests/test_native_oracle.py can show that wrong conventions are rejected."""
    h, w = atlas.shape[:2]
    u, v = stored_uv[:, 0], sample_v(stored_uv[:, 1])
    return bilinear(atlas.astype(np.float64), u * w - 0.5, v * h - 0.5)[:, :3]


def source_vertices(preset):
    """Per-vertex (chart index, source UV in image convention, source colour, chart rows) in the order of the written stream
    (gather order, then compaction), plus the layout; recomputed from the composed source parts, independent of chart_uv."""
    parts, _ = AC.compose(preset)
    for p in parts:
        p["charts"] = AS.charts_for_part(p, p.get("record"))
    charts = [c for p in parts for c in p["charts"]]
    atlas, layout = BA.build_atlas(charts)
    mp = bone_map.mapping()
    s, t = BA.space_fit()
    P, N, U, W, B, T, stats = BA.gather(parts, layout, mp, s, t)
    src_uv, chart_ix, colour = [], [], []
    for ci, c in enumerate(charts):
        used = np.unique(c["tris"])
        uv = np.clip(c["uv"][used], 0, 1)
        img = np.clip(c["image"], 0, 1)[..., :3] * 255.0
        h, w = img.shape[:2]
        src_uv.append(uv)
        chart_ix.append(np.full(len(used), ci))
        colour.append(bilinear(img, uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5))
    src_uv, chart_ix, colour = np.concatenate(src_uv), np.concatenate(chart_ix), np.concatenate(colour)
    assert len(src_uv) == len(U), "gather order assumption broken"
    keep = np.unique(T)
    return dict(uv=src_uv[keep], chart=chart_ix[keep], colour=colour[keep], layout=[layout[c["name"]] for c in charts],
                stored_rule_uv=U[keep], count=len(keep), atlas=atlas, parts=parts, layout_by_name=layout)


def read_models(zif_path, roots):
    zig_path = Path(zif_path).with_suffix(".zig")
    z = Zif(zif_path, zig_path)
    f = z.forests[0]
    out = {}
    for m in f["models"]:
        if m["root"] not in roots:
            continue
        rows = [r for r in z.meshes(f, m) if r["node"] == 0 and r["material_name"].startswith("AvatarXbox_")]
        assert len(rows) == 1, (m["root"], len(rows))
        row = rows[0]
        vd = row["vertex_descriptor"]
        stride, n, vo = vd[3], vd[4], f["gpu_base"] + vd[5]
        assert stride == 32 and [0, 20, 15, 0, 5, 0] in row["declaration"]
        raw = np.frombuffer(z.g, np.uint8, count=n * stride, offset=vo).reshape(n, stride)
        uv = raw[:, 20:24].copy().view("<f2").reshape(n, 2).astype(np.float64)
        tex = next(t for t in f["textures"] if t["pos"] - f["base"] == row["samplers"][0][1])
        im = Image.open(io.BytesIO(z.g[f["gpu_base"] + tex["header"][2]:]))
        im.load()
        atlas = np.array(im.convert("RGBA"))
        assert atlas.shape == (BA.ATLAS, BA.ATLAS, 4), atlas.shape
        out[m["root"]] = dict(uv=uv, atlas=atlas, count=n)
    assert set(out) == set(roots), (sorted(out), sorted(roots))
    return out


def fidelity(stored_uv, atlas, colour):
    err = np.abs(native_fetch(atlas, stored_uv) - colour).max(1)
    return dict(vertices=int(len(err)), mean=float(err.mean()), p95=float(np.percentile(err, 95)), max=float(err.max()),
                passed=bool(err.mean() <= MEAN_LIMIT and np.percentile(err, 95) <= P95_LIMIT))


def mutants(stored, src):
    """Candidate-derived failure mutants (vertex stream coordinates only)."""
    lay = src["layout"]
    ch = src["chart"]
    y_lo = np.array([(l["cell"][1] + l["gutter"]) for l in lay], float)[ch] / BA.ATLAS
    y_hi = np.array([(l["cell"][1] + l["gutter"] + l["content"]) for l in lay], float)[ch] / BA.ATLAS
    img_v = native_sample_v(stored[:, 1])          # stored V -> image V (the literal convention is an involution)
    m = {}
    m["V_not_flipped(baseline_rule)"] = np.stack([stored[:, 0], img_v], 1)
    m["U_mirrored"] = np.stack([1 - stored[:, 0], stored[:, 1]], 1)
    m["V_flipped_inside_chart_only"] = np.stack([stored[:, 0], native_sample_v(y_lo + y_hi - img_v)], 1)
    m["V_offset_past_gutter(+0.05)"] = np.stack([stored[:, 0], np.clip(stored[:, 1] + 0.05, 0, 1)], 1)
    return {k: v.astype("<f2").astype(np.float64) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    b = MODULE / "build"
    ap.add_argument("--baseline-racer", default=str(b / "baseline-bake/Avatar.zif"))
    ap.add_argument("--baseline-select", default=str(b / "baseline-select/Avatar.zif"))
    ap.add_argument("--candidate-racer", default=str(b / "bake/Avatar.zif"))
    ap.add_argument("--candidate-select", default=str(b / "select/Avatar.zif"))
    a = ap.parse_args()
    roots = list(BA.TARGETS)
    srcs = {}
    for root, preset in BA.TARGETS.items():
        srcs[root] = source_vertices(preset)
        print("source", root, preset, srcs[root]["count"], "vertices", flush=True)
    res = {"limits": dict(mean_limit=MEAN_LIMIT, p95_limit=P95_LIMIT), "source_level": {}, "files": {}, "checks": {}}

    # ---- source level: candidate rule vs the image-convention rule, no files
    for root in roots:
        s = srcs[root]
        cand = fidelity(s["stored_rule_uv"].astype("<f2").astype(np.float64), s["atlas"], s["colour"])
        orig = BA.chart_uv
        BA.chart_uv = BA.chart_uv_image          # the baseline rule (image-convention V written to the stream)
        try:
            P, N, U, W, B, T, _s = BA.gather(s["parts"], s["layout_by_name"], bone_map.mapping(), *BA.space_fit())
            base = fidelity(U[np.unique(T)].astype("<f2").astype(np.float64), s["atlas"], s["colour"])
        finally:
            BA.chart_uv = orig
        res["source_level"][root] = dict(candidate_rule=cand, image_convention_rule=base)
        print("source-level", root, "candidate", cand, "image-convention", base, flush=True)

    # ---- written files
    def run(label, racer, select, must_pass):
        models = read_models(racer, roots)
        models["Select:Avatar"] = read_models(select, ["Avatar"])["Avatar"]
        out = {}
        for key, md in models.items():
            s = srcs["Avatar" if key.startswith("Select") else key]
            assert md["count"] == s["count"], (key, md["count"], s["count"])
            out[key] = dict(fidelity=fidelity(md["uv"], md["atlas"], s["colour"]))
            if label == "candidate":
                out[key]["mutants"] = {k: fidelity(v, md["atlas"], s["colour"]) for k, v in mutants(md["uv"], s).items()}
        res["files"][label] = out
        ok = all(v["fidelity"]["passed"] == must_pass for v in out.values())
        print(label, {k: (round(v["fidelity"]["mean"], 2), v["fidelity"]["passed"]) for k, v in out.items()}, flush=True)
        return ok

    base_ok = run("baseline", a.baseline_racer, a.baseline_select, False)
    cand_ok = run("candidate", a.candidate_racer, a.candidate_select, True)
    cand_files = res["files"]["candidate"]
    res["checks"] = {
        "baseline_files_fail_native_fetch_fidelity": base_ok,
        "candidate_files_pass_native_fetch_fidelity": cand_ok,
        "all_candidate_mutants_fail": all(not m["passed"] for v in cand_files.values() for m in v["mutants"].values()),
        "source_level_candidate_rule_passes": all(v["candidate_rule"]["passed"] for v in res["source_level"].values()),
        "source_level_image_convention_rule_fails": all(not v["image_convention_rule"]["passed"] for v in res["source_level"].values()),
        "covers_driver_select_and_four_assistants": sorted(cand_files) == sorted(roots + ["Select:Avatar"]) and len(cand_files) == 6,
    }
    res["passed"] = all(res["checks"].values())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test_native_uv.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["checks"], indent=1))
    print("PASS" if res["passed"] else "FAIL")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
