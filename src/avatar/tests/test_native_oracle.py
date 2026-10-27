"""C253 finalization (root note06): the native-convention ORACLE of tests/test_native_uv.py must be independent of the production helper.

Problem fixed: native_fetch used to call BA.native_v, the same involution the producer (bake_avatar.chart_uv) uses. A correlated change such as
`1.01 - v` in the helper then cancelled (stored 1.01 - vi, sampled 1 - (1.01 - vi) via the same helper would be 1.01 - (1.01 - vi) = vi) and the
fidelity test still passed. native_fetch now uses a literal convention (test_native_uv.native_sample_v = 1.0 - v, tied to the original shader text
`o.tc%d.xy = float2(inputData._tc%d.%s,1.0f - inputData._tc%d.%s)`).

This focused test is fast and needs no game data, presets or built assets: it uses synthetic charts through the REAL BA.build_atlas / BA.chart_uv.
Checks:
  1 known_values_oracle      hand-written table: stored (u,v) -> expected atlas row/col colour; the oracle passes, wrong conventions
                             (identity, 1.01-v, 0.99-v, 1.0625-v, U mirrored) fail the same table
  2 helper_mutation_oracle   BA.native_v mutated (1.01-v, identity, 0.99-v, 1-v+1/1024): native_fetch and the candidate mutants are bit-identical
  3 producer_known_values    BA.chart_uv stored UV equals literal hand-computed numbers (head cell (0,0,512) g32; shirt cell (0,512,256) g32)
  4 producer_fidelity        stored half-float UV through the literal oracle reproduces the source chart colours (error ~0 on banded charts)
  5 producer_mutation_caught the same four helper mutations make the producer fail 3 and (all but the 1-texel one) 4
  6 old_cancellation_demo    OLD oracle (consumer through the mutated helper) accepts producer+consumer 1.01-v (vulnerability reproduced);
                             the independent oracle rejects it
  7 source_independence      AST: native_fetch, mutants, test_atlas_isolation.sampled_uv contain no reference to `native_v`
  8 literal_binding          the oracle literal equals the shader text root bound byte-for-byte to the EXE (skipped, flagged, if root's file is absent)
Result: tests/results/test_native_oracle.json.   Usage: python tests/test_native_oracle.py
"""
import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(MODULE / "tools"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import bake_avatar as BA  # noqa: E402
import test_atlas_isolation as TAI  # noqa: E402
import test_native_uv as T  # noqa: E402

RESULTS = HERE / "results"
ROOT_BINDING = MODULE.parents[1] / "c253-root-evidence-001" / "UV-LITERAL-EXE-BINDING.json"
HALF_ULP_HALF = 2.0 ** -11            # spacing of binary16 in [0.5, 1)

HELPER_MUTATIONS = {
    "1.01-v": lambda v: 1.01 - v,
    "identity(v)": lambda v: v + 0.0,
    "0.99-v": lambda v: 0.99 - v,
    "1-v+1/1024(one texel)": lambda v: 1.0 - v + 1.0 / 1024.0,
}
WRONG_CONSUMERS = {
    "identity": lambda a, s: T.native_fetch(a, s, sample_v=lambda v: np.asarray(v, np.float64)),
    "1.01-v": lambda a, s: T.native_fetch(a, s, sample_v=lambda v: 1.01 - np.asarray(v, np.float64)),
    "0.99-v": lambda a, s: T.native_fetch(a, s, sample_v=lambda v: 0.99 - np.asarray(v, np.float64)),
    "1.0625-v(half row)": lambda a, s: T.native_fetch(a, s, sample_v=lambda v: 1.0625 - np.asarray(v, np.float64)),
    "U_mirrored": lambda a, s: T.native_fetch(a, np.stack([1.0 - s[:, 0], s[:, 1]], 1)),
}


class patched:
    """Temporarily replace BA.native_v (the production helper); always restored."""
    def __init__(self, fn):
        self.fn, self.orig = fn, BA.native_v

    def __enter__(self):
        BA.native_v = self.fn

    def __exit__(self, *e):
        BA.native_v = self.orig


# ---------------------------------------------------------------- 1. hand-written known-value table (no production code involved)
def known_atlas():
    """8x8 atlas: R = 30 * row, G = 30 * column, B = 7; texel centres at (i + 0.5) / 8."""
    a = np.zeros((8, 8, 4), np.uint8)
    for r in range(8):
        for c in range(8):
            a[r, c] = (30 * r, 30 * c, 7, 255)
    return a


# stored (u, v) written by hand: row r is reached when the SAMPLED V is (r + 0.5) / 8, i.e. stored V = 1 - (r + 0.5) / 8.
KNOWN = [  # (stored_u, stored_v, expected row, expected col)
    (0.0625, 0.9375, 0, 0), (0.1875, 0.8125, 1, 1), (0.3125, 0.6875, 2, 2), (0.4375, 0.5625, 3, 3),
    (0.5625, 0.4375, 4, 4), (0.6875, 0.3125, 5, 5), (0.8125, 0.1875, 6, 6), (0.9375, 0.0625, 7, 7),
    (0.0625, 0.0625, 7, 0), (0.9375, 0.9375, 0, 7),            # off-diagonal: separates U and V errors
    (0.5625, 1.0, 0, 4), (0.4375, 0.0, 7, 3),                  # clamp ends: stored V = 1 -> sampled 0 (top row), stored V = 0 -> bottom row
]


def known_value_table(fetch):
    atlas = known_atlas()
    uv = np.array([[k[0], k[1]] for k in KNOWN], np.float64)
    got = fetch(atlas, uv)
    want = np.array([[30 * k[2], 30 * k[3], 7] for k in KNOWN], np.float64)
    err = np.abs(got - want).max(1)
    return dict(passed=bool(err.max() < 0.5), max_error=float(err.max()), failing=[i for i, e in enumerate(err) if e >= 0.5])


def check_known_values_oracle():
    ok = known_value_table(T.native_fetch)
    wrong = {k: known_value_table(f) for k, f in WRONG_CONSUMERS.items()}
    return dict(oracle=ok, wrong_conventions=wrong, passed=ok["passed"] and all(not w["passed"] for w in wrong.values()))


# ---------------------------------------------------------------- 2. mutating the production helper does not change the oracle
def check_helper_mutation_oracle():
    rng = np.random.default_rng(253)
    atlas = rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)
    stored = rng.random((500, 2))
    base_fetch = T.native_fetch(atlas, stored)
    fake_src = dict(layout=[dict(cell=(0, 0, 512), gutter=32, content=448), dict(cell=(0, 512, 256), gutter=32, content=192)],
                    chart=rng.integers(0, 2, 500))
    base_mut = T.mutants(stored, fake_src)
    out = {}
    for name, fn in HELPER_MUTATIONS.items():
        with patched(fn):
            applied = abs(float(BA.native_v(np.float64(0.25))) - 0.75) > 1e-9      # the mutation really changes the helper
            same_fetch = bool(np.array_equal(T.native_fetch(atlas, stored), base_fetch))
            m = T.mutants(stored, fake_src)
            same_mut = all(np.array_equal(m[k], base_mut[k]) for k in base_mut) and set(m) == set(base_mut)
            contract = known_value_table(T.native_fetch)["passed"]
        out[name] = dict(helper_changed=applied, native_fetch_identical=same_fetch, mutants_identical=same_mut, known_table_still_passes=contract)
    return dict(per_mutation=out, passed=all(all(v.values()) for v in out.values()))


# ---------------------------------------------------------------- synthetic production pipeline
BANDS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 64, 0), (64, 128, 255)]


def synthetic_chart(name, kind):
    """64x64 chart = 8x8 blocks; block (row band i, column group j) colour = (BANDS[i].r, 32*j, BANDS[i].b)."""
    img = np.zeros((64, 64, 4), np.float64)
    for i in range(8):
        for j in range(8):
            img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = (BANDS[i][0] / 255.0, 32 * j / 255.0, BANDS[i][2] / 255.0, 1.0)
    f = np.array([0.1, 0.5, 0.9])
    g = (np.arange(8)[:, None] + f[None, :]).ravel() / 8.0         # 24 positions, each 0.8 source rows from a block edge or centred
    uu, vv = np.meshgrid(g, g)
    uv = np.stack([uu.ravel(), vv.ravel()], 1)
    return dict(name=name, kind=kind, image=img, uv=uv, tris=np.zeros((0, 3), int))


def source_colour(chart):
    """Colour of the SOURCE chart image at the chart UV (image convention, top-origin rows): independent of the baked atlas."""
    img = np.clip(chart["image"], 0, 1)[..., :3] * 255.0
    h, w = img.shape[:2]
    return T.bilinear(img, chart["uv"][:, 0] * w - 0.5, chart["uv"][:, 1] * h - 0.5)


def pipeline():
    charts = [synthetic_chart("synthetic-head", "head"), synthetic_chart("synthetic-shirt", "shirt")]
    atlas, layout = BA.build_atlas(charts)
    return charts, atlas, layout


def fidelity(chart, stored_half, atlas, fetch=T.native_fetch):
    err = np.abs(fetch(atlas, stored_half) - source_colour(chart)).max(1)
    return dict(mean=float(err.mean()), p95=float(np.percentile(err, 95)), max=float(err.max()),
                passed=bool(err.mean() <= 1.5 and err.max() <= 8.0))


# hand-computed literal expectation: stored = (x0 + g + uv * content) / 1024 for U and 1 - (y0 + g + uv * content) / 1024 for V
LITERAL = {  # name -> (cell, gutter, content, [(uv_u, uv_v, stored_u, stored_v)])
    "synthetic-head": ((0, 0, 512), 32, 448, [(0.0, 0.0, 0.03125, 0.96875), (1.0, 1.0, 0.46875, 0.53125), (0.5, 0.5, 0.25, 0.75),
                                              (0.0, 1.0, 0.03125, 0.53125), (1.0, 0.0, 0.46875, 0.96875)]),
    "synthetic-shirt": ((0, 512, 256), 32, 192, [(0.0, 0.0, 0.03125, 0.46875), (1.0, 1.0, 0.21875, 0.28125), (0.5, 0.5, 0.125, 0.375),
                                                 (0.0, 1.0, 0.03125, 0.28125), (1.0, 0.0, 0.21875, 0.46875)]),
}


def producer_known_values(charts, layout):
    out, ok = {}, True
    for c in charts:
        cell, g, content, pts = LITERAL[c["name"]]
        lay = layout[c["name"]]
        layout_ok = tuple(lay["cell"]) == cell and lay["gutter"] == g and lay["content"] == content
        probe = dict(c, uv=np.array([[p[0], p[1]] for p in pts], np.float64))
        stored = BA.chart_uv(probe, lay).astype("<f2").astype(np.float64)       # as written to the vertex stream
        want = np.array([[p[2], p[3]] for p in pts], np.float64)
        err = float(np.abs(stored - want).max())
        # complementary-coordinate cross-check (the relation root's raw delta check uses): image V + stored V == 1 before rounding
        img = BA.chart_uv_image(probe, lay)
        comp = float(np.abs(img[:, 1] + BA.chart_uv(probe, lay)[:, 1] - 1.0).max())
        out[c["name"]] = dict(layout_matches_literal=layout_ok, max_error_vs_literal=err, max_complement_error=comp)
        ok = ok and layout_ok and err == 0.0 and comp < 1e-12
    return dict(per_chart=out, passed=ok)


def producer_run(charts, layout, atlas):
    res = {}
    for c in charts:
        stored = BA.chart_uv(c, layout[c["name"]]).astype("<f2").astype(np.float64)
        res[c["name"]] = (stored, fidelity(c, stored, atlas))
    return res


def check_producer_correct(charts, layout, atlas):
    kv = producer_known_values(charts, layout)
    fid = {k: v[1] for k, v in producer_run(charts, layout, atlas).items()}
    return dict(known_values=kv, fidelity=fid, passed=kv["passed"] and all(v["passed"] for v in fid.values()))


def check_producer_mutations(charts, layout, atlas):
    out = {}
    for name, fn in HELPER_MUTATIONS.items():
        with patched(fn):
            kv = producer_known_values(charts, layout)
            fid = {k: v[1] for k, v in producer_run(charts, layout, atlas).items()}
        fid_caught = all(not v["passed"] for v in fid.values())
        out[name] = dict(known_values_caught=not kv["passed"], fidelity_caught=fid_caught,
                         fidelity_mean={k: round(v["mean"], 2) for k, v in fid.items()})
    need_fid = [k for k in out if "one texel" not in k]          # a single texel stays inside a 56-texel band; the strict literal table catches it
    passed = all(v["known_values_caught"] for v in out.values()) and all(out[k]["fidelity_caught"] for k in need_fid)
    return dict(per_mutation=out, passed=passed,
                note="fidelity (colour) detection needs a shift that crosses a band; the one-texel mutation is caught by the literal table only")


def check_old_cancellation_demo(charts, layout, atlas):
    """Reproduce the vulnerability: consumer sampled through the (mutated) production helper, like the old native_fetch."""
    c = charts[0]
    with patched(HELPER_MUTATIONS["1.01-v"]):
        stored = BA.chart_uv(c, layout[c["name"]]).astype("<f2").astype(np.float64)
        old = fidelity(c, stored, atlas, fetch=lambda a, s: T.native_fetch(a, s, sample_v=BA.native_v))
        new = fidelity(c, stored, atlas)
    return dict(old_helper_sharing_oracle=old, independent_oracle=new, passed=old["passed"] and not new["passed"])


def check_wrong_consumers_on_correct_producer(charts, layout, atlas):
    c = charts[0]
    stored = BA.chart_uv(c, layout[c["name"]]).astype("<f2").astype(np.float64)
    out = {k: fidelity(c, stored, atlas, fetch=f) for k, f in WRONG_CONSUMERS.items()}
    return dict(per_convention={k: dict(mean=round(v["mean"], 2), passed=v["passed"]) for k, v in out.items()},
                passed=all(not v["passed"] for v in out.values()))


# ---------------------------------------------------------------- 7. no shared helper in the consumer functions
def refs_native_v(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [n.lineno for n in ast.walk(tree) if (isinstance(n, ast.Attribute) and n.attr == "native_v") or (isinstance(n, ast.Name) and n.id == "native_v")]


def check_source_independence():
    fns = {"test_native_uv.native_fetch": T.native_fetch, "test_native_uv.native_sample_v": T.native_sample_v,
           "test_native_uv.mutants": T.mutants, "test_atlas_isolation.sampled_uv": TAI.sampled_uv}
    refs = {k: refs_native_v(f) for k, f in fns.items()}
    # sampled_uv legitimately calls the PRODUCER (BA.chart_uv) but must not use the helper on the consumer side
    return dict(native_v_references=refs, passed=all(not v for v in refs.values()))


# ---------------------------------------------------------------- 8. literal binding
def check_literal_binding():
    if not ROOT_BINDING.exists():
        return dict(passed=True, status="UNAVAILABLE", detail=f"{ROOT_BINDING} not present; oracle literal not cross-checked here")
    d = json.loads(ROOT_BINDING.read_text())
    hits = [c for c in d.get("checks", []) if c.get("literal", "").strip() == T.SHADER_V_LITERAL and c.get("match")]
    return dict(passed=bool(hits) and d.get("status") == "ALL_LITERAL_BYTES_MATCH_ORIGINAL_EXE", status="CHECKED",
                va=[h["va"] for h in hits], exe_sha256=d.get("source_sha256"),
                scope="literal text bytes at the EXE offset only; not the active Avatar shader branch")


def main():
    charts, atlas, layout = pipeline()
    res = {"checks": {}, "detail": {}}
    runs = [("known_values_oracle", check_known_values_oracle, ()),
            ("helper_mutation_oracle", check_helper_mutation_oracle, ()),
            ("producer_correct", check_producer_correct, (charts, layout, atlas)),
            ("producer_mutation_caught", check_producer_mutations, (charts, layout, atlas)),
            ("old_cancellation_demo", check_old_cancellation_demo, (charts, layout, atlas)),
            ("wrong_consumers_fail_on_correct_producer", check_wrong_consumers_on_correct_producer, (charts, layout, atlas)),
            ("source_independence", check_source_independence, ()),
            ("literal_binding", check_literal_binding, ())]
    for name, fn, args in runs:
        r = fn(*args)
        res["detail"][name] = r
        res["checks"][name] = bool(r["passed"])
        print(("PASS " if r["passed"] else "FAIL ") + name, flush=True)
    res["passed"] = all(res["checks"].values())
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test_native_oracle.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res["checks"], indent=1))
    print("PASS" if res["passed"] else "FAIL")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
