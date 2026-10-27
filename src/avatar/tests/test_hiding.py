"""Hiding-template tests: every supplied template block is consumed exactly by the decoded layout, TOC links resolve to
templates of the matching body, mutated blocks are refused, the decoded triangle lists agree with independent
geometric coverage, and compose uses them for every preset. Results: tests/results/test_hiding.json."""
import json
import struct
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import avatar_compose as AC
import avatar_hiding as H
import avatar_model as AM
from catalog import catalog, decoded, raw_block

results = []
BODY = {0x10000: 0, 0x20001: 1}


def case(name, fn):
    try:
        d = fn()
        results.append((name, "pass", d or ""))
    except Exception as e:  # noqa: BLE001
        results.append((name, "FAIL", repr(e) + " " + traceback.format_exc().splitlines()[-2]))


def bodies():
    return {g: AM.decode(decoded(i))["meshes"][0] for g, i in BODY.items()}


def all_blocks():
    B = bodies()
    n = tris = ovr = 0
    for c in H.all_templates():
        t = H.load(c["index"])
        H.check_body(t, B[t["gender"]])
        n += 1
        tris += len(t["hidden_triangles"])
        ovr += len(t["vertices"])
    assert n == 200, n
    return f"{n} template blocks consumed exactly (byte length == layout), all bound to their body; {tris} hidden triangles, {ovr} vertex overrides"


def links():
    B = bodies()
    out = []
    for c in catalog():
        if c["type"] == H.HIDING_TYPE:
            continue
        t = H.template_for(c["index"])
        if t is None:
            continue
        tw = H._toc_words()[c["index"]][40] & 0xFFFF
        g = H.load(t)["gender"]
        assert BODY[g] == tw - 1, (c["index"], t, tw, hex(g))   # link gender (1 male / 2 female) == template body
        out.append((c["index"], t))
    assert len(out) == 287 and len({t for _, t in out}) == 197, (len(out), len({t for _, t in out}))
    return f"{len(out)} components link to {len({t for _, t in out})} templates; link gender == template body for all"


def refusals():
    d = bytearray(raw_block(1014))
    out = []
    for name, mut in (("truncated", lambda b: b[:-1]), ("extended", lambda b: b + b"\0"),
                      ("hidden count", lambda b: struct.pack_into("<I", b, 0, struct.unpack_from("<I", b, 0)[0] + 1) or b),
                      ("bits per entry", lambda b: struct.pack_into("<I", b, 28, 11) or b)):
        b = mut(bytearray(d))
        try:
            H.decode(bytes(b), AM.decode(decoded(0))["meshes"][0])
            out.append((name, False))
        except (H.HidingError, struct.error, ValueError, IndexError):
            out.append((name, True))
    try:
        H.load(1014, AM.decode(decoded(1))["meshes"][0])
        out.append(("other body", False))
    except H.HidingError:
        out.append(("other body", True))
    assert all(r for _, r in out), out
    return "refused: " + ", ".join(n for n, _ in out)


def coverage_agreement():
    """Independent check of the triangle semantics: triangles hidden by a top's template are (mostly) geometrically
    covered by that top, while triangles it leaves visible (hands, forearms, neck, legs) mostly are not."""
    rows = []
    for top, body_i in ((92, 0), (301, 1)):
        body = AM.decode(decoded(body_i))["meshes"][0]
        t = H.load(H.template_for(top), body)
        _, arec = AC.align_to_body(AC.load_part(top)["meshes"][0], body)
        moved, _ = AC.body_to_garment_frame(body, arec["delta"])
        keep, _ = AC.covered_keep_mask(moved, AC.load_part(top)["meshes"], reach=0.06, reach_inward=0.02)
        hidden = np.zeros(len(body["triangles"]), bool)
        hidden[t["hidden_triangles"]] = True
        P = AM.positions(moved)
        cy = P[body["triangles"]].mean(1)[:, 1]
        torso = (cy > 0.75) & (cy < 1.35)
        a = float((~keep[hidden]).mean())
        b = float((~keep[~hidden & ~torso]).mean())
        assert a > 0.6 and b < 0.1, (top, a, b)
        rows.append(f"{AC.NAMES[top]}: {a:.0%} of template-hidden triangles geometrically covered vs {b:.0%} of visible non-torso")
    return "; ".join(rows)


def presets_use_templates():
    rows = []
    for name, P in AC.PRESETS.items():
        body = AM.decode(decoded(P["body"]))["meshes"][0]
        worn = [P[r] for r in ("shirt", "pants", "shoes")]
        _, hidden, rec = AC.apply_hiding_templates(body, worn)
        assert {t["component"] for t in rec["templates"]} | {c["index"] for c in rec["without_template"]} == set(worn)
        assert all(r in {t["component"] for t in rec["templates"]} for r in (P["shirt"], P["pants"])), name
        rows.append(f"{name}: {len(rec['templates'])} templates, {int(hidden.sum())} hidden, {rec['overridden_vertices']} overrides"
                    + (f", no template: {[c['name'] for c in rec['without_template']]}" if rec["without_template"] else ""))
    return "; ".join(rows)


case("all 200 template blocks decode exactly", all_blocks)
case("TOC component -> template links", links)
case("mutated template blocks are refused", refusals)
case("hidden triangles agree with geometric coverage", coverage_agreement)
case("every preset top and trousers use their supplied template", presets_use_templates)

if __name__ == "__main__":
    fails = [r for r in results if r[1] != "pass"]
    for r in results:
        print(f"{r[1]:4}  {r[0]}  {str(r[2])[:300]}")
    (ROOT / "tests/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests/results/test_hiding.json").write_text(json.dumps(dict(suite="test_hiding", cases=len(results), failed=len(fails),
                                                                           results=[dict(name=a, status=b, detail=str(c)) for a, b, c in results]), indent=1) + "\n")
    sys.exit(1 if fails else 0)
