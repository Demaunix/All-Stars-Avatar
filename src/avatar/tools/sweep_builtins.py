"""Compose every X360 built-in avatar (the 17 descriptions in research/xbox-default-avatars.json) under the active decode profile
and record, per avatar, what composed, what the preset builder skipped, every part's measured offset against the body, the face-shape
records and any refusal. An exhaustive offline inventory (THE-AUDITOR rule 9) of what the bake can take as input today.
usage: sweep_builtins.py [--out notes/evidence/m10-rp6/builtin-sweep.json]
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import avatar_colours as ACOL
import avatar_compose as AC
import avatar_model as AM

MODULE = Path(__file__).resolve().parents[1]
PROJECT = MODULE.parents[1]


def one(k: int) -> dict:
    """Compose built-in avatar k and summarise the provenance (no bake, no files written)."""
    preset, colours, skipped = ACOL.default_avatar_preset(k)
    rec = dict(avatar=k, items={r: (i, AC.NAMES.get(i)) for r, i in preset.items() if r != "frame" and i is not None},
               skipped=[dict(name=s.get("name"), reason=s.get("reason")) if isinstance(s, dict) else str(s) for s in skipped])
    t0 = time.time()
    try:
        parts, prov = AC.compose(preset, colours=colours)
    except Exception as e:  # noqa: BLE001
        rec.update(status="REFUSED", error=repr(e), trace=traceback.format_exc().splitlines()[-3:])
        return rec
    offs = prov["frame"]["measured_offsets"]
    rec.update(status="composed", seconds=round(time.time() - t0, 1), parts=[p["role"] for p in parts],
               measured_offsets={n: dict(applied=a.get("applied"), delta=a.get("delta"), seams=a.get("seams"), reason=a.get("reason")) for n, a in offs.items()},
               max_abs_offset=max((max(abs(x) for x in a["delta"]) for a in offs.values() if a.get("delta")), default=None),
               face_shapes=prov.get("face_shapes"), hat_hair=prov.get("hat_hair"),
               hair_lift=prov.get("hair_over_head", {}).get("hair"), pants_tuck=[t["tuck"][0]["vertices_moved"] for t in prov.get("pants_under_top", [])],
               body_hiding=dict(hidden=prov["body_hiding"]["hidden_triangles"], dropped=prov["body_hiding"]["dropped_triangles"],
                                without_template=[c["name"] for c in prov["body_hiding"]["without_template"]]))
    return rec


def main() -> None:
    """Sweep all 17 built-in avatars and write the JSON record."""
    out = PROJECT / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "notes/evidence/m10-rp6/builtin-sweep.json")
    rows = []
    for k in range(17):
        r = one(k)
        rows.append(r)
        print(f"{k:2d} {r['status']:9s} items {len(r['items'])} skipped {[s['name'] if isinstance(s, dict) else s for s in r['skipped']]} "
              f"max|offset| {r.get('max_abs_offset')} {r.get('error', '')}")
        sys.stdout.flush()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(decode=dict(profile=AM.DECODE_NAME, factors=AM.K.tolist()), tolerance=AC.FRAME_TOLERANCE, avatars=rows), indent=1, default=str) + "\n")
    n_ok = sum(r["status"] == "composed" for r in rows)
    print(f"composed {n_ok}/17; refused {[r['avatar'] for r in rows if r['status'] != 'composed']}; wrote {out}")


if __name__ == "__main__":
    main()
