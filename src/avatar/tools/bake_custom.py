"""Bake coloured X360 built-in avatars into the PC racer forest (driver + the four Party Power friends) and the select model.

usage: bake_custom.py --driver 13 --friends 4,14,3,2 --out build/custom-13
Writes <out>/bake/Avatar.zif/.zig (+ .stored, atlases, BAKE-RECEIPT.json) and <out>/select/Avatar.zif/.zig like the C253 build,
using avatar_colours.default_avatar_preset for every model (items compose can bake, console colours).
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import avatar_colours as ACOL
import bake_avatar as BA
import build_select_model as BS

MODULE = Path(__file__).resolve().parents[1]


def main() -> None:
    """Bake and write the racer and select forests for the chosen built-in avatars."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", type=int, required=True)
    ap.add_argument("--friends", default="4,14,3,2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = MODULE / (a.out or f"build/custom-{a.driver:02d}")
    specs = {}
    for root, k in [("Avatar", a.driver)] + [(f"Avatar_Assistant_{i}", int(x)) for i, x in enumerate(a.friends.split(","))]:
        preset, colours, skipped = ACOL.default_avatar_preset(k)
        specs[root] = dict(avatar=k, preset=preset, skipped=skipped, target=(preset, colours))
    targets = {root: s["target"] for root, s in specs.items()}
    bake_dir = out / "bake"
    zif, zig, rec = BA.bake(MODULE / "work/pc-avatar/Racers_Avatar.zif", MODULE / "work/pc-avatar/Racers_Avatar.zig", targets, bake_dir, "BAKE-RECEIPT.json")
    (bake_dir / "Avatar.zif").write_bytes(zif)
    (bake_dir / "Avatar.zig").write_bytes(zig)
    (bake_dir / "Avatar.zif.stored").write_bytes(zlib.compress(zif, 9))
    (bake_dir / "Avatar.zig.stored").write_bytes(zlib.compress(zig, 9))
    sel_zif, sel_zig = BS.main(out / "select", driver=targets["Avatar"])
    summary = dict(driver=a.driver, friends=a.friends, models={root: dict(avatar=s["avatar"], preset=s["preset"], skipped=[x["name"] for x in s["skipped"]]) for root, s in specs.items()},
                   outputs=rec["outputs"], per_model=[dict(model=m["model"], vertices=m["vertices"], triangles=m["triangles"], palette=m["palette_size"],
                                                            cells={k: v["cell"] for k, v in m["layout"].items()}) for m in rec["models"]],
                   select=dict(zif_bytes=len(sel_zif), zig_bytes=len(sel_zig)))
    (out / "CUSTOM-BAKE.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
