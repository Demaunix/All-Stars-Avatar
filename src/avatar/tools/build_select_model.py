"""Build the PC character-select model Select/Avatar.zif/.zig (offline).

Evidence: the original Xbox Rev1 Select/Avatar.zif (root extraction, SHA a8652be8...) holds only AvatarCar with one clip
(WheelSpin); the console drew the avatar through the XTL avatar library. PC has no such library and no Select/Avatar member.
Stock PC select forests hold the driver + vehicle (+ optional Camera); several (Opa-Opa, House of the Dead, Ryo bike) have no
SelectIdle/Selected clips, so the select screen does not require them.

This build: start from the stock PC Racers/Avatar.zif/.zig (same forest name `Avatar.Forest`, as the Xbox charsel tuple expects),
bake the driver preset onto `Avatar` (bake_avatar.bake_model), restrict the forest's model table to [AvatarCar, Avatar], and add
clip-name aliases (new anim-table entries pointing at existing clip data): SelectIdle1..4 -> StartIdle1..4, Selected -> Go,
Idle -> StartIdle1, for both models, so whichever of these names the select screen requests resolves to a seated clip.
Which clip the PC select screen plays is NOT established offline (no literal reference found in scripts or the PC image).
"""
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import avatar_compose as AC
import bake_avatar as BA
from forest_append import ForestAppender
from smospkg import name_hash
from zif import Zif

MODULE = Path(__file__).resolve().parents[1]
ALIASES = {"SelectIdle1": "StartIdle1", "SelectIdle2": "StartIdle2", "SelectIdle3": "StartIdle3", "SelectIdle4": "StartIdle4",
           "Selected": "Go", "Idle": "StartIdle1"}
KEEP = ("AvatarCar", "Avatar")


def main(out=None, driver=None):
    """Writes Avatar.zif/.zig, Select-atlas.png and SELECT-RECEIPT.json to `out` (default build/select); returns the
    (zif, zig) bytes. `driver`: None = the C253 preset (byte-identical build), or (preset dict, Colours) for a coloured avatar."""
    zp, gp = MODULE / "work/pc-avatar/Racers_Avatar.zif", MODULE / "work/pc-avatar/Racers_Avatar.zig"
    zb, gb = zp.read_bytes(), gp.read_bytes()
    z = Zif(zp, gp)
    f = z.forests[0]
    fa = ForestAppender(zb, gb)
    s, t = BA.space_fit()
    tmpl = BA.template_material()
    out = Path(out) if out is not None else MODULE / "build" / "select"
    out.mkdir(parents=True, exist_ok=True)
    m = next(mm for mm in f["models"] if mm["root"] == "Avatar")
    spec, colours = (driver if driver is not None else (BA.TARGETS["Avatar"], None))
    parts, prov = AC.compose(spec, colours=colours)
    rec = BA.bake_model(fa, z, f, m, parts, prov, tmpl, s, t, "Select")
    Image.fromarray(rec.pop("atlas"), "RGBA").save(out / "Select-atlas.png")
    b = f["base"]
    fh = fa.words(0, 9)
    model_ptrs = fa.words(fh[1], fh[0])
    keep_models = [mm for mm in f["models"] if mm["root"] in KEEP]
    assert [mm["root"] for mm in keep_models] == ["AvatarCar", "Avatar"]
    keep_ptrs = [mm["pos"] - b for mm in keep_models]
    assert all(p in model_ptrs for p in keep_ptrs)
    table = fa.alloc(struct.pack(f"<{len(keep_ptrs)}I", *keep_ptrs), "select model table [AvatarCar, Avatar]")
    fa.put32(0, len(keep_ptrs), "select forest model count")
    fa.put32(4, table, "select forest model table")
    alias_report = {}
    for mm in keep_models:
        mp = mm["pos"] - b
        h = fa.words(mp, 25)
        entries = [fa.words(h[20] + 12 * j, 3) for j in range(h[19])]
        names = {a["name"]: (a["offset"], a["hash"]) for a in mm["anims"]}
        added = []
        for alias, target in ALIASES.items():
            if alias in names:
                continue
            if target not in names:
                raise SystemExit(f"{mm['root']}: alias target {target} missing")
            sp = fa.alloc(alias.encode() + b"\0", f"{mm['root']} anim alias name {alias}", 4)
            entries.append([names[target][0], sp, name_hash(alias)])
            added.append(dict(alias=alias, target=target, hash=f"{name_hash(alias):08X}", data_offset=names[target][0]))
        tab = fa.alloc(b"".join(struct.pack("<3I", *e) for e in entries), f"{mm['root']} anim table (+aliases)")
        fa.put32(mp + 4 * 19, len(entries), f"{mm['root']} anim count")
        fa.put32(mp + 4 * 20, tab, f"{mm['root']} anim table")
        alias_report[mm["root"]] = added
    zif_out, zig_out = fa.output()
    (out / "Avatar.zif").write_bytes(zif_out)
    (out / "Avatar.zig").write_bytes(zig_out)
    import hashlib
    sha = lambda d: hashlib.sha256(d).hexdigest()
    receipt = dict(source=dict(zif_sha256=sha(zb), zig_sha256=sha(gb)), driver=dict({k: v for k, v in rec.items() if k not in ("provenance",)}),
                   models=list(KEEP), aliases=alias_report, edits=fa.edits, allocations=fa.allocs,
                   outputs=dict(zif_sha256=sha(zif_out), zif_bytes=len(zif_out), zig_sha256=sha(zig_out), zig_bytes=len(zig_out),
                                zif_stored_sha256=sha(zlib.compress(zif_out, 9)), zig_stored_sha256=sha(zlib.compress(zig_out, 9))),
                   evidence="original Xbox Select/Avatar.zif holds only AvatarCar (root receipt W/avatar-original-rig-comparison-001)",
                   unknown="which clip names the PC select screen requests; aliases cover SelectIdle1..4/Selected/Idle")
    (out / "SELECT-RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    # re-read check
    z2 = Zif(out / "Avatar.zif", out / "Avatar.zig")
    f2 = z2.forests[0]
    print("select forest", f2["name"], [(mm["root"], len(mm["anims"])) for mm in f2["models"]])
    for mm in f2["models"]:
        names = [a["name"] for a in mm["anims"]]
        assert all(a in names for a in ALIASES), mm["root"]
    print(json.dumps(receipt["outputs"], indent=1))
    return zif_out, zig_out


if __name__ == "__main__":
    main()
