"""Bake tests (no renders): structure of the written racer and select files, select clip aliases, reproducibility (fresh rebuild of the
racer bake and the select model from source must be byte-identical to build/), and the receipts' recorded hashes.
Results: tests/results/test_bake.json."""
import hashlib
import json
import struct
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "vendor"))
import pcanim
import pcbody
from smospkg import name_hash
from zif import Zif

results = []
sha = lambda b: hashlib.sha256(b).hexdigest()
BAKE, SEL = ROOT / "build" / "bake", ROOT / "build" / "select"
SKINNED = [[0, 0, 2, 0, 0, 0], [0, 12, 16, 0, 3, 0], [0, 20, 15, 0, 5, 0], [0, 24, 8, 0, 1, 0], [0, 28, 5, 0, 2, 0]]


def case(name, fn):
    try:
        d = fn()
        results.append((name, "pass", d or ""))
    except Exception as e:  # noqa: BLE001
        results.append((name, "FAIL", repr(e) + " " + traceback.format_exc().splitlines()[-2]))


def check_model(z, f, m):
    rows = [r for r in z.meshes(f, m) if r["node"] == 0]
    assert len(rows) == 1
    row = rows[0]
    mesh = pcbody.read_mesh(z, row)
    pal, ib = pcbody.palette(z, f, row)
    Wb = pcanim.world(m, *pcanim.bind_locals(z, f, m))
    assert [list(x) for x in row["declaration"]] == SKINNED and row["vertex_descriptor"][9] == 531 and row["vertex_descriptor"][3] == 32
    assert 0 < len(pal) <= 66
    assert max(np.abs(np.linalg.inv(ib[k]) - Wb[pal[k]]).max() for k in range(len(pal))) < 1e-5
    w = np.rint(mesh["w"] * 255).astype(int)
    assert (w.sum(1) == 255).all() and (mesh["idx"].astype(int)[w > 0] < len(pal)).all() and mesh["tris"].max() < mesh["count"]
    uv = mesh["uv"]
    assert (uv >= 0).all() and (uv <= 1).all()
    assert row["material_header"][1] == 0x02020883 and row["material_header"][29] == 1
    return len(pal), int(mesh["count"])


def racer_structure():
    z = Zif(BAKE / "Avatar.zif", BAKE / "Avatar.zig")
    f = z.forests[0]
    out = []
    for root in ("Avatar", "Avatar_Assistant_0", "Avatar_Assistant_1", "Avatar_Assistant_2", "Avatar_Assistant_3"):
        m = next(mm for mm in f["models"] if mm["root"] == root)
        out.append((root,) + check_model(z, f, m))
    return "; ".join(f"{a}: palette {b}, {c} verts" for a, b, c in out)


def select_structure():
    z = Zif(SEL / "Avatar.zif", SEL / "Avatar.zig")
    f = z.forests[0]
    assert f["name"] == "Avatar.Forest" and [m["root"] for m in f["models"]] == ["AvatarCar", "Avatar"]
    m = f["models"][1]
    pal, nv = check_model(z, f, m)
    aliases = {"SelectIdle1": "StartIdle1", "SelectIdle2": "StartIdle2", "SelectIdle3": "StartIdle3", "SelectIdle4": "StartIdle4", "Selected": "Go", "Idle": "StartIdle1"}
    for mm in f["models"]:
        by = {a["name"]: a for a in mm["anims"]}
        for a, t in aliases.items():
            assert by[a]["offset"] == by[t]["offset"] and by[a]["hash"] == name_hash(a), (mm["root"], a)
    return f"[AvatarCar, Avatar]; driver palette {pal}, {nv} verts; 6 aliases per model resolve to stored clips with name_hash()"


def receipts():
    rc = json.loads((BAKE / "BAKE-RECEIPT.json").read_text())
    assert sha((BAKE / "Avatar.zif").read_bytes()) == rc["outputs"]["zif_sha256"] and sha((BAKE / "Avatar.zig").read_bytes()) == rc["outputs"]["zig_sha256"]
    sr = json.loads((SEL / "SELECT-RECEIPT.json").read_text())
    assert sha((SEL / "Avatar.zif").read_bytes()) == sr["outputs"]["zif_sha256"] and sha((SEL / "Avatar.zig").read_bytes()) == sr["outputs"]["zig_sha256"]
    assert rc["inputs"]["space_fit"]["scale"][0] == rc["inputs"]["space_fit"]["scale"][1] == rc["inputs"]["space_fit"]["scale"][2], "uniform fit expected"
    return "receipt hashes == files; uniform space fit recorded"


def reproducible():
    import bake_avatar as BA
    import build_select_model as BS
    with tempfile.TemporaryDirectory() as td:
        zif, zig, rec = BA.bake(ROOT / "work/pc-avatar/Racers_Avatar.zif", ROOT / "work/pc-avatar/Racers_Avatar.zig", BA.TARGETS, Path(td), "R.json")
        assert sha(zif) == sha((BAKE / "Avatar.zif").read_bytes()) and sha(zig) == sha((BAKE / "Avatar.zig").read_bytes()), "racer bake not reproducible"
    with tempfile.TemporaryDirectory() as td:
        zif, zig = BS.main(Path(td))
        assert sha(zif) == sha((SEL / "Avatar.zif").read_bytes()) and sha(zig) == sha((SEL / "Avatar.zig").read_bytes()), "select model not reproducible"
        assert (Path(td) / "SELECT-RECEIPT.json").read_bytes() == (SEL / "SELECT-RECEIPT.json").read_bytes(), "select receipt differs"
    return "fresh racer bake and fresh select-model build from current source == build/bake and build/select (zif+zig byte-identical, select receipt identical)"


case("racer file: five skinned avatar shapes", racer_structure)
case("select file: [AvatarCar, Avatar] + clip aliases", select_structure)
case("receipts bound to files", receipts)
case("reproducible racer bake and select model", reproducible)

if __name__ == "__main__":
    fails = [r for r in results if r[1] != "pass"]
    for r in results:
        print(f"{r[1]:4}  {r[0]}  {str(r[2])[:220]}")
    (ROOT / "tests/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests/results/test_bake.json").write_text(json.dumps(dict(suite="test_bake", cases=len(results), failed=len(fails),
                                                                         results=[dict(name=a, status=b, detail=str(c)) for a, b, c in results]), indent=1) + "\n")
    sys.exit(1 if fails else 0)
