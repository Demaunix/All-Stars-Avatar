"""Decoder tests: catalogue-wide structural decode, invariants, negative controls (mutated inputs must be refused, not guessed),
texture record rules and the bone-order source. Results: tests/results/test_decoder.json."""
import hashlib
import json
import struct
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import avatar_model as AM
import avatar_texture as AT
import bone_map
from catalog import catalog, decoded

results = []


def case(name, fn):
    try:
        d = fn()
        results.append((name, "pass", d or ""))
    except Exception as e:  # noqa: BLE001
        results.append((name, "FAIL", repr(e) + " " + traceback.format_exc().splitlines()[-2]))


def refuses(data, exc=(AM.FormatError, AssertionError, struct.error, ValueError, IndexError)):
    try:
        AM.decode(data)
    except exc:
        return True
    return False


def all_models():
    n = dec = empty = 0
    zero_normals = {}
    for c in catalog():
        if c["kind"] != "model":
            continue
        n += 1
        m = AM.decode(decoded(c["index"]))
        if m["empty"]:
            empty += 1
            continue
        dec += 1
        for me in m["meshes"]:
            assert me["triangles"].max() < me["vertex_count"]
            w = me["weights"].sum(1)
            assert (w == 255).all(), (c["index"], np.unique(w))
            assert (me["bones"] < 71).all()
            assert np.isfinite(AM.positions(me)).all()
            nl = np.linalg.norm(me["normal"], axis=1)
            zero = me["normal_raw"] == 0
            if zero.any():
                zero_normals[c["name"]] = int(zero.sum())
            assert np.abs(nl[~zero] - 1).max() < 0.02, (c["index"], np.abs(nl[~zero] - 1).max())
    assert (n, dec, empty) == (607, 605, 2), (n, dec, empty)
    return (f"{dec} decoded + {empty} empty of {n}; every mesh: indices < vertex count, UBYTE4N weights sum 255, bone index < 71, "
            f"HEND3N normals unit length except stored all-zero words {zero_normals}")


def negative_controls():
    d = bytearray(decoded(0))
    m = AM.decode(bytes(d))
    me = m["meshes"][0]
    out = []
    bad = bytearray(d); struct.pack_into("<f", bad, me["offset"] + 4, 1e-5)   # step constant
    out.append(("step", refuses(bytes(bad))))
    bad = bytearray(d); struct.pack_into("<H", bad, me["index_header"] + 6, me["index_width"] + 1)   # index width
    out.append(("index width", refuses(bytes(bad))))
    bad = bytearray(d); struct.pack_into("<I", bad, me["index_header"], me["index_count"] + 1)   # index count not multiple of 3 / header sum
    out.append(("index count", refuses(bytes(bad))))
    bad = bytearray(d); struct.pack_into("<I", bad, 0x14, 2)   # header mesh count
    out.append(("mesh count", refuses(bytes(bad))))
    bad = bytearray(d); struct.pack_into("<I", bad, 0x10, struct.unpack_from("<I", d, 0x10)[0] + 2)   # header index bytes
    out.append(("index bytes", refuses(bytes(bad))))
    bad = bytearray(d); s = (me["offset"] + 20) * 8 + 18 + 32 * 1   # normal bits field -> 40 (> 32)
    v = 40
    for k in range(32):
        byte, bit = divmod(s + k, 8)
        bad[byte] = (bad[byte] & ~(1 << bit)) | (((v >> k) & 1) << bit)
    out.append(("element width > 32", refuses(bytes(bad))))
    out.append(("truncated", refuses(bytes(d[:len(d) // 2]))))
    assert all(r for _, r in out), out
    return "refused: " + ", ".join(n for n, _ in out)


def textures():
    n = 0
    spans = {}
    for c in catalog():
        if c["kind"] != "texture":
            continue
        d = decoded(c["index"])
        r = AT.standalone(d)
        img = AT.decode_span(d, r, 0, tiled=False)
        assert img.shape == (r["height"], r["width"], 4)
        spans[r["spans"]] = spans.get(r["spans"], 0) + 1
        n += 1
    assert n == 187 and spans == {1: 88, 5: 27, 14: 72}, (n, spans)
    bad = bytearray(decoded(607)); struct.pack_into("<I", bad, 12, 0x1234)
    try:
        AT.standalone(bytes(bad))
        raise AssertionError("size mismatch accepted")
    except AT.TextureError:
        pass
    return f"187 standalone textures parse and decode (spans {spans}); size-mismatch control refused"


def bit8_is_not_tiling():
    """Format bit 8 is not a tiling flag: textures carrying it are smoother (mean neighbour difference) decoded linearly
    than through the Xbox 2D untile address formula; flat textures tie."""
    rough = lambda im: np.abs(np.diff(im[..., :3].astype(float), axis=1)).mean() + np.abs(np.diff(im[..., :3].astype(float), axis=0)).mean()
    n = flat = 0
    for c in catalog():
        if c["kind"] != "texture":
            continue
        d = decoded(c["index"])
        r = AT.standalone(d)
        if not r["tiled"]:
            continue
        lin, til = rough(AT.decode_span(d, r, 0, tiled=False)), rough(AT.decode_span(d, r, 0, tiled=True))
        n += 1
        if lin == til == 0:
            flat += 1
            continue
        assert lin < til, (c["index"], lin, til)
    assert n == 26, n
    return f"{n} standalone textures with bit 8: linear decode smoother than untiled for all {n - flat} non-flat ones ({flat} flat)"


def model_tails():
    n = 0
    for c in catalog():
        if c["kind"] != "model":
            continue
        d = decoded(c["index"])
        m = AM.decode(d)
        if m["empty"]:
            continue
        recs, trailer = AT.model_tail(d, m["tail_offset"])
        assert recs and trailer[-12:] == bytes.fromhex("01000000ffffffff00000000"), (c["index"], trailer[-12:].hex())
        n += 1
    return f"{n} model tails: texture records delimited, every trailer ends with the observed 12-byte terminator"


def bone_order():
    xb = bone_map.xbox_bone_names()
    assert len(xb) == 71 and xb[0] == "BASE" and xb[19] == "HEAD" and xb[70] == "RightThumb_03"
    mp = bone_map.mapping()
    assert len(set(mp)) == 71
    return "71 names from Xbox image table 0x826A6328 map 1:1 onto PC rig nodes"


def determinism():
    a = AM.decode(decoded(2))
    b = AM.decode(decoded(2))
    for x, y in zip(a["meshes"], b["meshes"]):
        assert np.array_equal(x["quantized"], y["quantized"]) and np.array_equal(x["triangles"], y["triangles"])
    return "repeat decode identical"


case("all 607 model blocks decode with invariants", all_models)
case("negative controls: mutated model blocks are refused", negative_controls)
case("standalone textures", textures)
case("format bit 8 is not tiling (linear vs untiled control)", bit8_is_not_tiling)
case("model tail texture records + trailer", model_tails)
case("bone order source", bone_order)
case("determinism", determinism)

if __name__ == "__main__":
    fails = [r for r in results if r[1] != "pass"]
    for r in results:
        print(f"{r[1]:4}  {r[0]}  {str(r[2])[:200]}")
    (ROOT / "tests/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests/results/test_decoder.json").write_text(json.dumps(dict(suite="test_decoder", cases=len(results), failed=len(fails),
                                                                            results=[dict(name=a, status=b, detail=str(c)) for a, b, c in results]), indent=1) + "\n")
    sys.exit(1 if fails else 0)
