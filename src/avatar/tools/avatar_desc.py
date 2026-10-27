"""The Xbox 360 avatar description (the 1000-byte "metadata" blob XAvatar uses) as decoded from the 17 built-in avatars
the All-Stars Xbox image carries at 0x82690B00 (copied by 0x8260AF50 for fallback avatars; found by the colour analysis of
2026-10-21). Layout (big-endian; CONFIRMED on all 17 blobs, meaning labels INFERRED where marked):
  +0x000 u32 0 | f32 a | f32 b                  (a, b in {-1, 0, 0.5, 1}: body height / weight sliders, INFERRED)
  +0x00C 15 x 16-byte item GUID {u32 slot bits, u16 TOC index, u16 gender, 8-byte pack tail} or zero: the face features
         (chin, nose, ears, mouth, eyes, eyebrows, facial hair, wrinkles / face paint...), one slot per feature type
  +0x0FC 9 x ARGB colours: skin, hair, mouth, eyes, eyebrows, eye shadow, facial hair, feature 1, feature 2
         (= XAvatar parameter IDs 0x0D..0x15 in order)
  +0x120 list A: up to 15 x 32-byte entries {GUID, u16 slot mask, u16 0, 3 x ARGB custom colours}: body, head, hair,
         clothes, shoes, glasses, jewellery...  (custom colours are non-zero only for colourable items)
  +0x300 list B: 4 x 32-byte entries, same layout (a second outfit set; meaning INFERRED: the items worn under / instead of
         a costume, e.g. list A top 69 "Male USA Sport Top" with list B top 98)
  +0x380 16 bytes trailer (unknown), then zeros to 1000.
Writes research/xbox-default-avatars.json (item names from the catalogue). Needs the Xbox image dump only to regenerate.
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from catalog import MODULE, catalog

TAIL = bytes.fromhex("c1c8f109a19cb2e0")
BLOBS_VA, BLOB_BYTES, NBLOBS = 0x82690B00, 1000, 17
COLOUR_NAMES = ["skin", "hair", "mouth", "eyes", "eyebrows", "eye_shadow", "facial_hair", "feature1", "feature2"]
ROWS = {c["index"]: c for c in catalog()}


def guid(b: bytes) -> dict | None:
    """Decode a 16-byte item GUID; None for an all-zero slot; raw hex when the pack tail is missing."""
    if b == bytes(16):
        return None
    t, i, g = struct.unpack(">IHH", b[:8])
    if b[8:] != TAIL:
        return dict(raw=b.hex())
    c = ROWS.get(i)
    return dict(type=f"{t:#x}", index=i, gender=g, name=c["name"] if c else None, type_matches=(c is not None and c["type"] == t))


def argb(b: bytes) -> str:
    """ARGB bytes -> '#RRGGBB' (alpha recorded separately when not 0xFF)."""
    return "#%02X%02X%02X" % (b[1], b[2], b[3]) + ("" if b[0] == 0xFF else f" (alpha {b[0]:#x})")


def parse_description(blob: bytes) -> dict:
    """Decode one 1000-byte avatar description."""
    assert len(blob) == BLOB_BYTES
    w0, fa, fb = struct.unpack_from(">Iff", blob, 0)
    features = [guid(blob[0x0C + 16 * j:0x0C + 16 * (j + 1)]) for j in range(15)]
    colours = {n: argb(blob[0xFC + 4 * k:0xFC + 4 * (k + 1)]) for k, n in enumerate(COLOUR_NAMES)}

    def entries(start: int, count: int) -> list:
        out = []
        for j in range(count):
            e = blob[start + 32 * j:start + 32 * (j + 1)]
            if e == bytes(32):
                continue
            g = guid(e[:16])
            mask16 = struct.unpack_from(">H", e, 16)[0]
            custom = [argb(e[20 + 4 * k:24 + 4 * k]) for k in range(3)] if e[20:32] != bytes(12) else []
            out.append(dict(slot=j, item=g, mask=f"{mask16:#x}", custom_colours=custom,
                            mask_matches=(g is not None and "index" in g and (int(g["type"], 16) & 0xFFFF) == mask16)))
        return out

    return dict(word0=w0, float_a=fa, float_b=fb, features=[f for f in features if f], feature_slots={j: f for j, f in enumerate(features) if f},
                colours=colours, items=entries(0x120, 15), items_b=entries(0x300, 4), trailer=blob[0x380:0x390].hex(),
                tail_zero=blob[0x390:] == bytes(BLOB_BYTES - 0x390))


def main() -> None:
    """Dump the 17 built-in descriptions from the Xbox image."""
    import xbimg
    out = []
    for k in range(NBLOBS):
        blob = xbimg.raw(BLOBS_VA + BLOB_BYTES * k, BLOB_BYTES)
        d = parse_description(blob)
        d["va"] = f"{BLOBS_VA + BLOB_BYTES * k:#x}"
        body = next((e["item"] for e in d["items"] if e["item"] and e["item"].get("type") == "0x2"), None)
        d["gender"] = {1: "male", 2: "female"}.get(body["gender"] if body else 0, "?")
        out.append(d)
        nm = lambda g: (g.get("name") or ("raw " + g["raw"] if "raw" in g else "?"))
        feats = ", ".join(nm(f) for f in d["features"])
        its = ", ".join(nm(e["item"]) + (" " + str(e["custom_colours"]) if e["custom_colours"] else "") for e in d["items"] if e["item"])
        print(f"[{k}] {d['gender']} a={d['float_a']} b={d['float_b']}  {feats}\n     items: {its}\n     B: " +
              ", ".join(nm(e["item"]) for e in d["items_b"] if e["item"]) + f"\n     colours: {d['colours']}")
    bad = [(k, e) for k, d in enumerate(out) for e in d["items"] + d["items_b"] if e["item"] and ("raw" in e["item"] or not e["item"]["type_matches"] or not e["mask_matches"])]
    badf = [(k, j, f) for k, d in enumerate(out) for j, f in d["feature_slots"].items() if "raw" in f or not f["type_matches"]]
    print("entries with problems:", bad, "feature slots with problems:", badf, "trailer zero:", all(d["tail_zero"] for d in out))
    (MODULE / "research" / "xbox-default-avatars.json").write_text(json.dumps(dict(
        source=dict(image=xbimg.ROOT + "xbox-image.bin", sha256=xbimg.SHA, va=f"{BLOBS_VA:#x}", count=NBLOBS, bytes_each=BLOB_BYTES),
        layout=__doc__, avatars=out), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
