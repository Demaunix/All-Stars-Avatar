"""Join the coordinator's pinned TOC / STRB / LZX receipts into one per-asset catalog (read-only).

Inputs (all in input/): AVATAR-TOC-STRUCTURE-REFERENCE.json, AVATAR-STRB-BOUNDARIES-REFERENCE.json,
AVATAR-LZX-CATALOG-REFERENCE.json and the pinned decoded outputs AVATAR-DECODED-<toc_index>.bin.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
INPUT = MODULE / "input"


@lru_cache(None)
def catalog():
    toc = json.loads((INPUT / "AVATAR-TOC-STRUCTURE-REFERENCE.json").read_text())["records"]
    strb = {a["toc_index"]: a for a in json.loads((INPUT / "AVATAR-STRB-BOUNDARIES-REFERENCE.json").read_text())["assets"]}
    lzx = {r["toc_index"]: r for r in json.loads((INPUT / "AVATAR-LZX-CATALOG-REFERENCE.json").read_text())["rows"]}
    out = []
    for r in toc:
        i = r["index"]
        if i not in strb:
            continue
        b = strb[i]["blocks"][0]
        out.append(dict(index=i, name=r["names"][0], names=r["names"], type=int(r["type_flags_uninterpreted"], 16),
                        variant=int(r["variant_flags_uninterpreted"], 16), kind=b["label_from_velocity"],
                        block_type=b["type"], toc_offset=int(b["toc_offset"], 16), block_bytes=b["bytes"],
                        decoded_sha256=lzx.get(i, {}).get("output_sha256"), decoded_bytes=lzx.get(i, {}).get("output_bytes")))
    return out


def by_name(name):
    return [c for c in catalog() if c["name"] == name]


def decoded(index, verify=True):
    c = next(c for c in catalog() if c["index"] == index)
    data = (INPUT / f"AVATAR-DECODED-{index:04d}.bin").read_bytes()
    if verify:
        assert hashlib.sha256(data).hexdigest() == c["decoded_sha256"], index
    return data


TOC = INPUT / "AvatarAssetPack.toc"


def raw_block(index):
    """Undecoded STRB block bytes (shape-override / animation blocks are not LZX-wrapped per the receipts)."""
    c = next(c for c in catalog() if c["index"] == index)
    with open(TOC, "rb") as f:
        f.seek(c["toc_offset"])
        return f.read(c["block_bytes"])


if __name__ == "__main__":
    rows = catalog()
    lines = ["index\tkind\ttype\tvariant\tdecoded_bytes\tname"]
    for c in rows:
        lines.append(f"{c['index']}\t{c['kind']}\t{c['type']:#x}\t{c['variant']:#x}\t{c['decoded_bytes']}\t{c['name']}")
    (MODULE / "research" / "catalog.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(rows))
