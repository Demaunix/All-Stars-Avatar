"""Xbox avatar item IDs for the Asset Package TOC records (M10 phase 1, step 2). Read-only.

Finding (2026-10-21): an Xbox 360 avatar item GUID is  {slot bits:08x}-{TOC index:04x}-{gender:04x}-c1c8-f109a19cb2e0
(gender 1 male, 2 female, 3 both; the tail is the pack constant also found in the TOC header and in each TOC record's
hiding-template link words 39..42). This script derives the GUID for every record and verifies it against
  * research/xbox-avatars-app/Assets.json (the IDs the installed Xbox Original Avatars app uses, with ColorSetID only: all
    666 base-pack IDs come from here) and AvatarClosetResponse.json (a sample closet; it carries names, but none of its IDs
    are base-pack IDs, so the name comparison below finds nothing to check: name_checked stays 0), and
  * the GUID stored in each payload's STRB header (bytes 6..21, per Velocity's STRBHeader layout): this turned out to be one
    pack-wide GUID (8be0bd59-0949-e44e-b991-ada87b7c0b6b in every block), not the item's, so it is only recorded.
Writes research/item-ids.json and prints the verification counts. Mismatches are listed, never hidden.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from catalog import INPUT, MODULE, TOC, catalog

TAIL = "c1c8-f109a19cb2e0"
GUID_RE = re.compile(r"^([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})$")
APP = MODULE / "research" / "xbox-avatars-app"
PAYLOAD = {r["index"]: (int(r["payload_offset"], 16), r["payload_size"]) for r in
           json.loads((INPUT / "AVATAR-TOC-STRUCTURE-REFERENCE.json").read_text())["records"]}
PACK = TOC.read_bytes()


def raw_block(index: int) -> bytes:
    """The record's STRB container bytes (payload_offset/size from the pinned TOC structure reference)."""
    off, size = PAYLOAD[index]
    return PACK[off:off + size]


def guid_for(c: dict) -> str:
    """Item GUID derived from a catalogue row."""
    return f"{c['type']:08x}-{c['index']:04x}-{(c['variant'] >> 24) & 0xFF:04x}-{TAIL}"


def walk(obj, out: list, name=None) -> None:
    """Collect every (guid, nearby name, extra) from a JSON tree."""
    if isinstance(obj, dict):
        nm = name
        for k in ("name", "Name", "displayName", "DisplayName", "title"):
            if isinstance(obj.get(k), str):
                nm = obj[k]
        for k, v in obj.items():
            if isinstance(v, str) and GUID_RE.match(v.lower()):
                out.append(dict(guid=v.lower(), name=nm, key=k, colorset=obj.get("ColorSetID", obj.get("colorSetId"))))
            else:
                walk(v, out, nm)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, out, name)


def block_guid(index: int) -> str:
    """GUID stored in the raw STRB block header, printed in the Windows text form (first three fields little-endian)."""
    b = raw_block(index)
    if b[:4] != b"STRB":
        return "no-STRB"
    g = b[6:22]
    return f"{int.from_bytes(g[0:4], 'little'):08x}-{int.from_bytes(g[4:6], 'little'):04x}-{int.from_bytes(g[6:8], 'little'):04x}-{g[8:10].hex()}-{g[10:16].hex()}"


def main() -> None:
    """Derive, verify and write the ID table."""
    rows = {c["index"]: c for c in catalog()}
    found = []
    for f in ("Assets.json", "AvatarClosetResponse.json"):
        walk(json.loads((APP / f).read_text(encoding="utf-8-sig")), found, None)
    app = {}
    for e in found:
        app.setdefault(e["guid"], []).append(e)
    stats = dict(app_ids=len(app), app_ids_with_tail=0, matched_toc=0, type_mismatch=[], gender_mismatch=[], index_missing=[], name_mismatch=[], name_checked=0,
                 strb_header_guids={}, not_in_app_by_kind={})
    table = []
    for i, c in sorted(rows.items()):
        g = guid_for(c)
        bg = block_guid(i)
        stats["strb_header_guids"][bg] = stats["strb_header_guids"].get(bg, 0) + 1
        entries = app.get(g, [])
        names = [e["name"] for e in entries if e["name"]]
        ok_name = None
        if names:
            stats["name_checked"] += 1
            toc_names = {n.strip().lower() for n in c["names"] if n.strip()}
            ok_name = any(n.strip().lower() in toc_names for n in names)
            if not ok_name:
                stats["name_mismatch"].append(dict(index=i, toc=c["name"], app=sorted(set(names))))
        if not entries:
            k = c["kind"] + (" (Hat)" if c["name"].endswith(" (Hat)") else "")
            stats["not_in_app_by_kind"][k] = stats["not_in_app_by_kind"].get(k, 0) + 1
        table.append(dict(index=i, guid=g, name=c["name"], type=f"{c['type']:#x}", gender=(c["variant"] >> 24) & 0xFF, kind=c["kind"],
                          in_app=bool(entries), app_names=sorted(set(names)), app_colorset=sorted({str(e["colorset"]) for e in entries if e["colorset"] is not None}),
                          app_name_matches=ok_name))
    for g, entries in app.items():
        m = GUID_RE.match(g)
        if not g.endswith(TAIL):
            continue
        stats["app_ids_with_tail"] += 1
        t, idx, gen = int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)
        c = rows.get(idx)
        if c is None:
            stats["index_missing"].append(dict(guid=g, names=sorted({e["name"] for e in entries if e["name"]})))
            continue
        if c["type"] != t:
            stats["type_mismatch"].append(dict(guid=g, toc_type=f"{c['type']:#x}", toc_name=c["name"]))
        elif ((c["variant"] >> 24) & 0xFF) != gen:
            stats["gender_mismatch"].append(dict(guid=g, toc_gender=(c["variant"] >> 24) & 0xFF, toc_name=c["name"]))
        else:
            stats["matched_toc"] += 1
    out = dict(rule=f"{{type:08x}}-{{toc_index:04x}}-{{gender:04x}}-{TAIL}", sources=[str(p.relative_to(MODULE)) for p in sorted(APP.glob('*.json'))],
               stats=stats, items=table)
    (MODULE / "research" / "item-ids.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else (len(v), v[:5])) for k, v in stats.items()}, indent=1))
    print("records:", len(table), "in app:", sum(1 for r in table if r["in_app"]), "not in app:", sum(1 for r in table if not r["in_app"]))
    byi = {r["index"]: r for r in table}
    for i in (0, 92, 144, 228, 449, 450, 607, 794):
        r = byi[i]
        print(r["index"], r["guid"], r["name"], "in app" if r["in_app"] else "not in app", "colorset", r["app_colorset"])
    print("model records not in the app (excluding (Hat) hair):", [r["index"] for r in table if not r["in_app"] and r["kind"] == "model" and not r["name"].endswith(" (Hat)")][:60])


if __name__ == "__main__":
    main()
