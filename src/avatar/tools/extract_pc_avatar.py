"""Copy the PC Avatar racer/assistant/car model members out of the installed xpacs (read-only) into work/pc-avatar/.

Adapted from mii-restoration-001/tools/extract_pc_mii.py (C234). Only reads the game folder; writes only in this module.
Records install-file hashes, row metadata and member hashes in research/pc-avatar-extract.json.
"""
import hashlib
import json
import struct
import zlib
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
GAME = Path("A:/SteamLibrary/steamapps/common/Sonic and SEGA All Stars Racing")
OUT = MODULE / "work" / "pc-avatar"


def ph(name):
    h = 0
    for c in name.upper().replace("/", "\\")[::-1]:
        h = (ord(c) + 131 * h) & 0xFFFFFFFF
    return h


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with open(path, "rb") as f:
        h = struct.unpack("<6I", f.read(24))
        assert h[0] == 0 and h[1] == 0 and h[4] == 0 and h[5] == 0, "not an xpac"
        return {r[0]: r for r in (struct.unpack("<5I", f.read(20)) for _ in range(h[3]))}


def member(path, row):
    with open(path, "rb") as f:
        f.seek(row[1])
        raw = f.read(row[3])
    try:
        return raw, zlib.decompress(raw), True
    except zlib.error:
        return raw, raw, False


WANTED = [("Racers.xpac", ".\\Resource/Racers/Avatar.zif"), ("Racers.xpac", ".\\Resource/Racers/Avatar.zig"),
          ("Select.xpac", ".\\Resource/Select/AvatarSelect.zif"), ("Select.xpac", ".\\Resource/Select/AvatarSelect.zig"),
          ("Select.xpac", ".\\Resource/Select/Avatar.zif"), ("Select.xpac", ".\\Resource/Select/Avatar.zig"),
          ("Racers.xpac", ".\\Resource/Racers/Mii.zif"), ("Racers.xpac", ".\\Resource/Racers/Mii.zig")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = dict(game=str(GAME), exe_sha256=sha_file(GAME / "Sonic & SEGA All-Stars Racing.exe"), archives={}, members=[])
    cache = {}
    for arc, name in WANTED:
        p = GAME / "Resource" / arc
        if arc not in cache:
            cache[arc] = rows(p)
            receipt["archives"][arc] = dict(sha256=sha_file(p), bytes=p.stat().st_size, rows=len(cache[arc]))
        key = ph(name)
        if key not in cache[arc]:
            receipt["members"].append(dict(archive=arc, name=name, key=f"{key:08X}", present=False))
            continue
        row = cache[arc][key]
        raw, data, compressed = member(p, row)
        dst = OUT / (arc.split(".")[0] + "_" + name.split("/")[-1])
        dst.write_bytes(data)
        receipt["members"].append(dict(archive=arc, name=name, key=f"{key:08X}", present=True, row=list(row), compressed=compressed,
                                       stored_sha256=hashlib.sha256(raw).hexdigest(), sha256=hashlib.sha256(data).hexdigest(),
                                       bytes=len(data), out=str(dst.relative_to(MODULE))))
    (MODULE / "research").mkdir(exist_ok=True)
    (MODULE / "research" / "pc-avatar-extract.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
