"""Read-only survey of stock PC racer models (Racers.xpac, in memory): skinned shape palette sizes, vertex flags,
material word patterns. Establishes the engine-accepted envelope for the Avatar bake. Output research/PC-PALETTE-SURVEY.json."""
import json
import os
import struct
import sys
import tempfile
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from zif import Zif

MODULE = Path(__file__).resolve().parents[1]
# The installed game (read-only; the bake clones the stock EggHands material from Racers.xpac). Override with ASR_GAME_DIR.
GAME = Path(os.environ.get("ASR_GAME_DIR", "A:/SteamLibrary/steamapps/common/Sonic and SEGA All Stars Racing"))


def members(arc):
    p = GAME / "Resource" / arc
    with open(p, "rb") as f:
        h = struct.unpack("<6I", f.read(24))
        rows = [struct.unpack("<5I", f.read(20)) for _ in range(h[3])]
        out = []
        for r in rows:
            f.seek(r[1])
            raw = f.read(r[3])
            try:
                out.append((r, zlib.decompress(raw)))
            except zlib.error:
                out.append((r, raw))
    return out


def main():
    ms = members("Racers.xpac")
    zifs = [(r, d) for r, d in ms if d[4:8] == b"FORE"]
    # pair each zif with the following zig by size heuristics is unreliable; use the GPU-less parse (meshes need only zif)
    rows_out = []
    pal = Counter()
    flags = Counter()
    matw = Counter()
    with tempfile.TemporaryDirectory() as td:
        for k, (r, d) in enumerate(zifs):
            fp = Path(td) / f"m{k}.zif"
            fp.write_bytes(d)
            try:
                z = Zif(fp)
            except AssertionError:
                continue
            for f in z.forests:
                for m in f["models"]:
                    for row in z.meshes(f, m):
                        sh = row["shape_header"]
                        if sh[6]:
                            pal[sh[6]] += 1
                        flags[(row["vertex_descriptor"][9], row["vertex_descriptor"][3], tuple(tuple(x) for x in row["declaration"]))] += 1
                        mh = row["material_header"]
                        matw[(mh[1], mh[3], mh[29])] += 1
                        if sh[6] >= 30:
                            rows_out.append(dict(forest=f["name"], model=m["root"], node=row["node_name"], palette=sh[6], material=row["material_name"],
                                                 vd_flags=row["vertex_descriptor"][9], stride=row["vertex_descriptor"][3]))
    out = dict(max_palette=max(pal), palette_histogram=dict(sorted(pal.items())),
               vertex_formats=[dict(flags=k[0], stride=k[1], decl=k[2], count=v) for k, v in flags.most_common(20)],
               material_words=[dict(w1=hex(k[0]), w3=k[1], samplers=k[2], count=v) for k, v in matw.most_common(25)],
               large_palettes=rows_out[:40])
    (MODULE / "research" / "PC-PALETTE-SURVEY.json").write_text(json.dumps(out, indent=1) + "\n")
    print("max palette", out["max_palette"], dict(sorted(pal.items())))
    for v in out["vertex_formats"][:10]:
        print(v)
    for v in out["material_words"][:15]:
        print(v)


if __name__ == "__main__":
    main()
