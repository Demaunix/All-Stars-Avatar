"""Read-only: stock PC racer materials used with the skinned 32-byte declaration (flags 531) and one sampler.
Output research/PC-MATERIAL-CANDIDATES.json (forest, model, material, header words, sampler words)."""
import json
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from survey_pc_palettes import members
from zif import Zif

MODULE = Path(__file__).resolve().parents[1]


def main():
    ms = members("Racers.xpac")
    out = defaultdict(list)
    with tempfile.TemporaryDirectory() as td:
        for k, (r, d) in enumerate(ms):
            if d[4:8] != b"FORE":
                continue
            fp = Path(td) / f"m{k}.zif"
            fp.write_bytes(d)
            try:
                z = Zif(fp)
            except AssertionError:
                continue
            for f in z.forests:
                for m in f["models"]:
                    for row in z.meshes(f, m):
                        vd = row["vertex_descriptor"]
                        mh = row["material_header"]
                        if vd[9] == 531 and mh[29] == 1 and row["shape_header"][6] > 0:
                            key = (hex(mh[1]), mh[3])
                            out[str(key)].append(dict(forest=f["name"], model=m["root"], material=row["material_name"], header=mh,
                                                      samplers=row["samplers"], palette=row["shape_header"][6], mesh_word40=row["mesh_header"][40],
                                                      mesh_words24_33=row["mesh_header"][24:34]))
    res = {k: dict(count=len(v), forests=sorted(set(x["forest"] for x in v)), example=v[0]) for k, v in out.items()}
    (MODULE / "research" / "PC-MATERIAL-CANDIDATES.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["count"]):
        print(k, v["count"], v["forests"][:12], v["example"]["material"])


if __name__ == "__main__":
    main()
