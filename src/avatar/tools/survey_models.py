"""Decode every model block in the catalog with avatar_model.decode; record per-model structural results."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from catalog import catalog, decoded
import avatar_model as AM

MODULE = Path(__file__).resolve().parents[1]


def main():
    rows = []
    fails = Counter()
    for c in catalog():
        if c["kind"] != "model":
            continue
        d = decoded(c["index"])
        r = dict(index=c["index"], name=c["name"], type=c["type"], bytes=len(d))
        try:
            m = AM.decode(d)
            r["status"] = "EMPTY" if m["empty"] else "DECODED"
            r["header"] = m["header"][:22]
            r["meshes"] = [dict(vertex_count=x["vertex_count"], index_count=x["index_count"], widths=x["widths"], record_bits=x["record_bits"],
                                tex_halves=x["tex_halves"], element_bits=x["element_bits"], element_base={k: hex(v) for k, v in x["element_base"].items()},
                                bones=sorted(set(np.unique(x["bones"]).tolist())), offset=x["offset"], end=x["end"]) for x in m["meshes"]]
            r["tail_offset"] = m.get("tail_offset")
            r["tail_bytes"] = len(d) - m["tail_offset"] if m.get("tail_offset") else None
        except AM.FormatError as e:
            r["status"] = "FAILED"
            r["error"] = str(e)
            fails[str(e)[:60]] += 1
        rows.append(r)
    st = Counter(r["status"] for r in rows)
    out = dict(counts=dict(st), failures=dict(fails), rows=rows)
    (MODULE / "research" / "MODEL-DECODE-SURVEY.json").write_text(json.dumps(out, indent=1) + "\n")
    print(st, fails.most_common(10))
    th = Counter(x["tex_halves"] for r in rows if r["status"] == "DECODED" for x in r["meshes"])
    print("tex halves", th)


if __name__ == "__main__":
    main()
