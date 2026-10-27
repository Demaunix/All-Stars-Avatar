"""Print the models, BASE bind TRS, non-unit scales and BASE scale channels of an Avatar .zif (read-only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import pcanim  # noqa: E402
import zif  # noqa: E402


def main(path: str) -> None:
    """Dump rig facts relevant to the BASE scale fix."""
    np.set_printoptions(precision=4, suppress=True)
    z = zif.Zif(path)
    for f in z.forests:
        print("forest", f["index"], f["name"], "base", hex(f["base"]))
        for m in f["models"]:
            names = [n["name"] for n in m["nodes"]]
            print(" model", m["index"], m["root"], "nodes", len(names), "anims", len(m["anims"]), "header", [hex(x) for x in m["header"][:8]])
            if len(names) < 2:
                continue
            T, Q, S = pcanim.bind_locals(z, f, m)
            par = pcanim.parents(m)
            for i in range(min(3, len(names))):
                print("   ", i, names[i], "par", par[i], "T", T[i][:3], "S", S[i][:3])
            nonunit = [(names[i], S[i][:3]) for i in range(len(names)) if not np.allclose(S[i][:3], 1, atol=1e-4)]
            print("    non-unit bind scales:", nonunit)
            if "BASE" in names:
                bi = names.index("BASE")
                W = pcanim.world(m, T, Q, S)
                ys = W[:, 3, 1]
                print("    BASE index", bi, "world y range", ys.min(), ys.max(), "HEAD y", ys[names.index("HEAD")] if "HEAD" in names else None)
                scale_clips, bad = [], []
                for k, a in enumerate(m["anims"]):
                    nxt = m["anims"][k + 1]["offset"] if k + 1 < len(m["anims"]) else None
                    if nxt is None:
                        bad.append(a["name"] + "(last)")
                        continue
                    try:
                        dd = pcanim.decode(z, f, m, a, nxt)
                    except Exception as e:  # undecoded clip kinds are reported, not fatal
                        bad.append(f"{a['name']}({str(e)[:30]})")
                        continue
                    for r in dd["channels"]:
                        if r["bit"] == 4 and r["node"] <= bi:
                            scale_clips.append((a["name"], names[r["node"]]))
                print("    scale channels on root/BASE:", scale_clips)
                print("    undecoded clips:", bad)


if __name__ == "__main__":
    main(sys.argv[1])
