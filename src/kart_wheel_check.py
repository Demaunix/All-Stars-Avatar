"""Report kart clips that move wheel/suspension nodes away from their bind position (read-only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import pcanim  # noqa: E402
import zif  # noqa: E402

WHEEL_WORDS = ("Susp", "WheelTurn", "WheelSpin")


def main(path: str, root: str, threshold: str = "0.02") -> None:
    """For each decodable clip of model `root`, print wheel nodes whose local translation or world position drifts."""
    thr = float(threshold)
    z = zif.Zif(path)
    for f in z.forests:
        for m in f["models"]:
            if m["root"] != root:
                continue
            names = [n["name"] for n in m["nodes"]]
            wheels = [i for i, n in enumerate(names) if any(w in n for w in WHEEL_WORDS)]
            bind = pcanim.bind_locals(z, f, m)
            Wb = pcanim.world(m, *bind)
            body = names.index("MainBody")
            print(f"{root}: {len(m['anims'])} clips; wheel nodes {[names[i] for i in wheels]}")
            for i in wheels:
                print(f"  bind {names[i]:14s} local T {np.round(bind[0][i, :3], 3)}  world {np.round(Wb[i, 3, :3], 3)}")
            skipped = []
            for k, a in enumerate(m["anims"]):
                nxt = m["anims"][k + 1]["offset"] if k + 1 < len(m["anims"]) else None
                if nxt is None:
                    skipped.append(a["name"] + "(last)")
                    continue
                try:
                    dd = pcanim.decode(z, f, m, a, nxt)
                except pcanim.AnimError as e:
                    skipped.append(f"{a['name']}({e})")
                    continue
                tch = sorted({names[r["node"]] for r in dd["channels"] if r["bit"] == 1 and r["node"] in wheels})
                sch = sorted({names[r["node"]] for r in dd["channels"] if r["bit"] == 4 and r["node"] in wheels})
                worst = (0.0, "", 0.0)
                for frac in np.linspace(0, 1, 11):
                    T, Q, S = pcanim.pose(m, bind, dd, frac)
                    W = pcanim.world(m, T, Q, S)
                    # wheel position in the kart body frame, compared with bind
                    inv_now = np.linalg.inv(W[body])
                    inv_b = np.linalg.inv(Wb[body])
                    for i in wheels:
                        if "WheelSpin" not in names[i]:
                            continue
                        p_now = (W[i] @ inv_now)[3, :3]
                        p_b = (Wb[i] @ inv_b)[3, :3]
                        dv = float(np.linalg.norm(p_now - p_b))
                        if dv > worst[0]:
                            worst = (dv, names[i], float(frac))
                if tch or sch or worst[0] > thr:
                    print(f"  {a['name']:14s} T-ch {tch} S-ch {sch} max wheel-vs-body drift {worst[0]:.3f} ({worst[1]} at {worst[2]:.1f})")
            print("  not decoded:", ", ".join(skipped))


if __name__ == "__main__":
    main(*sys.argv[1:])
