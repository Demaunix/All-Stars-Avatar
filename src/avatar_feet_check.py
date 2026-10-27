"""For each decodable clip of the avatar models, report BASE height and lowest foot height before/after fix A (read-only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import pcanim  # noqa: E402
import zif  # noqa: E402



def main(old_path: str, new_path: str, models: str) -> None:
    """Print per-clip (mid-clip) BASE y and min foot-joint y for old and patched files."""
    zo, zn = zif.Zif(old_path), zif.Zif(new_path)
    want = set(models.split(","))
    for fo, fn in zip(zo.forests, zn.forests):
        for mo, mn in zip(fo["models"], fn["models"]):
            if mo["root"] not in want:
                continue
            names = [n["name"] for n in mo["nodes"]]
            feet = [names.index(n) for n in ("LF_A", "LF_T", "RT_A", "RT_T")]
            print(mo["root"], "foot joints:", [names[i] for i in feet])
            bo, bn = pcanim.bind_locals(zo, fo, mo), pcanim.bind_locals(zn, fn, mn)
            for k, a in enumerate(mo["anims"]):
                nxt = mo["anims"][k + 1]["offset"] if k + 1 < len(mo["anims"]) else None
                if nxt is None:
                    continue
                try:
                    dd = pcanim.decode(zo, fo, mo, a, nxt)
                except Exception:
                    continue
                rows = []
                for frac in (0.0, 0.5):
                    Wo = pcanim.world(mo, *pcanim.pose(mo, bo, dd, frac))
                    Wn = pcanim.world(mn, *pcanim.pose(mn, bn, dd, frac))
                    rows.append((Wo[1, 3, 1], Wo[feet, 3, 1].min(), Wn[feet, 3, 1].min()))
                r = rows[1]
                print(f"  {a['name']:14s} BASE y {r[0]:6.3f}  min foot y old {r[1]:6.3f} new {r[2]:6.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
