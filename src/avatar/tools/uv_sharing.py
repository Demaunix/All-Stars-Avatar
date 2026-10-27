"""Measure how much of each preset garment's UV0 texture space is used by more than one 3D surface (mirrored halves,
repeated panels). A texel (128 x 128 grid, UV taken modulo 1 per triangle as wrap addressing does) is shared when the
triangles covering it have 3D centroids spread by more than 5 cm (excludes neighbours meeting at an edge).
Writes research/UV-SHARING.json (cited by research/SURFACE-NOTES.md)."""
import json
from pathlib import Path

import numpy as np

import avatar_compose as AC
import avatar_model as AM

MODULE = Path(__file__).resolve().parents[1]
RES = 128


def sharing(me):
    uv = me["texcoords"][:, 0:2]
    P = AM.positions(me)
    T = me["triangles"]
    shift = np.floor(uv[T].mean(1))                      # per-triangle wrap period
    uvt = (uv[T] - shift[:, None, :]).reshape(-1, 2)
    tris = np.arange(len(T) * 3).reshape(-1, 3)
    ys, xs, ts, _ = AC.raster_uv(uvt, tris, RES)
    cen = P[T].mean(1)[ts]
    key = ys * RES + xs
    lo = np.full((RES * RES, 3), np.inf)
    hi = np.full((RES * RES, 3), -np.inf)
    np.minimum.at(lo, key, cen)
    np.maximum.at(hi, key, cen)
    used = np.isfinite(lo[:, 0])
    spread = np.linalg.norm(hi - lo, axis=1)
    return float((spread[used] > 0.05).mean()), int(used.sum())


if __name__ == "__main__":
    out = {}
    for name, P in AC.PRESETS.items():
        for role in ("shirt", "pants", "shoes"):
            i = P[role]
            if str(i) in out:
                continue
            part = AC.load_part(i)
            rows = [sharing(me) for me in part["meshes"]]
            texels = sum(n for _, n in rows)
            out[str(i)] = dict(name=part["name"], role=role, shared_fraction=round(sum(f * n for f, n in rows) / max(texels, 1), 3),
                               texels_used=texels, grid=RES)
    (MODULE / "research" / "UV-SHARING.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        print(k, v["name"], v["shared_fraction"])
