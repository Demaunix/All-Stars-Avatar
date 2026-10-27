"""Offline multi-angle renders of a composed avatar preset in avatar space (decoded data only).

Two shadings of the same geometry:
  reference : per-pixel console-style shading (base UV0 wrap + decals UVk clamp-to-border; head six-layer bake);
  charts    : the single-UV/single-texture surface charts used by the PC bake (avatar_surface).
The mean absolute RGB difference between the two (same camera, coverage-masked) is the surface fidelity metric.
Uses the original softrender rasterizer (copied from C234 tools/softrender.py; renders what it is given)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import avatar_compose as AC
import avatar_model as AM
import avatar_surface as AS
from softrender import Renderer

MODULE = Path(__file__).resolve().parents[1]


def tex_shade(img):
    h, w = img.shape[:2]

    def shade(n, t):
        x = np.clip((t[:, 0] * w).astype(int), 0, w - 1)
        y = np.clip((t[:, 1] * h).astype(int), 0, h - 1)
        return img[y, x, :3] * 255.0, np.ones(len(n))
    return shade


def draw(r, parts, mode, light):
    for pt in parts:
        me = pt["mesh"]
        P, N = AM.positions(me), me["normal"]
        if mode == "reference":
            if pt["role"] in ("body", "hair"):
                c = AS.charts_for_part(pt, pt.get("record"))[0]
                T = c["tris"]
                r.draw(P, N, c["uv"], np.stack([T, T, T], -1), tex_shade(c["image"]), cull="back", front_ccw=True, light_dir=light)
                continue
            uv, shade = AS.reference_shade(pt, pt.get("record"))
            T = pt.get("keep_triangles", me["triangles"])
            r.draw(P, N, uv, np.stack([T, T, T], -1), shade, cull="back", front_ccw=True, light_dir=light)
        else:
            for c in pt["charts"]:
                T = c["tris"]
                r.draw(P, N, c["uv"], np.stack([T, T, T], -1), tex_shade(c["image"]), cull="back", front_ccw=True, light_dir=light)


def views(parts, mode, size=(300, 420), center=(0, 0.8, 0), dist=3.4, yaws=(0, 45, 90, 180, 270)):
    out = []
    for yaw in yaws:
        a = np.radians(yaw)
        eye = (center[0] + dist * np.sin(a), center[1] + 0.15, center[2] + dist * np.cos(a))
        r = Renderer(*size, ss=2)
        r.set_camera(eye, center, fovy=32)
        draw(r, parts, mode, (np.sin(a) + 0.3, 0.8, np.cos(a) + 0.3))
        out.append((yaw, r))
    return out


def sheet(rows, fn, title):
    size = (rows[0][1].out_size[0], rows[0][1].out_size[1])
    W = Image.new("RGB", (size[0] * len(rows), size[1] + 20), (40, 44, 52))
    d = ImageDraw.Draw(W)
    for k, (yaw, r) in enumerate(rows):
        W.paste(r.image(), (k * size[0], 20))
        d.text((k * size[0] + 4, 4), f"{title} yaw {yaw}", fill="white")
    W.save(fn)
    return hashlib.sha256(Path(fn).read_bytes()).hexdigest()


def fidelity(ref_rows, ch_rows):
    out = []
    for (yaw, a), (_, b) in zip(ref_rows, ch_rows):
        cov = np.isfinite(a.depth) & np.isfinite(b.depth)
        diff = np.abs(a.color - b.color).mean(-1)[cov]
        out.append(dict(yaw=yaw, mean_abs=float(diff.mean()), p95=float(np.percentile(diff, 95)), coverage_px=int(cov.sum()),
                        coverage_mismatch_px=int((np.isfinite(a.depth) ^ np.isfinite(b.depth)).sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--out", default="renders/avatar-space")
    ap.add_argument("--tag", default="v5")
    a = ap.parse_args()
    outd = MODULE / a.out
    outd.mkdir(parents=True, exist_ok=True)
    parts, prov = AC.compose(a.preset)
    for p in parts:
        p["charts"] = AS.charts_for_part(p, p.get("record"))
    ref = views(parts, "reference")
    ch = views(parts, "charts")
    rec = dict(preset=a.preset, provenance=prov,
               charts={p["role"]: [dict(name=c["name"], kind=c["kind"], triangles=int(len(c["tris"])), **({"overlap": c["overlap"]} if "overlap" in c else {}))
                                   for c in p["charts"]] for p in parts},
               reference=dict(file=f"{a.preset}-reference-{a.tag}.png", sha256=sheet(ref, outd / f"{a.preset}-reference-{a.tag}.png", a.preset + " reference")),
               charts_render=dict(file=f"{a.preset}-charts-{a.tag}.png", sha256=sheet(ch, outd / f"{a.preset}-charts-{a.tag}.png", a.preset + " charts")),
               fidelity=fidelity(ref, ch))
    hp = [p for p in parts if p["role"].startswith(("head", "hair"))]
    rec["head_close"] = sheet(views(hp, "charts", size=(300, 300), center=(0, 1.36, 0.02), dist=0.9), outd / f"{a.preset}-head-{a.tag}.png", a.preset + " head")
    (outd / f"{a.preset}-render-{a.tag}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(dict(fidelity=rec["fidelity"], charts=rec["charts"]), indent=1, default=str)[:2500])


if __name__ == "__main__":
    main()
