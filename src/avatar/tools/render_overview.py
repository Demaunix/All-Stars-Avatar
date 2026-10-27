"""One overview sheet per baked model, rendered from the written PC files: bind pose (front/side/back) plus decoded
stock PC animation key poses, each camera framed on the posed mesh. Unique output names per run (an earlier
viewer cached images by path; see CHECKPOINTS.md)."""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import pcanim
import pcbody
from softrender import Renderer
from verify_bake import skin
from zif import Zif

MODULE = Path(__file__).resolve().parents[1]


def frame(P, T, tex, yaw, size):
    used = np.unique(T)
    lo, hi = P[used].min(0), P[used].max(0)
    c = (lo + hi) / 2
    dist = max(4.0, 2.2 * float(np.linalg.norm(hi - lo)))
    a = np.radians(yaw)
    r = Renderer(*size, ss=2)
    r.set_camera((c[0] + dist * np.sin(a), c[1] + 0.15 * dist, c[2] + dist * np.cos(a)), tuple(c), fovy=30)
    return r, a


def render(P, N, U, T, tex, yaw, size=(220, 300)):
    r, a = frame(P, T, tex, yaw, size)
    h, w = tex.shape[:2]

    def shade(n, t):
        x = np.clip((t[:, 0] * w).astype(int), 0, w - 1)
        y = np.clip(((1.0 - t[:, 1]) * h).astype(int), 0, h - 1)   # native rule: sampled V = 1 - stored V
        return tex[y, x, :3].astype(float), np.ones(len(n))
    r.draw(P, N, U, np.stack([T, T, T], -1), shade, cull="back", front_ccw=True, light_dir=(np.sin(a) + 0.3, 0.8, np.cos(a) + 0.3))
    return r.image()


def main():
    zif_p, zig_p = Path(sys.argv[1]), Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else time.strftime("%H%M%S")
    outd = MODULE / "renders" / f"pc-{tag}"
    outd.mkdir(parents=True, exist_ok=True)
    z = Zif(zif_p, zig_p)
    f = z.forests[0]
    receipts = []
    for m in f["models"]:
        if not m["root"].startswith("Avatar") or m["root"] == "AvatarCar":
            continue
        row = [r for r in z.meshes(f, m) if r["node"] == 0][0]
        mesh = pcbody.read_mesh(z, row)
        pal, ib = pcbody.palette(z, f, row)
        tex = next(t for t in f["textures"] if t["pos"] - f["base"] == row["samplers"][0][1])
        atlas = np.array(Image.open(io.BytesIO(z.g[tex["header"][2]:])).convert("RGBA"))
        bind = pcanim.bind_locals(z, f, m)
        offs = sorted(set(a["offset"] for a in m["anims"]))
        ends = {o: (offs[i + 1] if i + 1 < len(offs) else m["header"][22]) for i, o in enumerate(offs)}
        P, N, U, T = mesh["pos"], mesh["nrm"][:, :3], mesh["uv"], mesh["tris"]
        bi = mesh["idx"].astype(int)
        tiles = [(f"bind yaw {y}", render(P, N, U, T, atlas, y)) for y in (0, 90, 180)]
        want = ("Speed1", "DriftL", "HardHitL", "Taunt1", "WIN", "RevengeOn") if m["root"] == "Avatar" else tuple(a["name"] for a in m["anims"])
        for an in want:
            a = next((x for x in m["anims"] if x["name"] == an), None)
            if a is None:
                continue
            try:
                dec = pcanim.decode(z, f, m, a, ends[a["offset"]])
            except Exception as e:  # noqa: BLE001
                tiles.append((f"{an}: {str(e)[:20]}", Image.new("RGB", (220, 300), (80, 0, 0))))
                continue
            Tq, Qq, Sq = pcanim.pose(m, bind, dec, 0.5)
            Wc = pcanim.world(m, Tq, Qq, Sq)
            Ps, Ns = skin(P, N, mesh["w"], bi, pal, ib, Wc)
            tiles.append((f"{an}@0.5 yaw 20", render(Ps, Ns, U, T, atlas, 20)))
        W = Image.new("RGB", (220 * len(tiles), 320), (40, 44, 52))
        d = ImageDraw.Draw(W)
        for k, (lab, im) in enumerate(tiles):
            W.paste(im, (220 * k, 20))
            d.text((220 * k + 4, 4), f"{m['root']} {lab}", fill="white")
        fn = outd / f"{m['root']}-overview-{tag}.png"
        W.save(fn)
        receipts.append(dict(model=m["root"], file=str(fn.relative_to(MODULE)), tiles=[t[0] for t in tiles]))
        print(fn)
    (outd / f"OVERVIEW-{tag}.json").write_text(json.dumps(dict(zif=str(zif_p), zig=str(zig_p), sheets=receipts), indent=1) + "\n")


if __name__ == "__main__":
    main()
