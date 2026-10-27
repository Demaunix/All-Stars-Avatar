"""Render the baked Avatar driver seated in the stock AvatarCar, both posed by the same-named stock clip (offline).

Reads the written PC files only. Car: LOD0 shapes of AvCarLODGrp (skinned to car nodes) + rigid shapes, first sampler texture
as diffuse. Driver: node-0 skinned shape. Both models share the forest origin (as stored in Avatar.Forest). Reports the
distance from each driver hand joint to the car's SteeringWheel node in each pose (evidence for the proportion decision)."""
import io
import json
import sys
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


def anim_world(z, f, m, name, frac):
    bind = pcanim.bind_locals(z, f, m)
    offs = sorted(set(a["offset"] for a in m["anims"]))
    ends = {o: (offs[i + 1] if i + 1 < len(offs) else m["header"][22]) for i, o in enumerate(offs)}
    a = next(x for x in m["anims"] if x["name"] == name)
    dec = pcanim.decode(z, f, m, a, ends[a["offset"]])
    T, Q, S = pcanim.pose(m, bind, dec, frac)
    return pcanim.world(m, T, Q, S), pcanim.world(m, *bind)


def tex_of(z, f, row):
    if not row["samplers"]:
        return np.full((4, 4, 4), 160, np.uint8)
    tex = next((t for t in f["textures"] if t["pos"] - f["base"] == row["samplers"][0][1]), None)
    if tex is None:
        return np.full((4, 4, 4), 160, np.uint8)
    return np.array(Image.open(io.BytesIO(z.g[tex["header"][2]:])).convert("RGBA"))


def draw_model(r, z, f, m, W, light, lod_node=None, rigid_skip=("Shadow",)):
    for row in z.meshes(f, m):
        if row["lod"] != 0 or any(s in row["node_name"] for s in rigid_skip) or "Shadow" in row["material_name"]:
            continue
        mesh = pcbody.read_mesh(z, row)
        pal, ib = pcbody.palette(z, f, row)
        P, N = mesh["pos"], mesh["nrm"][:, :3]
        if len(pal):
            Wt = mesh.get("w", np.c_[np.ones(len(P)), np.zeros((len(P), 3))])
            bi = mesh["idx"].astype(int) if "idx" in mesh else np.zeros((len(P), 4), int)
            P, N = skin(P, N, Wt, bi, pal, ib, W)
        else:
            M = W[row["node"]]
            P = (np.c_[P, np.ones(len(P))] @ M)[:, :3]
            N = N @ M[:3, :3]
        tex = tex_of(z, f, row)
        h, w = tex.shape[:2]
        uv = mesh.get("uv", np.zeros((len(P), 2)))

        def shade(n, t, tex=tex, h=h, w=w):
            x = np.clip((np.mod(t[:, 0], 1) * w).astype(int), 0, w - 1)
            y = np.clip((np.mod(t[:, 1], 1) * h).astype(int), 0, h - 1)
            return tex[y, x, :3].astype(float), np.ones(len(n))
        T = mesh["tris"]
        r.draw(P, N, uv, np.stack([T, T, T], -1), shade, cull="none", light_dir=light)


def main():
    zif_p, zig_p, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    z = Zif(zif_p, zig_p)
    f = z.forests[0]
    car = next(m for m in f["models"] if m["root"] == "AvatarCar")
    drv = next(m for m in f["models"] if m["root"] == "Avatar")
    cn = [n["name"] for n in car["nodes"]]
    dn = [n["name"] for n in drv["nodes"]]
    outd = MODULE / "renders" / f"kart-{tag}"
    outd.mkdir(parents=True, exist_ok=True)
    rows = []
    tiles = []
    for clip, frac in (("Speed1", 0.0), ("Sp1_TurnL", 0.5), ("DriftR", 0.5), ("StartIdle1", 0.0)):
        if not any(a["name"] == clip for a in car["anims"]) or not any(a["name"] == clip for a in drv["anims"]):
            continue
        try:
            Wc, _ = anim_world(z, f, car, clip, frac)
            Wd, _ = anim_world(z, f, drv, clip, frac)
        except pcanim.AnimError as e:  # e.g. kind-1 clips are not decoded by the reused C234 reader
            rows.append(dict(clip=clip, skipped=str(e)))
            continue
        wheel = Wc[cn.index("SteeringWheel"), 3, :3]
        lh, rh = Wd[dn.index("LeftHand"), 3, :3], Wd[dn.index("RightHand"), 3, :3]
        rows.append(dict(clip=clip, key_fraction=frac, steering_wheel=wheel.round(3).tolist(), left_hand_to_wheel=float(np.linalg.norm(lh - wheel)),
                         right_hand_to_wheel=float(np.linalg.norm(rh - wheel)), hands_span=float(np.linalg.norm(lh - rh))))
        for yaw in (25, 90, 160):
            a = np.radians(yaw)
            r = Renderer(260, 240, ss=2)
            c = np.array([0.0, 1.6, -0.6])
            r.set_camera(tuple(c + 7.0 * np.array([np.sin(a), 0.35, np.cos(a)])), tuple(c), fovy=34)
            light = (np.sin(a) + 0.3, 0.9, np.cos(a) + 0.3)
            draw_model(r, z, f, car, Wc, light)
            draw_model(r, z, f, drv, Wd, light)
            tiles.append((f"{clip} yaw {yaw}", r.image()))
    W = Image.new("RGB", (260 * 3, 260 * ((len(tiles) + 2) // 3)), (40, 44, 52))
    d = ImageDraw.Draw(W)
    for k, (lab, im) in enumerate(tiles):
        x, y = 260 * (k % 3), 260 * (k // 3)
        W.paste(im, (x, y + 20))
        d.text((x + 4, y + 4), lab, fill="white")
    fn = outd / f"kart-{tag}.png"
    W.save(fn)
    (outd / f"KART-{tag}.json").write_text(json.dumps(dict(zif=str(zif_p), image=str(fn.relative_to(MODULE)), poses=rows), indent=1) + "\n")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
