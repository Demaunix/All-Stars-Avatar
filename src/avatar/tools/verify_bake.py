"""Verify the baked Avatar.zif/.zig by re-reading them with the module's stock-format reader (zif.py, pcbody, pcanim),
independent of the bake code path, and render the PC files (bind pose + decoded PC animation key poses).

Checks (research/BAKE-VERIFY.json):
  * chunk structure parses; prefix/length words consistent; non-FORE chunks byte-identical to the stock file;
  * every stock FORE byte is unchanged except the edits recorded in the bake receipt;
  * GPU bytes [0, stock length) identical to the stock .zig;
  * each target model's node 0 shape: skinned 32-byte declaration (flags 531), palette <= 66, inverse binds equal
    inverse(bind FK) of the palette nodes, weights sum 255, palette indices < palette size, indices < vertex count;
  * the material sampler resolves to the new texture entry whose DDS decodes to the atlas size;
  * the 111 driver animations and 5 assistant animations still decode against the unchanged rig.
"""
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import pcanim
import pcbody
from softrender import Renderer
from zif import Zif

MODULE = Path(__file__).resolve().parents[1]
STOCK = (MODULE / "work/pc-avatar/Racers_Avatar.zif", MODULE / "work/pc-avatar/Racers_Avatar.zig")


def chunks(d):
    out, p = [], 4
    while p < len(d):
        tag = d[p:p + 4]
        size, used, mk = struct.unpack_from("<3I", d, p + 4)
        out.append((tag, p, size, used))
        p += size
    return out


def read_skinned(z, f, row):
    mesh = pcbody.read_mesh(z, row)
    pal, ib = pcbody.palette(z, f, row)
    return mesh, pal, ib


def skin(P, N, Wt, Bi, pal, ib, Wcur):
    M = np.array([ib[k] @ Wcur[pal[k]] for k in range(len(pal))])  # row-vector convention
    hp = np.concatenate([P, np.ones((len(P), 1))], 1)
    out = np.zeros_like(P)
    nout = np.zeros_like(N)
    for j in range(4):
        w = Wt[:, j:j + 1]
        Mj = M[Bi[:, j]]
        out += w * np.einsum("ni,nij->nj", hp, Mj)[:, :3]
        nout += w * np.einsum("ni,nij->nj", N, Mj[:, :3, :3])
    return out, nout / np.maximum(np.linalg.norm(nout, axis=1, keepdims=True), 1e-12)


def render_pose(P, N, U, T, tex, out, title, size=(300, 400), center=None, dist=None, yaws=(0, 45, 90, 180, 270)):
    """Camera orbits the posed mesh's own bounding-box centre (a fixed centre misframed translated poses)."""
    lo, hi = P[np.unique(T)].min(0), P[np.unique(T)].max(0)
    if center is None:
        center = tuple((lo + hi) / 2)
    if dist is None:
        dist = max(4.0, 2.4 * float(np.linalg.norm(hi - lo)))
    imgs = []
    h, w = tex.shape[:2]
    for yaw in yaws:
        a = np.radians(yaw)
        eye = (center[0] + dist * np.sin(a), center[1] + 0.15 * dist, center[2] + dist * np.cos(a))
        r = Renderer(*size, ss=2)
        r.set_camera(eye, center, fovy=32)

        def shade(n, t):
            x = np.clip((t[:, 0] * w).astype(int), 0, w - 1)
            y = np.clip(((1.0 - t[:, 1]) * h).astype(int), 0, h - 1)   # native rule: sampled V = 1 - stored V
            return tex[y, x, :3].astype(float), np.ones(len(n))
        tri = np.stack([T, T, T], -1)
        r.draw(P, N, U, tri, shade, cull="back", front_ccw=True, light_dir=(np.sin(a) + 0.3, 0.8, np.cos(a) + 0.3))
        imgs.append(r.image())
    W = Image.new("RGB", (size[0] * len(imgs), size[1] + 20), (40, 44, 52))
    d = ImageDraw.Draw(W)
    for k, im in enumerate(imgs):
        W.paste(im, (k * size[0], 20))
        d.text((k * size[0] + 4, 4), f"{title} yaw {yaws[k]}", fill="white")
    W.save(out)


def main():
    bake_dir = MODULE / "build/bake"
    rec = json.loads((bake_dir / "BAKE-RECEIPT.json").read_text())
    zif_p, zig_p = bake_dir / "Avatar.zif", bake_dir / "Avatar.zig"
    sz, sg = STOCK[0].read_bytes(), STOCK[1].read_bytes()
    nz, ng = zif_p.read_bytes(), zig_p.read_bytes()
    res = dict(checks={}, models=[])
    c = res["checks"]
    c["prefix_ok"] = struct.unpack_from("<I", nz)[0] == len(nz) - 4
    sc, nc = chunks(sz), chunks(nz)
    c["chunk_tags_same"] = [x[0] for x in sc] == [x[0] for x in nc]
    c["non_fore_chunks_identical"] = all(sz[a[1]:a[1] + a[2]] == nz[b[1]:b[1] + b[2]] for a, b in zip(sc[1:], nc[1:]))
    fore_stock_end = sc[0][1] + 16 + sc[0][3]
    diff = [i for i in range(16, fore_stock_end) if sz[i] != nz[i]]
    base = 0x34
    allowed = set()
    for e in rec["edits"]:
        allowed.update(range(base + e["forest_offset"], base + e["forest_offset"] + 4))
    c["fore_header_size_used_changed_only"] = all(i in range(8, 16) for i in diff if i < 16)
    bad = [i for i in diff if i >= 16 and i not in allowed]
    c["stock_fore_bytes_unchanged_except_edits"] = not bad
    c["unexpected_changed_bytes"] = bad[:20]
    c["zig_prefix_identical"] = ng[4:4 + len(sg) - 4] == sg[4:]
    z = Zif(zif_p, zig_p)
    f = z.forests[0]
    names = None
    for mrec in rec["models"]:
        m = next(mm for mm in f["models"] if mm["root"] == mrec["model"])
        rows = [r for r in z.meshes(f, m) if r["node"] == 0]
        assert len(rows) == 1
        row = rows[0]
        mesh, pal, ib = read_skinned(z, f, row)
        bind = pcanim.bind_locals(z, f, m)
        Wb = pcanim.world(m, *bind)
        names = [n["name"] for n in m["nodes"]]
        out = dict(model=m["root"], preset=mrec["preset"], decl=row["declaration"], vd_flags=row["vertex_descriptor"][9], stride=row["vertex_descriptor"][3],
                   vertices=int(mesh["count"]), triangles=int(len(mesh["tris"])), palette=len(pal))
        out["decl_skinned_32"] = [list(x) for x in row["declaration"]] == [[0, 0, 2, 0, 0, 0], [0, 12, 16, 0, 3, 0], [0, 20, 15, 0, 5, 0], [0, 24, 8, 0, 1, 0], [0, 28, 5, 0, 2, 0]]
        out["palette_le_66"] = len(pal) <= 66
        out["inverse_binds_match_bind_fk"] = float(max(np.abs(np.linalg.inv(ib[k]) - Wb[pal[k]]).max() for k in range(len(pal))))
        wraw = np.rint(mesh["w"] * 255).astype(int)
        bi = mesh["idx"].astype(int)
        out["weights_sum_255"] = bool((wraw.sum(1) == 255).all())
        out["palette_index_in_range"] = bool((bi[wraw > 0] < len(pal)).all())
        out["indices_in_range"] = bool(mesh["tris"].max() < mesh["count"])
        out["material"] = row["material_name"]
        smp = row["samplers"][0]
        tex = next(t for t in f["textures"] if t["pos"] - f["base"] == smp[1])
        im = Image.open(io.BytesIO(z.g[tex["header"][2]:]))
        im.load()
        out["texture"] = dict(name=tex["name"], size=im.size, mode=im.mode)
        atlas = np.array(im.convert("RGBA"))
        # animations still decode
        ok, bad_anims = 0, []
        offs = sorted(set(a["offset"] for a in m["anims"]))
        ends = {o: (offs[i + 1] if i + 1 < len(offs) else m["header"][22]) for i, o in enumerate(offs)}
        decoded = {}
        for a in m["anims"]:
            try:
                decoded[a["name"]] = pcanim.decode(z, f, m, a, ends[a["offset"]])
                ok += 1
            except Exception as e:  # noqa: BLE001 - record and continue
                bad_anims.append((a["name"], str(e)[:80]))
        out["animations_decoded"] = ok
        out["animations_failed"] = bad_anims[:5]
        # renders from the written PC file
        P, N, U, T = mesh["pos"], mesh["nrm"][:, :3], mesh["uv"], mesh["tris"]
        rd = MODULE / "renders/pc"
        rd.mkdir(parents=True, exist_ok=True)
        render_pose(P, N, U, T, atlas, rd / f"{m['root']}-bind.png", f"{m['root']} PC bind")
        poses = []
        for an, frac in (("Speed1", 0.0), ("DriftL", 0.5), ("Taunt1", 0.5), ("WIN", 0.5), ("HardHitL", 0.5), ("RevengeOn", 0.5)):
            if an not in decoded:
                continue
            T_, Q_, S_ = pcanim.pose(m, bind, decoded[an], frac)
            Wc = pcanim.world(m, T_, Q_, S_)
            Ps, Ns = skin(P, N, mesh["w"], bi, pal, ib, Wc)
            fn = rd / f"{m['root']}-pose-{an}.png"
            render_pose(Ps, Ns, U, T, atlas, fn, f"{m['root']} {an}@{frac}", yaws=(0, 60, 180))
            poses.append(dict(anim=an, key_fraction=frac, render=str(fn.relative_to(MODULE))))
        out["pose_renders"] = poses
        res["models"].append(out)
    (MODULE / "research" / "BAKE-VERIFY.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res["checks"], indent=1))
    for o in res["models"]:
        print({k: v for k, v in o.items() if k not in ("decl", "pose_renders")})


if __name__ == "__main__":
    main()
