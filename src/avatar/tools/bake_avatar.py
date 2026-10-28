"""Bake composed Xbox avatar presets into the PC `Avatar` racer models (offline, writes only build/).

Per target model (Avatar driver and Avatar_Assistant_0..3 in Racers/Avatar.zif):
  * geometry: avatar_compose parts (decoded supplied assets), retargeted to the PC bind space with the fitted
    diagonal map (research/SPACE-FIT.json): p_pc = s*p + t, n_pc = normalize(n / s);
  * skin: Xbox bone index -> PC node via the original Xbox bone-name table (bone_map), UBYTE4N weights preserved,
    palette = PC nodes with non-zero weight (<= 66, the stock maximum), inverse binds = inverse(bind FK world);
  * surface: one 1024x1024 DXT1 atlas (full mip chain, stock DDS header layout) of the baked part textures;
  * structures (appended inside FORE, existing bytes unchanged except the edits listed in the receipt): texture entry,
    texture table (+1), material cloned word-for-word from the stock eggman.Forest `EggHands` material (skinned,
    32-byte declaration, one sampler) with its sampler re-pointed, vertex descriptor + declaration (flags 531),
    mesh header (words 24..33/39..41 copied from the stock EggHands mesh), mesh list, palette, inverse binds, shape;
    node 0 of the model is re-pointed from the 4-vertex SHONKY_DEFAULT dummy to the new shape.
"""
import argparse
import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import avatar_compose as AC
import avatar_model as AM
import avatar_surface as AS
import bone_map
import pcanim
from forest_append import ForestAppender
from zif import Zif

MODULE = Path(__file__).resolve().parents[1]
ATLAS = 1024
CELLS = {"head0": (0, 0, 512), "head1": (512, 0, 512), "shirt": (0, 512, 256), "pants": (256, 512, 256), "shoes": (512, 512, 256),
         "body": (768, 512, 64), "hair": (832, 512, 64)}
MAX_PALETTE = 66
SKINNED_DECL = [(0, 0, 2, 0, 0, 0), (0, 12, 16, 0, 3, 0), (0, 20, 15, 0, 5, 0), (0, 24, 8, 0, 1, 0), (0, 28, 5, 0, 2, 0)]
TEMPLATE = dict(forest="eggman.Forest", model="Eggman", material="EggHands")
TARGETS = {"Avatar": "male_guest_1", "Avatar_Assistant_0": "male_guest_2", "Avatar_Assistant_1": "female_guest_1",
           "Avatar_Assistant_2": "male_guest_3", "Avatar_Assistant_3": "female_guest_2"}


class BakeError(Exception):
    pass


def sha(b):
    return hashlib.sha256(b).hexdigest()


SPACE_FIT_KEY = {"rp6": "uniform_rp6", "c253": "uniform_garment_frame"}   # research/SPACE-FIT.json key per decode profile


def space_fit():
    """Uniform PC-rig fit for the active decode profile (rp6: tools/fit_space_rp6.py, body + garment seams in one frame;
    c253: tools/fit_space_garment.py, garment-frame seams); returned as per-axis arrays for the shared code path."""
    d = json.loads((MODULE / "research" / "SPACE-FIT.json").read_text())[SPACE_FIT_KEY[AM.DECODE_NAME]]
    return np.full(3, d["scale"]), np.array(d["translation"])


def template_material():
    """Stock EggHands material words + sub-arrays + sampler, and mesh header constants (read-only from Racers.xpac)."""
    import tempfile
    from survey_pc_palettes import members
    for r, d in members("Racers.xpac"):
        if d[4:8] != b"FORE":
            continue
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "t.zif"
            fp.write_bytes(d)
            z = Zif(fp)
            for f in z.forests:
                if f["name"] != TEMPLATE["forest"]:
                    continue
                b = f["base"]
                m = next(mm for mm in f["models"] if mm["root"] == TEMPLATE["model"])
                row = next(rw for rw in z.meshes(f, m) if rw["material_name"] == TEMPLATE["material"])
                h = row["material_header"]
                return dict(zif_sha256=sha(d), header=h, raw80=z.d[row["material"] + 80:row["material"] + 112],
                            h4=z.us(b + h[4], h[3]), arrays={j: z.us(b + h[11 + j], h[5 + j]) for j in range(6) if h[5 + j]},
                            sampler=z.us(b + z.u(b + h[30]), 5), mesh_header=row["mesh_header"], vd=row["vertex_descriptor"],
                            decl=[tuple(x) for x in row["declaration"]])
    raise BakeError("template material not found")


def dds_dxt1(rgba):
    """DXT1 DDS with a full box-filtered mip chain; header fields as in the stock racer textures."""
    h, w = rgba.shape[:2]
    levels, cur = [], rgba.astype(np.float64)
    while True:
        levels.append(np.clip(np.floor(cur + 0.5), 0, 255).astype(np.uint8))
        if cur.shape[0] == 1 and cur.shape[1] == 1:
            break
        hh, ww = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
        cur = cur.reshape(hh, cur.shape[0] // hh, ww, cur.shape[1] // ww, 4).mean((1, 3))
    body = bytearray()
    for lv in levels:
        img = lv
        if img.shape[0] < 4 or img.shape[1] < 4:
            img = np.pad(img, ((0, max(0, 4 - img.shape[0])), (0, max(0, 4 - img.shape[1])), (0, 0)), mode="edge")
        bio = io.BytesIO()
        Image.fromarray(img, "RGBA").save(bio, "DDS", pixel_format="DXT1")
        blk = bio.getvalue()[128:]
        need = max(1, (lv.shape[1] + 3) // 4) * max(1, (lv.shape[0] + 3) // 4) * 8
        body += blk[:need]
    hd = bytearray(128)
    hd[:4] = b"DDS "
    struct.pack_into("<7I", hd, 4, 124, 0xA1007, h, w, (w // 4) * (h // 4) * 8, 0, len(levels))
    struct.pack_into("<2I4s", hd, 76, 32, 4, b"DXT1")
    struct.pack_into("<2I", hd, 108, 0x401008, 0)
    return bytes(hd) + bytes(body), len(levels)


CELL_SLOTS = [(0, 0, 512), (512, 0, 512), (0, 512, 256), (256, 512, 256), (512, 512, 256), (768, 512, 256),
              (0, 768, 256), (256, 768, 256), (512, 768, 64), (576, 768, 64), (640, 768, 64), (704, 768, 64),
              # step 4 (2026-10-21): 128-cells for the small items (hats, glasses, gloves, wrist, rings, earrings) in the area
              # Codex's layout left free, plus four more 64-cells; the first twelve slots keep the C253 layout unchanged
              (768, 768, 128), (896, 768, 128), (768, 896, 128), (896, 896, 128), (512, 832, 128), (640, 832, 128),
              (512, 960, 64), (576, 960, 64), (640, 960, 64), (704, 960, 64)]
GUTTER = {512: 32, 256: 32, 128: 16, 64: 16}


def chart_cell_size(c):
    """512 for head charts, 64 for solids, 128 for a small item's chart (avatar_surface marks them), else 256."""
    if c["kind"] == "head":
        return 512
    if c["kind"] == "solid":
        return 64
    return 128 if c.get("small") else 256


def build_atlas(charts):
    """Each chart in its own power-of-two-aligned cell; content = cell - 2*gutter; gutter = edge extrusion of the
    content (C234 atlas-isolation lesson: bilinear taps and low mips must not reach a neighbouring chart)."""
    atlas = np.zeros((ATLAS, ATLAS, 4), np.uint8)
    atlas[..., 3] = 255
    free = {size: [s for s in CELL_SLOTS if s[2] == size] for size in (512, 256, 128, 64)}
    layout = {}
    for c in charts:
        size = chart_cell_size(c)
        if not free[size]:
            raise BakeError(f"no free {size} cell for {c['name']}")
        x0, y0, _ = free[size].pop(0)
        g = GUTTER[size]
        content = size - 2 * g
        img = Image.fromarray((np.clip(c["image"], 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").resize((content, content), Image.BOX)
        arr = np.asarray(img).copy()
        arr[..., 3] = 255
        arr = np.pad(arr, ((g, g), (g, g), (0, 0)), mode="edge")
        atlas[y0:y0 + size, x0:x0 + size] = arr
        layout[c["name"]] = dict(cell=(x0, y0, size), gutter=g, content=content)
    return atlas, layout


def chart_uv_image(c, cell):
    """Atlas coordinates in image convention (V = row fraction from the top of the atlas PNG/DDS)."""
    x0, y0, size = cell["cell"]
    g, content = cell["gutter"], cell["content"]
    uv = np.clip(c["uv"], 0, 1)
    return np.stack([(x0 + g + uv[:, 0] * content) / ATLAS, (y0 + g + uv[:, 1] * content) / ATLAS], 1)


def native_v(v):
    """Stored vertex V <-> image V. The original engine's generated vertex shader samples at (u, 1 - v)
    ("o.tc%d.xy = float2(inputData._tc%d.%s, 1.0f - inputData._tc%d.%s)"), so stock vertex streams hold bottom-origin V.
    The map is an involution: it converts either way."""
    return 1.0 - v


def chart_uv(c, cell):
    """UV as written to the vertex stream: image-convention atlas coordinates with V stored as 1 - v (computed in float64
    before the half-float rounding in pack())."""
    uv = chart_uv_image(c, cell)
    uv[:, 1] = native_v(uv[:, 1])
    return uv


def gather(parts, layout, mp, s, t):
    """Vertices duplicated per chart (a part vertex used by two charts gets two atlas UVs)."""
    P, N, U, W, B, T = [], [], [], [], [], []
    base = 0
    stats = []
    for p in parts:
        me = p["mesh"]
        pos = AM.positions(me)
        nrm = me["normal"]
        for c in p["charts"]:
            tri = c["tris"]
            used = np.unique(tri)
            remap = -np.ones(len(pos), np.int64)
            remap[used] = np.arange(len(used))
            uv = chart_uv(dict(c, uv=c["uv"][used]), layout[c["name"]])
            if (c["uv"][used] < -1e-6).any() or (c["uv"][used] > 1 + 1e-6).any():
                raise BakeError(f"{c['name']}: chart UV outside [0,1]")
            P.append(pos[used] * s + t)
            n = nrm[used] / s
            N.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
            U.append(uv)
            W.append(me["weights"][used])
            B.append(np.vectorize(lambda x: mp[x])(me["bones"][used]))
            T.append(remap[tri] + base)
            stats.append(dict(chart=c["name"], kind=c["kind"], source=p["src"], vertices=int(len(used)), triangles=int(len(tri)), cell=layout[c["name"]]))
            base += len(used)
    return (np.concatenate(P), np.concatenate(N), np.concatenate(U), np.concatenate(W).astype(np.int64),
            np.concatenate(B).astype(np.int64), np.concatenate(T).astype(np.int64), stats)


def compact(P, N, U, W, B, T):
    """Drop vertices referenced by no kept triangle (hidden body skin)."""
    used = np.unique(T)
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    return P[used], N[used], U[used], W[used], B[used], remap[T]


def pack(P, N, U, W, Bpal):
    out = bytearray(len(P) * 32)
    nh = np.concatenate([N, np.ones((len(N), 1))], 1).astype("<f2")
    uh = U.astype("<f2")
    for i in range(len(P)):
        q = i * 32
        struct.pack_into("<3f", out, q, *P[i])
        out[q + 12:q + 20] = nh[i].tobytes()
        out[q + 20:q + 24] = uh[i].tobytes()
        out[q + 24:q + 28] = bytes(int(x) for x in W[i])
        out[q + 28:q + 32] = bytes(int(x) for x in Bpal[i])
    return bytes(out)


def mesh_bounds(P):
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    ext = (hi - lo) / 2
    r = float(np.linalg.norm(P - c, axis=1).max())
    return c, ext, r


def bake_model(fa, z, f, m, parts, prov, tmpl, s, t, label):
    mp = bone_map.mapping()
    for p in parts:
        p["charts"] = AS.charts_for_part(p, p.get("record"))
    charts = [c for p in parts for c in p["charts"]]
    atlas, layout = build_atlas(charts)
    P, N, U, W, B, T, part_stats = gather(parts, layout, mp, s, t)
    P, N, U, W, B, T = compact(P, N, U, W, B, T)
    if len(P) > 0xFFFF:
        raise BakeError("too many vertices for 16-bit indices")
    if (W.sum(1) != 255).any():
        raise BakeError("weights do not sum to 255")
    active = W > 0
    pal = sorted(set(B[active].tolist()))
    if len(pal) > MAX_PALETTE:
        raise BakeError(f"palette {len(pal)} > {MAX_PALETTE}")
    pidx = {n: i for i, n in enumerate(pal)}
    Bpal = np.where(active, np.vectorize(lambda x: pidx.get(x, 0))(B), 0)
    bind = pcanim.bind_locals(z, f, m)
    Wb = pcanim.world(m, *bind)
    IB = [np.linalg.inv(Wb[n]) for n in pal]
    names = [nd["name"] for nd in m["nodes"]]
    # ---- GPU data
    vb = pack(P, N, U, W, Bpal)
    ib = T.reshape(-1).astype("<u2").tobytes()
    dds, mips = dds_dxt1(atlas)
    vo = fa.alloc_gpu(vb, f"{label} vertices")
    io_ = fa.alloc_gpu(ib, f"{label} indices")
    to = fa.alloc_gpu(dds, f"{label} atlas DDS")
    # ---- forest structures
    tname = fa.alloc(f"AVATAR_XBOX_{label.upper()}.TGA".encode() + b"\0", f"{label} texture name", 4)
    tex_entry = fa.alloc(struct.pack("<4I", tname, 0, to, 0), f"{label} texture entry")
    fh = fa.words(0, 9)
    old_table = fa.words(fh[3], fh[2])
    new_table = fa.alloc(struct.pack(f"<{fh[2] + 1}I", *old_table, tex_entry), f"{label} texture table (+1)")
    fa.put32(8, fh[2] + 1, f"{label} forest texture count")
    fa.put32(12, new_table, f"{label} forest texture table")
    h = list(tmpl["header"])
    h4 = fa.alloc(struct.pack(f"<{len(tmpl['h4'])}I", *tmpl["h4"]), f"{label} material params (EggHands copy)")
    h[4] = h4
    for j, arr in tmpl["arrays"].items():
        h[11 + j] = fa.alloc(struct.pack(f"<{len(arr)}I", *arr), f"{label} material array {j} (EggHands copy)")
    smp = list(tmpl["sampler"])
    smp[1] = tex_entry
    sp = fa.alloc(struct.pack("<5I", *smp), f"{label} sampler")
    h[30] = fa.alloc(struct.pack("<I", sp), f"{label} sampler list")
    h[31] = fa.alloc(f"AvatarXbox_{label}".encode() + b"\0", f"{label} material name", 4)
    mat_blob = bytearray(struct.pack("<32I", *h))
    mat_blob.extend(tmpl["raw80"])
    mat = fa.alloc(bytes(mat_blob), f"{label} material (EggHands clone)")
    decl = b"".join(struct.pack("<HH4B", *d) for d in SKINNED_DECL) + struct.pack("<HH4B", 255, 0, 17, 0, 0, 0)
    if [tuple(x) for x in tmpl["decl"]] != SKINNED_DECL:
        raise BakeError("template declaration differs from the skinned 32-byte layout")
    dp = fa.alloc(decl, f"{label} vertex declaration")
    vd = [dp, 0, 0, 32, len(P), vo, 0, 0, 0, tmpl["vd"][9], 0, 0, 0, 0]
    vdp = fa.alloc(struct.pack("<14I", *vd) + bytes(80 - 56), f"{label} vertex descriptor")
    c, ext, r = mesh_bounds(P)
    mh = list(tmpl["mesh_header"])
    fl = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
    axes = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    mh[0:12] = [fl(x) for x in axes]
    mh[12:16] = [fl(c[0]), fl(c[1]), fl(c[2]), fl(1.0)]
    mh[16:20] = [fl(ext[0]), fl(ext[1]), fl(ext[2]), fl(r)]
    mh[20:24] = [fl(c[0]), fl(c[1]), fl(c[2]), fl(r)]
    mh[34], mh[35], mh[36], mh[38] = mat, len(T) * 3, io_, vdp
    mesh = fa.alloc(struct.pack("<42I", *mh), f"{label} mesh header")
    mlist = fa.alloc(struct.pack("<I", mesh), f"{label} mesh list")
    palp = fa.alloc(struct.pack(f"<{len(pal)}I", *pal), f"{label} palette")
    ibp = fa.alloc(np.array(IB, "<f4").reshape(-1).tobytes(), f"{label} inverse binds")
    node0 = m["nodes"][0]
    old_shape = z.u(node0["pos"] + 20)
    old_sh = fa.words(old_shape, 10)
    sname = old_sh[9]
    sh = [fl(c[0]), fl(c[1]), fl(c[2]), fl(r * 1.25), 1, mlist, len(pal), palp, ibp, sname]
    shp = fa.alloc(struct.pack("<10I", *sh), f"{label} shape")
    fa.put32(node0["pos"] - f["base"] + 20, shp, f"{label} node 0 shape -> baked avatar shape (was dummy {old_shape})")
    return dict(model=m["root"], label=label, preset=prov["preset"], provenance=prov, parts=part_stats, vertices=len(P), triangles=len(T), palette=[names[n] for n in pal],
                palette_size=len(pal), atlas_png_sha256=sha(atlas.tobytes()), dds_bytes=len(dds), dds_mips=mips, gpu=dict(vertices=vo, indices=io_, texture=to),
                bounds=dict(center=c.tolist(), half_extent=ext.tolist(), radius=r), atlas=atlas, layout=layout)


def bake(zif_path, zig_path, targets, out_dir, receipt_name):
    zb, gb = Path(zif_path).read_bytes(), Path(zig_path).read_bytes()
    z = Zif(zif_path, zig_path)
    f = z.forests[0]
    fa = ForestAppender(zb, gb)
    s, t = space_fit()
    tmpl = template_material()
    rec = dict(inputs=dict(zif=str(zif_path), zif_sha256=sha(zb), zig_sha256=sha(gb), template=dict(TEMPLATE, zif_sha256=tmpl["zif_sha256"]),
                           space_fit=dict(scale=s.tolist(), translation=t.tolist())), models=[])
    out_dir.mkdir(parents=True, exist_ok=True)
    for root, preset in targets.items():
        m = next(mm for mm in f["models"] if mm["root"] == root)
        colours = None
        if isinstance(preset, tuple):       # (preset dict, avatar_colours.Colours): a coloured custom avatar (step 3/4)
            preset, colours = preset
        parts, prov = AC.compose(preset, colours=colours)
        r = bake_model(fa, z, f, m, parts, prov, tmpl, s, t, root)
        Image.fromarray(r.pop("atlas"), "RGBA").save(out_dir / f"{root}-atlas.png")
        rec["models"].append(r)
    zif_out, zig_out = fa.output()
    rec["edits"] = fa.edits
    rec["allocations"] = fa.allocs
    rec["outputs"] = dict(zif_sha256=sha(zif_out), zif_bytes=len(zif_out), zig_sha256=sha(zig_out), zig_bytes=len(zig_out))
    for ext, data in (("zif", zif_out), ("zig", zig_out)):
        stored = zlib.compress(data, 9)
        assert zlib.decompress(stored) == data
        rec["outputs"][f"{ext}_stored_sha256"] = sha(stored)
        rec["outputs"][f"{ext}_stored_bytes"] = len(stored)
    (out_dir / receipt_name).write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return zif_out, zig_out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/bake")
    a = ap.parse_args()
    out = MODULE / a.out
    zif_out, zig_out, rec = bake(MODULE / "work/pc-avatar/Racers_Avatar.zif", MODULE / "work/pc-avatar/Racers_Avatar.zig", TARGETS, out, "BAKE-RECEIPT.json")
    (out / "Avatar.zif").write_bytes(zif_out)
    (out / "Avatar.zig").write_bytes(zig_out)
    (out / "Avatar.zif.stored").write_bytes(zlib.compress(zif_out, 9))
    (out / "Avatar.zig.stored").write_bytes(zlib.compress(zig_out, 9))
    print(json.dumps(rec["outputs"], indent=1))
    for mm in rec["models"]:
        print(mm["model"], mm["preset"], "verts", mm["vertices"], "tris", mm["triangles"], "palette", mm["palette_size"])


if __name__ == "__main__":
    main()
