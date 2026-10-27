"""Read PC Mii body meshes (skinned, D3D9 declarations) and skeleton from the extracted Mii.zif/.zig (read-only).

Declaration types observed (D3DDECLTYPE): 2 FLOAT3, 16 FLOAT16_4, 15 FLOAT16_2, 8 UBYTE4N, 5 UBYTE4;
usages: 0 POSITION, 3 NORMAL, 5 TEXCOORD, 1 BLENDWEIGHT, 2 BLENDINDICES. Index buffers: u16 triangle lists.
"""
import io
import struct

import numpy as np
from PIL import Image

from zif import Zif

TYPES = {2: ("<3f", 12), 16: ("<4e", 8), 15: ("<2e", 4), 8: ("<4B", 4), 5: ("<4B", 4), 4: ("<4B", 4)}
USAGE = {0: "pos", 3: "nrm", 5: "uv", 1: "w", 2: "idx", 6: "tan", 10: "col"}  # 6/10 added for the stock AvatarCar (C235)


def read_mesh(z, row):
    vd = row["vertex_descriptor"]
    stride, n, vo = vd[3], vd[4], vd[5]
    g = z.g
    out = {}
    for stream, off, typ, method, usage, uidx in row["declaration"]:
        fmt, size = TYPES[typ]
        vals = np.array([struct.unpack_from(fmt, g, vo + i * stride + off) for i in range(n)], float)
        if typ == 8:
            vals = vals / 255.0
        out[USAGE[usage]] = vals
    mh = row["mesh_header"]
    count, ib = mh[35], mh[36]
    idx = np.array(struct.unpack_from(f"<{count}H", g, ib), np.int64).reshape(-1, 3)
    out["tris"] = idx
    out["stride"], out["count"], out["vertex_offset"], out["index_offset"], out["index_count"] = stride, n, vo, ib, count
    return out


def palette(z, forest, row):
    b = forest["base"]
    sh = row["shape_header"]
    if not sh[6]:
        return [], np.zeros((0, 4, 4))
    pal = z.us(b + sh[7], sh[6])
    ib = np.array(z.f(b + sh[8], sh[6] * 16)).reshape(sh[6], 4, 4)
    return pal, ib


def texture(z, forest, name):
    t = next(t for t in forest["textures"] if t["name"] == name)
    off = t["header"][2]
    im = Image.open(io.BytesIO(z.g[off:]))
    im.load()
    return np.array(im.convert("RGBA")), t


def load(path_zif, path_zig):
    return Zif(path_zif, path_zig)


LOD_NODE = "WiiMii_Mesh"  # type-16 LOD switch; its entries 0..2 are the skinned LOD1..3 shapes
RIGID_LODS = ("WiiMii_LOD_4", "WiiMii_LOD_5")  # type-40 nodes holding static far LODs


def lod_rows(rows, lod):
    """Rows of one skinned LOD level (0-based) of the WiiMii_Mesh switch node only."""
    return [r for r in rows if r["node_name"] == LOD_NODE and r["lod"] == lod]
