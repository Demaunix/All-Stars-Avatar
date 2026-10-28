"""Original decoder for the supplied AvatarAssetPack hiding templates (TOC type 0x1000000, raw shape-override blocks)
and the garment -> template link in the TOC.

Layout (little-endian; established in this module from the bytes, research/HIDING-TEMPLATES.md):
  u32 n_hidden, u32 body_index_bytes (MaleBody 13584 / FemaleBody 14724), u32 2, u32 gender word (0x10000 male,
  0x20001 female), u32 x2 pack id, u32 0, u32 12 (bits per entry; 0 when n_hidden == 0: bands, heels),
  n_hidden x u12 (LSB-first) strictly ascending body TRIANGLE indices to hide, pad to byte;
  u32 n_override, u32 body vertex-buffer bytes (40-byte stride, 128-aligned), u32 2, u32 gender word, u32 x2 pack id,
  u32 0, u32 offset_bits (15/16), f32 step (=7.071068e-6), f32 min[3], then the mesh bitstream header of avatar_model
  (3 x u6 position widths, 4 x (u32 base, u32 bits) for NORMAL, BLENDWEIGHT, BLENDINDICES, COLOR) and n_override records
  `body vertex byte offset (offset_bits) | position | normal | weights | indices | colour` replacing those body vertices
  (texcoords are kept).
Every template block in the pack is consumed exactly by this layout (tests/test_hiding.py).

Link: TOC record word 39 == 0x1000000 marks a component with a hiding template; word 40 = (template TOC index << 16) |
gender (1 male, 2 female). The console's merge rule for several templates is not in the supplied data (see compose).
"""
import json
import struct
from functools import lru_cache

import numpy as np

import avatar_model as AM
from catalog import INPUT, TOC, catalog, raw_block

HIDING_TYPE = 0x1000000
VERTEX_STRIDE = 40


class HidingError(Exception):
    pass


@lru_cache(None)
def _toc_words():
    recs = json.loads((INPUT / "AVATAR-TOC-STRUCTURE-REFERENCE.json").read_text())["records"]
    data = TOC.read_bytes()
    return {r["index"]: struct.unpack_from(">62I", data, int(r["record_offset"], 16)) for r in recs}


def template_for(index):
    """TOC index of the hiding template linked from component `index`, or None."""
    w = _toc_words()[index]
    if w[0] == HIDING_TYPE or w[39] != 0x1000000:
        return None
    t = w[40] >> 16
    tw = _toc_words().get(t)
    if tw is None or tw[0] != HIDING_TYPE:
        raise HidingError(f"component {index} links to {t}, which is not a hiding template")
    return t


def decode(data, body_mesh=None):
    n, ibytes, two, gender = struct.unpack_from("<4I", data, 0)
    bits_per = struct.unpack_from("<I", data, 28)[0]
    if two != 2 or bits_per != (12 if n else 0) or gender not in (0x10000, 0x20001):   # n == 0: override-only (bands, heels)
        raise HidingError("bad hidden-triangle header")
    b = AM.Bits(data)
    tris = np.array([b.val(32 * 8 + 12 * k, 12) for k in range(n)], np.int64)
    if n and not (np.diff(tris) > 0).all():
        raise HidingError("hidden triangle list not strictly ascending")
    p = 32 + (n * 12 + 7) // 8
    nv, vbytes, two2, gender2 = struct.unpack_from("<4I", data, p)
    obits = struct.unpack_from("<I", data, p + 28)[0]
    step, mx, my, mz = struct.unpack_from("<4f", data, p + 32)
    if two2 != 2 or gender2 != gender or abs(step - AM.STEP) > 1e-12 or not 0 < obits <= 32:
        raise HidingError("bad override header")
    s = (p + 48) * 8
    widths = [b.val(s + 6 * k, 6) for k in range(3)]
    nb, nbits, wb, wbits, ib, ibits, cb, cbits = [b.val(s + 18 + 32 * k, 32) for k in range(8)]
    rec = obits + sum(widths) + nbits + wbits + ibits + cbits
    vstart = s + 274
    if (vstart + rec * nv + 7) // 8 != len(data):
        raise HidingError(f"override records do not consume the block ({(vstart + rec * nv + 7) // 8} != {len(data)})")
    st = vstart + rec * np.arange(nv)
    off = b.column(st, 0, obits)
    o = obits
    Q = np.stack([b.column(st, o + sum(widths[:k]), widths[k]) for k in range(3)], 1).astype(np.float64)
    o += sum(widths)
    normal_raw = (b.column(st, o, nbits) + nb) & 0xFFFFFFFF; o += nbits
    weights = AM.bytes4(b.column(st, o, wbits) + wb); o += wbits
    bones = AM.bytes4(b.column(st, o, ibits) + ib); o += ibits
    color = AM.bytes4(b.column(st, o, cbits) + cb)
    if (off % VERTEX_STRIDE).any():
        raise HidingError("override offset not on a vertex boundary")
    out = dict(hidden_triangles=tris, body_index_bytes=ibytes, body_vertex_bytes=vbytes, gender=gender,
               vertices=off // VERTEX_STRIDE, positions=np.array([mx, my, mz]) + Q * step * AM.K, quantized=Q, min=(mx, my, mz),
               normal_raw=normal_raw, normal=AM.hend3n(normal_raw), weights=weights, bones=bones, color=color)
    if (nv and ((weights.sum(1) != 255).any() or bones.max() >= 71)):
        raise HidingError("override skin out of range")
    if body_mesh is not None:
        check_body(out, body_mesh)
    return out


def check_body(t, body_mesh):
    nverts, ntris = len(body_mesh["quantized"]), len(body_mesh["triangles"])
    if t["body_index_bytes"] != body_mesh["index_count"] * 2:
        raise HidingError("template is for another body (index bytes)")
    if t["body_vertex_bytes"] != (nverts * VERTEX_STRIDE + 127) // 128 * 128:
        raise HidingError("template is for another body (vertex bytes)")
    if len(t["hidden_triangles"]) and t["hidden_triangles"].max() >= ntris:
        raise HidingError("hidden triangle out of range")
    if len(t["vertices"]) and t["vertices"].max() >= nverts:
        raise HidingError("override vertex out of range")


def load(index, body_mesh=None):
    return decode(raw_block(index), body_mesh)


def all_templates():
    return [c for c in catalog() if c["type"] == HIDING_TYPE]
