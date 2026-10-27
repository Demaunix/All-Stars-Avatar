"""Original decoder for the supplied Xbox 360 AvatarAssetPack model blocks (already LZX-expanded by the coordinator).

Format facts were established in this module from the bytes themselves (research/AVATAR-MODEL-FORMAT.md) and
cross-checked against the vertex declarations used by the game's own avatar renderer (Xbox image 0x82694D14..,
table 0x82694DD4, consumer 0x8260BF20). No third-party decoder was used.

Per mesh (all little-endian, bit fields LSB-first):
  u32 vertex_count, f32 step (=7.071068e-6), f32 min_x, min_y, min_z,
  bitstream: 3 x u6 position widths, then 4 x (u32 base, u32 bits) for NORMAL, BLENDWEIGHT, BLENDINDICES, COLOR,
  vertex records: pos(wx,wy,wz unsigned) | normal | weights | indices | color | texcoord halves (16 bits each),
  each packed element decodes as (base + stored) modulo 2**32,
  pad to byte, u32 index_count, u16 0, u16 index_width, index_count x index_width-bit indices, pad to byte.
Every structure is bounds-checked; the decoder refuses rather than guesses when the following index header does not
validate.
"""
import struct

import numpy as np

STEP = struct.unpack("<f", bytes.fromhex("0244ed36"))[0]


class FormatError(Exception):
    pass


class Bits:
    def __init__(self, data):
        self.data = data
        self.b = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little").astype(np.int64)

    def val(self, pos, width):
        if width == 0:
            return 0
        if pos < 0 or pos + width > len(self.b):
            raise FormatError(f"bit read out of range {pos}+{width}")
        return int((self.b[pos:pos + width] << np.arange(width)).sum())

    def column(self, starts, off, width):
        if width == 0:
            return np.zeros(len(starts), np.int64)
        idx = starts[:, None] + off + np.arange(width)[None, :]
        if idx.max() >= len(self.b):
            raise FormatError("column out of range")
        return (self.b[idx] << np.arange(width)).sum(1)


def hend3n(v):
    """Xbox D3DDECLTYPE_HEND3N: signed 11:11:10 normalized (x low bits)."""
    v = np.asarray(v, np.int64) & 0xFFFFFFFF
    x, y, z = v & 0x7FF, (v >> 11) & 0x7FF, (v >> 22) & 0x3FF
    x = np.where(x >= 1024, x - 2048, x) / 1023.0
    y = np.where(y >= 1024, y - 2048, y) / 1023.0
    z = np.where(z >= 512, z - 1024, z) / 511.0
    return np.stack([x, y, z], 1)


def bytes4(v):
    v = np.asarray(v, np.int64) & 0xFFFFFFFF
    return np.stack([(v >> s) & 0xFF for s in (0, 8, 16, 24)], 1)


def _mesh_at(bits, q, nverts_hint=None):
    d = bits.data
    nv = struct.unpack_from("<I", d, q)[0]
    step, mnx, mny, mnz = struct.unpack_from("<4f", d, q + 4)
    if abs(step - STEP) > 1e-12:
        raise FormatError(f"unexpected step {step!r} at {q:#x}")
    if not 0 < nv < 65536:
        raise FormatError(f"bad vertex count {nv}")
    s = (q + 20) * 8
    widths = [bits.val(s + 6 * k, 6) for k in range(3)]
    elem = [bits.val(s + 18 + 32 * k, 32) for k in range(8)]
    (nbase, nbits, wbase, wbits, ibase, ibits, cbase, cbits) = elem
    for w in widths + [nbits, wbits, ibits, cbits]:
        if w > 32:
            raise FormatError(f"field width {w} > 32")
    vstart = s + 274
    fixed = sum(widths) + nbits + wbits + ibits + cbits
    iw_expected = max(1, (nv - 1).bit_length())
    chosen = None
    for tex_halves in (2, 4, 6, 8, 10, 12):
        rec = fixed + 16 * tex_halves
        end = vstart + rec * nv
        ib = (end + 7) // 8
        if ib + 8 > len(d):
            continue
        cnt, zero, iw = struct.unpack_from("<IHH", d, ib)
        if zero == 0 and iw == iw_expected and cnt % 3 == 0 and 0 < cnt and (ib + 8) * 8 + cnt * iw <= len(d) * 8:
            if chosen is not None:
                raise FormatError("ambiguous texcoord width")
            chosen = (tex_halves, rec, ib, cnt, iw)
    if chosen is None:
        raise FormatError(f"no texcoord width validates the index header at mesh {q:#x}")
    tex_halves, rec, ib, cnt, iw = chosen
    starts = vstart + rec * np.arange(nv)
    o = 0
    q_xyz = []
    for w in widths:
        q_xyz.append(bits.column(starts, o, w))
        o += w
    raw_n = bits.column(starts, o, nbits); o += nbits
    raw_w = bits.column(starts, o, wbits); o += wbits
    raw_i = bits.column(starts, o, ibits); o += ibits
    raw_c = bits.column(starts, o, cbits); o += cbits
    halves = np.stack([bits.column(starts, o + 16 * k, 16) for k in range(tex_halves)], 1).astype(np.uint16)
    o += 16 * tex_halves
    assert o == rec
    istart = (ib + 8) * 8
    idx = bits.column(istart + iw * np.arange(cnt), 0, iw)
    if idx.max() >= nv:
        raise FormatError("index out of range")
    iend = (istart + cnt * iw + 7) // 8
    Q = np.stack(q_xyz, 1).astype(np.float64)
    mesh = dict(
        offset=q, vertex_count=nv, step=step, min=(mnx, mny, mnz), widths=widths,
        element_base=dict(normal=nbase, weights=wbase, indices=ibase, color=cbase),
        element_bits=dict(normal=nbits, weights=wbits, indices=ibits, color=cbits),
        record_bits=rec, tex_halves=tex_halves, vertex_bit_start=vstart, index_header=ib, index_count=cnt, index_width=iw,
        end=iend,
        quantized=Q,
        normal_raw=(raw_n + nbase) & 0xFFFFFFFF,
        weights=bytes4(raw_w + wbase),
        bones=bytes4(raw_i + ibase),
        color=bytes4(raw_c + cbase),
        texcoords=halves.view(np.float16).astype(np.float64),
        triangles=idx.reshape(-1, 3),
    )
    mesh["normal"] = hend3n(mesh["normal_raw"])
    return mesh


def positions(mesh, scale=2.0):
    """Dequantize: min + q * step * scale (scale=2 established by T-pose symmetry; see research notes).
    A composed copy may carry `pos_override` (documented per-vertex frame correction, avatar_compose)."""
    if "pos_override" in mesh:
        return mesh["pos_override"]
    return np.asarray(mesh["min"])[None, :] + mesh["quantized"] * mesh["step"] * scale


def find_meshes(data):
    """Mesh blocks are located by their exact (vertex_count, step) prefix and validated by full decode."""
    sig = bytes.fromhex("0244ed36")
    out = []
    p = data.find(sig)
    while p >= 4:
        out.append(p - 4)
        p = data.find(sig, p + 1)
    return out


def decode(data):
    if len(data) < 0x58:
        return dict(header=list(struct.unpack_from(f"<{len(data) // 4}I", data, 0)), meshes=[], empty=True)
    hdr = list(struct.unpack_from("<22I", data, 0))
    bits = Bits(data)
    meshes = []
    pos = 0
    for q in find_meshes(data):
        if q < pos:
            continue
        m = _mesh_at(bits, q)
        meshes.append(m)
        pos = m["end"]
    if len(meshes) != hdr[5]:
        raise FormatError(f"mesh count {len(meshes)} != header {hdr[5]}")
    total_idx = sum(m["index_count"] for m in meshes)
    if total_idx * 2 != hdr[4]:
        raise FormatError(f"index bytes {total_idx * 2} != header {hdr[4]}")
    return dict(header=hdr, meshes=meshes, empty=False, tail_offset=meshes[-1]["end"] if meshes else None)
