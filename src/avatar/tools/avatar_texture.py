"""Original decoder for AvatarAssetPack textures (standalone texture blocks and texture records in model tails).

Record layout (little-endian): u32 format word (0x1A2000xx; low 6 bits = Xenos format: 18 DXT1, 20 DXT4/5; bits 6-7 =
endian, 1 = 8-in-16 swap; bit 8 set on some records, meaning unknown - NOT tiling), u32 width, u32 height,
u32 total_bytes, u32 span_bytes, u32 spans, u32 pitch_info, u32 pitch, u8 extra, then the stored data. In model tails
each record is preceded by u32 gpu_offset, u32 gpu_size and the stored data is compact, GPU-aligned or absent
(model_tail). Validation: total == spans * span_bytes, span_bytes >= compact size, exact delimiting of the tail.

The 8-in-16 swap and linear decode were established by the coordinator probe (TEXTURE-PROBE-RECEIPT.json); every
supplied texture, including those with bit 8 set, decodes coherently as LINEAR data, so all callers pass tiled=False.
`untile` (the publicly documented XGAddress2DTiledX/Y formula) is kept only for the control in tests/test_decoder.py
(bit-8 textures are smoother decoded linearly than untiled); it is not used to build any output.
"""
import io
import struct

import numpy as np
from PIL import Image

FMT_BPB = {18: 8, 20: 16}   # bytes per 4x4 block
FMT_PIL = {18: 1, 20: 3}    # Pillow bcn decoder index


class TextureError(Exception):
    pass


def parse_record(d, p, stored_span=None):
    """stored_span None: standalone rule (span field == compact size). Model tails store compact spans while the
    span/total fields give GPU-aligned sizes; the caller passes the compact size it verified."""
    fmt, w, h, total, span, spans, pinfo, pitch = struct.unpack_from("<8I", d, p)
    if (fmt & 0xFFFFFE00) != 0x1A200000:
        raise TextureError(f"bad format word {fmt:#x} at {p:#x}")
    f6 = fmt & 0x3F
    if f6 not in FMT_BPB:
        raise TextureError(f"unsupported format {f6}")
    bpb = FMT_BPB[f6]
    need = (w // 4) * (h // 4) * bpb
    if total != span * spans or span < need:
        raise TextureError(f"size mismatch w{w} h{h} span{span} need{need} total{total} spans{spans}")
    if stored_span is None:
        if span != need:
            raise TextureError(f"standalone span {span} != compact {need}")
        stored_span = need
    data_at = p + 33
    if data_at + stored_span * spans > len(d):
        raise TextureError("data out of range")
    return dict(offset=p, format_word=fmt, fmt=f6, endian=(fmt >> 6) & 3, tiled=bool(fmt & 0x100), width=w, height=h,
                total=total, span_bytes=span, stored_span=stored_span, spans=spans, pitch_info=pinfo, pitch=pitch, extra=d[p + 32],
                data_at=data_at, end=data_at + stored_span * spans)


def standalone(d):
    r = parse_record(d, 0)
    if r["end"] != len(d):
        raise TextureError(f"standalone texture consumes {r['end']} of {len(d)}")
    return r


def model_tail(d, tail):
    """Texture records after the meshes; returns records and the unconsumed trailer bytes."""
    def is_hdr(q):
        return q + 12 <= len(d) and (struct.unpack_from("<I", d, q + 8)[0] & 0xFFFFFE00) == 0x1A200000

    TERM = bytes.fromhex("01000000ffffffff00000000")  # observed last 12 bytes of every model block
    recs = []
    p = tail
    while is_hdr(p):
        gpu_off, gpu_size = struct.unpack_from("<2I", d, p)
        fmt, w, h, total, span, spans = struct.unpack_from("<6I", d, p + 8)
        need = (w // 4) * (h // 4) * FMT_BPB.get(fmt & 0x3F, 0)
        nxt = p + 8 + 33
        options = [("placeholder", 0), ("compact", need), ("aligned", span)]
        # 1) exact delimiting: another record header follows (or the block ends) right after the stored data
        cands = [(n, sz) for n, sz in options if nxt + sz * spans == len(d) or is_hdr(nxt + sz * spans)]
        if not cands:
            # 2) last record: the remaining bytes must be a trailer that ends with the observed terminator
            #    (head/shirt seam tables). Stored data is preferred (compact, then aligned); header-only only when the data
            #    cannot fit (observed: Male/Female Gold Stud Earrings).
            last = [("compact", need), ("aligned", span), ("placeholder", 0)]
            cands = [(n, sz) for n, sz in last if nxt + sz * spans <= len(d) and d[nxt + sz * spans:].endswith(TERM)]
        if not cands:
            raise TextureError(f"cannot delimit texture record at {p:#x}")
        if recs and recs[-1]["placeholder"] and ("placeholder", 0) in cands:
            cands = [("placeholder", 0)]                     # head-style runs of header-only slots
        kind, sz = cands[0]
        if kind == "placeholder":
            r = parse_record(d, p + 8, stored_span=0)
            r.update(data_at=None, end=nxt, stored_span=need)
        else:
            r = parse_record(d, p + 8, stored_span=sz)
        r.update(gpu_offset=gpu_off, gpu_size=gpu_size, placeholder=kind == "placeholder", storage=kind)
        recs.append(r)
        p = r["end"]
    return recs, d[p:]


def _swap16(b):
    a = np.frombuffer(b, np.uint8).reshape(-1, 2)[:, ::-1]
    return a.tobytes()


def _xg_tiled_xy(offset, width_blocks, bpb):
    """XGAddress2DTiledX/Y for block-compressed data (texel = one 4x4 block of `bpb` bytes)."""
    aligned = (width_blocks + 31) & ~31
    log_bpp = (bpb >> 2) + ((bpb >> 1) >> (bpb >> 2))
    ob = offset << log_bpp
    ot = ((ob & ~4095) >> 3) + ((ob & 1792) >> 2) + (ob & 63)
    om = ot >> (7 + log_bpp)
    macro_x = (om % (aligned >> 5)) << 2
    tile_x = (((ot >> (5 + log_bpp)) & 2) + (ob >> 6)) & 3
    x = ((macro_x + tile_x) << 3) + (((((ot >> 1) & ~15) + (ot & 15)) & ((bpb << 3) - 1)) >> log_bpp)
    macro_y = (om // (aligned >> 5)) << 2
    tile_y = ((ot >> (6 + log_bpp)) & 1) + ((ob & 2048) >> 10)
    y = ((macro_y + tile_y) << 3) + ((((ot & (((bpb << 6) - 1) & ~31)) + ((ot & 15) << 1)) >> (3 + log_bpp)) & ~1) + ((ot & 16) >> 4)
    return x, y


def untile(blocks, w, h, bpb):
    bw, bh = w // 4, h // 4
    src = np.frombuffer(blocks, np.uint8).reshape(-1, bpb)
    out = np.zeros((bh, bw, bpb), np.uint8)
    aw = (bw + 31) & ~31
    ah = (bh + 31) & ~31
    n = min(len(src), aw * ah)
    for i in range(n):
        x, y = _xg_tiled_xy(i, bw, bpb)
        if x < bw and y < bh:
            out[y, x] = src[i]
    return out.reshape(-1).tobytes()


def decode_span(d, rec, span=0, tiled=None):
    if rec.get("data_at") is None:
        raise TextureError("placeholder slot has no data")
    a = rec["data_at"] + span * rec["stored_span"]
    raw = d[a:a + rec["stored_span"]]
    if rec["endian"] == 1:
        raw = _swap16(raw)
    elif rec["endian"] != 0:
        raise TextureError(f"endian mode {rec['endian']} not handled")
    if tiled is None:
        tiled = rec["tiled"]
    if tiled:
        raw = untile(raw, rec["width"], rec["height"], FMT_BPB[rec["fmt"]])
    im = Image.frombytes("RGBA", (rec["width"], rec["height"]), raw, "bcn", FMT_PIL[rec["fmt"]])
    return np.array(im)
