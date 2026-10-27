"""Append new structures to a PC Forest (.zif) forest and new GPU data to its .zig, without moving existing bytes.

Facts used (verified on the stock files in this module, research/FOREST-APPEND-NOTES.md):
- .zif = u32 (file length - 4) + chunks {tag, u32 size, u32 used, u32 0x44332211, payload}; sizes are multiples of 128.
- FORE payload: u32 forest count, then per forest {u32 name hash, u32 name offset, u32 base offset, u32 gpu base}
  relative to the payload start; every pointer inside a forest is a forest-relative offset (base + ptr). The FORE
  RELO chunk only covers the forest table words, so appending inside FORE after the existing data needs no new
  RELO entries.
- .zig = u32 length + GPU bytes; meshes/textures reference GPU bytes by offset (forest gpu base 0 here).
- Alignment: every forest structure the engine reads with 16-byte aligned SSE loads must be 16-aligned relative to the FORE payload
  start (absolute file position 4 mod 16), see ForestAppender.align_up. GPU data (`alloc_gpu`) has its own offset basis and is
  not affected.
"""
import struct


class AppendError(Exception):
    pass


class ForestAppender:
    def __init__(self, zif_bytes, zig_bytes):
        self.orig = bytes(zif_bytes)
        if struct.unpack_from("<I", zif_bytes, 0)[0] != len(zif_bytes) - 4:
            raise AppendError("zif length prefix mismatch")
        self.chunks = []
        p = 4
        while p < len(zif_bytes):
            tag = zif_bytes[p:p + 4]
            size, used, marker = struct.unpack_from("<3I", zif_bytes, p + 4)
            if marker != 0x44332211 or size % 128:
                raise AppendError(f"chunk {tag} at {p:#x} unexpected header")
            self.chunks.append([tag, p, size, used])
            p += size
        if p != len(zif_bytes) or self.chunks[0][0] != b"FORE":
            raise AppendError("unexpected chunk layout")
        fore = self.chunks[0]
        self.payload = 4 + 16
        nf, hs, no, bo, go = struct.unpack_from("<5I", zif_bytes, self.payload)
        if nf != 1 or go != 0:
            raise AppendError("expected exactly one forest with gpu base 0")
        self.base = self.payload + bo
        if bo % 16:
            raise AppendError("forest base is not 16-aligned relative to the FORE payload")
        self.fore_end_used = 4 + 16 + fore[3]  # absolute end of used payload
        self.data = bytearray(zif_bytes[:4 + fore[2]])  # FORE chunk incl. slack
        self.tail = bytes(zif_bytes[4 + fore[2]:])
        if struct.unpack_from("<I", zig_bytes, 0)[0] != len(zig_bytes) - 4:
            raise AppendError("zig length word mismatch")
        self.gpu = bytearray(zig_bytes[4:])
        self.orig_gpu_len = len(self.gpu)
        self.cursor = self.align_up(self.fore_end_used, 16)
        self.edits = []
        self.allocs = []

    # ---- forest data
    def rel(self, absolute):
        return absolute - self.base

    def align_up(self, absolute, align):
        """Smallest position >= `absolute` that is `align`-aligned relative to the FORE payload start (file position 20).

        The engine keeps the payload start 16-byte aligned in memory (stock shapes, meshes and inverse-bind arrays all sit at
        file position 4 mod 16), and its skinning code reads them with MOVAPS, so structure alignment is relative to the payload,
        not to the absolute file position (C252: aligning the absolute position left them at 12 mod 16 in memory).
        """
        return self.payload + ((absolute - self.payload + align - 1) & ~(align - 1))

    def alloc(self, data, why, align=16):
        self.cursor = self.align_up(self.cursor, align)
        at = self.cursor
        need = at + len(data)
        if need > len(self.data):
            self.data.extend(bytes(need - len(self.data)))
        self.data[at:at + len(data)] = data
        self.cursor = need
        self.allocs.append(dict(forest_offset=self.rel(at), bytes=len(data), why=why))
        return self.rel(at)

    def put32(self, rel_off, value, why):
        a = self.base + rel_off
        if a + 4 > self.fore_end_used and a < self.cursor:
            pass
        old = struct.unpack_from("<I", self.data, a)[0]
        struct.pack_into("<I", self.data, a, value)
        self.edits.append(dict(forest_offset=rel_off, old=old, new=value, why=why))

    def u32(self, rel_off):
        return struct.unpack_from("<I", self.data, self.base + rel_off)[0]

    def words(self, rel_off, n):
        return list(struct.unpack_from(f"<{n}I", self.data, self.base + rel_off))

    def raw(self, rel_off, n):
        return bytes(self.data[self.base + rel_off:self.base + rel_off + n])

    # ---- gpu data
    def alloc_gpu(self, data, why, align=16):
        self.gpu.extend(bytes((-len(self.gpu)) % align))
        at = len(self.gpu)
        self.gpu.extend(data)
        self.allocs.append(dict(gpu_offset=at, bytes=len(data), why=why))
        return at

    def output(self):
        used_end = self.cursor
        new_used = used_end - 4 - 16
        new_size = (16 + new_used + 127) // 128 * 128
        fore = bytearray(self.data[:4 + 16 + new_used])
        fore.extend(bytes(4 + new_size - len(fore)))
        struct.pack_into("<2I", fore, 4 + 4, new_size, new_used)
        body = bytes(fore[4:]) + self.tail
        zif = struct.pack("<I", len(body)) + body
        gpu = bytes(self.gpu)
        return zif, struct.pack("<I", len(gpu)) + gpu
