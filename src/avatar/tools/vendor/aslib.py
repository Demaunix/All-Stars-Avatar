"""Small dependency-free readers for Sonic & SEGA All-Stars Racing data (read-only use).

Formats are re-derived from the host loader source (xpac rows), prior SSRT parsers in earlier work
and direct inspection; every reader asserts its magic/layout so a wrong platform fails loudly.
"""
import struct, zlib, hashlib, os

T_STR, T_FLT, T_INT = 0xB7459825, 0xA45C088E, 0xFBE46712


def ph(name):
    h = 0
    for c in name.upper().replace("/", "\\")[::-1]:
        h = (ord(c) + 131 * h) & 0xFFFFFFFF
    return h


def name_hash(s, seed=0x04C11DB7):
    """Game string identifier: Jenkins lookup2 with seed 0x04C11DB7 (case-sensitive)."""
    if isinstance(s, str):
        s = s.encode()
    M = 0xFFFFFFFF
    a = b = 0x9E3779B9
    c = seed

    def mix(a, b, c):
        a = ((a - b - c) & M) ^ (c >> 13); b = ((b - c - a) & M) ^ ((a << 8) & M); c = ((c - a - b) & M) ^ (b >> 13)
        a = ((a - b - c) & M) ^ (c >> 12); b = ((b - c - a) & M) ^ ((a << 16) & M); c = ((c - a - b) & M) ^ (b >> 5)
        a = ((a - b - c) & M) ^ (c >> 3); b = ((b - c - a) & M) ^ ((a << 10) & M); c = ((c - a - b) & M) ^ (b >> 15)
        return a, b, c
    n = len(s); p = 0
    while n - p >= 12:
        a = (a + int.from_bytes(s[p:p + 4], "little")) & M; b = (b + int.from_bytes(s[p + 4:p + 8], "little")) & M
        c = (c + int.from_bytes(s[p + 8:p + 12], "little")) & M
        a, b, c = mix(a, b, c); p += 12
    c = (c + n) & M; t = s[p:]
    a = (a + int.from_bytes(t[:4], "little")) & M; b = (b + int.from_bytes(t[4:8], "little")) & M
    c = (c + (int.from_bytes(t[8:], "little") << 8)) & M
    return mix(a, b, c)[2]


assert name_hash("none") == 0x987B66C0


def sha256(b):
    return hashlib.sha256(b).hexdigest()


class Xpac:
    """PC .xpac: u32[6] header {0,0,tableEnd,count,0,0}; rows of 5 u32 {key,off,size,stored,flags}."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            h = struct.unpack("<6I", f.read(24))
            assert h[0] == 0 and h[1] == 0 and h[4] == 0 and h[5] == 0, "not an xpac"
            self.count = h[3]
            self.rows = [struct.unpack("<5I", f.read(20)) for _ in range(self.count)]
        self.by_key = {r[0]: r for r in self.rows}

    def raw(self, key):
        r = self.by_key[key]
        with open(self.path, "rb") as f:
            f.seek(r[1])
            return f.read(r[3])

    def data(self, key):
        raw = self.raw(key)
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return raw


def ssrt(data, endian="<"):
    """Return {table: {'rows': {rowname: {fieldhash: dict(type,bits,off,value)}}, 'order': [...]}}."""
    w = lambda p: struct.unpack_from(endian + "I", data, p)[0]
    z = lambda p: data[p:data.index(0, p)].decode("latin1")
    assert data[:4] == b"SSRT", "not SSRT"
    out = {}
    for i in range(w(4)):
        t = w(8 + 4 * i)
        rows, order = {}, []
        for j in range(w(t + 68)):
            r = w(t + 72 + 4 * j)
            fields = {}
            for k in range(w(r)):
                q = r + 72 + 12 * k
                h, typ, bits = struct.unpack_from(endian + "3I", data, q)
                if typ == T_STR:
                    val = z(bits)
                elif typ == T_FLT:
                    val = struct.unpack_from(endian + "f", data, q + 8)[0]
                else:
                    val = struct.unpack_from(endian + "i", data, q + 8)[0]
                fields[h] = dict(type=typ, bits=bits, off=q + 8, value=val)
            name = z(r + 8)
            rows[name] = fields
            order.append(name)
        out[z(t + 4)] = dict(rows=rows, order=order, off=t)
    return out


def smospkg(data):
    """FrontEnd/SetupObjects packages: 'smospkg2' + record table (32-byte records from 0x28).
    Returns list of dict(name, start, end, endian-of-module)."""
    assert data[:8] == b"smospkg2" or data[:8] == b"smospkg2", "not smospkg2"
    en = ">" if struct.unpack_from(">I", data, 8)[0] == 32 else "<"
    assert struct.unpack_from(en + "I", data, 8)[0] == 32
    n = struct.unpack_from(en + "I", data, 12)[0]
    recs = []
    for i in range(n):
        p = 40 + i * 32
        np_ = p + 20 + struct.unpack_from(en + "i", data, p + 20)[0]
        at = p + 28 + struct.unpack_from(en + "i", data, p + 28)[0]
        recs.append((data[np_:data.index(0, np_)].decode("latin1"), at))
    starts = sorted({a for _, a in recs}) + [len(data)]
    out = []
    for name, at in recs:
        end = next(x for x in starts if x > at)
        out.append(dict(name=name, start=at, end=end, data=data[at:end]))
    return out, en
