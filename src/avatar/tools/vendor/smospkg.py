"""smospkg2 package reader/writer and the game's string hash.

read_package/package: verbatim copies of the functions of the same names in the earlier project's
metal-sonic/separate-slot/build_scripts.py (SHA-256 0c6685f6ad76a6ab5eadd81ef7f528f7cf2e2dc3bb69fdbc6a5fd5a2fca95761),
used there to rebuild FrontEndScripts.sbn byte-exactly; copied here without that file's builder code.
name_hash: Jenkins lookup2 with seed 0x04C11DB7 as in C232 tools/aslib.py (game function 0x4947E0 / Wii sumohash).
"""
import struct

try:
    from smos_reloc import u32
except ImportError:  # imported as vendor.smospkg
    from .smos_reloc import u32


def name_hash(s, seed=0x04C11DB7):
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
    n = len(s)
    p = 0
    while n - p >= 12:
        a = (a + int.from_bytes(s[p:p + 4], "little")) & M
        b = (b + int.from_bytes(s[p + 4:p + 8], "little")) & M
        c = (c + int.from_bytes(s[p + 8:p + 12], "little")) & M
        a, b, c = mix(a, b, c)
        p += 12
    c = (c + n) & M
    t = s[p:]
    a = (a + int.from_bytes(t[:4], "little")) & M
    b = (b + int.from_bytes(t[4:8], "little")) & M
    c = (c + (int.from_bytes(t[8:], "little") << 8)) & M
    return mix(a, b, c)[2]


assert name_hash("none") == 0x987B66C0


def read_package(d):
    assert d[:8] == b'smospkg2' and u32(d, 8) == 32
    result = []
    for i in range(u32(d, 12)):
        p = 40 + 32 * i
        np = p + 20 + struct.unpack_from('<i', d, p + 20)[0]
        bp = p + 28 + struct.unpack_from('<i', d, p + 28)[0]
        assert p + 24 + struct.unpack_from('<i', d, p + 24)[0] == 0
        name = d[np:d.index(0, np)].decode()
        result.append(dict(name=name, hash=u32(d, p + 16), start=bp, entry=d[p:p + 32]))
    starts = sorted({r['start'] for r in result}) + [len(d)]
    for i, r in enumerate(result):
        end = next(p for p in starts if p > r['start'])
        r['data'] = d[r['start']:end]
        assert r['data'][:8] == b'smosbinL'
    return result


def package(d, replacements={}, additions={}):
    records = read_package(d)
    rootname = d[32 + u32(d, 32):d.index(0, 32 + u32(d, 32))]
    for r in records:
        if r['name'] in replacements: r['data'] = replacements[r['name']]
    for name, blob in additions.items():
        assert all(r['name'] != name for r in records)
        at = next((i for i, r in enumerate(records) if r['name'].casefold() > name.casefold()), len(records))
        records.insert(at, dict(name=name, hash=name_hash(name), entry=bytes(32), data=blob, start='new:' + name))
    count = len(records); names = bytearray(rootname + b'\0'); nameoffs = {}
    for r in records: nameoffs[r['name']] = 40 + 32 * count + len(names); names += r['name'].encode() + b'\0'
    header = bytearray(d[:40]); struct.pack_into('<I', header, 12, count)
    struct.pack_into('<I', header, 32, 8 + 32 * count)
    namesend = 40 + 32 * count + len(names); struct.pack_into('<I', header, 36, namesend)
    out = header + bytes(32 * count) + names; out += bytes((-len(out)) % 4)
    written = {}
    for i, r in enumerate(records):
        p = 40 + i * 32; out[p:p + 32] = r['entry']
        struct.pack_into('<I', out, p + 16, r['hash'])
        struct.pack_into('<i', out, p + 20, nameoffs[r['name']] - (p + 20))
        struct.pack_into('<i', out, p + 24, -(p + 24))
        if r['start'] in written:
            previous_at, previous_data = written[r['start']]
            assert previous_data == r['data'], 'Aliased records must retain shared payload'
            struct.pack_into('<i', out, p + 28, previous_at - (p + 28))
            continue
        struct.pack_into('<i', out, p + 28, len(out) - (p + 28))
        written[r['start']] = (len(out), r['data'])
        out += r['data']
        if i + 1 < count: out += bytes((-len(out)) % 4)
    return bytes(out)


def read_package_be(d):
    """Big-endian (Wii) smospkg2 reader: returns [(name, start, data)] with smosbinB modules."""
    e = ">"
    assert d[:8] == b"smospkg2" and struct.unpack_from(e + "I", d, 8)[0] == 32
    rows = []
    for i in range(struct.unpack_from(e + "I", d, 12)[0]):
        p = 40 + 32 * i
        np_ = p + 20 + struct.unpack_from(e + "i", d, p + 20)[0]
        bp = p + 28 + struct.unpack_from(e + "i", d, p + 28)[0]
        rows.append(dict(name=d[np_:d.index(0, np_)].decode(), start=bp))
    starts = sorted({r["start"] for r in rows}) + [len(d)]
    for r in rows:
        r["data"] = d[r["start"]:next(p for p in starts if p > r["start"])]
        assert r["data"][:8] == b"smosbinB"
    return rows
