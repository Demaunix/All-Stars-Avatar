"""Python port of the deployed loader's virtual XPAC composition (allstars-mod-manager-001/input/virtual_xpac.h, function virtual_xpac):
members keyed by u32 (std::map order = ascending key), header {0,0,align(24+20*n),n,0,0}, rows {key,off,len,len,0}, every member placed
at a 2048-aligned cursor and zero-padded to the next 2048 boundary; the composed byte stream is hashed with SHA-256.
Used to fill plan.json expected_composed_sha256/size/member_count; validated against Banjo alpha.7's shipped plan values."""
import hashlib
import struct


def align(n):
    a = (n + 2047) & ~2047
    assert a <= 0xFFFFFFFF, "XPAC exceeds 32-bit offsets"
    return a


def read_rows(path):
    with open(path, "rb") as f:
        h = list(struct.unpack("<6I", f.read(24)))
        assert h[0] == 0 and h[1] == 0 and h[4] == 0 and h[5] == 0 and 0 < h[3] < 1000000
        rows = [struct.unpack("<5I", f.read(20)) for _ in range(h[3])]
    for r in rows:
        assert r[2] == r[3] and r[4] == 0, "unsupported record"
    return h, rows


def member_bytes(path, row):
    with open(path, "rb") as f:
        f.seek(row[1])
        return f.read(row[3])


def compose(path, operations):
    """operations: list of dict(kind, key(int), payload(bytes)). Returns dict(sha256, size, member_count, base_stored_sha256={key: sha})."""
    h, rows = read_rows(path)
    members = {r[0]: ("orig", r) for r in rows}
    base_stored = {}
    seen = set()
    for o in operations:
        k = o["key"]
        assert k not in seen, "Duplicate member operation"
        seen.add(k)
        if o["kind"] == "replace":
            assert k in members, "Replacement member absent"
            base_stored[k] = hashlib.sha256(member_bytes(path, members[k][1])).hexdigest()
        else:
            assert o["kind"] == "add" and k not in members, "Added member already exists"
        members[k] = ("payload", o["payload"])
    keys = sorted(members)
    n = len(keys)
    meta = align(24 + 20 * n)
    hasher = hashlib.sha256()
    header = struct.pack("<6I", 0, 0, meta, n, 0, 0)
    table = bytearray()
    cursor = meta
    layout = []
    for k in keys:
        src = members[k]
        length = src[1][3] if src[0] == "orig" else len(src[1])
        table += struct.pack("<5I", k, cursor, length, length, 0)
        layout.append((k, cursor, length, src))
        cursor = align(cursor + length)
    hasher.update(header + bytes(table) + bytes(meta - 24 - len(table)))
    with open(path, "rb") as f:
        for k, off, length, src in layout:
            if src[0] == "orig":
                f.seek(src[1][1])
                left = length
                while left:
                    chunk = f.read(min(left, 1 << 20))
                    assert chunk
                    hasher.update(chunk)
                    left -= len(chunk)
            else:
                hasher.update(src[1])
            hasher.update(bytes(align(off + length) - off - length))
    return dict(sha256=hasher.hexdigest(), size=cursor, member_count=n, base_stored_sha256={f"{k:08X}": v for k, v in base_stored.items()})
