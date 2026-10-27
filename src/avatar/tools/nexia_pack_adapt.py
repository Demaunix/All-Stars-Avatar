"""Adapt the All-Stars Xbox 360 AvatarAssetPack.toc (version 1) to the record offset Nexia360 0.4.2 reads.

Why: the All-Stars disc copy of the Avatar Asset Package has a 0x20-byte header and its 1282 0xF8-byte records start at 0x20.
Nexia360 0.4.2 (`xna_avatar_format.cc`, `Catalog::Load`) assumes a version-1 pack's records start at 0x28 (and a
version-2 pack's at 0x30, after a locale count), so it would read every record 8 bytes late. This tool inserts 8 zero
bytes at 0x20 and adds 8 to every absolute offset the records hold (13 name pointers + the blob pointer). Nothing else
changes: the string table and every STRB item blob are copied byte for byte. `--check` proves the round trip.

This is a test-harness format adaptation for the emulator only (an Xbox 360 reference check); it is never shipped.
Usage:
  python nexia_pack_adapt.py <AvatarAssetPack.toc> <out.toc>
  python nexia_pack_adapt.py --check <AvatarAssetPack.toc> <out.toc>
"""
import hashlib
import struct
import sys
from pathlib import Path

FIRST_V1 = 0x20          # where the All-Stars pack's records really start
NEXIA_FIRST_V1 = 0x28    # where Nexia360 0.4.2 reads them for version 1
SHIFT = NEXIA_FIRST_V1 - FIRST_V1
LOCALES = 13
STRIDE = 8 + 0xB4 + 4 * LOCALES + 8
POINTERS = 8 + 0xB4      # name pointers, then blob pointer, then blob size
STOCK_TAIL = bytes.fromhex("c1c8f109a19cb2e0")


def be32(data: bytes, at: int) -> int:
    """Read a big-endian u32."""
    return struct.unpack_from(">I", data, at)[0]


def check_source(data: bytes) -> int:
    """Refuse anything that is not the version-1 layout this tool understands; return the record count."""
    version, count = be32(data, 0x18), be32(data, 0x1C)
    if version != 1:
        raise SystemExit(f"refusing: version {version}, this tool only adapts version 1")
    end = FIRST_V1 + count * STRIDE
    if end > len(data):
        raise SystemExit("refusing: records run past the end of the file")
    for i in range(count):
        base = FIRST_V1 + i * STRIDE
        for k in range(LOCALES + 1):
            value = be32(data, base + POINTERS + 4 * k)
            if value and not end <= value < len(data):
                raise SystemExit(f"refusing: record {i} pointer {k} = {value:#x} is outside the data area")
        blob = be32(data, base + POINTERS + 4 * LOCALES)
        if blob and data[blob:blob + 4] != b"STRB":
            raise SystemExit(f"refusing: record {i} blob at {blob:#x} is not STRB")
    return count


def adapt(data: bytes) -> bytes:
    """Return the pack with its records moved to 0x28 and their offsets shifted to match."""
    count = check_source(data)
    out = bytearray(data[:FIRST_V1] + bytes(SHIFT) + data[FIRST_V1:])
    for i in range(count):
        base = NEXIA_FIRST_V1 + i * STRIDE
        for k in range(LOCALES + 1):
            at = base + POINTERS + 4 * k
            value = be32(out, at)
            if value:
                struct.pack_into(">I", out, at, value + SHIFT)
    return bytes(out)


def nexia_view(data: bytes) -> list[tuple[str, bytes, bytes]]:
    """Read records the way Nexia360 0.4.2 does for version 1: (English name, asset id, blob bytes) per record."""
    count = be32(data, 0x1C)
    rows = []
    for i in range(count):
        base = NEXIA_FIRST_V1 + i * STRIDE
        name_at = be32(data, base + POINTERS)
        blob = be32(data, base + POINTERS + 4 * LOCALES)
        size = be32(data, base + POINTERS + 4 * LOCALES + 4)
        name = ""
        if name_at and name_at < len(data):
            end = name_at
            while end + 1 < len(data) and data[end:end + 2] != b"\0\0":
                end += 2
            name = data[name_at:end].decode("utf-16-be", "replace")
        asset_id = data[base + 8 + 0x94:base + 8 + 0x94 + 16]
        body = data[blob:blob + size] if blob and size > 0x3C and data[blob:blob + 4] == b"STRB" else b""
        rows.append((name, asset_id, body))
    return rows


def source_view(data: bytes) -> list[tuple[str, bytes, bytes]]:
    """Read the original pack's records at their true offset, for comparison with nexia_view."""
    shifted = data[:FIRST_V1] + bytes(SHIFT) + data[FIRST_V1:]
    rows = []
    count = be32(data, 0x1C)
    for i in range(count):
        base = FIRST_V1 + i * STRIDE
        name_at = be32(data, base + POINTERS)
        blob = be32(data, base + POINTERS + 4 * LOCALES)
        size = be32(data, base + POINTERS + 4 * LOCALES + 4)
        name = ""
        if name_at:
            end = name_at
            while end + 1 < len(data) and data[end:end + 2] != b"\0\0":
                end += 2
            name = data[name_at:end].decode("utf-16-be", "replace")
        asset_id = data[base + 8 + 0x94:base + 8 + 0x94 + 16]
        body = data[blob:blob + size] if blob and size > 0x3C and data[blob:blob + 4] == b"STRB" else b""
        rows.append((name, asset_id, body))
    del shifted
    return rows


def unadapt(data: bytes) -> bytes:
    """Undo adapt(): drop the inserted bytes and shift the offsets back."""
    count = be32(data, 0x1C)
    out = bytearray(data)
    for i in range(count):
        base = NEXIA_FIRST_V1 + i * STRIDE
        for k in range(LOCALES + 1):
            at = base + POINTERS + 4 * k
            value = be32(out, at)
            if value:
                struct.pack_into(">I", out, at, value - SHIFT)
    return bytes(out[:FIRST_V1] + out[NEXIA_FIRST_V1:])


def check(source: bytes, adapted: bytes) -> None:
    """Prove the adaptation is lossless and that Nexia's reading of it equals the true reading of the source."""
    if unadapt(adapted) != source:
        raise SystemExit("FAIL: round trip does not give back the source")
    a, b = source_view(source), nexia_view(adapted)
    if a != b:
        bad = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        raise SystemExit(f"FAIL: record {bad} differs")
    named = sum(1 for n, _, _ in b if n)
    blobs = sum(1 for _, _, body in b if body)
    ids = sum(1 for _, aid, _ in b if aid[8:] == STOCK_TAIL)
    print(f"PASS: {len(b)} records, {named} named, {blobs} STRB blobs identical, {ids} stock asset ids; round trip exact")


def main(argv: list[str]) -> None:
    """Command line entry point."""
    if argv[:1] == ["--check"] and len(argv) == 3:
        source, adapted = Path(argv[1]).read_bytes(), Path(argv[2]).read_bytes()
        check(source, adapted)
        return
    if len(argv) != 2:
        raise SystemExit(__doc__)
    source = Path(argv[0]).read_bytes()
    adapted = adapt(source)
    check(source, adapted)
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(adapted)
    print(f"wrote {out} ({len(adapted)} bytes, sha256 {hashlib.sha256(adapted).hexdigest()})")
    print(f"source sha256 {hashlib.sha256(source).hexdigest()}")


if __name__ == "__main__":
    main(sys.argv[1:])
