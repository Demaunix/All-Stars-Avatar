"""Read-only access to the decoded Xbox 360 All-Stars image (prior work, pinned by SHA-256) for static analysis.

Source: metal-sonic/xenon-probe/xbox-image.bin (base 0x82000000), SHA-256 188c4946...b773a (W/AVATAR-STATE-STATIC-REFERENCE.json).
Capstone (stock) decodes PPC but not VMX128; such words print as .long.
"""
import bisect
import hashlib
import json
import struct
from functools import lru_cache

import capstone

ROOT = "<codex-archive>/Documents-Codex/2026-09-10/referenced-chatgpt-conversation-this-is-an-2/work/metal-sonic/xenon-probe/"
SHA = "188c4946832eb1ebc4f409bb18bbd66317005873f8bc339b7f10f569e40b773a"


@lru_cache(None)
def image():
    d = open(ROOT + "xbox-image.bin", "rb").read()
    assert hashlib.sha256(d).hexdigest() == SHA
    meta = json.load(open(ROOT + "xbox-image.json"))
    return d, meta


def base():
    return image()[1]["base"]


def u32(va):
    d, m = image()
    return struct.unpack_from(">I", d, va - m["base"])[0]


def raw(va, n):
    d, m = image()
    return d[va - m["base"]:va - m["base"] + n]


def cstr(va, limit=256):
    b = raw(va, limit)
    return b[:b.index(0)].decode("latin1") if 0 in b else b.decode("latin1")


@lru_cache(None)
def functions():
    """(start, end) pairs from .pdata (PPC: begin VA, then packed word with function length in instructions in bits 8..29)."""
    d, m = image()
    sec = next(s for s in m["sections"] if s["name"] == ".pdata")
    out = []
    for p in range(sec["base"], sec["base"] + sec["size"], 8):
        b, w = struct.unpack_from(">II", d, p - m["base"])
        if b == 0:
            continue
        n = (w >> 8) & 0x3FFFFF
        out.append((b, b + 4 * n))
    out.sort()
    return out


def func_of(va):
    fs = functions()
    i = bisect.bisect_right([f[0] for f in fs], va) - 1
    if i >= 0 and fs[i][0] <= va < fs[i][1]:
        return fs[i]
    return None


_md = capstone.Cs(capstone.CS_ARCH_PPC, capstone.CS_MODE_32 | capstone.CS_MODE_BIG_ENDIAN)


def dis(va, n):
    out = []
    for i in range(n):
        a = va + 4 * i
        w = raw(a, 4)
        ins = list(_md.disasm(w, a))
        out.append((a, w.hex(), f"{ins[0].mnemonic} {ins[0].op_str}" if ins else ".long"))
    return out


def dis_func(va):
    f = func_of(va)
    assert f, hex(va)
    return dis(f[0], (f[1] - f[0]) // 4)


def find_bytes(pat, section=None):
    d, m = image()
    res = []
    i = d.find(pat)
    while i >= 0:
        res.append(m["base"] + i)
        i = d.find(pat, i + 1)
    return res


def text_range():
    m = image()[1]
    s = next(s for s in m["sections"] if s["name"] == ".text")
    return s["base"], s["base"] + s["size"]


def scan_lis_ori(hi, lo):
    """Find 'lis rX,hi' followed within 8 instructions by 'ori/addi rX,rX,lo' in .text."""
    d, m = image()
    t0, t1 = text_range()
    hits = []
    for a in range(t0, t1, 4):
        w = struct.unpack_from(">I", d, a - m["base"])[0]
        if (w >> 26) == 15 and ((w >> 16) & 31) == 0 and (w & 0xFFFF) == hi:
            r = (w >> 21) & 31
            for k in range(1, 9):
                w2 = struct.unpack_from(">I", d, a + 4 * k - m["base"])[0]
                op = w2 >> 26
                if op in (24, 14) and (w2 & 0xFFFF) == lo and ((w2 >> 16) & 31) == r if op == 14 else ((w2 >> 21) & 31) == r:
                    hits.append(a)
                    break
    return hits
