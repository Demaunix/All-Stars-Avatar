"""Read-only access to the saved in-memory PC game image (prior work; SHA-256 a77d6149..., base 0x400000) used by the
C232/C234 native tests. Capstone x86-32 disassembly helpers for static reference checks."""
import hashlib
import struct
import sys
from functools import lru_cache

LIBS = r"<codex-archive>/Documents-Codex/2026-09-10/referenced-chatgpt-conversation-this-is-an-2/work/freecam-20260914-v4/freeze-review/libs"
IMG = r"<codex-archive>/Documents-Codex/2026-09-10/referenced-chatgpt-conversation-this-is-an-2/work/game-loaded-image.bin"
IMG_SHA = "a77d614904d9ad96da9b9c17bd95136fce62ccb82ce62c8b71ef3fe13f22f92a"
BASE = 0x400000

import capstone


@lru_cache(None)
def image():
    d = open(IMG, "rb").read()
    assert hashlib.sha256(d).hexdigest() == IMG_SHA
    return d


def find(pat):
    d = image()
    out, i = [], d.find(pat)
    while i >= 0:
        out.append(BASE + i)
        i = d.find(pat, i + 1)
    return out


def refs32(va):
    return find(struct.pack("<I", va))


_md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)


def dis(va, n=40):
    d = image()
    off = va - BASE
    return [(i.address, i.mnemonic, i.op_str) for i in _md.disasm(d[off:off + n * 8], va)][:n]


def func_start(va, max_back=0x2000):
    """Heuristic: nearest preceding int3/ret padding boundary followed by a typical prologue."""
    d = image()
    off = va - BASE
    for k in range(off, max(off - max_back, 0), -1):
        if d[k - 1] in (0xCC, 0xC3) and d[k] in (0x55, 0x83, 0x81, 0x56, 0x53, 0x57, 0x8B, 0x51, 0x6A, 0xA1, 0x64):
            return BASE + k
    return None
