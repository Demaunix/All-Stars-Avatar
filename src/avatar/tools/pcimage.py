"""Read-only access to the existing saved in-memory PC game image (0x400000.., used by C232's native emulation) with
x86 disassembly. Path/hash identical to C232 tests/native_world.py."""
import hashlib
import struct
import sys
from pathlib import Path

IMG = Path("<codex-archive>/Documents-Codex/2026-09-10/referenced-chatgpt-conversation-this-is-an-2/work/game-loaded-image.bin")
BASE = 0x400000
LIBS = Path("<codex-archive>/Documents-Codex/2026-09-10/referenced-chatgpt-conversation-this-is-an-2/work/freecam-20260914-v4/freeze-review/libs")


class Image:
    def __init__(self):
        self.d = IMG.read_bytes()
        self.sha256 = hashlib.sha256(self.d).hexdigest()

    def read(self, va, n):
        return self.d[va - BASE:va - BASE + n]

    def u32(self, va):
        return struct.unpack_from("<I", self.d, va - BASE)[0]

    def disasm(self, va, n=0x100, stop_at_ret=True):
        if str(LIBS) not in sys.path:
            sys.path.insert(0, str(LIBS))
        import capstone
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
        out = []
        for ins in md.disasm(self.read(va, n), va):
            out.append(f"{ins.address:08x} {ins.bytes.hex():<16} {ins.mnemonic} {ins.op_str}")
            if stop_at_ret and ins.mnemonic in ("ret", "jmp") and ins.address > va + 4 and ins.mnemonic == "ret":
                break
        return out
