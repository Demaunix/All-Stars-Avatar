"""SSRT (RacerParams/Shopping) structure guard + exact native PC relocation and shop-eligibility emulation.

guarded_relocate / native_emulation / native_filter are copied from the earlier project's metal-sonic
identity3-startup-fix/validate_ssrt_relocation.py (sha256 of that file recorded in research/REUSED-CODE.json) and
separate-slot/add_shopping_registration.py, changed only to use the absolute saved-image path (tools/pcimage.py) and
the module-local Unicorn/hash imports. They execute the ORIGINAL PC routines 0x5432B0 (relocation) and 0x5453D0
(profile/shop eligibility filter) from the saved in-memory image in isolated Unicorn memory.
"""
import collections
import hashlib
import struct
import sys

from pcimage import IMG, LIBS
sys.path.insert(0, str(LIBS))
import unicorn as U  # noqa: E402
from unicorn.x86_const import UC_X86_REG_EBX, UC_X86_REG_ESI, UC_X86_REG_EDI, UC_X86_REG_EBP, UC_X86_REG_ESP, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_EAX  # noqa: E402
from vendor.smospkg import name_hash  # noqa: E402

DATA = 0x10000000; CTX = 0x20000000; STACK = 0x30000000; RET = 0x40000000
ENTRY = 0x5432b0
TYPES = {name_hash(n): n for n in ('float', 'string', 'int', 'bool')}
STRING = name_hash('string')
assert STRING == 0xb7459825
sha = lambda b: hashlib.sha256(b).hexdigest()
u = lambda d, p: struct.unpack_from('<I', d, p)[0]
_IMAGE = None


def image():
    global _IMAGE
    if _IMAGE is None:
        _IMAGE = IMG.read_bytes()
        assert sha(_IMAGE) == "a77d614904d9ad96da9b9c17bd95136fce62ccb82ce62c8b71ef3fe13f22f92a"
    return _IMAGE


def guarded_relocate(d):
    if d[:4] != b'SSRT': raise ValueError('Expected raw SSRT magic; compressed/archive bytes are not accepted')
    N = len(d); out = bytearray(d); fixups = {}; tables = []; rows_seen = set(); table_seen = set(); types = collections.Counter()

    def bounds(p, n, label):
        if p < 0 or n < 0 or p + n > N: raise ValueError(f'{label} outside raw data: offset={p:#x}, length={n}, size={N}')

    def text(p, limit, label):
        bounds(p, 1, label)
        end = d.find(b'\0', p, min(N, p + limit))
        if end < 0: raise ValueError(f'{label} lacks terminator inside allowed range')
        return d[p:end].decode('ascii')

    def fix(p, target, label):
        bounds(p, 4, label + ' pointer'); bounds(target, 1, label + ' target')
        if p in fixups: raise ValueError('Native loader would relocate a pointer more than once')
        value = DATA + target
        if value > 0xffffffff: raise ValueError('32-bit relocated pointer overflow')
        struct.pack_into('<I', out, p, value); fixups[p] = target
    bounds(0, 8, 'header'); nt = u(d, 4); bounds(8, 4 * nt, 'table pointer array')
    for i in range(nt):
        tp = 8 + 4 * i; t = u(d, tp); bounds(t, 72, 'table header')
        if t in table_seen: raise ValueError('Repeated table would be relocated twice')
        table_seen.add(t); fix(tp, t, 'table')
        name = text(t + 4, 64, 'table name'); nr = u(d, t + 68); bounds(t + 72, 4 * nr, 'row pointer array')
        rowreports = []
        for j in range(nr):
            rp = t + 72 + 4 * j; r = u(d, rp); bounds(r, 72, 'row header')
            if r in rows_seen: raise ValueError('Repeated row would relocate its strings twice')
            rows_seen.add(r); fix(rp, r, 'row')
            rname = text(r + 8, 64, 'row name'); nf = u(d, r); bounds(r + 72, 12 * nf, 'field array')
            string_count = 0
            for k in range(nf):
                p = r + 72 + 12 * k; typ = u(d, p + 4); types[TYPES.get(typ, f'unknown:{typ:08X}')] += 1
                if typ == STRING:
                    sp = u(d, p + 8); text(sp, N - sp, 'string field'); fix(p + 8, sp, 'string'); string_count += 1
            rowreports.append({'name': rname, 'offset': r, 'fields': nf, 'strings': string_count})
        tables.append({'name': name, 'offset': t, 'rows': rowreports})
    return bytes(out), fixups, {'tables': tables, 'table_count': nt, 'row_count': len(rows_seen), 'field_types': dict(types), 'fixup_count': len(fixups), 'all_pointer_targets_and_strings_in_bounds': True}


def native_emulation(d, expected=None, fixups=None):
    blob = image()
    code = blob[0x143000:0x144000]
    mu = U.Uc(U.UC_ARCH_X86, U.UC_MODE_32)
    mu.mem_map(0x543000, 4096); mu.mem_write(0x543000, code)
    mu.mem_map(DATA, (len(d) + 4095) & ~4095); mu.mem_write(DATA, d)
    for a in (CTX, STACK, RET): mu.mem_map(a, 4096)
    mu.mem_write(CTX, struct.pack('<7I', DATA, 0, 0, name_hash('float'), STRING, name_hash('int'), name_hash('bool')))
    sp = STACK + 0x800; mu.mem_write(sp, struct.pack('<I', RET)); mu.reg_write(UC_X86_REG_ESP, sp); mu.reg_write(UC_X86_REG_ECX, CTX)
    registers = {UC_X86_REG_EBX: 0x11223344, UC_X86_REG_ESI: 0x22334455, UC_X86_REG_EDI: 0x33445566, UC_X86_REG_EBP: 0x44556677}
    for reg, value in registers.items(): mu.reg_write(reg, value)
    writes = []; bad = []

    def memory(uc, access, address, size, value, user):
        if DATA <= address < DATA + ((len(d) + 4095) & ~4095):
            if address + size > DATA + len(d): bad.append({'address': hex(address), 'size': size}); uc.emu_stop(); return
            if access == U.UC_MEM_WRITE: writes.append((address - DATA, size))
    mu.hook_add(U.UC_HOOK_MEM_READ | U.UC_HOOK_MEM_WRITE, memory)
    failure = None
    try: mu.emu_start(ENTRY, RET, count=1000000)
    except U.UcError as ex: failure = str(ex)
    assert failure is None and not bad, (failure, bad)
    assert mu.reg_read(UC_X86_REG_EIP) == RET and mu.reg_read(UC_X86_REG_ESP) == sp + 4
    assert all(mu.reg_read(reg) == value for reg, value in registers.items())
    result = bytes(mu.mem_read(DATA, len(d)))
    assert result == expected, 'Exact native relocated buffer differs from guarded result'
    assert collections.Counter(writes) == collections.Counter((p, 4) for p in fixups)
    return {'exact_native_full_buffer_matches_guarded_result': True, 'native_relocation_writes': len(writes),
            'all_data_reads_and_writes_within_exact_payload': True, 'stack_and_callee_saved_registers_preserved': True,
            'relocated_sha256': sha(result), 'routine_bytes_sha256': sha(blob[0x143250:0x1432f4])}


def native_filter(data, table, racer):
    mu = U.Uc(U.UC_ARCH_X86, U.UC_MODE_32)
    blob = image()
    mu.mem_map(0x400000, (len(blob) + 4095) & ~4095); mu.mem_write(0x400000, blob)
    mu.mem_map(0xd21000, 4096)
    for a in (DATA, CTX, STACK, RET):
        mu.mem_map(a, ((len(data) + 4095) & ~4095) if a == DATA else 4096)
    mu.mem_write(DATA, data)
    for a, v in ((0xd214a8, name_hash('PC')), (0xd214b0, name_hash('All'))):
        mu.mem_write(a, struct.pack('<I', v))
    sp = STACK + 0x800
    mu.mem_write(sp, struct.pack('<3I', RET, DATA + table, racer))
    mu.reg_write(UC_X86_REG_ESP, sp); mu.reg_write(UC_X86_REG_ECX, CTX)
    mu.emu_start(0x5453d0, RET, count=1000000)
    assert mu.reg_read(UC_X86_REG_EIP) == RET and mu.reg_read(UC_X86_REG_ESP) == sp + 12
    return bool(mu.reg_read(UC_X86_REG_EAX) & 255)
