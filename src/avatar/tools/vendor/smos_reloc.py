"""Offline SMOS parser/relocator derived from PC interpreter 0x48F990.

No game files are opened. Branch targets, inline string references and public
function entries are relocated separately, with integer-range assertions.
"""
import struct

FORMATS = {op: () for op in range(0x5e)}
for op in (1,2,3,6,0xa,0xe,0x26,0x2d,0x2f,0x5d): FORMATS[op]=(4,)
for op in (5,8,9,0xc,0x10,0x13,0x14,0x15,0x27,0x29,0x2c,0x52,0x54,0x56): FORMATS[op]=(2,)
for op in (7,0xb,0xd,0xf,0x16,0x19,0x51,0x53,0x55,0x57,0x58,0x5b): FORMATS[op]=(1,)
FORMATS.update({0x12:(1,2),0x1a:(1,4),0x1c:(1,4),0x22:(1,1,4),0x24:(1,2),
                0x28:(1,2),0x2e:(1,4),0x32:(1,2,2,2),0x33:(2,2),0x34:(1,4,2,2),
                0x35:(1,4,2,2),0x5c:(1,2,2)})
REL16={5,0x13,0x14,0x15,0x2c}
REL32={0x1a,0x34}
# Native SMOS header offsets which point elsewhere in this module. In particular
# 0x1C is the optional global-symbol table consumed by 0x491630, not a count.
# 0x20 is a fixed VM value size; 0x24 is a module identity; 0x28 is a local-slot
# count. None of those three are pointers. 0x38/0x4C are runtime state/count.
HEADER_POINTER_FIELDS=(0x0C,0x14,0x18,0x1C,0x2C,0x34)
METADATA_POINTER_FIELDS=HEADER_POINTER_FIELDS[1:]

def u32(d,p,e='<'): return struct.unpack_from(e+'I',d,p)[0]

def instructions(d,start=None,end=None,e=None):
    if e is None: e='<' if d[:8]==b'smosbinL' else '>'
    if start is None: start=u32(d,12,e)
    if end is None: end=start+u32(d,16,e)
    p=start;result=[]
    while p<end:
        begin=p;op=d[p];p+=1;fields=[]
        if op in (4,0x2b):
            q=d.index(0,p,end)+1;fields.append((p,q-p,'string'));p=q
        else:
            assert op in FORMATS,(hex(begin),hex(op))
            for n in FORMATS[op]: fields.append((p,n,'integer'));p+=n
        assert p<=end,(hex(begin),hex(p),hex(end))
        rec=dict(offset=begin,end=p,opcode=op,fields=fields)
        if op in REL16:
            rel=struct.unpack_from(e+'h',d,begin+1)[0]
            rec.update(relative_at=begin+1,relative_width=2,target=p+rel,kind='string' if op in(5,0x2c) else 'branch')
        elif op in REL32:
            rel=struct.unpack_from(e+'i',d,begin+2)[0]
            rec.update(relative_at=begin+2,relative_width=4,target=p+rel,kind='call')
        result.append(rec)
    assert p==end
    return result

def endian_code(d,start,end,e='>'):
    out=bytearray(d[start:end])
    for rec in instructions(d,start,end,e):
        for p,n,t in rec['fields']:
            if n>1 and t=='integer':out[p-start:p-start+n]=d[p:p+n][::-1]
    return bytes(out)

def validate(d):
    assert d[:8]==b'smosbinL'
    size=u32(d,8);code_start=u32(d,12);code_end=code_start+u32(d,16)
    assert 80<=code_start<=code_end<=size<=len(d) and len(d)-size<4
    for p in HEADER_POINTER_FIELDS:
        value=u32(d,p)
        if p==0x1C and value==0:continue
        assert 0<value<size,(hex(p),hex(value))
    ops=instructions(d);points={r['offset'] for r in ops}|{ops[-1]['end']}
    strings=[(p,p+n) for r in ops for p,n,t in r['fields'] if t=='string']
    for r in ops:
        if 'target' not in r:continue
        if r['kind']=='string':assert any(a<=r['target']<b for a,b in strings),r
        else:assert r['target'] in points,r
    pub=u32(d,20);count=u32(d,pub)
    assert code_end<=pub and pub+4+8*count<=size,'Export table is outside metadata bounds'
    functions=[]
    for j in range(count):
        h,off=struct.unpack_from('<II',d,pub+4+8*j)
        assert off+u32(d,12) in points,(hex(h),hex(off))
        functions.append((h,off))
    assert functions==sorted(functions)
    constants=u32(d,24);constant_count=u32(d,constants)
    assert code_end<=constants and constants+4+8*constant_count<=size,'Typed constants are outside metadata bounds'
    globals_pointer=u32(d,0x1C);global_count=0
    if globals_pointer:
        assert code_end<=globals_pointer and globals_pointer+4<=size,'Global table points inside bytecode or outside module'
        global_count=u32(d,globals_pointer)
        assert globals_pointer+4+4*global_count<=size,'Global symbol count exceeds module bounds'
    assert code_end<=u32(d,44) and u32(d,44)+4*u32(d,48)<=size
    for j in range(u32(d,48)):
        p=u32(d,u32(d,44)+4*j);assert 0<p<u32(d,8) and d.index(0,p)<u32(d,8)
    assert d.index(0,u32(d,52),size)<size
    return dict(instructions=len(ops),relative_references=sum('target'in r for r in ops),public_functions=count,
                global_table_offset=globals_pointer,global_symbols=global_count,
                inline_strings=len(strings),code_start=u32(d,12),code_length=u32(d,16))

def insert(d,insertions):
    """Insert instruction chunks before original positions; preserve original code."""
    assert d[:8]==b'smosbinL'
    validate(d)
    ops=instructions(d);boundaries={r['offset'] for r in ops}|{ops[-1]['end']}
    for p,chunk in insertions.items():
        assert p in boundaries,hex(p)
        instructions(chunk,0,len(chunk),'<')
    def loc(p):return p+sum(len(v) for k,v in insertions.items() if k<=p)
    out=bytearray();last=0
    for p,chunk in sorted(insertions.items()):out+=d[last:p]+chunk;last=p
    out+=d[last:]
    fixes=[]
    for rec in ops:
        if 'target' not in rec:continue
        # Instruction end is derived from relocated start: an insertion exactly
        # at its old end must not change the previous instruction's end address.
        end=loc(rec['offset'])+rec['end']-rec['offset']
        target=loc(rec['target'])
        value=target-end;w=rec['relative_width']
        assert -(1<<(8*w-1))<=value<(1<<(8*w-1)),rec
        struct.pack_into('<h' if w==2 else '<i',out,loc(rec['relative_at']),value)
        if value!=rec['target']-rec['end']:fixes.append(dict(old=rec['offset'],new=loc(rec['offset']),target=target,kind=rec['kind']))
    delta=sum(map(len,insertions.values()))
    for p in (8,16):struct.pack_into('<I',out,p,u32(d,p)+delta)
    # Code start stays fixed: an insertion at the first instruction is part of
    # the new code region. All metadata after the code moves with its contents.
    for p in METADATA_POINTER_FIELDS:
        old=u32(d,p)
        # A zero optional-global pointer must remain the native null sentinel.
        struct.pack_into('<I',out,p,loc(old) if old else 0)
    # The symbol offsets are relative to code start, not the module's base.
    oldpub=u32(d,20);newpub=loc(oldpub)
    for j in range(u32(d,oldpub)):
        p=oldpub+8+8*j;target=u32(d,p)+u32(d,12)
        struct.pack_into('<I',out,newpub+8+8*j,loc(target)-u32(d,12))
    assert u32(d,u32(d,24))==0,'Static typed constants need separate relocation support'
    for j in range(u32(d,48)):
        p=u32(d,44)+4*j;struct.pack_into('<I',out,loc(p),loc(u32(d,p)))
    # Keep module size aligned for the package; module+8 remains unpadded length.
    out=out[:u32(out,8)];out+=bytes((-len(out))%4)
    return bytes(out),dict(inserted_bytes=delta,relative_fixes=fixes,verification=validate(out))
