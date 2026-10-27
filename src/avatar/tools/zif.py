"""Read-only parser for PC racer model files (.zif + .zig), little-endian.

Layout facts reused from the earlier project's verified converters (banjo-kazooie-port-20260926/converter:
parse_console_forest.py, native_layout_probe.py, console_model.py) and C232 tools; re-implemented here.
Every structure read is bounds-checked; unknown words are kept raw.
"""
import struct
from pathlib import Path


class Zif:
    def __init__(self, zif_path, zig_path=None, en="<"):
        self.path = Path(zif_path)
        self.d = self.path.read_bytes()
        self.g = Path(zig_path).read_bytes()[4:] if zig_path else None
        self.en = en
        self.chunks = []
        # PC .zif: u32 prefix then chunks; Wii .sif (big-endian): chunks from offset 0.
        p = 0 if self.d[:4] == b"FORE" else 4
        while p < len(self.d):
            tag = self.d[p:p + 4]
            size, used, marker = self.us(p + 4, 3)
            assert marker == 0x44332211 and size >= 16 and p + size <= len(self.d), (p, tag, size, used, hex(marker))
            self.chunks.append(dict(tag=tag.decode("latin1"), pos=p, size=size, used=used))
            p += size
        assert p == len(self.d)
        self.forests = self._forests()

    def u(self, p):
        return struct.unpack_from(self.en + "I", self.d, p)[0]

    def us(self, p, n):
        return list(struct.unpack_from(self.en + str(n) + "I", self.d, p))

    def f(self, p, n=1):
        v = struct.unpack_from(self.en + str(n) + "f", self.d, p)
        return v[0] if n == 1 else list(v)

    def s(self, p):
        assert 0 <= p < len(self.d)
        return self.d[p:self.d.index(0, p)].decode("ascii", errors="replace")

    def _forests(self):
        fore = next(c for c in self.chunks if c["tag"] == "FORE")
        q = fore["pos"] + 16
        out = []
        for fi in range(self.u(q)):
            hs, no, bo, go = self.us(q + 4 + fi * 16, 4)
            b = q + bo
            hdr = self.us(b, 9)
            count, models, nt, tex = hdr[:4]
            f = dict(index=fi, name=self.s(q + no), base=b, gpu_base=go, header=hdr, textures=[], models=[])
            for i in range(nt):
                tp = b + self.u(b + tex + i * 4)
                th = self.us(tp, 15)
                f["textures"].append(dict(index=i, pos=tp, name=self.s(b + th[0]), header=th))
            for mi in range(count):
                mp = b + self.u(b + models + mi * 4)
                mh = self.us(mp, 25)
                m = dict(index=mi, pos=mp, header=mh, nodes=[], anims=[])
                for ni in range(mh[2]):
                    np_ = b + self.u(b + mh[3] + ni * 4)
                    nh = self.us(np_, 10)
                    h16 = struct.unpack_from(self.en + "4H", self.d, np_)
                    m["nodes"].append(dict(index=ni, pos=np_, header=nh, h16=h16, type=h16[3], name=self.s(b + nh[3])))
                for j in range(mh[19]):
                    a, n, h = self.us(b + mh[20] + 12 * j, 3)
                    m["anims"].append(dict(index=j, offset=a, name=self.s(b + n), hash=h))
                m["root"] = m["nodes"][0]["name"] if m["nodes"] else None
                f["models"].append(m)
            out.append(f)
        return out

    def node_shapes(self, b, node):
        np_ = node["pos"]
        if node["type"] in (8, 9, 40, 41):
            return [b + self.u(np_ + 20)]
        if node["type"] == 16:
            return [b + self.u(b + self.u(np_ + 24) + 12 * i + 4) for i in range(self.u(np_ + 20))]
        return []

    def meshes(self, forest, model):
        b = forest["base"]
        rows = []
        for n in model["nodes"]:
            for lod, sp in enumerate(self.node_shapes(b, n)):
                if sp == b:
                    continue
                sh = self.us(sp, 10)
                for i in range(sh[4]):
                    mp = b + self.u(b + sh[5] + 4 * i)
                    mat = b + self.u(mp + 136)
                    mh = self.us(mat, 32)
                    vdp = b + self.u(mp + 152)
                    vd = self.us(vdp, 14)
                    decl = []
                    p = b + vd[0]
                    while True:
                        dec = list(struct.unpack_from(self.en + "HH4B", self.d, p))
                        p += 8
                        if dec[0] == 255:
                            break
                        decl.append(dec)
                    rows.append(dict(node=n["index"], node_name=n["name"], lod=lod, shape=sp, shape_header=sh, shape_name=self.s(b + sh[9]),
                                     mesh=mp, mesh_header=self.us(mp, 42), material=mat, material_header=mh, material_name=self.s(b + mh[31]),
                                     samplers=[self.us(b + self.u(b + mh[30] + 4 * j), 5) for j in range(mh[29])],
                                     vertex_descriptor=vd, vertex_descriptor_offset=vdp, declaration=decl))
        return rows
