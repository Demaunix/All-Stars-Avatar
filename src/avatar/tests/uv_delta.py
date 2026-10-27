"""C253: locality check of the native-V correction. Raw bytes only (ZIF/ZIG pairs), no rendering, no game access.

`check_pair(baseline_zif, candidate_zif)` asserts, for the Avatar skinned meshes (material `AvatarXbox_*`, 32-byte vertices, TEXCOORD half2 at +20):
  * the ZIF is byte-identical (alignment, material, mesh and identity structures untouched: the C252 correction is preserved);
  * the ZIG has the same length and prefix and differs ONLY inside the V half-float of vertices of those meshes;
  * every candidate V is the native complement of the baseline V: |V_base + V_cand - 1| <= 2^-11 (two independently rounded halves);
  * every U half-float is unchanged.
It returns per-mesh facts. Complements (does not replace) the coordinator's verify_uv_delta.py."""
import hashlib
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.dont_write_bytecode = True
from zif import Zif  # noqa: E402

HALF_TOL = 2.0 ** -11
sha = lambda b: hashlib.sha256(b).hexdigest()


def check_pair(base_zif, cand_zif):
    base_zif, cand_zif = Path(base_zif), Path(cand_zif)
    a, b = base_zif.read_bytes(), cand_zif.read_bytes()
    assert a == b, "ZIF changed (candidate must keep every structure/offset of the baseline)"
    x, y = base_zif.with_suffix(".zig").read_bytes(), cand_zif.with_suffix(".zig").read_bytes()
    assert len(x) == len(y) and x[:4] == y[:4], "ZIG length/prefix changed"
    z = Zif(base_zif, base_zif.with_suffix(".zig"))
    allowed, rows = [], []
    for f in z.forests:
        for m in f["models"]:
            for row in z.meshes(f, m):
                if not row["material_name"].startswith("AvatarXbox_"):
                    continue
                stride, count, offset = row["vertex_descriptor"][3:6]
                assert stride == 32 and row["node"] == 0 and [0, 20, 15, 0, 5, 0] in row["declaration"]
                start = 4 + f["gpu_base"] + offset
                worst, changed = 0.0, 0
                for i in range(count):
                    p = start + stride * i + 20            # U at +20, V at +22
                    assert p + 4 <= len(x)
                    uo, vo = struct.unpack_from("<2e", x, p)
                    un, vn = struct.unpack_from("<2e", y, p)
                    assert x[p:p + 2] == y[p:p + 2], "U changed"
                    assert 0 <= vo <= 1 and 0 <= vn <= 1
                    err = abs(vo + vn - 1.0)
                    assert err <= HALF_TOL, (m["root"], i, vo, vn)
                    worst = max(worst, err)
                    changed += x[p + 2:p + 4] != y[p + 2:p + 4]
                    allowed.append((p + 2, p + 4))
                rows.append(dict(model=m["root"], material=row["material_name"], vertices=count, changed_v_values=changed,
                                 max_complement_error=worst))
    assert rows and all(r["changed_v_values"] for r in rows), "an Avatar mesh has no changed V"
    cursor = 0
    for lo, hi in sorted(allowed):
        assert lo >= cursor, "overlapping V ranges"
        assert x[cursor:lo] == y[cursor:lo], f"non-V bytes differ at or after {cursor}"
        cursor = hi
    assert x[cursor:] == y[cursor:], "non-V bytes differ in the tail"
    return dict(zif_sha256=sha(a), baseline_zig_sha256=sha(x), candidate_zig_sha256=sha(y), v_bytes_changed_only=True, meshes=rows)


if __name__ == "__main__":
    import json
    print(json.dumps(check_pair(*sys.argv[1:3]), indent=1))
