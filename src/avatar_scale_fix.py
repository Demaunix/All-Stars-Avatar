"""Avatar size fix: set the driver model's BASE joint bind scale to 1/s in the race and select files, s = the bake's PC-rig fit.

Why: the Xbox 360 game draws the avatar at native XAvatar size (no scale; see notes/avatar-fit.md), but the bake fits the avatar
to the PC rig at s (src/avatar/research/SPACE-FIT.json: `uniform_garment_frame` 2.1679 for the C253 / rp4 / rp5 bakes,
`uniform_rp6` for rp6). A uniform BASE bind scale of 1/s shrinks everything below the hips back to XAvatar size while the pelvis
keeps the position the rig clips give it. No clip animates BASE scale (checked: bit 4 never set on BASE).

usage: avatar_scale_fix.py <installed payload dir> <out dir> [rp6|c253]
Reads the stored (zlib) payloads, writes patched raw + stored copies and fix-a-report.json. Only 12 bytes per model change.
"""
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import pcanim  # noqa: E402
import zif  # noqa: E402

SPACE_FIT = Path(__file__).resolve().parent / "avatar" / "research" / "SPACE-FIT.json"
SCALE_KEY = {"rp6": "uniform_rp6", "c253": "uniform_garment_frame"}     # = bake_avatar.SPACE_FIT_KEY
# Driver (race + select) only. The four Party Power assistants stay at rig size: the official Xbox trailer
# (YouTube t77AoLLh4JU, 56.6-57.6 s) shows them big, holding the kart, feet on the road; shrunk they looked tiny in game.
AVATAR_MODELS = {"Avatar"}
FILES = {"Avatar.zif": "race model (with 4 All-Star assistants)", "AvatarSelect.zif": "select-screen model"}
sha = lambda b: hashlib.sha256(b).hexdigest()


def fit_scale(decode: str = "rp6") -> float:
    """The uniform PC-rig scale the bake used for this decode profile (research/SPACE-FIT.json)."""
    return float(json.loads(SPACE_FIT.read_text())[SCALE_KEY[decode]]["scale"])


def patch(raw: bytes, tmp: Path, k: np.float32) -> tuple[bytes, list]:
    """Return the raw zif with BASE bind scale set to k in each avatar model, and a per-model report."""
    tmp.write_bytes(raw)
    z = zif.Zif(str(tmp))
    out = bytearray(raw)
    rep = []
    for f in z.forests:
        for m in f["models"]:
            if m["root"] not in AVATAR_MODELS:
                continue
            names = [n["name"] for n in m["nodes"]]
            bi = names.index("BASE")
            assert bi == 1 and pcanim.parents(m)[bi] == 0, (m["root"], "unexpected BASE position")
            off = f["base"] + m["header"][6] + bi * 16          # S array: n x float32[4]
            old = struct.unpack_from("<4f", raw, off)
            assert np.allclose(old[:3], 1.0), (m["root"], "BASE bind scale is not 1", old)
            struct.pack_into("<3f", out, off, k, k, k)
            rep.append(dict(model=m["root"], file_offset=off, old=list(old), new=[float(k)] * 3 + [old[3]]))
    return bytes(out), rep


def verify(raw_old: bytes, raw_new: bytes, tmp_old: Path, tmp_new: Path, rep: list, k: np.float32) -> list:
    """Check only the intended bytes changed and every joint below BASE moved to BASE + k*(p - BASE) in the bind pose."""
    diff = [i for i in range(len(raw_old)) if raw_old[i] != raw_new[i]] if len(raw_old) == len(raw_new) else None
    assert diff is not None, "size changed"
    allowed = {r["file_offset"] + j for r in rep for j in range(12)}
    assert set(diff) <= allowed, "bytes outside the BASE scale fields changed"
    tmp_old.write_bytes(raw_old)
    tmp_new.write_bytes(raw_new)
    zo, zn = zif.Zif(str(tmp_old)), zif.Zif(str(tmp_new))
    checks = []
    for fo, fn in zip(zo.forests, zn.forests):
        for mo, mn in zip(fo["models"], fn["models"]):
            if mo["root"] not in AVATAR_MODELS:
                continue
            Wo = pcanim.world(mo, *pcanim.bind_locals(zo, fo, mo))
            Wn = pcanim.world(mn, *pcanim.bind_locals(zn, fn, mn))
            po, pn = Wo[:, 3, :3], Wn[:, 3, :3]
            b = po[1]
            want = b + float(k) * (po - b)
            err = float(np.abs(pn[2:] - want[2:]).max())
            names = [n["name"] for n in mo["nodes"]]
            h = names.index("HEAD")
            assert err < 1e-4 and np.allclose(pn[:2], po[:2]), (mo["root"], err)
            checks.append(dict(model=mo["root"], max_err=err, head_y_old=float(po[h, 1]), head_y_new=float(pn[h, 1]), base_y=float(b[1])))
    return checks


def main(src: str, out: str, decode: str = "rp6") -> None:
    """Patch both payloads with 1/s of the given decode profile's fit, recompress like build_package.py (zlib level 9), write a report."""
    s = fit_scale(decode)
    k = np.float32(1.0 / s)
    src_p, out_p = Path(src), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    report = dict(decode=decode, k=float(k), fit_scale=s, files={})
    for name, what in FILES.items():
        stored_old = (src_p / name).read_bytes()
        raw_old = zlib.decompress(stored_old)
        assert zlib.compress(raw_old, 9) == stored_old, "stored payload is not plain zlib level 9"
        raw_new, rep = patch(raw_old, out_p / f"_tmp_{name}", k)
        checks = verify(raw_old, raw_new, out_p / f"_old_{name}", out_p / f"_new_{name}", rep, k)
        stored_new = zlib.compress(raw_new, 9)
        assert zlib.decompress(stored_new) == raw_new
        (out_p / f"raw-{name}").write_bytes(raw_new)
        (out_p / name).write_bytes(stored_new)
        for t in out_p.glob(f"_*_{name}"):
            t.unlink()
        report["files"][name] = dict(what=what, stored_old_sha256=sha(stored_old), stored_new_sha256=sha(stored_new), stored_new_size=len(stored_new),
                                     raw_old_sha256=sha(raw_old), raw_new_sha256=sha(raw_new), patched=rep, checks=checks)
    (out_p / "fix-a-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"BASE scale 1/{s:.6f} = {float(k):.6f} ({decode})")
    for n, r in report["files"].items():
        print(n, r["stored_old_sha256"][:12], "->", r["stored_new_sha256"][:12], "models", [c["model"] for c in r["checks"]])
        for c in r["checks"]:
            print("   ", c)


if __name__ == "__main__":
    main(*sys.argv[1:4])
