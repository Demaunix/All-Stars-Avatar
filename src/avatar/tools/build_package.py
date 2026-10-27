"""Assemble the Avatar (Xbox 360 racer) loader package from the verified build outputs (deterministic; installed game files are read-only).

Adapted from the reviewed C234 package builder (mii-restoration-001/tools/build_package.py). Differences: the select model is ADDED to
Select.xpac (PC has no Select/Avatar member; loader op `add`), the racer file replaces Racers/Avatar.zif/.zig.

package/mods/Avatar (Xbox 360 racer)/{mod.ini, plan.json, allstars-mod.json, README.txt, CHECKSUMS.sha256, runtime/AvatarXbox.dll, payloads/*}
package/Avatar (Xbox 360 racer) - candidate <version>.zip   (same tree under mods/...)
"""
import hashlib
import json
import sys
import zipfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
from xpac_compose import compose  # noqa: E402
from aslib import ph  # noqa: E402

MODULE = HERE.parent
GAME = Path("A:/SteamLibrary/steamapps/common/Sonic and SEGA All Stars Racing")
VERSION = "0.1.0-alpha.2"
FOLDER = "Avatar (Xbox 360 racer)"
MOD_ID = "avatar-xbox360"
OUT = MODULE / "package"
sha = lambda b: hashlib.sha256(b).hexdigest()
STOCK = {"Resource/Base.xpac": "c992ab834f3903563709a5fcaf21775538f492966c0e67d93dad743d7ea82b9f",
         "Resource/Select.xpac": "5b12ab70e10dc9af36527c31bff42691e099d9c5a65a9239217fc5515c3da310",
         "Resource/Racers.xpac": "4e2b76cb1a68c36369caef210f927f0e1d2c0da828cfcc0a6483bb328cd87abb",
         "Scripts/FrontEndScripts.sbn": "c1f5ac488c4036e151429e1295ccbb90043064970908f14adb7a6aa5dccd9c35"}


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stored(p):
    data = p.read_bytes()
    z = zlib.compress(data, 9)
    assert zlib.decompress(z) == data
    return z


def main():
    for rel, want in STOCK.items():
        assert file_sha(GAME / rel) == want, f"installed {rel} is not the pinned stock file"
    bake, sel, selm = MODULE / "build" / "bake", MODULE / "build" / "selection", MODULE / "build" / "select"
    payloads = {
        "payloads/RacerParams.dat": (sel / "RacerParams.dat").read_bytes(),
        "payloads/Shopping.dat": (sel / "Shopping.dat").read_bytes(),
        "payloads/FrontEndScripts.sbn": (sel / "FrontEndScripts.sbn").read_bytes(),
        "payloads/Avatar.zif": stored(bake / "Avatar.zif"),
        "payloads/Avatar.zig": stored(bake / "Avatar.zig"),
        "payloads/AvatarSelect.zif": stored(selm / "Avatar.zif"),
        "payloads/AvatarSelect.zig": stored(selm / "Avatar.zig"),
    }
    members = {  # archive -> [(kind, key, member name, payload)]
        "Resource/Base.xpac": [("replace", 0xFECF5CFC, "RacerParams.dat", "payloads/RacerParams.dat"), ("replace", 0xAA924689, "Shopping.dat", "payloads/Shopping.dat")],
        "Resource/Racers.xpac": [("replace", ph(".\\Resource/Racers/Avatar.zif"), "Racers/Avatar.zif", "payloads/Avatar.zif"),
                                 ("replace", ph(".\\Resource/Racers/Avatar.zig"), "Racers/Avatar.zig", "payloads/Avatar.zig")],
        "Resource/Select.xpac": [("add", ph(".\\Resource/Select/Avatar.zif"), "Select/Avatar.zif", "payloads/AvatarSelect.zif"),
                                 ("add", ph(".\\Resource/Select/Avatar.zig"), "Select/Avatar.zig", "payloads/AvatarSelect.zig")],
    }
    groups, composed = [], {}
    for arc, ops in members.items():
        c = compose(GAME / arc, [dict(kind=kind, key=k, payload=payloads[p]) for kind, k, _, p in ops])
        composed[arc] = c
        opl = []
        for kind, k, _, p in ops:
            o = {"kind": kind, "key": f"{k:08X}", "payload": p, "payload_sha256": sha(payloads[p]), "stored_size": len(payloads[p]), "entry_flags": 0}
            if kind == "replace":
                o["base_stored_sha256"] = c["base_stored_sha256"][f"{k:08X}"]
            opl.append(o)
        groups.append({"path": arc, "format": "xpac", "base_sha256": STOCK[arc], "operations": opl,
                       "expected_composed_sha256": c["sha256"], "expected_composed_size": c["size"], "expected_member_count": c["member_count"]})
    groups.append({"path": "Scripts/FrontEndScripts.sbn", "format": "file", "base_sha256": STOCK["Scripts/FrontEndScripts.sbn"],
                   "payload": "payloads/FrontEndScripts.sbn", "payload_sha256": sha(payloads["payloads/FrontEndScripts.sbn"])})
    plan = {"schema": 1, "archive_groups": groups}
    plan_bytes = (json.dumps(plan, indent=2) + "\n").encode("ascii")
    runtime = (MODULE / "runtime" / "build" / "AvatarXbox.dll").read_bytes()
    ini = ("[mod]\r\nschema=1\r\nid=%s\r\nname=%s\r\nversion=%s\r\nruntime=runtime\\AvatarXbox.dll\r\nplan=plan.json\r\nplan_sha256=%s\r\nruntime_sha256=%s\r\n"
           % (MOD_ID, FOLDER, VERSION, sha(plan_bytes), sha(runtime))).encode("ascii")
    files = {"mod.ini": ini, "plan.json": plan_bytes, "runtime/AvatarXbox.dll": runtime, **payloads,
             "README.txt": (MODULE / "docs" / "README.txt").read_bytes()}
    manifest = json.loads((MODULE / "docs" / "allstars-mod.template.json").read_text(encoding="utf-8"))
    manifest["version"] = VERSION
    manifest["touches"]["archive_members"] = [dict(archive=arc, op=kind, key=f"{k:08X}", name=n, payload=p, payload_sha256=sha(payloads[p]))
                                              for arc, ops in members.items() for kind, k, n, p in ops]
    manifest["touches"]["scripts"][0]["payload_sha256"] = sha(payloads["payloads/FrontEndScripts.sbn"])
    manifest["artifacts"] = [{"path": k, "sha256": sha(v), "bytes": len(v)} for k, v in sorted(files.items())]
    files["allstars-mod.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    files["CHECKSUMS.sha256"] = "".join("%s *%s\n" % (sha(v), k) for k, v in sorted(files.items())).encode("ascii")
    root = OUT / "mods" / FOLDER
    if root.exists():
        for p in sorted(root.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    for k, v in files.items():
        p = root / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(v)
    zpath = OUT / f"{FOLDER} - candidate {VERSION}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for k in sorted(files):
            zi = zipfile.ZipInfo(f"mods/{FOLDER}/{k}", date_time=(2026, 10, 15, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, files[k])
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and sorted(z.namelist()) == sorted(f"mods/{FOLDER}/{k}" for k in files)
        for k in files:
            assert sha(z.read(f"mods/{FOLDER}/{k}")) == sha(files[k])
    report = dict(version=VERSION, mod_id=MOD_ID, folder=FOLDER, zip=str(zpath.relative_to(MODULE)), zip_bytes=zpath.stat().st_size, zip_sha256=sha(zpath.read_bytes()),
                  plan_sha256=sha(plan_bytes), runtime_sha256=sha(runtime), composed={k: {kk: vv for kk, vv in v.items() if kk != "base_stored_sha256"} for k, v in composed.items()},
                  files={k: dict(sha256=sha(v), bytes=len(v)) for k, v in sorted(files.items())})
    (OUT / "package-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
