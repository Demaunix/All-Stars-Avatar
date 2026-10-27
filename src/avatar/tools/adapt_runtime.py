"""One-time, recorded adaptation of the reviewed C234 Mii runtime (mii-restoration-001/runtime, review-repo-002 313010b) to the Avatar
identity. Only identity tokens change; the patch state machine, trampoline, guards and tests stay the same. Writes research/RUNTIME-ADAPTATION.json
with source/output hashes and the substitution table."""
import hashlib
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
RT = MODULE / "runtime"
RENAMES = {"src/mii_mod.cpp": "src/avatar_mod.cpp", "src/mii_pure.hpp": "src/avatar_pure.hpp",
           "tests/test_mii_native.py": "tests/test_avatar_native.py", "tests/test_production_mii.py": "tests/test_production_avatar.py"}
# ordered: specific tokens first, then the generic racer word
SUBS = [
    ("MiiHash = 0xD7BC0C4A", "AvatarHash = 0x0168AFAB"),
    ('lookup2("mii")', 'lookup2("avatar")'),
    ("assert MII == 0xD7BC0C4A", "assert AVATAR == 0x0168AFAB"),
    ("Mii (Wii racer)", "Avatar (Xbox 360 racer)"),
    ("MiiWii", "AvatarXbox"), ("MiiFixture", "AvatarFixture"), ("MiiEmu", "AvatarEmu"),
    ("Mii selection data (RacerParams Settings/Mii PlatformUnlocked WII -> ALL)", "Avatar selection data (RacerParams Settings/Avatar PlatformUnlocked X360 -> ALL)"),
    ("the Mii Settings row precedes Banjo's", "the Avatar Settings row precedes Mii's and Banjo's"),
    ("shop Item10 platform All", "shop Item9 platform All"),
    ("Settings/Mii PlatformUnlocked=ALL", "Settings/Avatar PlatformUnlocked=ALL"),
    ("never used on PC before", "never used on PC before; kept from the reviewed adapter"),
    ("MII", "AVATAR"), ("Mii", "Avatar"), ("mii", "avatar"),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    rec = dict(source="mii-restoration-001/runtime (C234 reviewed 313010b; copied byte-for-byte first)", files=[], substitutions=SUBS)
    for p in sorted(RT.rglob("*")):
        if not p.is_file() or "build" in p.parts:
            continue
        rel = p.relative_to(RT).as_posix()
        src = p.read_bytes()
        try:
            text = src.decode("utf-8")
        except UnicodeDecodeError:
            continue
        out = text
        for a, b in SUBS:
            out = out.replace(a, b)
        dst_rel = RENAMES.get(rel, rel)
        dst = RT / dst_rel
        dst.write_bytes(out.encode("utf-8"))
        if dst_rel != rel:
            p.unlink()
        rec["files"].append(dict(source=rel, output=dst_rel, source_sha256=sha(src), output_sha256=sha(out.encode()), changed=out != text))
    (MODULE / "research" / "RUNTIME-ADAPTATION.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps([(f["source"], f["output"], f["changed"]) for f in rec["files"]], indent=0))


if __name__ == "__main__":
    main()
