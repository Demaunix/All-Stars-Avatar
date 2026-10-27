"""Build the PC selection-layer payloads for the Xbox Avatar racer (offline; installed files are read-only inputs).

Adapted from the reviewed C234 Mii builder (mii-restoration-001/tools/build_selection.py, review-repo-002 313010b): same
SSRT edit method, same SMOS insertion/relocation tools, same PC insertion points (asserted byte-for-byte).

RacerParams.dat  Settings/Avatar PlatformUnlocked 'X360' -> pointer to the table's existing 'ALL' string (only field changed).
Shopping.dat     Item9 characterbio_avatar Platform hash Xbox360 -> All (only word changed); native filter 0x5453D0 flips for
                 the Avatar only (original PC routine emulated).
FrontEndScripts  Original Xbox 360 Rev1 Avatar registration rows (resolved-source FrontEndScripts.sbn, SHA-256 7eb772b0...),
                 operand-endian converted, inserted into the PC modules where PC keeps the equivalent stock rows:
                 fe_charselscreen tuple (\\Avatar.zif, 13 fields) + carousel literal (Xbox order: Opa-Opa, Banjo, Avatar; PC has no
                 Banjo, so after Opa-Opa); fe_network_game / fe_trackselect / fe_license_records / fe_leaderboards dictionary rows +
                 leaderboard label; fe_game_bridge identity + loading alias rows CONSTRUCTED (the Xbox module has none), following
                 the PC pattern and the shipped PC Loading/character_avatar_*.stz.
Not ported: the two Xbox-only gp_podium branches on the avatar hash (console avatar handling; PC podium uses the generic path).
Writes build/selection/{RacerParams.dat,Shopping.dat,FrontEndScripts.sbn} and research/SELECTION-BUILD.json.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
from smos_reloc import instructions, validate, insert, endian_code  # noqa: E402
from smospkg import read_package, read_package_be, package, name_hash  # noqa: E402
from aslib import Xpac, ssrt, ph  # noqa: E402
import ssrt_native  # noqa: E402

MODULE = HERE.parent
GAME = Path("A:/SteamLibrary/steamapps/common/Sonic and SEGA All Stars Racing")
XBOX_FE = Path("<codex-archive>/Documents-Codex/2026-09-10/referenced-chatgpt-conversation-this-is-an-2/work/banjo-kazooie-port-20260926/resolved-source/Scripts/FrontEndScripts.sbn")
XBOX_FE_SHA = "7eb772b050e3c740f8459b2446fb63237f6a58eec0baea604bc7d4bf555e538a"
OUT = MODULE / "build" / "selection"
AVATAR = name_hash("avatar")
sha = lambda d: hashlib.sha256(d).hexdigest()
F_PLATFORM_UNLOCKED = name_hash("PlatformUnlocked")
SHOP_UNLOCK, SHOP_PLATFORM = 0xAC5CC462, 0x874FA16E
STOCK = dict(base_xpac="c992ab834f3903563709a5fcaf21775538f492966c0e67d93dad743d7ea82b9f",
             frontend="c1f5ac488c4036e151429e1295ccbb90043064970908f14adb7a6aa5dccd9c35")
# Xbox Rev1 module row ranges (verified by evaluate(): first value is the avatar literal)
XROWS = dict(fe_network_game=(0xBEB, 0xC0B), fe_trackselect=(0xA31, 0xA3B), fe_license_records=(0x685, 0x68F), fe_leaderboards=(0x8F7, 0x90D))
XLABEL = (0xBAB, 0xBC9)
XTUPLE_START = 0x21CA
# PC stock constructor positions/counts (identical to the reviewed Mii builder; asserted below)
PCROWS = dict(fe_network_game=(0xBCB, 42), fe_trackselect=(0xA63, 42), fe_license_records=(0x6B7, 42), fe_leaderboards=(0x8E1, 42))


def field_offsets(d):
    u = lambda p: struct.unpack_from("<I", d, p)[0]
    s = lambda p: d[p:d.index(0, p)].decode()
    for i in range(u(4)):
        t = u(8 + 4 * i)
        for j in range(u(t + 68)):
            r = u(t + 72 + 4 * j)
            yield s(t + 4), s(r + 8), {u(r + 72 + 12 * k): r + 72 + 12 * k for k in range(u(r))}


def build_racerparams(orig):
    out = bytearray(orig)
    changed = []
    for table, name, fields in field_offsets(orig):
        if table == "Settings" and name == "Avatar":
            p = fields[F_PLATFORM_UNLOCKED] + 8
            old = struct.unpack_from("<I", orig, p)[0]
            assert orig[old:orig.index(0, old)] == b"X360", "Avatar platform is not X360"
            rows = ssrt(orig)["Settings"]["rows"]
            alls = sorted({f[F_PLATFORM_UNLOCKED]["bits"] for r in rows.values() for f in [r] if f[F_PLATFORM_UNLOCKED]["value"] == "ALL"})
            assert alls, "no existing ALL string"
            struct.pack_into("<I", out, p, alls[0])
            changed.append(dict(field_offset=p, old_pointer=old, new_pointer=alls[0]))
    assert len(changed) == 1
    before, after = ssrt(orig), ssrt(bytes(out))
    for table, t in before.items():
        for name, fields in t["rows"].items():
            for key, v in fields.items():
                if (table, name, key) != ("Settings", "Avatar", F_PLATFORM_UNLOCKED):
                    assert after[table]["rows"][name][key]["value"] == v["value"], (table, name, hex(key))
    assert after["Settings"]["rows"]["Avatar"][F_PLATFORM_UNLOCKED]["value"] == "ALL"
    order = list(after["Settings"]["rows"].keys())
    all_rows = [n for n in order if after["Settings"]["rows"][n][F_PLATFORM_UNLOCKED]["value"] == "ALL"]
    diff = [i for i, (a, b) in enumerate(zip(orig, out)) if a != b]
    assert len(orig) == len(out) and all(changed[0]["field_offset"] <= i < changed[0]["field_offset"] + 4 for i in diff)
    expected, fixups, details = ssrt_native.guarded_relocate(bytes(out))
    emu = ssrt_native.native_emulation(bytes(out), expected, fixups)
    return bytes(out), dict(edits=changed, changed_bytes=diff, native_relocation=emu, tables=details["table_count"], rows=details["row_count"],
                            settings_order=order, all_platform_rows_in_order=all_rows, avatar_record_slot=all_rows.index("Avatar"))


def build_shopping(orig):
    out = bytearray(orig)
    t = struct.unpack_from("<I", orig, 8)[0]
    edits = []
    for _, name, fields in field_offsets(orig):
        if SHOP_UNLOCK in fields and struct.unpack_from("<I", orig, fields[SHOP_UNLOCK] + 8)[0] == AVATAR:
            p = fields[SHOP_PLATFORM] + 8
            assert struct.unpack_from("<I", orig, p)[0] == name_hash("Xbox360")
            struct.pack_into("<I", out, p, name_hash("All"))
            edits.append(dict(row=name, platform_offset=p))
    assert len(edits) == 1
    assert all(a == b for i, (a, b) in enumerate(zip(orig, out)) if not edits[0]["platform_offset"] <= i < edits[0]["platform_offset"] + 4)
    oldrel, _, _ = ssrt_native.guarded_relocate(orig)
    newrel, fixups, _ = ssrt_native.guarded_relocate(bytes(out))
    emu = ssrt_native.native_emulation(bytes(out), newrel, fixups)
    checks = []
    for _, name, fields in field_offsets(orig):
        if SHOP_UNLOCK not in fields:
            continue
        key = struct.unpack_from("<I", orig, fields[SHOP_UNLOCK] + 8)[0]
        a = ssrt_native.native_filter(oldrel, t, key)
        c = ssrt_native.native_filter(newrel, t, key)
        if key != AVATAR:
            assert a == c, name
        checks.append(dict(row=name, unlock_id=f"{key:08X}", before=a, after=c))
    av = [c for c in checks if c["unlock_id"] == f"{AVATAR:08X}"][0]
    assert av["before"] is False and av["after"] is True
    return bytes(out), dict(edits=edits, native_relocation=emu, eligibility=checks)


def ctor(kind, n):
    return b"\x22" + bytes((kind, 0)) + struct.pack("<I", n)


def literal(h):
    return b"\x02" + struct.pack("<I", h)


def evaluate(chunk, en="<"):
    stack = []
    for r in instructions(chunk, 0, len(chunk), en):
        p, op = r["offset"], r["opcode"]
        if op in (1, 2, 3):
            stack.append((op, struct.unpack_from(en + "I", chunk, p + 1)[0]))
        elif op == 4:
            stack.append(("string", chunk[p + 1:r["end"] - 1].decode()))
        elif op == 0x31:
            stack.append(None)
        elif op == 0x22:
            kind, n = chunk[p + 1], struct.unpack_from(en + "I", chunk, p + 3)[0]
            assert n <= len(stack)
            v = stack[-n:] if n else []
            stack = stack[:-n] if n else stack
            stack.append(dict(zip(v[::2], v[1::2])) if kind == 9 else tuple(v))
        else:
            raise AssertionError((hex(p), hex(op)))
    return stack


def expression_before(data, ctor_at):
    ops = instructions(data)
    idx = next(i for i, r in enumerate(ops) if r["offset"] == ctor_at)
    j = idx
    while j > 0 and ops[j - 1]["opcode"] in (1, 2, 3, 4, 0x31, 0x22):
        j -= 1
    vals = evaluate(data[ops[j]["offset"]:ops[idx]["end"]])
    assert len(vals) == 1 and isinstance(vals[0], dict), vals[:3]
    return vals[0]


def xbox_rows(xb, name, start, end):
    raw = xb[name][start:end]
    out = endian_code(raw, 0, len(raw))
    assert endian_code(out, 0, len(out), "<") == raw, "endian conversion not involutive"
    assert evaluate(raw, ">") == evaluate(out), "conversion changed values"
    return raw, out


def tuple_end(xd, start):
    """End of the avatar charsel registration: smallest end where the expression is (avatar literal, 13-field tuple)."""
    ops = instructions(xd)
    i0 = next(i for i, r in enumerate(ops) if r["offset"] == start)
    for k in range(i0, len(ops)):
        try:
            v = evaluate(xd[start:ops[k]["end"]], ">")
        except AssertionError:
            continue
        if len(v) == 2 and v[0] == (2, AVATAR) and isinstance(v[1], tuple) and len(v[1]) == 13:
            return ops[k]["end"]
    raise AssertionError("avatar registration tuple not found")


def apply(original, additions, counts):
    d = bytearray(original)
    for at, kind, old, new in counts:
        assert original[at:at + 7] == ctor(kind, old), (hex(at), original[at:at + 7].hex())
        assert not any(r.get("target") == at for r in instructions(original)), "constructor is a branch target"
        struct.pack_into("<I", d, at + 3, new)
    out, report = insert(bytes(d), additions)
    a, b = instructions(original), instructions(out)

    def loc(p):
        return p + sum(len(c) for at, c in additions.items() if at <= p)
    newops = {r["offset"]: r for r in b}
    for r in a:
        nr = newops[loc(r["offset"])]
        assert r["opcode"] == nr["opcode"]
        if "target" in r:
            assert loc(r["target"]) == nr["target"]
        elif r["offset"] not in {x[0] for x in counts}:
            assert original[r["offset"]:r["end"]] == out[nr["offset"]:nr["end"]]
    return out, report


def sumo_keys():
    import zlib
    x = Xpac(str(GAME / "Resource/SumoToolResources.xpac"))
    keys = {}

    def sections(d):
        h = struct.unpack_from("<18I", d)
        out = []
        for i in range(6):
            p, n, c = h[i * 3:i * 3 + 3]
            b = zlib.decompress(d[p:p + c]) if c else b""
            assert len(b) == n
            out.append(b)
        return out
    for r in x.rows:
        try:
            ss = sections(x.data(r[0]))
        except Exception:
            continue
        for si in (0, 3):
            p = 0
            d = ss[si]
            while p < len(d):
                n = struct.unpack_from("<I", d, p + 4)[0]
                c = d[p:p + n]
                if c[:4] == b"PTEX":
                    for j in range(struct.unpack_from("<I", c, 16)[0]):
                        keys.setdefault(struct.unpack_from("<I", c, 28 + j * 40)[0], set()).add(r[0])
                elif c[:4] == b"TEXT":
                    for j in range(struct.unpack_from("<I", c, 24)[0]):
                        keys.setdefault(struct.unpack_from("<I", c, 28 + j * 8)[0], set()).add(r[0])
                p += n
    return x, keys


def build_frontend(stock):
    rows = read_package(stock)
    assert package(stock) == stock, "no-op repack must be byte exact"
    records = {r["name"]: r for r in rows}
    xbytes = XBOX_FE.read_bytes()
    assert sha(xbytes) == XBOX_FE_SHA, "Xbox FrontEndScripts source is not the pinned file"
    xb = {r["name"]: r["data"] for r in read_package_be(xbytes)}
    changes, audit = {}, {}

    def commit(name, adds, counts, extra):
        rec = records[name]
        out, check = apply(rec["data"], adds, counts)
        validate(out)
        aliases = [r["name"] for r in rows if r["start"] == rec["start"]]
        changes.update({a: out for a in aliases})
        audit[name] = dict(aliases=aliases, source_sha256=sha(rec["data"]), output_sha256=sha(out), relocation=check, **extra)
        return out

    # fe_charselscreen: tuple appended to the 28-pair registration dictionary; carousel literal after Opa-Opa.
    xd = xb["fe_charselscreen"]
    end = tuple_end(xd, XTUPLE_START)
    raw, row = xbox_rows(xb, "fe_charselscreen", XTUPLE_START, end)
    sel = evaluate(row)
    assert sel[0] == (2, AVATAR) and sel[1][1] == ("string", "\\Avatar.zif")
    pc = records["fe_charselscreen"]["data"]
    assert pc[0x2528:0x252F] == ctor(9, 56) and pc[0x259B:0x25A2] == ctor(8, 21)
    pc_list = [v[1] for v in evaluate(pc[0x2532:0x259B])]
    x_ops = instructions(xd)
    lit_at = 0x268A
    assert struct.unpack_from(">I", xd, lit_at + 1)[0] == AVATAR and xd[lit_at] == 2
    k = next(i for i, r in enumerate(x_ops) if r["offset"] == lit_at)
    x_list = []
    j = k
    while x_ops[j]["opcode"] == 2:
        j -= 1
    j += 1
    while x_ops[j]["opcode"] == 2:
        x_list.append(struct.unpack_from(">I", xd, x_ops[j]["offset"] + 1)[0])
        j += 1
    banjo = name_hash("banjo")
    assert [v for v in x_list if v not in (AVATAR, banjo)] == pc_list, "PC carousel differs from Xbox carousel apart from Banjo/Avatar"
    prev = x_list[x_list.index(AVATAR) - 1]
    if prev == banjo:
        prev = x_list[x_list.index(banjo) - 1]
    assert prev == name_hash("opaopa")
    at_list = 0x2532 + 5 * (pc_list.index(prev) + 1)
    commit("fe_charselscreen", {0x2528: row, at_list: literal(AVATAR)}, [(0x2528, 9, 56, 58), (0x259B, 8, 21, 22)],
           dict(xbox_source=dict(offset=hex(XTUPLE_START), end=hex(end), bytes=len(raw), sha256=sha(raw)), registration=repr(sel),
                carousel_after="opaopa", xbox_carousel_neighbours="opaopa, banjo, avatar", carousel_insert_offset=hex(at_list)))
    for name, (xs, xe) in XROWS.items():
        raw, row = xbox_rows(xb, name, xs, xe)
        vals = evaluate(row)
        assert vals[0] == (2, AVATAR), (name, vals)
        at, count = PCROWS[name]
        data = records[name]["data"]
        size = xe - xs
        old = evaluate(data[at - 21 * size:at])
        assert len(old) == 42 and (2, AVATAR) not in old[::2], name
        adds, counts = {at: row}, [(at, 9, count, count + 2)]
        extra = dict(xbox_source=dict(offset=hex(xs), bytes=size, sha256=sha(raw)), row=repr(vals))
        if name == "fe_leaderboards":
            lraw, label = xbox_rows(xb, name, *XLABEL)
            assert evaluate(label) == [((2, AVATAR), ("string", "CHARACTER_AVATAR"))]
            assert data[0xB5B:0xB62] == ctor(8, 21)
            adds[0xB5B] = label
            counts.append((0xB5B, 8, 21, 22))
            extra["label_source"] = dict(offset=hex(XLABEL[0]), bytes=len(lraw), sha256=sha(lraw))
        commit(name, adds, counts, extra)
    # fe_game_bridge: identity and loading rows (constructed; the Xbox module has none for the Avatar)
    x = Xpac(str(GAME / "Resource/SumoToolResources.xpac"))
    alias = "character_avatar"
    loading_members = [f".\\Resource/SumoToolResources/Loading/{alias}_{lang}.stz" for lang in ("EN", "FR", "GE", "IT", "JP", "SP", "US")]
    present = [m for m in loading_members if ph(m) in x.by_key]
    assert present, "no PC loading screen for the Avatar"
    assert AVATAR not in [struct.unpack_from(">I", xb["fe_game_bridge"], i)[0] for i in range(len(xb["fe_game_bridge"]) - 3)], "Xbox bridge unexpectedly has avatar rows"
    data = records["fe_game_bridge"]["data"]
    assert data[0x275:0x27C] == ctor(9, 40) and data[0x7DA:0x7E1] == ctor(9, 40)
    ident = expression_before(data, 0x275)
    loads = expression_before(data, 0x7DA)
    assert len(ident) == 20 and len(loads) == 20
    assert (2, AVATAR) not in ident and (2, AVATAR) not in loads
    identity = literal(AVATAR) + literal(AVATAR) + ctor(8, 1)
    loading = literal(AVATAR) + b"\x04" + alias.encode() + b"\0"
    commit("fe_game_bridge", {0x275: identity, 0x7DA: loading}, [(0x275, 9, 40, 42), (0x7DA, 9, 40, 42)],
           dict(constructed=True, reason="Xbox fe_game_bridge has no Avatar identity/loading rows; PC pattern (20 stock singleton rows) followed",
                loading_alias=alias, loading_members_present=[m.split("/")[-1] for m in present]))
    result = package(stock, changes)
    newrows = read_package(result)
    for r in newrows:
        validate(r["data"])
        if r["name"] not in changes:
            assert r["data"] == records[r["name"]]["data"]
    for name in audit:
        src = xb.get(name, b"")
        audit[name]["avatar_identity_references"] = dict(xbox=src.count(struct.pack(">I", AVATAR)), pc_stock=records[name]["data"].count(struct.pack("<I", AVATAR)),
                                                         candidate=changes[name].count(struct.pack("<I", AVATAR)))
    _, keys = sumo_keys()
    tup = evaluate(row if False else changes["fe_charselscreen"][0x2528:0x2528 + (end - XTUPLE_START)])[1]
    flat = []

    def walk(v):
        if isinstance(v, tuple) and len(v) == 2 and v[0] == 2:
            flat.append(v[1])
        elif isinstance(v, tuple):
            for x_ in v:
                walk(x_)
    walk(tup)
    ui = dict(keys=[f"{k:08X}" for k in flat], present_in_pc_stz=[f"{k:08X}" for k in flat if k in keys],
              absent_from_pc_stz=[f"{k:08X}" for k in flat if k not in keys])
    return result, dict(modules=audit, unchanged_modules=sum(r["name"] not in changes for r in rows), ui_key_audit=ui,
                        not_ported=dict(gp_podium="two Xbox-only branches comparing the racer with the avatar hash (0x848, 0xF1B); console avatar handling, not ported"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = GAME / "Resource/Base.xpac"
    fe = GAME / "Scripts/FrontEndScripts.sbn"
    base_sha = sha(base.read_bytes())
    fe_bytes = fe.read_bytes()
    assert base_sha == STOCK["base_xpac"], "installed Base.xpac is not the pinned stock file"
    assert sha(fe_bytes) == STOCK["frontend"], "installed FrontEndScripts.sbn is not the pinned stock file"
    x = Xpac(str(base))
    rp0, sh0 = x.raw(0xFECF5CFC), x.raw(0xAA924689)
    assert rp0[:4] == b"SSRT" and sh0[:4] == b"SSRT", "members are expected raw (uncompressed)"
    rp, rp_rep = build_racerparams(rp0)
    sh, sh_rep = build_shopping(sh0)
    fe2, fe_rep = build_frontend(fe_bytes)
    for n, d in (("RacerParams.dat", rp), ("Shopping.dat", sh), ("FrontEndScripts.sbn", fe2)):
        (OUT / n).write_bytes(d)
    rep = dict(inputs=dict(base_xpac_sha256=base_sha, frontend_sha256=STOCK["frontend"], racerparams_stored_sha256=sha(rp0), shopping_stored_sha256=sha(sh0),
                           xbox_frontend_sha256=XBOX_FE_SHA, xbox_frontend_path=str(XBOX_FE)),
               outputs={n: dict(sha256=sha((OUT / n).read_bytes()), bytes=(OUT / n).stat().st_size) for n in ("RacerParams.dat", "Shopping.dat", "FrontEndScripts.sbn")},
               racerparams=rp_rep, shopping=sh_rep, frontend=fe_rep)
    (MODULE / "research" / "SELECTION-BUILD.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep["outputs"], indent=1))
    print("avatar slot", rp_rep["avatar_record_slot"], "ALL rows", len(rp_rep["all_platform_rows_in_order"]))
    print("ui keys", json.dumps(fe_rep["ui_key_audit"]))
    print("unchanged FE modules", fe_rep["unchanged_modules"], {k: v["avatar_identity_references"] for k, v in fe_rep["modules"].items()})


if __name__ == "__main__":
    main()
