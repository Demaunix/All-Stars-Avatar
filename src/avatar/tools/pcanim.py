"""Decode PC racer skeletal animation key poses (quantized format, kind 6) and evaluate forward kinematics.

Format facts (header and channel tables re-derived from the earlier project's animation.py byte-order converter;
value semantics determined and validated here):
  header u32[6] = kind(6), ?, nodes, uv_channels, scalar_channels, value_start (forest-relative);
  flag bits: 4 per node {1: translation(3), 2: rotation quaternion xyzw(4), 4: scale(3), 8: toggle(0)};
  per channel: u16 a, u16 keys, u16 times[keys];  values per channel: f32 min, f32 range, then
  (3*keys+1) interleaved records of `components` u16, value = min + u/65535*range; record 3k is key k
  (record 0 at time 0), records 3k+1/3k+2 are curve handles (not needed for key poses).
Validation: every rotation key record is a unit quaternion (tests/test_pc_anim.py); bind-pose FK equals the
inverse of the stored inverse-bind matrices.
"""
import struct

import numpy as np


def quat_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    # row-vector convention (v' = v @ M)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
                     [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
                     [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]])


def local_matrix(t, q, s):
    M = np.eye(4)
    M[:3, :3] = np.diag(s[:3]) @ quat_matrix(np.asarray(q, float))
    M[3, :3] = t[:3]
    return M


def bind_locals(z, forest, model):
    b, h, n = forest["base"], model["header"], len(model["nodes"])
    T, Q, S = (np.array(z.f(b + h[k], n * 4)).reshape(n, 4) for k in (4, 5, 6))
    return T, Q, S


def parents(model):
    return [nd["h16"][0] if nd["h16"][0] != 0xFFFF else -1 for nd in model["nodes"]]


def world(model, T, Q, S):
    par = parents(model)
    W = [None] * len(par)
    for i in range(len(par)):
        L = local_matrix(T[i], Q[i], S[i])
        W[i] = L if par[i] < 0 else L @ W[par[i]]
    return np.array(W)


class AnimError(ValueError):
    pass


def decode(z, forest, model, anim, next_offset):
    b = forest["base"]
    d = z.d[b + anim["offset"]:b + next_offset]
    h = struct.unpack_from("<6I", d)
    if h[0] not in (4, 6):  # 6: quantized u16 with per-channel min/range; 4: same layout with raw f32 values
        raise AnimError(f"{anim['name']}: kind {h[0]} not decoded")
    n = h[2]
    if n != len(model["nodes"]):
        raise AnimError("node count mismatch")
    fw = (n * 4 + h[3] * 4 + h[4] + 31) // 32
    bits = sum(w << (32 * i) for i, w in enumerate(struct.unpack_from(f"<{fw}I", d, 24)))
    ch = [(i, bit, c) for i in range(n) for bit, c in ((1, 3), (2, 4), (4, 3), (8, 0)) if (bits >> (4 * i)) & bit]
    if h[3] or h[4]:
        raise AnimError("uv/scalar channels not handled")
    q = 24 + fw * 4
    recs = []
    for i, bit, c in ch:
        a, keys = struct.unpack_from("<2H", d, q)
        times = struct.unpack_from(f"<{keys}H", d, q + 4)
        recs.append(dict(node=i, bit=bit, comps=c, a=a, keys=keys, times=times))
        q += 4 + keys * 2
    vp = h[5] - anim["offset"]
    if any(d[q:vp]):
        raise AnimError("nonzero header padding")
    for r in recs:
        if r["comps"] == 0:
            continue
        cnt = (r["keys"] * 3 + 1) * r["comps"]
        if h[0] == 6:
            lo, rng = struct.unpack_from("<2f", d, vp)
            u = np.array(struct.unpack_from(f"<{cnt}H", d, vp + 8), float)
            r["values"] = (lo + u / 65535.0 * rng).reshape(r["keys"] * 3 + 1, r["comps"])
            vp += (8 + cnt * 2 + 3) & ~3
        else:
            r["values"] = np.array(struct.unpack_from(f"<{cnt}f", d, vp), float).reshape(r["keys"] * 3 + 1, r["comps"])
            vp += cnt * 4
    if vp != len(d):
        raise AnimError(f"value block ends at {vp}, blob is {len(d)}")
    return dict(name=anim["name"], header=h, channels=recs)


def pose(model, bind, decoded, key_fraction):
    """Local TRS at a key point: each channel uses key record round(key_fraction * keys)."""
    T, Q, S = (x.copy() for x in bind)
    for r in decoded["channels"]:
        if r["comps"] == 0:
            continue
        k = int(round(key_fraction * r["keys"]))
        v = r["values"][3 * k]
        if r["bit"] == 1:
            T[r["node"], :3] = v
        elif r["bit"] == 2:
            Q[r["node"]] = v
        elif r["bit"] == 4:
            S[r["node"], :3] = v
    return T, Q, S
