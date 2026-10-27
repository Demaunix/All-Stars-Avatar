"""Surface charts for a composed avatar: single-UV/single-texture representation of the console's multi-UV shading.

Why: garment UV0 is heavily shared (mirrored halves/panels: 80-100% of used texels map to several 3D surfaces in the
preset garments, tools/uv_sharing.py -> research/UV-SHARING.json; research/SURFACE-NOTES.md), so decals that use another UV set cannot be baked into UV0. Each part becomes charts:
  head  : one chart per head mesh in UV0 (low overlap), six-layer bake (avatar_compose.bake_layers);
  solid : body/hair default colour (one chart);
  base  : garment triangles not touched by a decal; image = base texture resampled over the UV0 wrap domain
          (exactly what wrap addressing samples);
  decal : garment triangles where a decal has coverage; chart in that decal's UV set, image = per-texel
          base(UV0, wrap) composited with decal(UVk, clamp-to-border). Refused if the decal UV set overlaps.
The reference for fidelity tests is the per-pixel multi-UV shading (reference_shade) of the same part.
"""
import numpy as np
from PIL import Image

import avatar_compose as AC
import avatar_model as AM


class SurfaceError(Exception):
    pass


SMALL_ROLES = ("hat", "glasses", "gloves", "wrist", "ring", "earrings")


def decal_layers(part):
    out = []
    for s in part.get("samplers", [])[1:]:
        t = part["textures"][s["texture"]]
        if t is None or s["texture"] == 0:
            continue
        if np.ptp(t[..., 3]) > 25:
            out.append((t.astype(float) / 255.0, s["uv_set"], s["texture"]))
    return out


def resample_wrap(img, lo, hi, res):
    """Image over UV domain [lo,hi] sampled from `img` with wrap addressing (bilinear)."""
    u = lo[0] + (np.arange(res) + 0.5) / res * (hi[0] - lo[0])
    v = lo[1] + (np.arange(res) + 0.5) / res * (hi[1] - lo[1])
    U, V = np.meshgrid(u, v)
    s = AC.sample(img, np.stack([U.ravel(), V.ravel()], 1), wrap=True)
    return s.reshape(res, res, 4)


def tri_touches(decal_img, uvk, tris, thr=0.02, grow=2):
    """Conservative and exact: a triangle has decal coverage if it intersects (closed sets, separating-axis test) the
    cell of any texel within `grow` texels of a texel with alpha > thr (bilinear support plus a margin). Thin and
    degenerate triangles are handled (zero-length edges give no separating axis). Outside [0,1] the decal is transparent
    (clamp-to-border): no cells exist there. Over-inclusion only moves base-coloured triangles into the decal chart,
    whose bake composites base + decal per texel."""
    if not np.isfinite(uvk[np.unique(tris)]).all():
        raise SurfaceError("non-finite decal texcoords")   # refuse rather than silently classify as uncovered
    h, w = decal_img.shape[:2]
    m = decal_img[..., 3] > thr
    for _ in range(grow):
        mp = np.pad(m, 1)
        m = mp[1:-1, 1:-1] | mp[:-2, 1:-1] | mp[2:, 1:-1] | mp[1:-1, :-2] | mp[1:-1, 2:]
        m |= mp[:-2, :-2] | mp[:-2, 2:] | mp[2:, :-2] | mp[2:, 2:]
    oy, ox = np.nonzero(m)
    cw, ch = 1.0 / w, 1.0 / h                             # texel cell size in UV
    bx0, by0 = ox * cw, oy * ch
    hit = np.zeros(len(tris), bool)
    eps = 1e-12
    for t, tri in enumerate(tris):
        V = uvk[tri]
        lo, hi = V.min(0), V.max(0)
        sel = (bx0 + cw >= lo[0] - eps) & (bx0 <= hi[0] + eps) & (by0 + ch >= lo[1] - eps) & (by0 <= hi[1] + eps)
        if not sel.any():
            continue
        x0, y0 = bx0[sel], by0[sel]
        ok = np.ones(len(x0), bool)
        for k in range(3):
            e = V[(k + 1) % 3] - V[k]
            n = np.array([-e[1], e[0]])
            pt = V @ n
            c0 = x0 * n[0] + y0 * n[1]
            bmin = c0 + min(0.0, n[0] * cw) + min(0.0, n[1] * ch)
            bmax = c0 + max(0.0, n[0] * cw) + max(0.0, n[1] * ch)
            ok &= ~((bmax < pt.min() - eps) | (bmin > pt.max() + eps))
        hit[t] = ok.any()
    return hit


def overlap_fraction(ident, uv, tris, res=256, eps=0.03):
    """Fraction of texels of `uv` shared by triangles whose identity (per-vertex array, e.g. base UV0 or position)
    differs by more than eps: a conflict for baking. Mirrored pairs that also share base UV0 are not conflicts."""
    py, px, tt, bb = AC.raster_uv(uv, tris, res)
    if len(py) == 0:
        return 0.0
    key = py * res + px
    cen = (ident[tris[tt]] * bb[:, :, None]).sum(1)
    order = np.argsort(key, kind="stable")
    key, cen = key[order], cen[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    groups = np.split(np.arange(len(key)), bounds)
    bad = sum(1 for g in groups if len(g) > 1 and np.ptp(cen[g], axis=0).max() > eps)
    return bad / len(groups)


def charts_for_part(p, part_record, res_base=256, res_decal=256):
    """p: composed part (avatar_compose output entry); part_record: load_part() result for garments (or None)."""
    me = p["mesh"]
    tris = p.get("keep_triangles", me["triangles"])
    P = AM.positions(me)
    if p["role"].startswith("head"):
        lo, hi = (np.array(x) for x in p["uv_domain"])
        return [dict(name=p["role"], kind="head", tris=tris, uv=(me["texcoords"][:, 0:2] - lo) / (hi - lo), image=p["texture"])]
    base_rgba = part_record["textures"][0].astype(float) / 255.0 if part_record else None
    solid = AC.solid_colour(base_rgba) if base_rgba is not None else None
    decs = decal_layers(part_record) if part_record else []
    # step 4 (2026-10-21): texture 1 is the item's tint mask. With alpha variance it is listed by decal_layers; it is a decal only
    # in the sense that it lives on UV set 1. A coloured avatar (avatar_colours.tint_part set `tint_uv1`) bakes it as a UV-1 layer
    # with the console's three-colour mix; without colours it is dropped (the C253 presets never have such a mask).
    tint = part_record.get("tint_uv1") if part_record else None
    layers = [(img, k, None) for img, k, ti in decs if ti != 1]          # real decals; the mask is never a plain decal
    if tint is not None and part_record and len(part_record["textures"]) > 1 and part_record["textures"][1] is not None:
        mk = part_record["textures"][1].astype(float) / 255.0
        if AC.solid_colour(mk) is None:                                   # an image mask on its own UV set (hair ties: alpha 1, R = hair weight)
            k1 = next((s["uv_set"] for s in part_record["samplers"] if s["texture"] == 1), 1)
            layers.append((mk, k1, AC.mix3(*tint)))
    decs = [(img, k, None) for img, k, _ in layers]
    lo, hi = AC.uv_domain(me, 0, tris)
    uv0n = (me["texcoords"][:, 0:2] - lo) / (hi - lo)
    if solid is not None and not decs:
        img = np.tile(solid, (32, 32, 1))
        return [dict(name=p["role"], kind="solid", tris=tris, uv=np.clip(uv0n, 0, 1), image=img, colour=solid)]
    if len({k for _, k, _ in decs}) > 1:
        raise SurfaceError(f"{p['role']}: decals on several UV sets are not supported")
    charts = []
    # step 4 (2026-10-21): the small items get 128-cells in the atlas (their diffuses are 128x64 or smaller); the C253 roles keep 256
    small = p["role"].startswith(SMALL_ROLES)
    dtri = np.zeros(len(tris), bool)
    if decs:
        img_d, k, tex_index = decs[0]
        uvk = me["texcoords"][:, 2 * k:2 * k + 2]
        for d_img, _, _ in decs:
            dtri |= tri_touches(d_img, uvk, tris)
    rest = tris[~dtri]
    if len(rest):
        charts.append(dict(name=p["role"] + ":base", kind="base", tris=rest, uv=uv0n, image=resample_wrap(base_rgba, lo, hi, res_base),
                           domain=(lo.tolist(), hi.tolist()), small=small))
    if dtri.any():
        dt = tris[dtri]
        uvk = me["texcoords"][:, 2 * k:2 * k + 2]
        # conflict = same decal texel but different base texel (identity = interpolated UV0, eps = 2 base texels)
        ov = overlap_fraction(me["texcoords"][:, 0:2], uvk, dt, eps=2.0 / base_rgba.shape[0])
        if ov > 0.02 and solid is None:                                    # a solid base cannot conflict under a shared decal texel
            if not p.get("allow_shared_decal"):
                raise SurfaceError(f"{p['role']}: decal texels map to different base texels ({ov:.1%}); cannot chart")
            # step 4 fallback (2026-10-21): bake the decal into the shared UV0 chart instead. Where UV0 is mirrored (both shoes
            # share texels) the decal lands on every surface sharing them; the Xbox sneakers' logo is on both shoes anyway.
            img = AC.bake_layers(me, 0, res_base, base_rgba, layers, domain=(lo, hi), base_sample_pair=0)[0]
            return [dict(name=p["role"] + ":base", kind="base", tris=tris, uv=uv0n, image=img, domain=(lo.tolist(), hi.tolist()), small=small,
                         fallback=f"decal baked into shared UV0 (decal-space overlap {ov:.1%})", uv_set=k)]
        klo = np.floor(uvk[np.unique(dt)].min(0) * 64) / 64
        khi = np.ceil(uvk[np.unique(dt)].max(0) * 64) / 64
        me_k = dict(me)
        tc = me["texcoords"].copy()
        me_k["texcoords"] = tc
        me_k["triangles"] = dt
        # raster in UVk, base sampled at interpolated UV0 (wrap), decals lerped by alpha (clamp-to-border)
        img, _ = AC.bake_layers(me_k, k, res_decal, base_rgba, layers, domain=(klo, khi), base_sample_pair=0)
        charts.append(dict(name=p["role"] + ":decal", kind="decal", tris=dt, uv=(uvk - klo) / (khi - klo), image=img,
                           domain=(klo.tolist(), khi.tolist()), uv_set=k, overlap=ov, triangles=int(len(dt)), small=small))
    return charts


def reference_shade(p, part_record):
    """Per-pixel console-style shading function for softrender: returns (uv_array, shade_fn) for the part."""
    me = p["mesh"]
    if p["role"].startswith("head"):
        lo, hi = (np.array(x) for x in p["uv_domain"])
        uv = (me["texcoords"][:, 0:2] - lo) / (hi - lo)
        tex = p["texture"]
        res = tex.shape[0]

        def shade(n, t):
            s = AC.sample(tex, t[:, 0:2])
            return s[:, :3] * 255, np.ones(len(n))
        return uv, shade
    base = part_record["textures"][0].astype(float) / 255.0
    decs = decal_layers(part_record)
    UV = np.concatenate([me["texcoords"][:, 0:2]] + [me["texcoords"][:, 2 * k:2 * k + 2] for _, k, _ in decs], 1)

    def shade(n, t):
        col = AC.sample(base, t[:, 0:2], wrap=True)[:, :3]
        for j, (img, _, _) in enumerate(decs):
            q = t[:, 2 + 2 * j:4 + 2 * j]
            s = AC.sample(img, q)
            out = (q[:, 0] < 0) | (q[:, 0] > 1) | (q[:, 1] < 0) | (q[:, 1] > 1)
            s[out, 3] = 0
            col = col * (1 - s[:, 3:4]) + s[:, :3] * s[:, 3:4]
        return col * 255, np.ones(len(n))
    return UV, shade
