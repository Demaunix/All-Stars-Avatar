"""Compose an Xbox avatar preset from the supplied AvatarAssetPack (decoded blocks only) in avatar space.

Sources of behaviour:
- geometry/skin/texcoords: avatar_model (original decoder of the supplied blocks);
- textures: avatar_texture; sampler -> (texture, UV set) records from each model header
  `(1, sampler, u16 texture, u16 uv_set, 3, 0, 0)`;
- head layering: the game's own avatar head shader text (PC exe 0x2DD900.., Xbox image 0x820BA91C..):
  matColour = C2; layer.rgb = r*Ca + g*Cb + b*Cc; matColour = lerp(matColour, layer.rgb, layer.a) for
  skinFeatures(uv0.xy) facialHair(uv0.zw) eyeBrow(uv1.xy) eye(uv1.zw) mouth(uv2.xy) eyeShadow(uv2.zw).
- garments: texture 0 (UV0) diffuse; a texture whose alpha is not constant (e.g. guest number, cap logo) is a decal
  composited over the diffuse with its own UV set. 32x32 tint masks / black maps are recorded, not applied (their
  colour slots belong to the console avatar description, which is not in the supplied data).
Colour parameters not present in the supplied data are explicit, labelled preset choices (PRESET_COLOURS).
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import avatar_colours as ACOL
import avatar_model as AM
import avatar_texture as AT
from catalog import catalog, decoded

NAMES = {c["index"]: c["name"] for c in catalog()}
KIND = {c["index"]: c["kind"] for c in catalog()}

# Explicit preset choices (not console palette data; the avatar description palettes are not in the supplied pack).
PRESET_COLOURS = dict(
    iris=(0.33, 0.21, 0.12),        # C9
    sclera=(0.96, 0.96, 0.94),      # C11
    eye_g=(0.0, 0.0, 0.0),          # C10 (G channel unused by the chosen eyes: all zero)
    lips=(0.78, 0.45, 0.40),        # C12
    mouth_inner=(0.93, 0.92, 0.88), # C13 (teeth / inner highlight)
    mouth_line=(0.30, 0.12, 0.10),  # C14
)

PRESETS = {
    "male_guest_1": dict(body=0, head=2, hair=449, shirt=92, pants=144, shoes=46, eyebrows=607, eyes=634, mouth=742, frame=0),
    # shoes: the Xbox sneakers (53/54) are not chartable for one texture (decal over differing base texels)
    "male_guest_2": dict(body=0, head=2, hair=474, shirt=93, pants=169, shoes=55, eyebrows=608, eyes=634, mouth=742, frame=0),
    "male_guest_3": dict(body=0, head=2, hair=462, shirt=94, pants=145, shoes=49, eyebrows=607, eyes=634, mouth=742, frame=0),
    "female_guest_1": dict(body=1, head=2, hair=502, shirt=301, pants=344, shoes=252, eyebrows=608, eyes=634, mouth=742, frame=0),
    "female_guest_2": dict(body=1, head=2, hair=522, shirt=302, pants=365, shoes=253, eyebrows=608, eyes=634, mouth=742, frame=0),
}


def load_part(index):
    d = decoded(index)
    m = AM.decode(d)
    if m["empty"]:
        return dict(index=index, name=NAMES[index], meshes=[], textures=[], samplers=[])
    recs, trailer = AT.model_tail(d, m["tail_offset"])
    tex = []
    for r in recs:
        tex.append(None if r["placeholder"] else AT.decode_span(d, r, 0, tiled=False))
    samp = []
    q = 0x50
    first = m["meshes"][0]["offset"]
    while q + 24 <= first:
        a, s, t, uv, three, z1, z2 = struct.unpack_from("<IIHHIII", d, q)
        if a == 1 and three == 3 and z1 == 0 and z2 == 0 and t < len(recs) and s < 64:
            samp.append(dict(sampler=s, texture=t, uv_set=uv))
            q += 24
        else:
            q += 1
    return dict(index=index, name=NAMES[index], data_sha=None, meshes=m["meshes"], textures=tex, texture_records=recs,
                samplers=samp, trailer_len=len(trailer))


def standalone_texture(index, span):
    d = decoded(index)
    r = AT.standalone(d)
    if not 0 <= span < r["spans"]:
        raise ValueError("span out of range")
    return AT.decode_span(d, r, span, tiled=False).astype(np.float64) / 255.0


def sample(img, uv, wrap=False):
    """Bilinear; clamp (default) or wrap addressing; img float HxWx4 in 0..1; uv Nx2 (u right, v down)."""
    h, w = img.shape[:2]
    if wrap:
        x = np.mod(uv[:, 0] * w - 0.5, w)
        y = np.mod(uv[:, 1] * h - 0.5, h)
        x0, y0 = np.floor(x).astype(int) % w, np.floor(y).astype(int) % h
        x1, y1 = (x0 + 1) % w, (y0 + 1) % h
        fx, fy = (x - np.floor(x))[:, None], (y - np.floor(y))[:, None]
        return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def raster_uv(uv, tris, res):
    """Pixels of a res x res texture covered by triangles in UV space: returns (pix_y, pix_x, tri, bary)."""
    ys, xs, ts, bs = [], [], [], []
    U = uv * res - 0.5
    for t, (a, b, c) in enumerate(tris):
        pa, pb, pc = U[a], U[b], U[c]
        x0, x1 = int(np.floor(min(pa[0], pb[0], pc[0]))), int(np.ceil(max(pa[0], pb[0], pc[0])))
        y0, y1 = int(np.floor(min(pa[1], pb[1], pc[1]))), int(np.ceil(max(pa[1], pb[1], pc[1])))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, res - 1), min(y1, res - 1)
        if x0 > x1 or y0 > y1:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        px, py = gx.ravel().astype(float), gy.ravel().astype(float)
        det = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(det) < 1e-12:
            continue
        l1 = ((pb[1] - pc[1]) * (px - pc[0]) + (pc[0] - pb[0]) * (py - pc[1])) / det
        l2 = ((pc[1] - pa[1]) * (px - pc[0]) + (pa[0] - pc[0]) * (py - pc[1])) / det
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not inside.any():
            continue
        ys.append(gy.ravel()[inside]); xs.append(gx.ravel()[inside]); ts.append(np.full(inside.sum(), t))
        bs.append(np.stack([l1[inside], l2[inside], l3[inside]], 1))
    if not ys:
        return (np.zeros(0, int),) * 3 + (np.zeros((0, 3)),)
    return np.concatenate(ys), np.concatenate(xs), np.concatenate(ts), np.concatenate(bs)


def dilate(rgba, filled, iters=8):
    """Grow covered texels outward (gutter) so bilinear/mip sampling at seams stays on-surface colour."""
    img, f = rgba.copy(), filled.copy()
    for _ in range(iters):
        if f.all():
            break
        acc = np.zeros_like(img); cnt = np.zeros(f.shape)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sf = np.roll(f, (dy, dx), (0, 1)); si = np.roll(img, (dy, dx), (0, 1))
            acc += si * sf[..., None]; cnt += sf
        grow = (~f) & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow][:, None]
        f = f | grow
    return img, f


def uv_domain(mesh, pair=0, triangles=None):
    """[lo, hi] per axis covering [0,1] and every used UV of the pair (garments rely on wrap addressing)."""
    T = mesh["triangles"] if triangles is None else triangles
    uv = mesh["texcoords"][np.unique(T), 2 * pair:2 * pair + 2]
    lo = np.minimum(0.0, np.floor(uv.min(0) * 64) / 64)
    hi = np.maximum(1.0, np.ceil(uv.max(0) * 64) / 64)
    return lo, hi


def bake_layers(mesh, base_pair, res, base_rgba, layers, domain=None, base_sample_pair=None):
    """Texture over the raster UV pair's domain [lo,hi] (default [0,1]): base (image sampled with wrap at
    `base_sample_pair` (default: the raster pair), or a constant colour) then layers lerped by alpha.
    layers: list of (img float HxWx4, uv_pair index, colour_fn(rgba)->rgb or None)."""
    sample_pair = base_pair if base_sample_pair is None else base_sample_pair
    uvs = mesh["texcoords"]
    T = mesh["triangles"]
    lo, hi = (np.zeros(2), np.ones(2)) if domain is None else domain
    base_uv = (uvs[:, 2 * base_pair:2 * base_pair + 2] - lo) / (hi - lo)
    py, px, tt, bb = raster_uv(base_uv, T, res)
    out = np.zeros((res, res, 4)); filled = np.zeros((res, res), bool)
    if len(py) == 0:
        return out, filled
    corner = T[tt]
    def interp(pair):
        q = uvs[:, 2 * pair:2 * pair + 2]
        return (q[corner] * bb[:, :, None]).sum(1)
    if callable(base_rgba):
        col = base_rgba(len(py))
    elif isinstance(base_rgba, np.ndarray) and base_rgba.ndim == 3:
        col = sample(base_rgba, interp(sample_pair), wrap=True)[:, :3]
    else:
        col = np.tile(np.asarray(base_rgba, float)[:3], (len(py), 1))
    for img, pair, cfn in layers:
        q = interp(pair)
        s = sample(img, q)
        # clamp-to-border transparent: the feature/decal layers only exist inside their UV square (verified visually:
        # clamp-to-edge smears mouth edge texels across the ear, research/RENDER-NOTES.md)
        outside = (q[:, 0] < 0) | (q[:, 0] > 1) | (q[:, 1] < 0) | (q[:, 1] > 1)
        s[outside, 3] = 0.0
        rgb = cfn(s) if cfn else s[:, :3]
        a = s[:, 3:4]
        col = col * (1 - a) + rgb * a
    out[py, px, :3] = col
    out[py, px, 3] = 1.0
    filled[py, px] = True
    # Degenerate (collapsed) UV triangles, e.g. an ear mapped to a single point, are not rasterized; fill every
    # remaining texel from its nearest covered neighbour so such points sample surface colour (observed defect, fixed).
    out, full = dilate(out, filled, iters=res)
    return out, filled


def mix3(c0, c1, c2):
    c0, c1, c2 = (np.asarray(c, float) for c in (c0, c1, c2))
    return lambda s: s[:, 0:1] * c0 + s[:, 1:2] * c1 + s[:, 2:3] * c2


def solid_colour(img):
    """A texture that is a single colour (the 32x32 defaults): return it, else None."""
    flat = img.reshape(-1, 4)
    if np.ptp(flat[:, :3], axis=0).max() <= 4.5 / 255:  # DXT endpoint rounding leaves <= 3 LSB variation
        return np.median(flat, axis=0)
    return None


def ray_hits(origins, dirs, V, T, max_t):
    """For each ray, whether it hits any triangle (V[T]) within (1e-5, max_t]."""
    return np.isfinite(ray_first(origins, dirs, V, T, max_t))


def ray_first(origins, dirs, V, T, max_t):
    """Distance to the nearest triangle (V[T]) hit within (1e-5, max_t], inf if none. Moller-Trumbore per ray chunk."""
    A, B, Cc = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = B - A, Cc - A
    hit = np.full(len(origins), np.inf)
    for s in range(0, len(origins), 64):
        o, dvec = origins[s:s + 64, None, :], dirs[s:s + 64, None, :]
        pv = np.cross(dvec, e2[None])
        det = (e1[None] * pv).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        tv = o - A[None]
        u = (tv * pv).sum(-1) * inv
        qv = np.cross(tv, e1[None])
        v = (dvec * qv).sum(-1) * inv
        t = (e2[None] * qv).sum(-1) * inv
        h = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-5) & (t <= max_t)
        hit[s:s + 64] = np.where(h, t, np.inf).min(1)
    return hit


def tuck_under(mesh, covering, depth=0.05, margin=0.006, side=-1, toward_axis=False):
    """Layering between separately authored parts (the console rule is not in the supplied data; TOC word 38 is a
    dashboard sort key). side=-1: vertices of `mesh` lying outside a covering part (a ray along the inward normal meets
    it within `depth`) are moved along that ray to `margin` inside it (trousers under a top). side=+1: vertices lying
    inside an underlying part are lifted to `margin` outside it (hair shell over the head). toward_axis: the ray is
    horizontal toward the vertical axis through the covering part's lower 15 cm instead of along the normal (belt loops
    and waistband edges whose normals miss the top). Returns (mesh copy, record)."""
    P = AM.positions(mesh)
    V = np.concatenate([AM.positions(g) for g in covering])
    if toward_axis:
        c = V[V[:, 1] < V[:, 1].min() + 0.15].mean(0)
        N = np.stack([c[0] - P[:, 0], np.zeros(len(P)), c[2] - P[:, 2]], 1)
        reach = np.linalg.norm(N, axis=1)
        N = N / np.maximum(reach, 1e-12)[:, None]
    else:
        N = side * mesh["normal"] / np.maximum(np.linalg.norm(mesh["normal"], axis=1, keepdims=True), 1e-12)
        reach = np.full(len(P), np.inf)
    off = np.cumsum([0] + [len(g["quantized"]) for g in covering])[:-1]
    T = np.concatenate([g["triangles"] + o for g, o in zip(covering, off)])
    t = ray_first(P, N, V, T, depth)
    out = np.isfinite(t) & (t < reach)
    m2 = dict(mesh)
    m2["pos_override"] = np.where(out[:, None], P + N * (np.where(out, t, 0) + margin)[:, None], P)
    return m2, dict(vertices_moved=int(out.sum()), depth_median=round(float(np.median(t[out])), 4) if out.any() else 0.0,
                    depth_max=round(float(t[out].max()), 4) if out.any() else 0.0, search_depth=depth, margin=margin,
                    direction="toward top axis" if toward_axis else ("outward normal" if side > 0 else "inward normal"))


def hide_covered_body(body_mesh, garment_meshes, reach=0.04, reach_inward=0.0):
    """Geometric coverage (used where no supplied hiding template applies, and for trousers under tops): a vertex is
    covered when a ray along its outward normal meets a covering mesh within `reach` (or, if reach_inward > 0, a ray
    along the inward normal meets it within reach_inward: interpenetrating layers); triangles with all corners covered
    are dropped."""
    keep, stats = covered_keep_mask(body_mesh, garment_meshes, reach, reach_inward)
    return body_mesh["triangles"][keep], stats


def covered_keep_mask(body_mesh, garment_meshes, reach=0.04, reach_inward=0.0):
    P = AM.positions(body_mesh)
    N = body_mesh["normal"] / np.maximum(np.linalg.norm(body_mesh["normal"], axis=1, keepdims=True), 1e-12)
    if not garment_meshes:
        return np.ones(len(body_mesh["triangles"]), bool), dict(covered_vertices=0, dropped_triangles=0)
    V = np.concatenate([AM.positions(g) for g in garment_meshes])
    off = np.cumsum([0] + [len(g["quantized"]) for g in garment_meshes])[:-1]
    T = np.concatenate([g["triangles"] + o for g, o in zip(garment_meshes, off)])
    covered = ray_hits(P, N, V, T, reach)
    if reach_inward > 0:
        covered |= ray_hits(P, -N, V, T, reach_inward)
    tri = body_mesh["triangles"]
    keep = ~covered[tri].all(1)
    return keep, dict(covered_vertices=int(covered.sum()), dropped_triangles=int((~keep).sum()), kept_triangles=int(keep.sum()), reach=reach,
                           reach_inward=reach_inward)


def apply_hiding_templates(body_mesh, components):
    """Supplied hiding templates (avatar_hiding) of the worn components: replace the overridden body vertices
    (position, normal, skin, colour; texcoords kept) and hide the union of their body triangles. The console's merge
    rule for a vertex overridden by several templates is not in the supplied data; the most inward override (along the
    original body normal) is used, and the disagreement is recorded. Returns (body copy, hidden triangle mask, record)."""
    import avatar_hiding as H
    P = AM.positions(body_mesh).copy()
    N0 = body_mesh["normal"]
    out = dict(body_mesh)
    for k in ("normal_raw", "normal", "weights", "bones", "color"):
        out[k] = body_mesh[k].copy()
    hidden = np.zeros(len(body_mesh["triangles"]), bool)
    best = {}
    rec = dict(method="supplied hiding templates (TOC link word 40)", templates=[], without_template=[])
    for comp in components:
        t = H.template_for(comp)
        if t is None:
            rec["without_template"].append(dict(index=comp, name=NAMES[comp]))
            continue
        T = H.load(t, body_mesh)
        hidden[T["hidden_triangles"]] = True
        for k, v in enumerate(T["vertices"]):
            depth = float((T["positions"][k] - P[v]) @ N0[v])
            best.setdefault(int(v), []).append((depth, T, k))
        rec["templates"].append(dict(component=comp, template=t, name=NAMES[t], hidden_triangles=len(T["hidden_triangles"]),
                                     vertex_overrides=len(T["vertices"])))
    Q = P.copy()
    disagree = 0.0
    for v, cands in best.items():
        cands.sort(key=lambda c: c[0])
        if len(cands) > 1:
            disagree = max(disagree, max(float(np.abs(c[1]["positions"][c[2]] - cands[0][1]["positions"][cands[0][2]]).max()) for c in cands))
        _, T, k = cands[0]
        Q[v] = T["positions"][k]
        for f in ("normal_raw", "normal", "weights", "bones", "color"):
            out[f][v] = T[f][k]
    out["pos_override"] = Q
    rec.update(hidden_triangles=int(hidden.sum()), overridden_vertices=len(best),
               shared_override_vertices=sum(len(c) > 1 for c in best.values()), max_shared_disagreement=round(disagree, 4),
               merge_rule="most inward override per vertex (console merge rule not supplied)")
    return out, hidden, rec


def _seams(me, names):
    P = AM.positions(me)
    dom = me["bones"][np.arange(len(P)), me["weights"].argmax(1)]
    T = me["triangles"]
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    out = {}
    for a, b in E:
        ba, bb = int(dom[a]), int(dom[b])
        if ba != bb:
            out.setdefault(tuple(sorted((names[ba], names[bb]))), []).append((P[a] + P[b]) / 2)
    return {k: np.mean(v, 0) for k, v in out.items() if len(v) >= 4}


def align_to_body(garment_mesh, body_mesh, min_seams=3, max_spread=0.03):
    """Measure (and, in the returned copy, remove) the rigid offset of a garment against its body over their shared seams.

    rp6: with the per-axis decode every garment type sits in the body's frame (research/GARMENT-OFFSETS.json: per-type median
    |dy| <= 0.008, worst item 0.022 in y; dz up to 0.045 on two heels), so compose only records the measurement. The factor-2 decode showed category "frames"
    (tops -0.15, gloves -0.20: research/GARMENT-OFFSETS-c253-decode.json), which the c253 reproduction path still corrects.
    Returns (copy with the median offset removed, record)."""
    import bone_map
    names = bone_map.xbox_bone_names()
    g, b = _seams(garment_mesh, names), _seams(body_mesh, names)
    common = sorted(set(g) & set(b))
    if len(common) < min_seams:
        return garment_mesh, dict(applied=False, seams=len(common), reason="too few shared seams")
    ds = np.array([g[k] - b[k] for k in common])
    med = np.median(ds, 0)
    spread = np.median(np.abs(ds - med), 0)
    if spread.max() > max_spread:
        return garment_mesh, dict(applied=False, seams=len(common), delta=med.round(4).tolist(), spread=spread.round(4).tolist(), reason="offset not rigid")
    m2 = dict(garment_mesh)
    # positions = min + q*step*K (avatar_model.positions); shifting min moves every vertex by -delta whatever K is
    m2["min"] = tuple(np.asarray(garment_mesh["min"]) - med)
    return m2, dict(applied=True, seams=len(common), delta=med.round(4).tolist(), spread=spread.round(4).tolist())


EXTRA_ROLES = ("costume", "hat", "glasses", "gloves", "wrist", "ring", "earrings")
# rp6 (2026-10-27): with the per-axis decode (avatar_model.DECODES["rp6"]) every worn part is authored in the body's own frame
# (research/GARMENT-OFFSETS.json: per-type median |dy| <= 0.008, worst single item 0.022; head items have no body seams and sit
# with the head). Nothing is shifted any more. Every part's measured offset goes into the provenance, and a median seam offset
# beyond FRAME_TOLERANCE (rigid or not) refuses the bake: it would mean a part in another frame, as the factor-2 decode's 15 cm
# "garment frame" was (checked: with K forced to 2 the rp6 path refuses Male Formal Suit at dy -0.14).
FRAME_TOLERANCE = 0.05
# c253 reproduction only (avatar_model.set_decode("c253")): the C253 presets' keys and Codex's body shift for the factor-2 decode.
C253_KEYS = {"body", "head", "hair", "shirt", "pants", "shoes", "eyebrows", "eyes", "mouth", "frame"}
TOP_OFFSET_FALLBACK = {0: (0.0, -0.1549, 0.0), 1: (0.0, -0.1499, 0.0)}  # GARMENT-OFFSETS-c253-decode.json medians, 0x8 male/female


def hat_variant(hair_index):
    """TOC index of a hair style's "(Hat)" cut (worn under hats), or None: by name, else catalog_slots' adjacency overrides."""
    import catalog_slots as CS
    name = NAMES[hair_index] + " (Hat)"
    for c in catalog():
        if c["type"] == 0x4 and c["name"] == name:
            return c["index"]
    for hat_i, base in CS.HAT_PARTNER_OVERRIDES.items():
        if str(hair_index) in base.split(","):
            return hat_i
    return None


def upper_bones():
    """Xbox bone indices of BACKA and its descendants (PC/Xbox hierarchy is identical by name). c253 reproduction only."""
    import bone_map
    import pcanim
    xb = bone_map.xbox_bone_names()
    _, _, m, _, _ = bone_map.pc_rig()
    names = [n["name"] for n in m["nodes"]]
    par = pcanim.parents(m)
    root = names.index("BACKA")
    up = set()
    for i in range(len(names)):
        k = i
        while k >= 0:
            if k == root:
                up.add(names[i])
                break
            k = par[k]
    return {j for j, n in enumerate(xb) if n in up}


def body_to_garment_frame(body_mesh, delta):
    """c253 reproduction only: move the body's upper half (BACKA and descendants) by `delta`, blended by the skin weights.

    Codex's correction for the factor-2 decode (research/FRAME-NOTES.md): under it every top seam sat ~0.15 below the body's
    own seam while trousers/shoes matched, so the body's upper half was moved down to the "garment frame". rp6's per-axis
    decode removes the offset at the source (avatar_model.DECODES), so the production path never calls this."""
    up = upper_bones()
    P = AM.positions(body_mesh)
    w = body_mesh["weights"].astype(float) / 255.0
    is_up = np.isin(body_mesh["bones"], list(up)).astype(float)
    frac = (w * is_up).sum(1, keepdims=True)
    m2 = dict(body_mesh)
    m2["pos_override"] = P + frac * np.asarray(delta, float)[None, :]
    return m2, dict(delta=list(map(float, delta)), upper_bones=len(up), vertices_moved=int((frac[:, 0] > 0).sum()),
                    vertices_partial=int(((frac[:, 0] > 0) & (frac[:, 0] < 1)).sum()))


def compose(preset_name, head_res=1024, garment_res=512, colours=None):
    """Returns parts in avatar space with baked textures, plus a provenance record.

    preset_name: a PRESETS key, or a preset dict (body, head, hair, shirt, pants, shoes, eyebrows, eyes, mouth, frame, and
    optionally facial_hair, face_paint, eye_shadow: TOC indices).
    colours: None keeps the C235..C253 behaviour (default textures + PRESET_COLOURS, Codex's channel reading; reproduces the
    installed bake). An avatar_colours.Colours applies the console's rule instead: head layers R = colour, G = white,
    B = skin shading, and body / hair / garment tint masks with CustomColor1..3 (research/AVATAR-SHADERS.md)."""
    P = PRESETS[preset_name] if isinstance(preset_name, str) else dict(preset_name)
    legacy = AM.DECODE_NAME == "c253"
    if legacy and {k for k, v in P.items() if v is not None} - C253_KEYS:
        raise ValueError("the c253 reproduction path takes the C253 presets only (no face shapes, hats, glasses, hand items, costumes)")
    prov = dict(preset=preset_name if isinstance(preset_name, str) else "custom", items={k: v for k, v in P.items() if k != "frame"},
                components={}, colours={}, notes=[], decode=dict(profile=AM.DECODE_NAME, factors=AM.K.round(6).tolist()))
    parts = []
    body = load_part(P["body"])
    head = load_part(P["head"])
    if any(P.get(k) is not None for k in ("nose", "chin", "ears")):
        # step 4 / rp6: face shape overrides as stored. With the per-axis decode they sit on the stock head (the mean offset per
        # region is below 4 mm and is recorded per block in the provenance), so the old "shape only" correction is gone.
        import avatar_faceshape as FS
        head["meshes"], prov["face_shapes"] = FS.apply(head["meshes"], [P.get("nose"), P.get("chin"), P.get("ears")])
    skin_rgba = body["textures"][0].astype(float) / 255.0
    skin = solid_colour(skin_rgba)
    if skin is None:
        raise ValueError("body default texture is not a solid colour")
    prov["colours"]["skin_C2"] = dict(value=skin[:3].round(4).tolist(), source=f"{body['name']} texture 0 (solid default)")
    hair_index = P.get("hair")
    if P.get("hat") is not None and hair_index is not None:
        hv = hat_variant(hair_index)
        prov["hat_hair"] = dict(hair=hair_index, hair_name=NAMES[hair_index], hat_variant=hv, hat_variant_name=NAMES.get(hv),
                                note="hair swapped for its (Hat) cut" if hv is not None else "no (Hat) cut for this style: hair kept under the hat")
        if hv is not None:
            hair_index = hv
    hair_part = load_part(hair_index) if hair_index is not None else None
    hair_col = None
    if hair_part and hair_part["meshes"]:
        hair_col = solid_colour(hair_part["textures"][0].astype(float) / 255.0)
        prov["colours"]["hair"] = dict(value=hair_col[:3].round(4).tolist(), source=f"{hair_part['name']} texture 0 (solid default)")
    brow_col = hair_col[:3] if hair_col is not None else np.array([0.10, 0.08, 0.06])
    prov["colours"]["eyebrow_C6"] = dict(value=np.round(brow_col, 4).tolist(), source="hair default colour" if hair_col is not None else "preset choice")
    for k, v in PRESET_COLOURS.items():
        prov["colours"][k] = dict(value=list(v), source="explicit preset choice (console palette not supplied)")
    # --- head: bake six layers into each half's skin-features UV parameterization (pair 0)
    fr = P["frame"]
    brow = standalone_texture(P["eyebrows"], 0)
    eye = standalone_texture(P["eyes"], fr)
    mouth = standalone_texture(P["mouth"], fr)
    C = PRESET_COLOURS
    layer_rec = dict(eyebrows=(P["eyebrows"], NAMES[P["eyebrows"]], 0), eyes=(P["eyes"], NAMES[P["eyes"]], fr), mouth=(P["mouth"], NAMES[P["mouth"]], fr))
    if colours is None:
        head_layers = [(brow, 2, mix3(brow_col, (0, 0, 0), (0, 0, 0))),
                       (eye, 3, mix3(C["iris"], C["eye_g"], C["sclera"])),
                       (mouth, 4, mix3(C["lips"], C["mouth_inner"], C["mouth_line"]))]
        empty = ["skinFeatures", "facialHair", "eyeShadow"]
    else:
        # console rule: skin base = SkinTone; layers in shader order, each r*colour + g*white + b*skin (eye shadow: r only)
        skin = np.array([*colours.skin, 1.0])
        prov["colours"] = dict(console=colours.provenance(), rule="research/AVATAR-SHADERS.md slot table")
        L = colours.head_layer_colours
        head_layers, empty = [], []
        for key, pair, layer in (("face_paint", 0, "skin_features"), ("facial_hair", 1, "facial_hair")):
            if P.get(key) is not None:
                head_layers.append((standalone_texture(P[key], 0), pair, mix3(*L(layer))))
                layer_rec[key] = (P[key], NAMES[P[key]], 0)
            else:
                empty.append(layer)
        head_layers += [(brow, 2, mix3(*L("eyebrows"))), (eye, 3, mix3(*L("eyes"))), (mouth, 4, mix3(*L("mouth")))]
        if P.get("eye_shadow") is not None:
            head_layers.append((standalone_texture(P["eye_shadow"], 0), 5, mix3(*L("eye_shadow"))))
            layer_rec["eye_shadow"] = (P["eye_shadow"], NAMES[P["eye_shadow"]], 0)
        else:
            empty.append("eyeShadow")
    prov["components"]["head"] = dict(index=head["index"], name=head["name"], layers=layer_rec, empty_slots=empty)
    for k, me in enumerate(head["meshes"]):
        dom = uv_domain(me)
        tex, filled = bake_layers(me, 0, head_res, skin, head_layers, domain=dom)
        parts.append(dict(role=f"head{k}", src=head["index"], mesh=me, uv_pair=0, texture=tex, filled=filled, uv_domain=(dom[0].tolist(), dom[1].tolist())))
    # --- body, hair, garments
    roles = [("body", body)] + ([("hair", hair_part)] if hair_part and hair_part["meshes"] else [])
    for role in ("shirt", "pants", "shoes") + EXTRA_ROLES:
        if P.get(role) is not None:
            roles.append((role, load_part(P[role])))
    if colours is not None:
        # console material rule on every worn part: lerp(diffuse, mask.rgb . CustomColor1..3, mask.a); body = skin, hair = hair
        prov["colours"]["tint"] = {}
        for role, part in roles:
            prov["colours"]["tint"][role] = dict(ACOL.tint_part(part, colours.custom_for(part["index"], role)), item=part["name"])
    body_mesh = body["meshes"][0]
    worn = [P[r] for r in ("shirt", "pants", "shoes", "costume", "gloves") if P.get(r) is not None]
    templated_body, hidden, hrec = apply_hiding_templates(body_mesh, worn)
    if legacy:
        # c253 reproduction (factor-2 decode): measure the preset top against the body, then move the body's upper half
        top_part = next((p for r, p in roles if r == "shirt"), None)
        gender = 0 if P["body"] == 0 else 1
        delta, src = np.array(TOP_OFFSET_FALLBACK[gender]), "category median (no top in preset)"
        if top_part and top_part["meshes"]:
            _, arec = align_to_body(top_part["meshes"][0], body_mesh)
            if arec.get("applied"):
                delta, src = np.array(arec["delta"]), f"measured {top_part['name']} vs {body['name']} ({arec['seams']} seams, spread {arec['spread']})"
        final_body, brec = body_to_garment_frame(templated_body, delta)
        prov["frame"] = dict(frame="garment/head frame (c253 reproduction: factor-2 decode + body upper-half shift)", body_upper_shift=brec, delta_source=src)
    else:
        # rp6: one frame for everything. Every part stays where it is authored; its measured offset against the body is recorded
        # (head items share no body seams: "too few shared seams") and any measured median offset beyond FRAME_TOLERANCE refuses the bake.
        final_body = templated_body
        prov["frame"] = dict(frame="body frame (per-axis decode: every part as authored, nothing shifted)", tolerance=FRAME_TOLERANCE, measured_offsets={})
    for role, part in roles:
        prov["components"][role] = dict(index=part["index"], name=part["name"], samplers=part["samplers"], textures=len(part["textures"]))
        for k, me in enumerate(part["meshes"]):
            name = role if len(part["meshes"]) == 1 else f"{role}{k}"
            if role == "body":
                me = final_body
            elif not legacy:
                _, arec = align_to_body(me, body_mesh)
                prov["frame"]["measured_offsets"][name] = arec
                d = arec.get("delta")      # present whenever enough seams are shared, rigid or not
                if d is not None and max(abs(x) for x in d) > FRAME_TOLERANCE:
                    raise ValueError(f"{part['name']} ({name}) sits {d} from the body: another frame? (tolerance {FRAME_TOLERANCE})")
            elif role in ("pants", "shoes"):
                _, arec = align_to_body(me, body_mesh)  # recorded only: lower-body garments match the body (|dy|<0.016)
                prov["components"][role].setdefault("measured_offset_vs_body_not_applied", []).append(arec)
            # surfaces are built by avatar_surface.charts_for_part (UV0 of garments is shared/mirrored, so decals
            # cannot be baked into it); the decoded record travels with the part
            parts.append(dict(role=name, src=part["index"], mesh=me, uv_pair=0, record=part,
                              allow_shared_decal=colours is not None))   # step 4: coloured (new) avatars accept the shared-UV0 decal fallback
    untemplated = {c["index"] for c in hrec["without_template"]}
    fallback = [p["mesh"] for p in parts if p["role"].startswith(("shirt", "pants", "shoes")) and p["src"] in untemplated]
    tops = [p["mesh"] for p in parts if p["role"].startswith("shirt")]
    heads = [p["mesh"] for p in parts if p["role"].startswith("head")]
    for p in parts:
        if p["role"] == "hair":
            # close-cropped hair shells sit within ~5 mm of the scalp, partly inside it (Number One: 69/146 vertices)
            p["mesh"], lrec = tuck_under(p["mesh"], heads, depth=0.03, margin=0.003, side=+1)
            prov["hair_over_head"] = dict(method="geometric lift + scalp tuck (head/hair layering rule not supplied)", hair=lrec, head=[])
            # scalp still crossing the shell between hair vertices: tuck those head vertices under the lifted shell
            for q in parts:
                if q["role"].startswith("head"):
                    q["mesh"], hrec2 = tuck_under(q["mesh"], [p["mesh"]], depth=0.02, margin=0.003)
                    prov["hair_over_head"]["head"].append(hrec2)
        if p["role"] == "body":
            tri = p["mesh"]["triangles"]
            keep = ~hidden
            if fallback:
                # components without a supplied template (e.g. High Top Sneakers): geometric coverage for those only
                fkeep, stats = covered_keep_mask(p["mesh"], fallback)
                keep &= fkeep
                hrec["fallback_geometric"] = dict(components=sorted(untemplated), **stats)
            p["keep_triangles"] = tri[keep]
            prov["body_hiding"] = dict(hrec, kept_triangles=int(keep.sum()), dropped_triangles=int((~keep).sum()))
        elif p["role"].startswith("pants") and tops:
            # untucked tops overlap the trouser waistband (garment frame: hem ~0.65, waistband up to ~0.93) and the
            # authored waistband lies up to ~5 cm outside the top in places (root v2 review; work/scratch/poke_t1_*):
            # tuck those trouser vertices just inside the top, then hide the trouser triangles the top covers
            p["mesh"], trec = tuck_under(p["mesh"], tops)
            p["mesh"], trec2 = tuck_under(p["mesh"], tops, toward_axis=True)
            p["keep_triangles"], stats = hide_covered_body(p["mesh"], tops, reach=0.04, reach_inward=0.03)
            prov.setdefault("pants_under_top", []).append(dict(method="geometric tuck + ray coverage (garment layering rule not supplied)",
                                                               tuck=[trec, trec2], hiding=stats))
    return parts, prov
