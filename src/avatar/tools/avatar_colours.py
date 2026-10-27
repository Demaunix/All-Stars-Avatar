"""Avatar colours the console's way (M10 phase 1, step 3).

Sources (research/AVATAR-SHADERS.md, CONFIRMED in the Xbox code unless marked):
- Head: base = SkinTone; each feature texture layer is `r * colour + g * white + b * SkinTone`, lerped in by its alpha, in the
  order skinFeatures (feature1, feature2, skin), facialHair, eyeBrow, eye, mouth (each: colour, white, skin), eyeShadow (r only).
- Garments, hair, body: `lerp(diffuse, mask.r * C1 + mask.g * C2 + mask.b * C3, mask.a)` with the item's CustomColor1..3; the
  body and the hair get their skin / hair colour as CustomColor1 (INFERRED; their masks are solid (1,0,0,1)).
- Byte -> float: the palette byte is used as the texel value (XAM's conversion is UNKNOWN: plain /255 or sRGB; labelled guess).
Colours come from the game's own 17 avatar descriptions (research/xbox-default-avatars.json, decoded from the X360 image by
avatar_desc.py); nothing outside the All-Stars X360 files is used (user rule 2026-10-21).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from catalog import MODULE, catalog

WHITE, BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
FIELDS = ("skin", "hair", "mouth", "eyes", "eyebrows", "eye_shadow", "facial_hair", "feature1", "feature2")
DEFAULTS_JSON = MODULE / "research" / "xbox-default-avatars.json"
# preset keys by slot bit for the items compose() can bake today (step 4 adds the rest)
SLOT_KEYS = {0x2: "body", 0x1: "head", 0x4: "hair", 0x8: "shirt", 0x208: "shirt", 0x10: "pants", 0x20: "shoes",
             0x4000: "eyebrows", 0x2000: "eyes", 0x8000: "mouth", 0x10000: "facial_hair", 0x20000: "face_paint", 0x40000: "eye_shadow",
             # step 4 roles
             0x40: "hat", 0x100: "glasses", 0x80: "gloves", 0x880: "gloves", 0x200: "wrist", 0x800: "ring", 0x400: "earrings",
             0x800018: "costume", 0x800218: "costume", 0x800238: "costume", 0x800a98: "costume",
             0x80000: "nose", 0x100000: "chin", 0x200000: "ears"}


def hex_rgb(h: str) -> tuple:
    """'#RRGGBB' -> (r, g, b) floats, byte / 255 (labelled guess for XAM's conversion)."""
    h = h.strip().split()[0].lstrip("#")
    return tuple(int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


def is_solid(img: np.ndarray) -> np.ndarray | None:
    """Median RGBA of a texture that is one colour (DXT rounding tolerance), else None. Works on uint8 or float images."""
    f = img.reshape(-1, img.shape[-1]).astype(float)
    tol = 4.5 if img.dtype == np.uint8 else 4.5 / 255
    return np.median(f, axis=0) if np.ptp(f[:, :3], axis=0).max() <= tol else None


class Colours:
    """One avatar's colour description: the nine XAvatar colours plus per-item custom colours (TOC index -> 3 RGB)."""

    def __init__(self, source: str = "explicit", custom: dict | None = None, **fields):
        missing = [f for f in FIELDS if f not in fields]
        if missing:
            raise ValueError(f"missing colours: {missing}")
        for f in FIELDS:
            v = fields[f]
            setattr(self, f, hex_rgb(v) if isinstance(v, str) else tuple(float(x) for x in v[:3]))
        self.custom = {int(k): tuple(hex_rgb(c) if isinstance(c, str) else tuple(float(x) for x in c[:3]) for c in v) for k, v in (custom or {}).items()}
        self.source = source

    @classmethod
    def from_description(cls, desc: dict, source: str = "description") -> "Colours":
        """From one decoded 1000-byte description (avatar_desc.parse_description output)."""
        custom = {e["item"]["index"]: e["custom_colours"] for e in desc["items"] + desc.get("items_b", []) if e["item"] and "index" in e["item"] and e["custom_colours"]}
        return cls(source=source, custom=custom, **desc["colours"])

    @classmethod
    def default_avatar(cls, k: int) -> "Colours":
        """The colours of built-in avatar k (0..16) of the X360 game."""
        d = json.loads(DEFAULTS_JSON.read_text(encoding="utf-8"))["avatars"][k]
        return cls.from_description(d, source=f"X360 built-in avatar {k} ({d['va']})")

    def head_layer_colours(self, layer: str) -> tuple:
        """(colour for R, colour for G, colour for B) of a head feature layer, per the constant-slot table."""
        return {"skin_features": (self.feature1, self.feature2, self.skin), "facial_hair": (self.facial_hair, WHITE, self.skin),
                "eyebrows": (self.eyebrows, WHITE, self.skin), "eyes": (self.eyes, WHITE, self.skin), "mouth": (self.mouth, WHITE, self.skin),
                "eye_shadow": (self.eye_shadow, BLACK, BLACK)}[layer]

    def custom_for(self, index: int, role: str) -> tuple:
        """CustomColor1..3 of a worn item: skin for the body, hair colour for hair, else the item's own custom colours (black
        when none are recorded, which only matters for items with a real tint mask)."""
        if role == "body":
            return (self.skin, self.skin, self.skin)
        if role == "hair":
            return (self.hair, self.hair, self.hair)
        return self.custom.get(index, (BLACK, BLACK, BLACK))

    def provenance(self) -> dict:
        d = {f: [round(x, 4) for x in getattr(self, f)] for f in FIELDS}
        d["custom"] = {k: [[round(x, 4) for x in c] for c in v] for k, v in self.custom.items()}
        d["source"] = self.source
        d["byte_to_float"] = "byte / 255 (XAM conversion unknown; labelled guess)"
        return d


def tint_texture(diffuse: np.ndarray, mask: np.ndarray | None, colours3: tuple, same_uv: bool = False) -> tuple:
    """Apply the console material rule: lerp(diffuse, m.r*C1 + m.g*C2 + m.b*C3, m.a) per texel. diffuse/mask are uint8 RGBA.
    A constant mask applies everywhere; an image mask is applied per texel only when the item's UV set 1 equals its UV set 0
    (`same_uv`, true for the colourable sport tops), by resampling the mask to the diffuse size; otherwise it is refused.
    Returns (new uint8 RGBA diffuse, record)."""
    if mask is None:
        return diffuse, dict(applied=False, reason="no mask texture")
    m = is_solid(mask)
    c1, c2, c3 = (np.asarray(c, float) for c in colours3)
    if m is None:
        amax = int(mask[..., 3].max())
        if amax <= 2:      # an image mask whose alpha is zero everywhere tints nothing (e.g. Male Formal Suit's 128x64 mask)
            return diffuse, dict(applied=False, reason="image mask with zero alpha", mask_shape=list(mask.shape[:2]))
        if not same_uv:
            # the mask lives on UV set 1 with its own layout: avatar_surface charts it as a UV-1 layer with the three-colour mix
            return diffuse, dict(applied=False, deferred=True, reason="image mask on its own UV set: baked by avatar_surface as a UV-1 layer",
                                 mask_shape=list(mask.shape[:2]), colours=[[round(x, 4) for x in c] for c in (c1, c2, c3)])
        from PIL import Image
        h, w = diffuse.shape[:2]
        mk = np.asarray(Image.fromarray(mask, "RGBA").resize((w, h), Image.BILINEAR)).astype(float) / 255.0
        tint = mk[..., 0:1] * c1 + mk[..., 1:2] * c2 + mk[..., 2:3] * c3
        d = diffuse.astype(float) / 255.0
        d[..., :3] = d[..., :3] * (1 - mk[..., 3:4]) + tint * mk[..., 3:4]
        out = np.clip(np.floor(d * 255 + 0.5), 0, 255).astype(np.uint8)
        return out, dict(applied=True, mask="image (UV set 1 == UV set 0, resampled to the diffuse)", mask_shape=list(mask.shape[:2]),
                         colours=[[round(x, 4) for x in c] for c in (c1, c2, c3)], alpha_coverage=round(float((mk[..., 3] > 0.5).mean()), 3))
    m = m / 255.0
    if m[3] <= 0.0:
        return diffuse, dict(applied=False, reason="mask alpha 0 (fixed-colour item)", mask=[round(x, 3) for x in m])
    tint = m[0] * c1 + m[1] * c2 + m[2] * c3
    d = diffuse.astype(float) / 255.0
    d[..., :3] = d[..., :3] * (1 - m[3]) + tint[None, None, :] * m[3]
    out = np.clip(np.floor(d * 255 + 0.5), 0, 255).astype(np.uint8)
    return out, dict(applied=True, mask=[round(x, 3) for x in m], tint=[round(x, 4) for x in tint], alpha=round(float(m[3]), 3))


def tint_part(part: dict, colours3: tuple, uv_tol: float = 0.02) -> dict:
    """Tint a loaded part (avatar_compose.load_part output) in place: texture 0 (diffuse) by texture 1 (the mask)."""
    if not part["textures"] or part["textures"][0] is None:
        return dict(applied=False, reason="no diffuse texture")
    mask = part["textures"][1] if len(part["textures"]) > 1 else None
    same_uv = all(me["tex_halves"] >= 4 and float(np.abs(me["texcoords"][:, 0:2] - me["texcoords"][:, 2:4]).max()) <= uv_tol for me in part["meshes"])
    part["textures"][0], rec = tint_texture(part["textures"][0], mask, colours3, same_uv=same_uv)
    if rec.get("deferred"):
        part["tint_uv1"] = tuple(tuple(float(x) for x in c) for c in colours3)
    return rec


def default_avatar_preset(k: int) -> tuple:
    """(compose preset dict, Colours, skipped items) for built-in avatar k: only the slots compose() can bake today."""
    d = json.loads(DEFAULTS_JSON.read_text(encoding="utf-8"))["avatars"][k]
    names = {c["index"]: c["name"] for c in catalog()}
    preset, skipped = dict(frame=0), []
    for g in [e["item"] for e in d["items"] if e["item"]] + d["features"]:
        if "index" not in g:
            continue
        t = int(g["type"], 16)
        key = SLOT_KEYS.get(t)
        if key is None:
            skipped.append(dict(index=g["index"], name=names.get(g["index"]), type=g["type"]))
        elif key in preset:
            skipped.append(dict(index=g["index"], name=names.get(g["index"]), type=g["type"], reason=f"{key} already set"))
        else:
            preset[key] = g["index"]
    for key in ("body", "head", "eyebrows", "eyes", "mouth"):
        if key not in preset:
            raise ValueError(f"built-in avatar {k} lacks {key}")
    return preset, Colours.from_description(d, source=f"X360 built-in avatar {k} ({d['va']})"), skipped


if __name__ == "__main__":
    for k in range(17):
        p, c, s = default_avatar_preset(k)
        print(k, p, "skipped:", [x["name"] for x in s])
