"""Small original software rasterizer (NumPy) for offline inspection renders of decoded assets.

Draw items carry world-space positions/normals/uvs, triangle corner indices, a shading function and blend/cull
settings. Perspective camera, z-buffer, per-pixel barycentric interpolation, Lambert + ambient lighting,
straight-alpha blending for translucent passes. It renders what it is given; it invents no geometry.
"""
import math

import numpy as np


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    V = np.eye(4)
    V[0, :3], V[1, :3], V[2, :3] = r, u, -f
    V[:3, 3] = -V[:3, :3] @ eye
    return V


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    P = np.zeros((4, 4))
    P[0, 0], P[1, 1] = f / aspect, f
    P[2, 2], P[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    P[3, 2] = -1
    return P


class Renderer:
    def __init__(self, w, h, bg=(40, 44, 52), ss=2):
        self.W, self.H, self.ss = w * ss, h * ss, ss
        self.out_size = (w, h)
        self.color = np.zeros((self.H, self.W, 3))
        self.color[:] = bg
        self.depth = np.full((self.H, self.W), np.inf)
        self.stats = dict(items=0, triangles=0, fragments=0)

    def set_camera(self, eye, target, fovy=30.0, near=0.05, far=100.0, up=(0, 1, 0)):
        self.eye = np.asarray(eye, float)
        self.V = look_at(eye, target, up)
        self.P = perspective(fovy, self.W / self.H, near, far)

    def draw(self, pos, nrm, uv, tris, shade, blend=False, cull="back", front_ccw=True, light_dir=(0.3, 0.6, 0.8), ambient=0.45):
        """tris: int array [T,3,k] with k=2 (pos,nrm) or 3 (pos,nrm,uv) corner indices.
        shade(normal_world[N,3], uv[N,2]) -> (rgb[N,3] 0..255, alpha[N] 0..1)."""
        self.stats["items"] += 1
        if len(tris) == 0:
            return
        L = np.asarray(light_dir, float)
        L /= np.linalg.norm(L)
        hp = np.concatenate([pos, np.ones((len(pos), 1))], 1)
        clip = (self.P @ self.V @ hp.T).T
        ndc = clip[:, :3] / clip[:, 3:4]
        sx = (ndc[:, 0] * 0.5 + 0.5) * self.W
        sy = (1 - (ndc[:, 1] * 0.5 + 0.5)) * self.H
        sz = ndc[:, 2]
        wv = clip[:, 3]
        for tri in tris:
            pi = tri[:, 0]
            if np.any(wv[pi] <= 0):
                continue
            x, y, z = sx[pi], sy[pi], sz[pi]
            area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            if abs(area) < 1e-12:
                continue
            # screen y is down: a CCW world triangle appears with negative area here
            facing_front = (area < 0) if front_ccw else (area > 0)
            if cull == "back" and not facing_front:
                continue
            if cull == "front" and facing_front:
                continue
            x0, x1 = max(int(math.floor(x.min())), 0), min(int(math.ceil(x.max())), self.W - 1)
            y0, y1 = max(int(math.floor(y.min())), 0), min(int(math.ceil(y.max())), self.H - 1)
            if x0 > x1 or y0 > y1:
                continue
            ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px, py = xs + 0.5, ys + 0.5
            w0 = ((x[1] - px) * (y[2] - py) - (x[2] - px) * (y[1] - py)) / area
            w1 = ((x[2] - px) * (y[0] - py) - (x[0] - px) * (y[2] - py)) / area
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            # perspective-correct weights
            iw = 1.0 / wv[pi]
            b = np.stack([w0 * iw[0], w1 * iw[1], w2 * iw[2]], -1)
            b /= b.sum(-1, keepdims=True)
            zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
            sub_d = self.depth[y0:y1 + 1, x0:x1 + 1]
            ok = inside & (zz <= sub_d + 1e-9)
            if not ok.any():
                continue
            bb = b[ok]
            n = bb @ nrm[tri[:, 1]]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            t = bb @ uv[tri[:, 2]] if tri.shape[1] > 2 else np.zeros((len(bb), 2))
            rgb, alpha = shade(n, t)
            lit = ambient + (1 - ambient) * np.clip(n @ L, 0, 1)
            rgb = np.clip(rgb * lit[:, None], 0, 255)
            sub_c = self.color[y0:y1 + 1, x0:x1 + 1]
            if blend:
                a = alpha[:, None]
                keep = alpha > 0
                idx = np.argwhere(ok)[keep]
                cur = sub_c[ok][keep]
                sub_c[idx[:, 0], idx[:, 1]] = rgb[keep] * a[keep] + cur * (1 - a[keep])
                sub_d[idx[:, 0], idx[:, 1]] = zz[ok][keep]
            else:
                sub_c[ok] = rgb
                sub_d[ok] = zz[ok]
            self.stats["triangles"] += 1
            self.stats["fragments"] += int(ok.sum())

    def image(self):
        from PIL import Image
        img = Image.fromarray(np.clip(self.color, 0, 255).astype(np.uint8))
        return img.resize(self.out_size, Image.LANCZOS) if self.ss > 1 else img
