"""Offline software renders of an X360 built-in avatar (avatar_colours.default_avatar_preset + the console colour rule),
or of a legacy PRESETS entry, using render_avatar's cameras: a 5-yaw body sheet (charts shading) and a head close-up.
usage: render_default.py <k | preset name> [--legacy] [--out build/renders] [--tag t]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import avatar_colours as ACOL
import avatar_compose as AC
import avatar_surface as AS
import render_avatar as RA

MODULE = Path(__file__).resolve().parents[1]


def main() -> None:
    """Render and write <name>-body-<tag>.png, <name>-head-<tag>.png and a provenance JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--legacy", action="store_true", help="compose without colours (C253 behaviour)")
    ap.add_argument("--out", default="build/renders")
    ap.add_argument("--tag", default="c1")
    a = ap.parse_args()
    outd = MODULE / a.out
    outd.mkdir(parents=True, exist_ok=True)
    if a.what.isdigit():
        preset, colours, skipped = ACOL.default_avatar_preset(int(a.what))
        name = f"x360-default-{int(a.what):02d}"
    else:
        preset, colours, skipped, name = a.what, None, [], a.what
    if a.legacy:
        colours, name = None, name + "-legacy"
    parts, prov = AC.compose(preset, colours=colours)
    for p in parts:
        p["charts"] = AS.charts_for_part(p, p.get("record"))
    body = RA.sheet(RA.views(parts, "charts"), outd / f"{name}-body-{a.tag}.png", name)
    hp = [p for p in parts if p["role"].startswith(("head", "hair"))]
    head = RA.sheet(RA.views(hp, "charts", size=(300, 300), center=(0, 1.36, 0.02), dist=0.9), outd / f"{name}-head-{a.tag}.png", name + " head")
    (outd / f"{name}-{a.tag}.json").write_text(json.dumps(dict(preset=preset, skipped=skipped, provenance=prov, body_png_sha256=body, head_png_sha256=head),
                                                           indent=1, default=str) + "\n")
    print(name, "body", body[:12], "head", head[:12], "skipped", [s["name"] for s in skipped])


if __name__ == "__main__":
    main()
