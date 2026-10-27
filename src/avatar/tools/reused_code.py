"""Provenance of code reused from the C234 Mii module (read-only sibling): writes research/REUSED-CODE.json with the
SHA-256 of every same-named file here and there, marking identical copies and adapted files."""
import hashlib
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
C234 = MODULE.parent / "mii-restoration-001"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

if __name__ == "__main__":
    rows = []
    files = sorted(list(MODULE.glob("tools/**/*.py")) + list(MODULE.glob("tests/*.py")) + list(MODULE.glob("runtime/src/**/*"))
                   + list(MODULE.glob("runtime/tests/*")))
    for f in files:
        rel = f.relative_to(MODULE)
        o = C234 / rel
        if f.is_file() and "build" not in rel.parts and o.is_file():
            a, b = sha(f), sha(o)
            rows.append(dict(path=rel.as_posix(), status="identical copy" if a == b else "adapted", sha256=a, c234_sha256=b))
    out = dict(source_module="../mii-restoration-001 (C234, read-only)", files=rows,
               note=("tools/ssrt_native.py and tools/vendor/* are byte-identical C234 copies; their docstrings inherit C234's "
                     "statement about the earlier metal-sonic origin. The upstream metal-sonic file hashes were not re-derived here; "
                     "this record binds the copies to the C234 files they were taken from."))
    (MODULE / "research" / "REUSED-CODE.json").write_text(json.dumps(out, indent=1) + "\n")
    print(sum(r["status"] == "identical copy" for r in rows), "identical,", sum(r["status"] == "adapted" for r in rows), "adapted")
