"""Stage the source-only review repository (no supplied/derived assets, renders, binaries or private payloads).

  python tools/stage_review_repo.py base <repo>   -> files reused from C234, at their C234 bytes (research/REUSED-CODE.json)
  python tools/stage_review_repo.py head <repo>   -> this module's allowlisted sources, tests, results, docs, receipts and
                                                    ARTIFACT-HASHES.json (sha256 of every omitted build/package artifact)
  python tools/stage_review_repo.py identity <repo> <base> <head> -> review/<repo-name>-IDENTITY.json
  python tools/stage_review_repo.py bind <repo> <reviewed-head>   -> review/REVIEWED-SOURCE-BINDING.json
Git commits are made by the caller between the stages.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
C234 = MODULE.parent / "mii-restoration-001"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

TOP = ["ACK-C235.md", "ACK-C235-R1.md", "ASSIGNMENT-C235.md", "CHECKPOINTS.md", "ROOT-INBOX.md", "ROOT-INBOX-ACK.md"]
GLOBS = ["tools/*.py", "tools/vendor/*.py", "tests/*.py", "tests/results/*.json", "runtime/src/**/*", "runtime/tests/**/*",
         "docs/*", "research/*.md", "research/*.json", "research/catalog.tsv", "manager-acceptance/*.cs",
         "manager-acceptance/*.csproj", "build/bake/BAKE-RECEIPT.json", "build/select/SELECT-RECEIPT.json",
         "package/package-report.json", "package/mods/*/allstars-mod.json", "package/mods/*/mod.ini",
         "package/mods/*/plan.json", "package/mods/*/README.txt", "package/mods/*/CHECKSUMS.sha256"]
BINARY = {".dll", ".exe", ".obj", ".lib", ".exp", ".pdb", ".png", ".zif", ".zig", ".sbn", ".dat", ".zip", ".bin", ".stored"}
ARTIFACTS = ["build/**/*", "package/**/*", "runtime/build/*.dll", "runtime/build/*.exe"]


def head_files():
    out = [MODULE / p for p in TOP]
    for g in GLOBS:
        out += [p for p in MODULE.glob(g) if p.is_file() and "__pycache__" not in p.parts and p.suffix.lower() not in BINARY]
    return sorted(set(out))


def copy(src, repo, rel):
    dst = Path(repo) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def stage_base(repo):
    rows = json.loads((MODULE / "research/REUSED-CODE.json").read_text())["files"]
    for r in rows:
        src = C234 / r["path"]
        assert sha(src) == r["c234_sha256"], r["path"]
        copy(src, repo, r["path"])
    (Path(repo) / "README.md").write_text("# C235 review base\n\nFiles this module reused from C234 (Mii), at their C234 bytes. "
                                          "The head commit adds the Avatar module; adapted files show as diffs.\n")
    print(len(rows), "base files")


def stage_head(repo):
    files = head_files()
    for f in files:
        copy(f, repo, f.relative_to(MODULE).as_posix())
    (Path(repo) / "ARTIFACT-HASHES.json").write_text(json.dumps(dict(
        note="Private/derived artifacts are omitted from this repository (supplied Xbox/PC data); their hashes bind them to this source.",
        artifacts=artifact_hashes(set(files))), indent=1) + "\n")
    shutil.copyfile(MODULE / "review/REVIEW-README.md", Path(repo) / "README.md")
    print(len(files), "head files,", len(artifact_hashes(set(files))), "artifact hashes")


def artifact_hashes(files):
    arts = {}
    for g in ARTIFACTS:
        for p in MODULE.glob(g):
            if p.is_file() and p.suffix.lower() in BINARY | {".json"} and p not in files:
                arts[p.relative_to(MODULE).as_posix()] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return dict(sorted(arts.items()))


def bind(repo, reviewed):
    """Final source/artifact binding: every repo file equals the module file, the recorded artifact hashes equal the
    current artifacts, and the final head differs from the reviewed head only in documentation (*.md)."""
    repo = Path(repo)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout
    head = git("rev-parse", "HEAD").strip()
    names = [n for n in git("ls-files").split("\n") if n]
    mismatch = [n for n in names if n not in ("README.md", "ARTIFACT-HASHES.json") and sha(repo / n) != sha(MODULE / n)]
    recorded = json.loads((repo / "ARTIFACT-HASHES.json").read_text())["artifacts"]
    current = artifact_hashes(set(head_files()))
    art_diff = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
    since = [n for n in git("diff", "--name-only", reviewed, head).split("\n") if n]
    tooling = {"tools/stage_review_repo.py"}            # this review-staging/binding tool; builds no artifact
    non_doc = [n for n in since if not n.endswith(".md") and n not in tooling]
    out = dict(repo=str(repo), reviewed_head=reviewed, final_head=head, files=len(names), module_mismatches=mismatch,
               artifact_differences=art_diff, changed_since_review=since, non_doc_changes_since_review=non_doc,
               allowed_tooling_changes=sorted(tooling & set(since)),
               key_artifacts={k: current[k] for k in current if k.startswith(("build/bake/Avatar.zi", "build/select/Avatar.zi", "build/selection/"))
                              or k.endswith("alpha.2.zip") or k.endswith("AvatarXbox.dll")},
               bound=not mismatch and not art_diff and not non_doc)
    (MODULE / "review" / "REVIEWED-SOURCE-BINDING.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "key_artifacts"}, indent=1))
    return out["bound"]


def identity(repo, base, head):
    repo = Path(repo)
    names = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    names = [n for n in names if n]
    out = dict(repo=str(repo), base=base, head=head, files=len(names), source_sha256={n: sha(repo / n) for n in names})
    (MODULE / "review" / f"{repo.name}-IDENTITY.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(names), "files bound")


if __name__ == "__main__":
    {"base": lambda: stage_base(sys.argv[2]), "head": lambda: stage_head(sys.argv[2]),
     "identity": lambda: identity(sys.argv[2], sys.argv[3], sys.argv[4]),
     "bind": lambda: sys.exit(0 if bind(sys.argv[2], sys.argv[3]) else 1)}[sys.argv[1]]()
