"""Install or restore a staged overlay (CHANGES.json from stage_avatar_fix.py) in the game folder, fail-closed.

install: refuses if the game is running, or if ANY target's live SHA-256 differs from CHANGES.json old_sha256 (so a change by
another project since staging stops it); an entry with old_sha256 null is a new file and must not exist yet; an entry with
new_sha256 null deletes the live file (backed up like any other). Backs up every live target to <backup>/files/... with a receipt, writes each staged file
via a temp file + rename, then re-hashes everything; on any error it restores the backup.
restore: puts the backed-up files back (only if the live files are still the ones this install wrote, unless --force).
check: only the pre-install checks (nothing written).
usage: install_stage.py install|check <game dir> <stage dir> <backup dir>
       install_stage.py restore <game dir> <backup dir> [--force]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

GAME_EXE = "Sonic & SEGA All-Stars Racing.exe"


def sha_file(p: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def game_running() -> bool:
    """True if the game process is listed by tasklist (CSV output: the table form cuts names to 25 characters)."""
    out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {GAME_EXE}", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
    return f'"{GAME_EXE.lower()}"' in out.lower()


def precheck(game: Path, stage: Path) -> list:
    """Return the CHANGES entries after verifying game closed, staged files intact and live files at old_sha256."""
    changes = json.loads((stage / "CHANGES.json").read_text())["files"]
    problems = []
    if game_running():
        problems.append("the game is running")
    for c in changes:
        src, dst = stage / c["path"], game / c["path"]
        if c["new_sha256"] is None:  # file to delete: nothing staged
            if c["old_sha256"] is None:
                problems.append(f"entry neither adds nor deletes: {c['path']}")
        elif not src.exists() or sha_file(src) != c["new_sha256"]:
            problems.append(f"staged file wrong/missing: {c['path']}")
        if c["old_sha256"] is None:  # added file: must not exist yet
            if dst.exists():
                problems.append(f"file to add already exists: {c['path']}")
        elif not dst.exists() or sha_file(dst) != c["old_sha256"]:
            problems.append(f"live file is not the expected old version: {c['path']} ({sha_file(dst)[:12] if dst.exists() else 'missing'} vs {c['old_sha256'][:12]})")
    if problems:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(problems))
    return changes


def restore(game: Path, backup: Path, force: bool = False) -> None:
    """Copy the backed-up files back after checking they are what install replaced."""
    receipt = json.loads((backup / "receipt.json").read_text())
    if game_running():
        raise SystemExit("REFUSED: the game is running")
    for c in receipt["files"]:
        dst = game / c["path"]
        if not force and dst.exists() and sha_file(dst) not in (c["new_sha256"], c["old_sha256"]):
            raise SystemExit(f"REFUSED: {c['path']} was changed by something else since install (use --force to overwrite)")
        if c["old_sha256"] is not None and sha_file(backup / "files" / c["path"]) != c["old_sha256"]:
            raise SystemExit(f"REFUSED: backup copy damaged: {c['path']} (nothing restored)")
    for c in receipt["files"]:
        src, dst = backup / "files" / c["path"], game / c["path"]
        if c["old_sha256"] is None:  # file was added by install: remove it
            if dst.exists():
                dst.unlink()
            continue
        assert sha_file(src) == c["old_sha256"], ("backup copy damaged", c["path"])
        tmp = dst.with_name(dst.name + ".restore-tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    bad = [c["path"] for c in receipt["files"]
           if (game / c["path"]).exists() != (c["old_sha256"] is not None)
           or (c["old_sha256"] is not None and sha_file(game / c["path"]) != c["old_sha256"])]
    if bad:
        raise SystemExit(f"RESTORE INCOMPLETE: {bad}")
    print(f"restored {len(receipt['files'])} files from {backup}")


def install(game: Path, stage: Path, backup: Path) -> None:
    """Back up, write and verify every staged file; roll back on failure."""
    changes = precheck(game, stage)
    if backup.exists():
        raise SystemExit(f"REFUSED: backup folder already exists: {backup}")
    backup.mkdir(parents=True)
    for c in changes:
        if c["old_sha256"] is None:
            continue
        dst = backup / "files" / c["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(game / c["path"], dst)
        assert sha_file(dst) == c["old_sha256"]
    (backup / "receipt.json").write_text(json.dumps(dict(at=datetime.now().isoformat(timespec="seconds"), game=str(game), stage=str(stage),
                                                         files=changes), indent=1) + "\n")
    print(f"backed up {len(changes)} files to {backup}")
    try:
        for c in changes:
            dst = game / c["path"]
            if c["new_sha256"] is None:  # backed up above; restore puts it back
                dst.unlink()
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".install-tmp")
            shutil.copyfile(stage / c["path"], tmp)
            os.replace(tmp, dst)
        bad = [c["path"] for c in changes if ((game / c["path"]).exists() if c["new_sha256"] is None
                                              else sha_file(game / c["path"]) != c["new_sha256"])]
        if bad:
            raise RuntimeError(f"post-install hash mismatch: {bad}")
    except BaseException as e:  # includes Ctrl+C: always roll back
        print(f"INSTALL FAILED ({e!r}); restoring backup")
        for c in changes:
            (game / c["path"]).with_name(Path(c["path"]).name + ".install-tmp").unlink(missing_ok=True)
        restore(game, backup, force=True)
        raise SystemExit(1)
    print(f"installed and verified {len(changes)} files")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "install":
        install(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif cmd == "check":
        precheck(Path(sys.argv[2]), Path(sys.argv[3]))
        print("pre-install checks pass")
    elif cmd == "restore":
        restore(Path(sys.argv[2]), Path(sys.argv[3]), "--force" in sys.argv)
    else:
        raise SystemExit(__doc__)
