"""Offline test of src/install_stage.py on a throwaway fake game folder: replace + add, install, restore, refusals."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import install_stage as ist

ist.game_running = lambda: False  # the real game may be open; this test never touches it


def h(b: bytes) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    """Run the scenario and assert every state."""
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        game, stage = t / "game", t / "stage"
        (game / "mods/M").mkdir(parents=True)
        (game / "mods/M/a.txt").write_bytes(b"old")
        (stage / "mods/M/new").mkdir(parents=True)
        (stage / "mods/M/a.txt").write_bytes(b"NEW")
        (stage / "mods/M/new/b.dat").write_bytes(b"added")
        files = [dict(path="mods/M/a.txt", old_sha256=h(b"old"), new_sha256=h(b"NEW"), bytes=3),
                 dict(path="mods/M/new/b.dat", old_sha256=None, new_sha256=h(b"added"), bytes=5)]
        (stage / "CHANGES.json").write_text(json.dumps(dict(files=files)))
        ist.install(game, stage, t / "bk")
        assert (game / "mods/M/a.txt").read_bytes() == b"NEW" and (game / "mods/M/new/b.dat").read_bytes() == b"added"
        try:
            ist.install(game, stage, t / "bk2")
            raise AssertionError("second install should be refused")
        except SystemExit as e:
            assert "REFUSED" in str(e)
        ist.restore(game, t / "bk")
        assert (game / "mods/M/a.txt").read_bytes() == b"old" and not (game / "mods/M/new/b.dat").exists()
        (game / "mods/M/new/b.dat").write_bytes(b"squatter")
        try:
            ist.precheck(game, stage)
            raise AssertionError("existing file to add should be refused")
        except SystemExit as e:
            assert "already exists" in str(e)
        # Deleting a file: backed up, removed, and put back by restore; refused if the live file is not the old one.
        (game / "mods/M/new/b.dat").unlink()
        (game / "mods/M/gone.rtsc").write_bytes(b"seed")
        files.append(dict(path="mods/M/gone.rtsc", old_sha256=h(b"seed"), new_sha256=None, bytes=0))
        (stage / "CHANGES.json").write_text(json.dumps(dict(files=files)))
        ist.install(game, stage, t / "bk3")
        assert not (game / "mods/M/gone.rtsc").exists() and (game / "mods/M/a.txt").read_bytes() == b"NEW"
        ist.restore(game, t / "bk3")
        assert (game / "mods/M/gone.rtsc").read_bytes() == b"seed" and (game / "mods/M/a.txt").read_bytes() == b"old"
        (game / "mods/M/gone.rtsc").write_bytes(b"changed")
        try:
            ist.precheck(game, stage)
            raise AssertionError("a changed file to delete should be refused")
        except SystemExit as e:
            assert "not the expected old version" in str(e)
    print("PASS install/restore with replaced + added + deleted files, refusals")


if __name__ == "__main__":
    main()
