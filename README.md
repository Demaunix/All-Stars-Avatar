# All-Stars Avatar

Source code for the **Xbox 360 Avatar** racer in **Sonic & SEGA All-Stars Racing** on Steam PC (2010).

On the Xbox 360, players could race as their own Xbox Avatar, with the **Party Power** All-Star move. The PC version still holds the Avatar's rig, car and animations, but its avatar model is a four-vertex placeholder, because the Xbox 360 built the avatar while the game ran. This mod rebuilds the avatar from the **Avatar Asset Package that ships with the Xbox 360 game**: body, head, hair, clothes and colours. It uses no other avatar source and needs no Microsoft servers.

This repository holds **only our own code and tools**. It contains no game files, no console files and no built mod. Everything that comes from a game has to be made from **your own copies**. The scripts check those inputs by SHA-256 before using them.

## Status

Not released yet.

| Part | State |
|---|---|
| Racing | Drives and races on PC. The driver is the console's native avatar size. |
| Party Power | Works, with the four helpers at the size the Xbox 360 trailer shows. |
| Open | Some ear and sleeve clipping. The podium scene is untested. |
| In progress | Custom avatars: pick items and colours from the Xbox 360 avatar set, then race as them. |

Nothing here is called "1:1 with the Xbox 360" until every difference has been checked in game.

## What is in here

| Path | What it is |
|---|---|
| `src/avatar/tools/` | The pipeline. It decodes the Xbox 360 avatar models and textures (`avatar_model.py`, `avatar_texture.py`, `catalog.py`) and applies the console's colour rule (`avatar_colours.py`). It reads the Xbox avatar description format (`avatar_desc.py`), fits the parts to the PC rig (`bone_map.py`, `fit_space.py`, `garment_offsets.py`) and bakes the race and selection-screen models (`bake_avatar.py`, `bake_custom.py`, `build_selection.py`). Offline renders and checks are included too. |
| `src/avatar/tools/nexia_pack_adapt.py` | Lets the Nexia360 Xbox 360 emulator read the same Avatar Asset Package, for reference checks against the Xbox 360 game. |
| `src/avatar/tests/` | Offline tests of the decoder, the bake and the texture atlas. |
| `src/avatar_scale_fix.py` | Sets the driver to the console's avatar size. |
| `src/avatar_rig_info.py`, `src/avatar_feet_check.py`, `src/kart_wheel_check.py` | Read-only checks of the rig, the feet and the kart. |
| `src/vendor/` | Readers for the PC game's model, animation and archive formats. |
| `src/install_stage.py`, `tests/test_install_stage.py` | Installer that refuses to act when unsure, backs up every file it replaces, and can restore it. |

## What is not in here, and why

- **Game and console files:** models, textures, animations, archives and executables of any version. They belong to their owners. Bring your own.
- **Data made from console files:** the decoded Avatar Asset Package blocks (`src/avatar/input/`) and the fit and reference tables (`src/avatar/research/`).
- **The built mod and its package files.** The pipeline makes the Avatar's model files. Packaging them as an installable mod is not part of this repository yet.
- **Work notes.** Some files mention `notes\...` files or scripts that are not included. Those belong to the author's private work setup.

### Inputs not included

| Needs | Where it comes from |
|---|---|
| Decoded Avatar Asset Package blocks and `AvatarAssetPack.toc` (`src/avatar/input/`) | the Xbox 360 game's own Avatar Asset Package. Every file is listed with its SHA-256 in `src/avatar/INPUT-PINS.sha256`. |
| Fit and reference tables (`src/avatar/research/`, such as `BONE-MAP.json`, `SPACE-FIT.json`, `GARMENT-OFFSETS.json`, `SELECTION-BUILD.json`, `xbox-default-avatars.json`) | built from the Xbox 360 game's data. Not every tool that made them is included yet. |
| The PC game's own Avatar files | your Steam PC game folder (`extract_pc_avatar.py`; some scripts read the folder from `ASR_GAME_DIR`). |

## Building (for developers)

You need:
- the Steam PC game and the Xbox 360 game files;
- Python 3.12 with NumPy and Pillow;
- [ChaoGarage](https://github.com/Demaunix/ChaoGarage/releases/latest), the mod loader (separate download, never bundled).

Some scripts still point at the original author's folders: the game folder on drive A:, and private folders shown as `<codex-archive>` or `<reference-library>`. Change them to your own locations before running. A wrong input is refused instead of producing a broken model.

## Legal

Not affiliated with or endorsed by SEGA, Sumo Digital or Microsoft. Sonic & SEGA All-Stars Racing belongs to SEGA. Xbox Avatars belong to Microsoft. Our code is released under the MIT licence (see `LICENSE`). That licence covers none of those games or their files.
