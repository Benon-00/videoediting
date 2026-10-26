# Contributing

Thanks for helping! The most wanted contribution is **background voice-over dubbing** (see the README), but everything below is welcome.

## Ways to help
- **Voice-over/dubbing** — TTS/voice-clone backends, dub-to-shot alignment, speech-aware ducking, mix quality.
- **Tests** — the story-arc parser, `plan_ducking`, DNA blending, config read/write in `server.py`.
- **Docs** — tutorials, a sample project with *legally free* footage/music, GIFs.
- **Platforms** — Linux/macOS/Apple-silicon testing; fix path/encoding bugs.
- **Performance** — faster shot scoring, GPU paths, smarter caching.
- **Edit Studio** — config validation, remix-script editor, progress bars, dark/light polish.

## Setup
```
git clone https://github.com/<your-username>/<repo>.git
python -m venv venv && activate it
pip install -r requirements.txt
```
Create a branch: `git checkout -b feat/short-name`.

## Guidelines
- **Open an issue first** for anything big (especially the dubbing design) so we agree on direction.
- Keep PRs small and focused; describe *what, why, and how you tested*.
- Match the existing style: scripts are single-file with a `CONFIG` block at the top. New settings go in `CONFIG` with a clear name — and add its one-line description in `descriptions.js`.
- Keep `CONFIG` values plain Python literals (Edit Studio reads them with `ast`, it never executes the file).
- Backends must be **optional** and degrade gracefully if their package is missing (see how mediapipe is handled).
- Prefer local/offline tools; no hard dependency on a paid API.
- **Never add copyrighted media** (footage, music, transcripts, voices). Use CC0/CC-BY samples and cite them.
- Don't commit generated folders (`outputs/`, `shot_cache/`, `*.bak`, logs) — see `.gitignore`.

## Reporting bugs
Include: OS, Python version, ffmpeg version, GPU/CPU, the command, the **log** (`beat_sync_editor.log`), and your `CONFIG` diff.

## Conduct
Be kind and constructive. 
