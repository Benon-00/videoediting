# Roadmap & starter issues
Copy each into a GitHub issue; label `good first issue` / `help wanted` / `bounty`.

## Voice-over dubbing (help wanted)
1. Design doc: pluggable `dub_backend` interface (text, voice, language → wav + word timings).
2. Backend: local TTS adapter (e.g. Piper/Coqui) behind that interface.
3. Backend: optional voice-cloning adapter with a consent notice.
4. Align dub lines to arc stages and shots; re-time cuts around speech.
5. Speech-following sidechain ducking (replace fixed windows when a dub exists).
6. Remove/replace original clip dialogue under a dub (Demucs, `vocal_cache/`).
7. Mix QA: loudness target, overlap checks, intelligibility score.

## Good first issues
- Add `requirements.txt` pins and test on Python 3.11/3.12.
- Unit tests for the story-arc header parser.
- Linux/macOS run report.
- Sample project using CC0 footage + music.
- Edit Studio: per-field "reset to default" and input validation hints.
- Split `beat_sync_editor_new.py` (3.4k lines) into modules without changing behaviour.
