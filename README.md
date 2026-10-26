# Procedural Video Editing (Anime-Edit)

**Turn long footage + a music track into a beat-synced, story-arc edit — automatically, and tune every knob from a browser.**

Point it at your source footage, a song, and a short script (Hook → Rising Action → Conflict → Comeback → Payoff). It detects shots, cuts on the beat, picks shots whose dialogue matches your script, copies the pacing/transition/colour style of any reference edit you feed it, ducks the music, and renders the video.

> **Looking for contributors:** the biggest unsolved piece is **background voice-over dubbing while the music and VFX keep playing**. See [The big open problem](#the-big-open-problem-background-voice-over-dubbing) — it's where help matters most.

<!-- Add a GIF/screenshot of Edit Studio and a 20s sample edit here. Nothing sells a repo like a demo. -->

## Features
- **Beat-synced cutting** — librosa beat/onset/energy analysis; fast cuts on loud parts, slow on quiet parts.
- **Story-arc scripts** — a plain `.txt` with Hook / Rising Action / Conflict / Comeback / Payoff sections drives *what* is shown *when*; shots are chosen by fuzzy-matching the script to the movie's transcript.
- **Editing DNA** — `edit_analyzer_full.py` measures ~50 dimensions of a reference edit (cut rate, transitions, camera motion, colour, audio sync) and the editor imitates it.
- **Transition engine** — 200+ unique recipes per edit (ffmpeg xfade, procedural expressions, ~120 gl-transitions shaders); a ledger guarantees no recipe repeats across edits. Fade-to-black is banned by default.
- **VFX library** — colour grades, shakes, zoom-punches, chosen by energy.
- **Audio ducking** — periodic voice-over windows (default 5 s window / gap configurable) or per-stage levels.
- **Content-aware shot scoring** — motion (optical flow) and face detection (MediaPipe).
- **Resumable** — shot cache + checkpoints, so interrupted runs continue.
- **Edit Studio** — a local web UI (`index.html` + `server.py`) to edit every `CONFIG` value with a one-line explanation, save it into the scripts, and run them with live logs.

## The pipeline

| # | Script | Does | Reads → Writes |
|---|---|---|---|
| 1 | `transcribe.py` | Whisper transcript with timestamps, speakers, music flags | media → `<name>_transcript.txt` |
| 2 | `clip_video.py` | Cuts the moments named in quote-style remix scripts | `remix/*.txt` + movie + transcript → `video_cutaways/<script>/` |
| 3 | `clean_audio.py` | Demucs voice isolation + loudness normalisation (optional) | `video_cutaways/` → `cleaned_clips/` |
| 4 | `edit_analyzer_full.py` | Measures a reference edit's style | `videos_to_analyze/<niche>/` → `dna_modules/<niche>/*.json` |
| 5 | `beat_sync_editor_new.py` | Builds the final edit | `source_footage/` + `music/` + `remix/<track>.txt` + transcripts + DNA → `outputs/` |

Helpers: `webrtcvad.py` (stub so Resemblyzer imports without compiling a C extension — keep it next to `transcribe.py`), `yolov8n.pt` (YOLO weights used by the analyzer).

> Two script formats share `remix/`: `clip_video.py` wants quote lines (optionally ending in `[mm:ss–mm:ss]`); `beat_sync_editor_new.py` wants arc sections and is matched to the music file by name (`music/song.mp3` → `remix/song.txt`).

## Install

Requirements: **Python 3.9+** and **ffmpeg + ffprobe on PATH**. A CUDA GPU helps (Whisper, Demucs) but isn't required.

```powershell
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
python -m venv venv
.\venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-analyzer.txt   # only if you'll run edit_analyzer_full.py
```

## Quick start

1. Put your movie/episode in `source_footage/`, a song in `music/`.
2. Transcribe the footage: `python transcribe.py` (set `SCAN_FOLDER` first; see Edit Studio below).
3. Map it in `beat_sync_editor_new.py`: `TRANSCRIPT_MAP = {"Movie.mp4": "Movie_transcript.txt"}`.
4. Write `remix/<songname>.txt`:
   ```
   Hook
   Your target's at 12 o'clock.
   Rising Action
   Because it's one thing to steal the loot.
   Conflict
   ...
   Comeback
   ...
   Payoff
   ...
   ```
   Repeat the cycle (a new `Hook`) every ~30 s of the edit.
5. *(Optional)* put reference edits in `videos_to_analyze/<niche>/`, run `python edit_analyzer_full.py`, then set `DNA_MODULES = ["<niche>"]`.
6. Render: `python beat_sync_editor_new.py --duration 60` → `outputs/`.

Useful flags: `--duration 60`, `--start 30`, `--dna NAME[,NAME2]`, `--dna-join inner|left|right|full_outer|all`, `--build-transition-sets 30`, `--transition-report`, `--online-refresh`.

## Edit Studio (the frontend)

Editing `CONFIG` blocks by hand is tedious, so there's a local UI. From the project folder:

```powershell
python server.py --project .
```
then open **http://127.0.0.1:8765**. Every setting has a one-line explanation; **Save** writes into the `.py` file (a `.bak` backup is made once) and **Run** streams the logs. It only listens on `127.0.0.1`, runs only the five scripts, and uses the standard library only.

## The big open problem: background voice-over dubbing

The goal: while the music and VFX play, a **voice-over (dub) runs underneath** — a character line, a narrator, or a translated/re-voiced track — placed automatically and mixed so the music ducks cleanly and the picture stays on the beat.

**What exists today**
- Music ducking windows (`DUCK_MODE`, `VO_WINDOW_SEC`, `VO_GAP_SEC`, `PERIODIC_DUCK_LEVEL`, EQ "voice pocket").
- The edit's own clip dialogue can be brought up inside those windows (`FINAL_AUDIO_MODE = "mix"`).
- A separate voice file can be supplied in `voiceover/<track>.wav` and blended (`FINAL_AUDIO_MODE = "blend"`), with speech-segment detection.

**What's missing (this is where you come in)**
- Generating the dub: TTS / voice cloning / translation from the arc script, with a pluggable backend (local-first, no mandatory paid API).
- Aligning dub lines to the right shots and to the beat — time-stretching or re-timing the *edit* around speech, not just ducking around fixed windows.
- Keeping original clip dialogue out of the way (or replacing it) when a dub speaks — Demucs separation (`clean_audio.py`, `vocal_cache/`) is the starting point.
- Mixing quality: sidechain-style ducking that follows actual speech, de-essing, loudness (-14 LUFS), no pumping.
- Automatic checks: dub never overlaps a lyric hook; intelligibility scoring.

If you have ideas, a prototype or even a benchmark clip, please open an issue or a draft PR. Small, well-scoped contributions are welcome — a backend adapter, a test, a better ducking curve.

## Contributing

All skill levels welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Good first areas: docs, tests for the script parser/ducking planner, a GPU/CPU benchmark table, new transition recipes, Linux/macOS testing, config validation in Edit Studio.

## Support the project

If this saves you time, you can support development (from **$10**) via **[GitHub Sponsors](https://github.com/sponsors/<your-username>)** · **[Ko-fi](https://ko-fi.com/<your-name>)**. Donations fund compute for testing, and bounties for the voice-over work. Thank you!

## Legal / responsible use
This tool is a *processor*. **Do not commit or distribute copyrighted footage, music or transcripts** — the repo ships none. You are responsible for having rights to the media you edit and for complying with platform rules and local law (fair use/fair dealing varies). If you build voice cloning, only clone voices you have consent to use.

## License
MIT — see [LICENSE](LICENSE). Third-party tools keep their own licences (ffmpeg, Demucs, Whisper/faster-whisper, YOLOv8 [AGPL-3.0 via Ultralytics], MediaPipe, gl-transitions, xfade-easing [MIT]). Note: YOLOv8/Ultralytics is AGPL-3.0 — check that it fits how you distribute.
