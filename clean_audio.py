"""
clean_audio.py

Run this AFTER clip_video.py. It scans video_cutaways/ (recursively, since
clip_video.py saves clips into per-remix-script subfolders) and for every
clip:
  1. Extracts its audio track.
  2. Separates vocals from background music/noise using Demucs — keeping
     ONLY the speech.
  3. Normalizes the loudness of the isolated voice so it sounds level and
     consistent, similar to a freshly-recorded voice track.
  4. Remuxes the cleaned audio back onto the original video (video stream
     untouched, only the audio is replaced).

Cleaned clips are saved to a separate folder (cleaned_clips/), mirroring
the same per-remix-script subfolder structure, and clips already cleaned
are skipped automatically on re-runs.

Usage:
    python clean_audio.py

No arguments — everything is controlled by the CONFIG section below.

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
pip install demucs pyloudnorm soundfile tqdm

ffmpeg must already be installed and on PATH.

First run downloads the Demucs model automatically (~80MB, one-time).
Separation runs on CPU by default and is noticeably slower than
transcription — it'll use your GPU automatically if you have a
CUDA-capable one set up with torch.
--------------------------------------------------------------------------
"""

import os
import sys
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm

# ============================== CONFIG ===================================
INPUT_FOLDER = "video_cutaways"      # where clip_video.py saved its output
OUTPUT_FOLDER = "cleaned_clips"      # where cleaned clips get saved

DEMUCS_MODEL = "htdemucs"
TARGET_LUFS = -16.0                  # target loudness for the isolated voice

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
# ===========================================================================


def find_clips(folder: str):
    """Recursively find video clips under folder (per-remix subfolders)."""
    files = []
    for root, _dirs, names in os.walk(folder):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
                files.append(os.path.join(root, name))
    return sorted(files)


def output_path_for(input_path: str) -> str:
    # Mirror the same subfolder structure under OUTPUT_FOLDER.
    rel_path = os.path.relpath(input_path, INPUT_FOLDER)
    return os.path.join(OUTPUT_FOLDER, rel_path)


def already_cleaned(input_path: str) -> bool:
    out_path = output_path_for(input_path)
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def extract_audio(video_path: str, out_wav_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "2", "-ar", "48000",
        out_wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[-500:]}")


def separate_vocals(input_wav_path: str, work_dir: str) -> str:
    """Run Demucs on the extracted audio, isolating vocals. Returns path to vocals.wav."""
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", DEMUCS_MODEL,
        "-o", work_dir,
        input_wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-800:]}")

    base = os.path.splitext(os.path.basename(input_wav_path))[0]
    vocals_path = os.path.join(work_dir, DEMUCS_MODEL, base, "vocals.wav")
    if not os.path.isfile(vocals_path):
        raise RuntimeError(f"Expected vocals output not found at {vocals_path}")
    return vocals_path


def normalize_loudness(vocals_path: str, out_wav_path: str):
    data, rate = sf.read(vocals_path)
    meter = pyln.Meter(rate)

    if np.max(np.abs(data)) < 1e-6:
        sf.write(out_wav_path, data, rate)
        return

    current_loudness = meter.integrated_loudness(data)
    normalized = pyln.normalize.loudness(data, current_loudness, TARGET_LUFS)
    normalized = np.clip(normalized, -1.0, 1.0)
    sf.write(out_wav_path, normalized, rate)


def remux_with_cleaned_audio(video_path: str, cleaned_audio_path: str, out_video_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", cleaned_audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",       # video re-encode already happened in clip_video.py — just copy
        "-c:a", "aac",
        "-shortest",
        out_video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Remux failed: {result.stderr[-500:]}")


def clean_one(video_path: str):
    out_path = output_path_for(video_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tempfile.TemporaryDirectory() as work_dir:
        extracted_wav = os.path.join(work_dir, "extracted.wav")
        extract_audio(video_path, extracted_wav)

        vocals_path = separate_vocals(extracted_wav, work_dir)

        cleaned_wav = os.path.join(work_dir, "cleaned.wav")
        normalize_loudness(vocals_path, cleaned_wav)

        remux_with_cleaned_audio(video_path, cleaned_wav, out_path)

    return out_path


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        print("Run clip_video.py first to produce clips to clean.")
        sys.exit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_clips = find_clips(INPUT_FOLDER)
    if not all_clips:
        print(f"No video clips found in: {INPUT_FOLDER}")
        return

    to_process = [c for c in all_clips if not already_cleaned(c)]
    skipped = [c for c in all_clips if already_cleaned(c)]

    print(f"Found {len(all_clips)} clip(s) in {INPUT_FOLDER}.")
    if skipped:
        print(f"Skipping {len(skipped)} already-cleaned clip(s).")

    if not to_process:
        print("Nothing new to clean. All done.")
        return

    print(f"{len(to_process)} clip(s) to clean (vocal isolation + loudness normalization).\n")

    succeeded, failed = [], []
    with tqdm(total=len(to_process), desc="Cleaning clips", unit="clip", ncols=80) as pbar:
        for clip_path in to_process:
            rel_name = os.path.relpath(clip_path, INPUT_FOLDER)
            try:
                clean_one(clip_path)
                succeeded.append(clip_path)
            except Exception as e:
                tqdm.write(f"  ERROR cleaning {rel_name}: {e}")
                failed.append(clip_path)
            pbar.update(1)

    print("\n" + "=" * 60)
    print(f"Done. {len(succeeded)} cleaned, {len(failed)} failed, {len(skipped)} skipped.")
    print(f"Cleaned clips saved to: {os.path.abspath(OUTPUT_FOLDER)}")
    if failed:
        print("\nFailed clips:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()