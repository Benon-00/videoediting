"""
clip_video.py

Reads remix script .txt files from the remix/ folder and cuts the
corresponding VIDEO segments (with their original audio intact) out of
the source .mp4. This replaces the old trim_clips.py + clip_video.py split
— video clips now carry their own audio, which clean_audio.py then cleans
up afterward.

Each .txt file in remix/ is processed independently, with its clips saved
into their own subfolder under video_cutaways/, named after the file
(e.g. remix/Enjoy The Lollipop.txt -> video_cutaways/Enjoy The Lollipop/).

Each line in a remix script can either:
  - End with an explicit timestamp range in brackets, e.g.:
      "Your target's at 12 o'clock." [47:35–47:45]
    Accepted: MM:SS or H:MM:SS, separated by a hyphen (-) or en-dash (–).
  - Have NO timestamp, e.g.:
      Because it's one thing to steal the loot.
    In this case the script searches the matching transcript (from
    transcribe.py) for the closest-matching stretch of speech.

Every cut is snapped to the transcript's own segment boundaries (so it
never lands mid-word/mid-sentence), then a padding buffer is added and
capped so it never overlaps into a neighboring cue's footage.

Usage:
    python clip_video.py

No arguments — everything is controlled by the CONFIG section below.

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
ffmpeg/ffprobe must be installed and on PATH.
pip install tqdm   (already installed if you've run the earlier scripts)
--------------------------------------------------------------------------
"""

import os
import re
import sys
import json
import difflib
import subprocess

from tqdm import tqdm

# ============================== CONFIG ===================================
# Folder containing one or more remix script .txt files, e.g.:
#   remix/Enjoy The Lollipop.txt
#   remix/Medium Friendly.txt
REMIX_FOLDER = "remix"

# The source video to cut from.
SOURCE_VIDEO_PATH = "The_Bad_Guys_2_1080P.mp4"

# The timestamped transcript already produced by transcribe.py for THIS
# video's audio track.
TRANSCRIPT_PATH = "The_Bad_Guys_2_1080P_transcript.txt"

# Base folder for output. Each remix .txt gets its own subfolder here.
OUTPUT_BASE_FOLDER = "video_cutaways"

# Buffer added before/after every clip, in seconds (capped so it never
# overlaps a neighboring clip's footage).
PADDING_SEC = 3.0

FUZZY_MATCH_WARN_THRESHOLD = 0.45
MAX_SEGMENT_WINDOW = 6

VIDEO_CODEC = "libx264"
VIDEO_CRF = "18"
VIDEO_PRESET = "veryfast"
AUDIO_CODEC = "aac"
# ===========================================================================

TIMESTAMP_BRACKET_RE = re.compile(r"\[\s*([0-9:]+)\s*[–-]\s*([0-9:]+)\s*\]\s*$")
TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\]\s*(?:Speaker\s*\d+:\s*)?(.*)$"
)


def parse_clock_timestamp(ts: str) -> float:
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unrecognized timestamp format: {ts}")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_transcript(path: str):
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = TRANSCRIPT_LINE_RE.match(line.strip())
            if not m:
                continue
            h1, m1, s1, ms1, h2, m2, s2, ms2, text = m.groups()
            start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
            end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
            text = text.strip()
            if not text or text == "[MUSIC / NON-SPEECH]":
                continue
            segments.append({"start": start, "end": end, "text": text})
    return segments


def parse_remix_script(path: str):
    """Ordered list of cues: {"text", "start" or None, "end" or None}."""
    cues = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            m = TIMESTAMP_BRACKET_RE.search(line)
            if m:
                start_str, end_str = m.groups()
                text = line[:m.start()].strip()
                try:
                    start = parse_clock_timestamp(start_str)
                    end = parse_clock_timestamp(end_str)
                except ValueError:
                    start = end = None
                cues.append({"text": text, "start": start, "end": end})
            else:
                cues.append({"text": line, "start": None, "end": None})
    return cues


def snap_to_segment_boundaries(start: float, end: float, segments):
    """Expand (start, end) to fully cover every overlapping transcript segment."""
    overlapping = [s for s in segments if s["end"] > start and s["start"] < end]
    if not overlapping:
        return start, end, []
    new_start = min(s["start"] for s in overlapping)
    new_end = max(s["end"] for s in overlapping)
    return new_start, new_end, [s["text"] for s in overlapping]


def find_best_match(cue_text: str, segments):
    """Best-matching contiguous transcript window for a sentence with no timestamp."""
    norm_cue = normalize(cue_text)
    if not norm_cue or not segments:
        return None, None, 0.0
    norm_segments = [normalize(s["text"]) for s in segments]

    best_idx, best_score = 0, -1.0
    for i, seg_text in enumerate(norm_segments):
        score = difflib.SequenceMatcher(None, norm_cue, seg_text).ratio()
        if score > best_score:
            best_idx, best_score = i, score

    best_window = (best_idx, best_idx)
    for window_size in range(1, MAX_SEGMENT_WINDOW + 1):
        end_idx = min(best_idx + window_size - 1, len(segments) - 1)
        combined = " ".join(norm_segments[best_idx:end_idx + 1])
        score = difflib.SequenceMatcher(None, norm_cue, combined).ratio()
        if score > best_score:
            best_score = score
            best_window = (best_idx, end_idx)

    start_idx, end_idx = best_window
    return segments[start_idx]["start"], segments[end_idx]["end"], best_score


def get_video_duration_sec(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(json.loads(result.stdout)["format"]["duration"])


def compute_capped_padding(results, padding_sec: float, video_duration_sec: float):
    """Cap each cue's padding so it never overlaps a neighboring cue's clip."""
    timed = [c for c in results if c["start"] is not None and c["end"] is not None]
    timed.sort(key=lambda c: c["start"])
    for i, cue in enumerate(timed):
        gap_before = timed[i - 1]["end"] if i > 0 else 0.0
        gap_before = max(0.0, cue["start"] - gap_before)
        gap_after = timed[i + 1]["start"] if i < len(timed) - 1 else video_duration_sec
        gap_after = max(0.0, gap_after - cue["end"])
        pad_before = min(padding_sec, gap_before / 2)
        pad_after = min(padding_sec, gap_after / 2)
        cue["padded_start"] = max(0.0, cue["start"] - pad_before)
        cue["padded_end"] = min(video_duration_sec, cue["end"] + pad_after)


def cut_clip(source_path: str, start: float, end: float, out_path: str, log_path: str):
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{duration:.3f}",
        "-c:v", VIDEO_CODEC, "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET,
        "-c:a", AUDIO_CODEC, "-ac", "2",  # downmix to stereo — aac can't handle raw 5.1/6ch
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        error_lines = [
            line for line in result.stderr.splitlines()
            if any(kw in line for kw in ("Error", "error", "Invalid", "No such", "Permission denied", "Cannot", "cannot"))
        ]
        summary = "\n".join(error_lines[-5:]) if error_lines else result.stderr[-300:]
        raise RuntimeError(f"{summary}\n  (full log: {log_path})")

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"ffmpeg reported success but output file is missing/empty: {out_path}")

    try:
        os.remove(log_path)
    except OSError:
        pass


def process_remix_video_file(script_path: str, output_folder: str, video_duration_sec: float, segments):
    os.makedirs(output_folder, exist_ok=True)

    print(f"\nParsing remix script: {script_path}")
    cues = parse_remix_script(script_path)
    print(f"  {len(cues)} line(s) to cut.")
    if not cues:
        print("  No lines found. Skipping.")
        return

    warnings = []
    results = []

    # --- Pass 1: resolve a start/end timestamp for every cue ---
    with tqdm(total=len(cues), desc="Locating timestamps", unit="line", ncols=80) as pbar:
        for i, cue in enumerate(cues, start=1):
            num = f"{i:03}"

            if cue["start"] is not None and cue["end"] is not None:
                snapped_start, snapped_end, matched_texts = snap_to_segment_boundaries(
                    cue["start"], cue["end"], segments
                )
                if matched_texts:
                    cue["start"], cue["end"] = snapped_start, snapped_end
                    cue["source"] = "explicit (snapped to whole words)"
                    cue["score"] = 1.0
                else:
                    cue["source"] = "explicit (no transcript overlap found)"
                    cue["score"] = 0.0
            else:
                start, end, score = find_best_match(cue["text"], segments)
                cue["start"], cue["end"] = start, end
                cue["source"] = "matched"
                cue["score"] = score

            if cue["start"] is None or cue["end"] is None:
                tqdm.write(f"  [{num}] SKIPPED — could not determine timestamp for: \"{cue['text'][:60]}\"")
                warnings.append(cue)
            elif cue["end"] <= cue["start"]:
                tqdm.write(f"  [{num}] SKIPPED — end time is not after start time for: \"{cue['text'][:60]}\"")
                warnings.append(cue)
                cue["start"] = None
            elif cue["score"] < FUZZY_MATCH_WARN_THRESHOLD:
                tqdm.write(f"  [{num}] LOW-CONFIDENCE ({cue['source']}, score {cue['score']:.2f}) for: \"{cue['text'][:60]}\"")
                warnings.append(cue)

            results.append(cue)
            pbar.update(1)

    compute_capped_padding(results, PADDING_SEC, video_duration_sec)

    # --- Pass 2: cut and export ---
    saved = 0
    with tqdm(total=len(results), desc="Cutting video clips", unit="clip", ncols=80) as pbar:
        for i, cue in enumerate(results, start=1):
            num = f"{i:03}"
            if cue["start"] is None or cue["end"] is None:
                pbar.update(1)
                continue

            out_path = os.path.join(output_folder, f"{num}.mp4")
            log_path = os.path.join(output_folder, f"{num}_ffmpeg_error.log")
            try:
                cut_clip(SOURCE_VIDEO_PATH, cue["padded_start"], cue["padded_end"], out_path, log_path)
                saved += 1
            except Exception as e:
                tqdm.write(f"  [{num}] ERROR cutting clip: {e}")
                warnings.append(cue)

            pbar.update(1)

    print(f"  Done. {saved} clip(s) saved to: {os.path.abspath(output_folder)}")
    if warnings:
        print(f"  {len(warnings)} line(s) need a manual check:")
        for cue in warnings:
            print(f"    - \"{cue['text'][:70]}\"")


def main():
    if not os.path.isdir(REMIX_FOLDER):
        print(f"ERROR: Remix folder not found: {REMIX_FOLDER}")
        sys.exit(1)
    if not os.path.isfile(SOURCE_VIDEO_PATH):
        print(f"ERROR: Source video not found at: {SOURCE_VIDEO_PATH}")
        sys.exit(1)
    if not os.path.isfile(TRANSCRIPT_PATH):
        print(f"ERROR: Transcript not found at: {TRANSCRIPT_PATH}")
        print("Run transcribe.py on this video's audio first.")
        sys.exit(1)

    script_files = sorted(
        f for f in os.listdir(REMIX_FOLDER)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(REMIX_FOLDER, f))
    )
    if not script_files:
        print(f"No .txt remix scripts found in: {REMIX_FOLDER}")
        return

    print(f"Found {len(script_files)} remix script(s) in {REMIX_FOLDER}:")
    for f in script_files:
        print(f"  - {f}")

    print(f"\nReading video duration: {SOURCE_VIDEO_PATH}")
    video_duration_sec = get_video_duration_sec(SOURCE_VIDEO_PATH)

    print(f"\nParsing transcript: {TRANSCRIPT_PATH}")
    segments = parse_transcript(TRANSCRIPT_PATH)
    print(f"  {len(segments)} speech segment(s) available for lookup.")

    os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

    for script_file in script_files:
        script_path = os.path.join(REMIX_FOLDER, script_file)
        folder_name = os.path.splitext(script_file)[0]
        output_folder = os.path.join(OUTPUT_BASE_FOLDER, folder_name)

        print("\n" + "=" * 60)
        print(f"Processing: {script_file}  ->  {output_folder}")
        process_remix_video_file(script_path, output_folder, video_duration_sec, segments)

    print("\n" + "=" * 60)
    print("All remix scripts processed.")


if __name__ == "__main__":
    main()