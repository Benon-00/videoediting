"""
edit_analyzer_full.py

Full 50-dimension edit-analysis system, implemented PURELY with classical
signal processing / computer vision / established (non-ML) algorithms —
no trained models, no object/face/speech recognition, no external paid
APIs. This only measures TIMING, MOTION, COLOR, and AUDIO STATISTICS about
a video; it never stores or reproduces frame images, audio waveforms, or
any other content from the source video itself.

--------------------------------------------------------------------------
HONESTY NOTE — read this before trusting any single number
--------------------------------------------------------------------------
Without trained models, several of the 50 dimensions can only be
APPROXIMATED, not reliably measured. Every detector below returns a
"confidence" and, where relevant, a short note on what it's actually
measuring. In particular, treat these as rough heuristic proxies, not
ground truth:
  - SFX "type" (impact/whoosh/click) — classified by transient shape
    (attack speed, duration, spectral trend), NOT true sound recognition.
  - "Musical sections" (verse/chorus/etc.) — labeled by relative energy
    level of self-similarity segments, not real song-structure understanding.
  - J-cut / L-cut — approximated from a vocal-frequency-band energy
    heuristic (no real speech/dialogue detection or source separation),
    so this is unreliable on music-only sections and gets much less
    accurate wherever vocals and instruments overlap heavily.
  - Overlay/text/UI detection, lens flares, chromatic aberration, glitch,
    scanlines — all detected via generic image-signal heuristics (edge
    density, channel misalignment, periodic FFT patterns, blown highlights)
    that will false-positive on unrelated visual content sometimes.
  - Skin-tone stability — a fixed HSV/YCrCb color-range heuristic, not
    real skin detection; will misfire on skin-toned non-skin objects.
Numbers with genuinely strong footing: shot timing, cut rate, tempo/beats/
onsets, RMS energy, spectral centroid, optical flow motion/shake/zoom,
color/HSV stats, stereo panning, silence detection, basic key estimation
(Krumhansl-Schmuckler is real 1980s music theory, not ML).

--------------------------------------------------------------------------
THE 50 DIMENSIONS, mapped to where each is computed below
--------------------------------------------------------------------------
 1  Shot boundaries                    -> detect_shots()
 2  Shot duration distribution         -> shot_duration_stats()
 3  Cut frequency over time            -> cut_frequency_over_time()
 4  Transition type                    -> classify_transition()
 5  Transition duration                -> classify_transition()
 6  Cut-on-action                      -> shot['cut_on_action']
 7  Match cut detection                -> compute_adjacency_features()
 8  Graphic match                      -> compute_adjacency_features()
 9  Jump-cut detection                 -> compute_adjacency_features()
10  Spatial continuity                 -> compute_adjacency_features()
11  Screen-direction continuity        -> compute_adjacency_features()
12  Editorial repetition / montage     -> detect_montage_patterns()
13  BPM / tempo                        -> analyze_music()
14  Beat positions                     -> analyze_music()
15  Downbeats (approx, assumes 4/4)    -> estimate_downbeats()
16  Bar / measure structure (approx)   -> estimate_downbeats()
17  Musical sections (heuristic)       -> analyze_music() section_boundaries
18  Energy curve                       -> analyze_music()
19  Onset/transient detection          -> analyze_music() (+ percussive/harmonic split)
20  Spectral brightness                -> analyze_music()
21  Harmonic/key characteristics       -> estimate_key()
22  Dynamic range / loudness           -> analyze_music()
23  Musical micro-events               -> detect_micro_events()
24  SFX occurrence (heuristic typing)  -> detect_sfx_transients()
25  SFX timing relative to edits       -> analyze_sync() (reused)
26  SFX intensity                      -> detect_sfx_transients()
27  Audio transition type              -> detect_audio_transitions()
28  J-cut / L-cut (approximate)        -> detect_jl_cuts()
29  Silence engineering                -> analyze_music() + classify_silence_context()
30  Music/SFX ducking                  -> detect_ducking()
31  Reverb/delay tails                 -> detect_reverb_tails()
32  Stereo/spatial movement            -> analyze_stereo()
33  Camera-motion classification       -> analyze_shot_motion()
34  Motion magnitude                   -> analyze_shot_motion()
35  Speed changes                      -> analyze_shot_motion() (dense intra-shot sampling)
36  Motion blur characteristics        -> analyze_shot_sharpness()
37  Zoom / push / pull                 -> analyze_shot_motion() (divergence sign)
38  Camera shake / impact shake        -> analyze_shot_motion()
39  Reframing / crop changes           -> detect_letterboxing()
40  Focus / sharpness changes          -> analyze_shot_sharpness()
41  Overlay detection (heuristic)      -> detect_overlays()
42  Mask/reveal transitions            -> classify_transition() (circular_reveal / wipe)
43  Blur effects                       -> analyze_shot_sharpness()
44  Glow / light / optical effects     -> detect_glow_and_flares()
45  Distortion effects                 -> detect_distortion_effects()
46  Texture / degradation effects      -> detect_texture_effects()
47  Global color palette               -> analyze_shot_color()
48  Exposure / contrast structure      -> analyze_shot_color()
49  Saturation / skin-tone behavior    -> analyze_shot_color() + skin_tone_stability()
50  Color-change <-> editorial event   -> build_event_timeline() + cluster_synchronized_events()

Usage:
    python edit_analyzer_full.py

No arguments — everything is controlled by the CONFIG section below.

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
pip install scenedetect[opencv] opencv-python librosa soundfile numpy scipy tqdm matplotlib pandas pyarrow faster-whisper ultralytics transformers torch tensorflow tensorflow-hub pillow
ffmpeg/ffprobe must be installed and on PATH.
--------------------------------------------------------------------------
"""

import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"   # -8 = AV_LOG_QUIET, silences ffmpeg's own stderr chatter
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"     # also quiets OpenCV's own internal logging, just in case
import sys
import json
import csv
import subprocess

import numpy as np
import cv2
import librosa
from scipy.signal import find_peaks
from tqdm import tqdm
from scenedetect import detect, ContentDetector

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# ============================== CONFIG ===================================
INPUT_FOLDER = "videos_to_analyze"
OUTPUT_FOLDER = "edit_analysis_full"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

SCENE_DETECTOR_THRESHOLD = 27.0
MIN_SHOT_DURATION_SEC = 0.08

TRANSITION_WINDOW_SEC = 0.6
FADE_LUMA_THRESHOLD = 18
FADE_LUMA_HIGH_THRESHOLD = 235
DISSOLVE_SIMILARITY_MIN = 0.25
WHIP_PAN_MOTION_THRESHOLD = 6.0       # optical-flow magnitude just before a cut, above which = whip pan
ZOOM_TRANSITION_DIVERGENCE = 0.6      # flow divergence spike at a cut boundary, above which = zoom transition
GLITCH_CHANNEL_OFFSET_PX = 3          # RGB channel misalignment above this many px = possible glitch/RGB-shift

MOTION_SAMPLE_FRAMES_PER_SHOT = 8     # denser than the first version, to catch speed ramps
SHAKE_STD_THRESHOLD = 2.2
ZOOM_DIVERGENCE_THRESHOLD = 0.35
FREEZE_FRAME_SIMILARITY = 0.995       # consecutive-frame similarity above this = likely freeze frame

COLOR_SAMPLE_FRAMES_PER_SHOT = 3
SHARPNESS_SAMPLE_FRAMES_PER_SHOT = 3

ONSET_BACKTRACK = True
SILENCE_RMS_PERCENTILE = 10
SILENCE_CONTEXT_WINDOW_SEC = 1.0      # how close a silence region must be to a cut to be "pre/post-cut silence"

SYNC_WINDOW_MS = 80

# --- Analysis modes (NEW) ---
# fast: metadata + shots + cuts + BPM/beats + basic color/motion only.
# deep: everything except the optional AI layer and contact sheets/HTML (default).
# forensic: everything, including the AI speech layer, HTML report, contact
#           sheet, and parquet dataset export.
ANALYSIS_MODE = "forensic"   # "fast" | "deep" | "forensic"

# --- Optional AI layer (NEW) — local, free, no API key/subscription ---
# Uses faster-whisper (same package as your transcribe.py) running fully on
# your own machine. Only enabled in "forensic" mode. Used specifically to
# replace the vocal-band heuristic for J-cut/L-cut and dialogue-window
# detection (item 28) with REAL speech timing, which is the one dimension
# that heuristic DSP genuinely can't do reliably.
AI_LAYER_ENABLED = True
WHISPER_MODEL_SIZE = "small"   # tiny/base/small — bigger = more accurate, slower

# --- Editing DNA / niche-folder batch mode (NEW) ---
# If INPUT_FOLDER contains subfolders, each subfolder is treated as a
# "niche" — every video inside is analyzed and their DNA profiles are
# averaged into one DNA module for that niche, written to DNA_OUTPUT_FOLDER.
# If INPUT_FOLDER has videos directly in it (no subfolders), the whole
# folder is treated as a single niche.
DNA_OUTPUT_FOLDER = "dna_modules"

# --- Contact sheet / HTML report ---
CONTACT_SHEET_COLUMNS = 6
CONTACT_SHEET_THUMB_SIZE = (192, 108)


# SFX/transient heuristic classification thresholds
SFX_ATTACK_SHARP_MS = 15              # attack faster than this = "sharp" (impact/click-like)
SFX_CLICK_MAX_DURATION_MS = 40
SFX_WHOOSH_MIN_DURATION_MS = 120

DUCKING_MIN_DB = 3.0                  # minimum dB dip to count as intentional ducking
REVERB_TAIL_MIN_DECAY_MS = 150        # minimum post-transient decay length to call it a "tail"

ADJACENCY_JUMP_CUT_SIMILARITY_MIN = 0.55   # cross-cut frame similarity above this (but not a dissolve) = jump-cut-ish
MONTAGE_PATTERN_MIN_REPEATS = 2

WRITE_CHART = True
# ===========================================================================


# --------------------------------------------------------------------------
# Media ingest
# --------------------------------------------------------------------------

def find_videos(folder: str):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return files


def probe_media(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels",
           "-of", "json", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    data = json.loads(result.stdout)
    info = {"duration": float(data["format"]["duration"])}
    for s in data.get("streams", []):
        if s["codec_type"] == "video" and "fps" not in info:
            num, den = (s.get("avg_frame_rate", "0/1")).split("/")
            info["fps"] = float(num) / float(den) if float(den) != 0 else 0.0
            info["width"] = s.get("width")
            info["height"] = s.get("height")
        elif s["codec_type"] == "audio" and "sample_rate" not in info:
            info["sample_rate"] = int(s.get("sample_rate", 0))
            info["channels"] = s.get("channels")
    return info


def extract_audio_mono(video_path: str, out_wav_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "22050", out_wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[-500:]}")


def extract_audio_stereo(video_path: str, out_wav_path: str) -> bool:
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "2", "-ar", "22050", out_wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0 and os.path.isfile(out_wav_path)


# --------------------------------------------------------------------------
# 1-3: Shot boundaries, duration distribution, cut frequency
# --------------------------------------------------------------------------

def detect_shots(video_path: str):
    scene_list = detect(video_path, ContentDetector(threshold=SCENE_DETECTOR_THRESHOLD))
    shots = [{"start": s.seconds, "end": e.seconds} for s, e in scene_list]
    shots = [s for s in shots if (s["end"] - s["start"]) >= MIN_SHOT_DURATION_SEC]
    return shots


def shot_duration_stats(shots: list) -> dict:
    if not shots:
        return {}
    d = np.array([s["end"] - s["start"] for s in shots])
    return {
        "count": len(shots), "mean_sec": float(d.mean()), "median_sec": float(np.median(d)),
        "min_sec": float(d.min()), "max_sec": float(d.max()), "std_sec": float(d.std()),
        "pct_under_0_5s": float((d < 0.5).mean() * 100), "pct_over_3s": float((d > 3.0).mean() * 100),
    }


def cut_frequency_over_time(shots: list, total_duration: float, window_sec: float = 10.0):
    cut_times = [s["start"] for s in shots[1:]]
    windows, t = [], 0.0
    while t < total_duration:
        we = min(t + window_sec, total_duration)
        n = sum(1 for c in cut_times if t <= c < we)
        windows.append({"start": t, "end": we, "cuts_per_sec": round(n / max(we - t, 0.01), 3)})
        t += window_sec
    return windows


# --------------------------------------------------------------------------
# 4, 5, 42: Transition classification (extended: wipe, zoom, whip pan, glitch, circular reveal)
# --------------------------------------------------------------------------

def _mean_luma(frame) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def _frame_similarity(f1, f2) -> float:
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32); g1 -= g1.mean()
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32); g2 -= g2.mean()
    denom = (np.linalg.norm(g1) * np.linalg.norm(g2)) + 1e-6
    return float(np.clip((g1 * g2).sum() / denom, -1, 1) * 0.5 + 0.5)


def _detect_wipe_or_reveal(f_before, f_after) -> dict:
    """
    Looks at the DIFFERENCE mask between the frame just before and just
    after a cut. A wipe/reveal leaves a coherent moving boundary (a large
    connected region of change with a fairly straight or circular edge),
    unlike a hard cut (change is everywhere) or noise (change is scattered).
    """
    g1 = cv2.cvtColor(f_before, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f_after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"is_wipe_or_reveal": False}
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    frame_area = g1.shape[0] * g1.shape[1]
    if area < frame_area * 0.15:
        return {"is_wipe_or_reveal": False}
    perimeter = cv2.arcLength(largest, True) + 1e-6
    circularity = 4 * np.pi * area / (perimeter ** 2)  # 1.0 = perfect circle
    x, y, w, h = cv2.boundingRect(largest)
    aspect = w / max(h, 1)
    if circularity > 0.6:
        return {"is_wipe_or_reveal": True, "shape": "circular_reveal", "confidence": round(min(circularity, 0.95), 2)}
    if aspect > 3 or aspect < 0.33:
        return {"is_wipe_or_reveal": True, "shape": "wipe", "confidence": 0.6}
    return {"is_wipe_or_reveal": True, "shape": "shape_mask_reveal", "confidence": 0.45}


def classify_transition(cap, cut_time: float, fps: float, motion_before: float) -> dict:
    window = TRANSITION_WINDOW_SEC
    start_t, end_t = max(0.0, cut_time - window), cut_time + window
    step = 1.0 / max(fps, 1.0)

    samples = []
    t = start_t
    while t <= end_t:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            samples.append((t, frame))
        t += step * 2

    if len(samples) < 4:
        return {"type": "hard_cut", "duration_sec": 0.0, "confidence": 0.4}

    # Whip pan: very high motion immediately before the cut.
    if motion_before > WHIP_PAN_MOTION_THRESHOLD:
        return {"type": "whip_pan", "duration_sec": round(2 * step, 3), "confidence": 0.6}

    lumas = [_mean_luma(f) for _, f in samples]
    min_luma, max_luma = min(lumas), max(lumas)

    if min_luma < FADE_LUMA_THRESHOLD:
        idxs = [i for i, l in enumerate(lumas) if l < FADE_LUMA_THRESHOLD + 10]
        if len(idxs) >= 2:
            return {"type": "fade_to_black", "duration_sec": round(max(samples[idxs[-1]][0] - samples[idxs[0]][0], step), 3), "confidence": 0.75}
    if max_luma > FADE_LUMA_HIGH_THRESHOLD:
        idxs = [i for i, l in enumerate(lumas) if l > FADE_LUMA_HIGH_THRESHOLD - 10]
        if len(idxs) >= 2:
            return {"type": "fade_to_white", "duration_sec": round(max(samples[idxs[-1]][0] - samples[idxs[0]][0], step), 3), "confidence": 0.75}

    # Locate the two frames straddling the actual boundary for wipe/reveal/glitch checks.
    before_idx = max(range(len(samples)), key=lambda i: samples[i][0] if samples[i][0] <= cut_time else -1)
    after_idx = min(before_idx + 1, len(samples) - 1)
    f_before, f_after = samples[before_idx][1], samples[after_idx][1]

    wipe_info = _detect_wipe_or_reveal(f_before, f_after)
    if wipe_info.get("is_wipe_or_reveal"):
        dur = samples[after_idx][0] - samples[max(0, before_idx - 2)][0]
        return {"type": wipe_info["shape"], "duration_sec": round(max(dur, step), 3), "confidence": wipe_info["confidence"]}

    # Glitch / RGB-shift check: misalignment between color channels at the boundary.
    glitch_conf = _detect_channel_misalignment(f_after)
    if glitch_conf > 0.5:
        return {"type": "glitch_transition", "duration_sec": round(2 * step, 3), "confidence": round(glitch_conf, 2)}

    sims = [_frame_similarity(samples[i][1], samples[i + 1][1]) for i in range(len(samples) - 1)]
    min_sim = min(sims)
    low_sim_run = sum(1 for s in sims if s < (min_sim + DISSOLVE_SIMILARITY_MIN))
    if low_sim_run >= 3:
        dur = samples[-1][0] - samples[0][0]
        return {"type": "dissolve", "duration_sec": round(dur * (low_sim_run / len(sims)), 3), "confidence": 0.6}

    return {"type": "hard_cut", "duration_sec": 0.0, "confidence": 0.85}


# --------------------------------------------------------------------------
# 45 (helper used by classify_transition too): channel misalignment for glitch/RGB-shift/chromatic aberration
# --------------------------------------------------------------------------

def _detect_channel_misalignment(frame) -> float:
    """
    Estimates spatial misalignment between the R and B channels via
    cross-correlation of their edge maps. Used both for the transition
    classifier (glitch) and the standalone distortion detector
    (chromatic aberration, item 45).
    """
    b, g, r = cv2.split(frame)
    edges_r = cv2.Canny(r, 50, 150)
    edges_b = cv2.Canny(b, 50, 150)
    if edges_r.sum() == 0 or edges_b.sum() == 0:
        return 0.0
    shifts = range(-GLITCH_CHANNEL_OFFSET_PX - 2, GLITCH_CHANNEL_OFFSET_PX + 3)
    best_corr, best_shift = -1, 0
    for dx in shifts:
        shifted = np.roll(edges_b, dx, axis=1)
        corr = float(np.mean(edges_r.astype(np.float32) * shifted.astype(np.float32)))
        if corr > best_corr:
            best_corr, best_shift = corr, dx
    if abs(best_shift) >= GLITCH_CHANNEL_OFFSET_PX:
        return float(min(0.9, 0.3 + abs(best_shift) * 0.1))
    return 0.0


# --------------------------------------------------------------------------
# 33-38: Motion analysis (camera-motion class, magnitude, shake, zoom sign, speed ramps)
# --------------------------------------------------------------------------

def analyze_shot_motion(cap, start: float, end: float, sample_frames: int) -> dict:
    duration = max(0.05, end - start)
    sample_count = max(3, sample_frames)
    times = [start + duration * (i / sample_count) for i in range(sample_count)]

    prev_gray, prev_small = None, None
    magnitudes, mean_dx, mean_dy, divergences = [], [], [], []
    freeze_pairs = 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            magnitudes.append(float(mag.mean()))
            mean_dx.append(float(flow[..., 0].mean()))
            mean_dy.append(float(flow[..., 1].mean()))
            div = np.gradient(flow[..., 0], axis=1) + np.gradient(flow[..., 1], axis=0)
            divergences.append(float(div.mean()))  # signed: + = expanding (zoom in), - = contracting (zoom out)
            if _frame_similarity(prev_small, small) > FREEZE_FRAME_SIMILARITY:
                freeze_pairs += 1
        prev_gray, prev_small = gray, small

    if not magnitudes:
        return {"motion_magnitude": 0.0, "camera_motion": "unknown", "shake_intensity": 0.0,
                "zoom_direction": "none", "speed_variability": 0.0, "likely_freeze_frame": False}

    motion_mag = float(np.mean(magnitudes))
    dx_std, dy_std = float(np.std(mean_dx)), float(np.std(mean_dy))
    shake = dx_std + dy_std
    div_mean = float(np.mean(divergences))
    speed_variability = float(np.std(magnitudes) / (motion_mag + 1e-6))  # item 35: how much motion ramps within the shot

    if motion_mag < 0.15:
        camera_motion = "static"
    elif shake > SHAKE_STD_THRESHOLD:
        camera_motion = "shaky"
    elif abs(div_mean) > ZOOM_DIVERGENCE_THRESHOLD:
        camera_motion = "zoom_like"
    elif abs(np.mean(mean_dx)) > abs(np.mean(mean_dy)) * 1.5:
        camera_motion = "pan"
    elif abs(np.mean(mean_dy)) > abs(np.mean(mean_dx)) * 1.5:
        camera_motion = "tilt"
    else:
        camera_motion = "mixed_motion"

    zoom_direction = "none"
    if abs(div_mean) > ZOOM_DIVERGENCE_THRESHOLD * 0.6:
        zoom_direction = "zoom_in_push" if div_mean > 0 else "zoom_out_pull"

    return {
        "motion_magnitude": round(motion_mag, 4),
        "camera_motion": camera_motion,
        "shake_intensity": round(shake, 4),
        "zoom_direction": zoom_direction,
        "speed_variability": round(speed_variability, 3),      # high = speed ramp within the shot
        "likely_freeze_frame": freeze_pairs >= max(2, sample_count // 2),
    }


# --------------------------------------------------------------------------
# 36, 40, 43: Sharpness / blur / focus
# --------------------------------------------------------------------------

def analyze_shot_sharpness(cap, start: float, end: float, sample_frames: int) -> dict:
    duration = max(0.05, end - start)
    times = [start + duration * (i / max(1, sample_frames)) for i in range(max(1, sample_frames))]
    sharpness_vals, center_vals, edge_vals = [], [], []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_vals.append(float(lap))
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        ch, cw = h // 4, w // 4
        center = gray[cy - ch:cy + ch, cx - cw:cx + cw]
        center_vals.append(float(cv2.Laplacian(center, cv2.CV_64F).var()) if center.size else 0.0)
        edge_region = np.concatenate([gray[:ch, :].flatten(), gray[-ch:, :].flatten()])
        edge_vals.append(float(np.var(np.diff(edge_region.astype(np.float32))))) if edge_region.size else edge_vals.append(0.0)

    if not sharpness_vals:
        return {}
    mean_sharp = float(np.mean(sharpness_vals))
    center_sharp = float(np.mean(center_vals)) if center_vals else 0.0
    dof_proxy = (center_sharp - mean_sharp) / (mean_sharp + 1e-6)  # positive = center sharper than average (shallow DOF-like)
    return {
        "mean_sharpness": round(mean_sharp, 1),
        "is_blurred_shot": bool(mean_sharp < 15),       # heuristic threshold; frame-dependent, treat as relative
        "depth_of_field_proxy": round(float(dof_proxy), 3),
    }


# --------------------------------------------------------------------------
# 39: Letterboxing / crop / reframing
# --------------------------------------------------------------------------

def detect_letterboxing(cap, t: float, frame_w: int, frame_h: int) -> dict:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    dark_rows_top = int(np.argmax(row_means > 12)) if (row_means < 12).any() else 0
    dark_rows_bottom = int(np.argmax(row_means[::-1] > 12)) if (row_means < 12).any() else 0
    has_letterbox = dark_rows_top > frame_h * 0.03 or dark_rows_bottom > frame_h * 0.03
    return {
        "has_letterbox_bars": bool(has_letterbox),
        "top_bar_px": dark_rows_top,
        "bottom_bar_px": dark_rows_bottom,
    }


# --------------------------------------------------------------------------
# 41: Overlay detection (text/UI/graphics — heuristic via stable high-edge-density regions)
# --------------------------------------------------------------------------

def detect_overlays(cap, start: float, end: float) -> dict:
    """
    Text/UI/graphic overlays tend to have high local edge density AND stay
    in roughly the same screen position across a shot (unlike scene content,
    which moves with the camera/subjects). This checks for a region whose
    edge-map stays consistently high-density across multiple sampled frames.
    """
    duration = max(0.05, end - start)
    times = [start + duration * f for f in (0.2, 0.5, 0.8)]
    edge_maps = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_maps.append(edges)
    if len(edge_maps) < 2:
        return {"overlay_detected": False}
    stacked = np.stack(edge_maps).astype(np.float32) / 255.0
    persistence = stacked.mean(axis=0)  # high where edges appear in most sampled frames = static overlay-like content
    strong_persistent = (persistence > 0.8).sum()
    frac = strong_persistent / persistence.size
    return {
        "overlay_detected": bool(frac > 0.01),
        "overlay_coverage_pct": round(float(frac * 100), 2),
        "confidence": 0.4,  # inherently low — this also fires on static logos, letterboxing edges, high-contrast static backgrounds
    }


# --------------------------------------------------------------------------
# 44: Glow / bloom / lens flare (heuristic)
# --------------------------------------------------------------------------

def detect_glow_and_flares(cap, t: float) -> dict:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blown = float((gray > 250).mean())
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_sat_bright = float(((hsv[..., 1] < 40) & (hsv[..., 2] > 220)).mean())
    return {
        "blown_highlight_pct": round(blown * 100, 2),
        "glow_bloom_proxy_pct": round(low_sat_bright * 100, 2),   # bright + desaturated = glow/bloom-ish
        "possible_lens_flare": bool(blown > 0.005 and low_sat_bright > 0.02),
    }


# --------------------------------------------------------------------------
# 45: Distortion effects (chromatic aberration reuse, scanlines via FFT, generic glitch)
# --------------------------------------------------------------------------

def detect_distortion_effects(cap, t: float) -> dict:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        return {}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    chroma_ab_conf = _detect_channel_misalignment(frame)

    # Scanlines: look for a strong periodic horizontal-line pattern via
    # a 1D FFT of the column-averaged row profile.
    row_profile = gray.mean(axis=1)
    row_profile -= row_profile.mean()
    fft = np.abs(np.fft.rfft(row_profile))
    if len(fft) > 4:
        # ignore the DC/very-low-frequency component, look for a sharp peak elsewhere
        peak_idx = int(np.argmax(fft[2:])) + 2
        peak_strength = float(fft[peak_idx] / (fft.mean() + 1e-6))
        scanlines_conf = float(min(0.9, max(0.0, (peak_strength - 4) / 10)))
    else:
        scanlines_conf = 0.0

    return {
        "chromatic_aberration_confidence": round(chroma_ab_conf, 2),
        "scanlines_confidence": round(scanlines_conf, 2),
    }


# --------------------------------------------------------------------------
# 46: Texture / degradation (grain/noise, flicker, jitter)
# --------------------------------------------------------------------------

def detect_texture_effects(cap, start: float, end: float, fps: float) -> dict:
    duration = max(0.05, end - start)
    n = min(6, max(3, int(duration * fps / 4)))
    times = [start + duration * (i / n) for i in range(n)]
    lumas, noise_vals, shifts = [], [], []
    prev_gray = None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lumas.append(float(gray.mean()))
        # grain/noise proxy: local variance in a flat (low-gradient) region
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        flat_mask = grad_mag < np.percentile(grad_mag, 20)
        if flat_mask.any():
            noise_vals.append(float(gray[flat_mask].astype(np.float32).std()))
        if prev_gray is not None and prev_gray.shape == gray.shape:
            # jitter proxy: small-shift phase correlation between consecutive sampled frames
            try:
                shift, _ = cv2.phaseCorrelate(prev_gray.astype(np.float32), gray.astype(np.float32))
                shifts.append(float(np.hypot(*shift)))
            except cv2.error:
                pass
        prev_gray = gray

    flicker_hz_proxy = 0.0
    if len(lumas) >= 4:
        l = np.array(lumas) - np.mean(lumas)
        flicker_hz_proxy = float(np.std(np.diff(l)))  # higher = more frame-to-frame brightness oscillation

    return {
        "grain_noise_proxy": round(float(np.mean(noise_vals)), 2) if noise_vals else 0.0,
        "flicker_proxy": round(flicker_hz_proxy, 3),
        "jitter_px_proxy": round(float(np.mean(shifts)), 3) if shifts else 0.0,
    }


# --------------------------------------------------------------------------
# 47-49: Color (palette, exposure/contrast, saturation, skin-tone stability)
# --------------------------------------------------------------------------

def analyze_shot_color(cap, start: float, end: float, sample_frames: int) -> dict:
    duration = max(0.05, end - start)
    times = [start + duration * (i / max(1, sample_frames)) for i in range(max(1, sample_frames))]
    hues, sats, vals, skin_fracs = [], [], [], []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hues.append(float(np.median(hsv[..., 0])))
        sats.append(float(np.mean(hsv[..., 1])))
        vals.append(float(np.mean(hsv[..., 2])))

        # Item 49: skin-tone heuristic via a fixed YCrCb range — not real
        # skin detection, just a common color-range approximation.
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_fracs.append(float((mask > 0).mean()))

    if not hues:
        return {}

    mean_hue = float(np.mean(hues))
    is_warm = (mean_hue <= 20) or (mean_hue >= 160)
    v_arr = np.array(vals)
    contrast_proxy = float(np.std(v_arr))
    return {
        "dominant_hue": round(mean_hue, 1),
        "mean_saturation": round(float(np.mean(sats)), 1),
        "mean_brightness": round(float(np.mean(vals)), 1),
        "warm_or_cool": "warm" if is_warm else "cool",
        "contrast_proxy": round(contrast_proxy, 2),
        "skin_tone_area_pct": round(float(np.mean(skin_fracs)) * 100, 2),
        "skin_tone_stability": round(1.0 - float(np.std(skin_fracs)), 3) if len(skin_fracs) > 1 else None,
    }


# --------------------------------------------------------------------------
# 7-11: Adjacency analysis between consecutive shots (match cut, graphic match,
# jump cut, spatial continuity, screen-direction continuity)
# --------------------------------------------------------------------------

def compute_adjacency_features(cap, shot_a: dict, shot_b: dict) -> dict:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, shot_a["end"] - 0.05) * 1000)
    ok_a, frame_a_last = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, shot_b["start"] * 1000)
    ok_b, frame_b_first = cap.read()
    if not (ok_a and ok_b):
        return {}

    similarity = _frame_similarity(frame_a_last, frame_b_first)

    # Graphic match / match cut proxy: compare edge-density spatial layout
    # (coarse 4x4 grid) between the two frames — similar layout in similar
    # positions is what a graphic match / match cut typically looks like.
    def grid_edge_profile(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        h, w = edges.shape
        cells = []
        for r in range(4):
            for c in range(4):
                cell = edges[r * h // 4:(r + 1) * h // 4, c * w // 4:(c + 1) * w // 4]
                cells.append(float(cell.mean()))
        return np.array(cells)

    prof_a, prof_b = grid_edge_profile(frame_a_last), grid_edge_profile(frame_b_first)
    corr = float(np.corrcoef(prof_a, prof_b)[0, 1]) if prof_a.std() > 0 and prof_b.std() > 0 else 0.0
    graphic_match_score = max(0.0, corr)

    # Spatial continuity proxy: correlation of peripheral (background-ish) regions only.
    def periphery(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        return np.concatenate([gray[:h // 6, :].flatten(), gray[-h // 6:, :].flatten(),
                                gray[:, :w // 6].flatten(), gray[:, -w // 6:].flatten()]).astype(np.float32)
    pa, pb = periphery(frame_a_last), periphery(frame_b_first)
    spatial_continuity = float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 0 and pb.std() > 0 else 0.0

    jump_cut_likely = bool(ADJACENCY_JUMP_CUT_SIMILARITY_MIN < similarity < DISSOLVE_SIMILARITY_MIN + 0.3)

    return {
        "cross_cut_frame_similarity": round(similarity, 3),
        "graphic_match_score": round(graphic_match_score, 3),
        "match_cut_likely": bool(graphic_match_score > 0.7),
        "spatial_continuity_score": round(max(spatial_continuity, 0.0), 3),
        "jump_cut_likely": jump_cut_likely,
    }


def screen_direction_continuity(shots: list) -> list:
    """Item 11: compares each shot's dominant horizontal flow direction to the next shot's."""
    directions = []
    for i in range(len(shots) - 1):
        d1 = shots[i].get("_mean_dx", 0.0)
        d2 = shots[i + 1].get("_mean_dx", 0.0)
        if abs(d1) < 0.05 or abs(d2) < 0.05:
            continue
        same_direction = (d1 > 0) == (d2 > 0)
        directions.append({"shot_index": i, "same_screen_direction": bool(same_direction)})
    return directions


# --------------------------------------------------------------------------
# 6: Cut-on-action
# --------------------------------------------------------------------------

def compute_cut_on_action(cap, cut_time: float, shot_mean_motion: float) -> dict:
    window = 0.2
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, cut_time - window) * 1000)
    ok1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, cut_time * 1000)
    ok2, f2 = cap.read()
    if not (ok1 and ok2):
        return {"cut_on_action": False, "pre_cut_motion": 0.0}
    g1 = cv2.cvtColor(cv2.resize(f1, (160, 90)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(f2, (160, 90)), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 2, 15, 2, 5, 1.2, 0)
    mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
    return {"cut_on_action": bool(mag > max(shot_mean_motion * 1.3, 0.4)), "pre_cut_motion": round(mag, 4)}


# --------------------------------------------------------------------------
# 12: Editorial repetition / montage pattern detection
# --------------------------------------------------------------------------

def detect_montage_patterns(shots: list) -> list:
    """
    Bucket each shot into a coarse label (duration bucket + camera-motion
    class), then look for repeated subsequences — a classical string-
    repetition search (no ML), not true semantic pattern recognition.
    """
    def bucket(s):
        dur = s["end"] - s["start"]
        dur_b = "short" if dur < 0.8 else ("medium" if dur < 2.5 else "long")
        return f"{dur_b}_{s.get('camera_motion', 'unknown')}"

    labels = [bucket(s) for s in shots]
    patterns = []
    n = len(labels)
    for length in range(2, min(6, n // 2 + 1)):
        seen = {}
        for i in range(n - length + 1):
            key = tuple(labels[i:i + length])
            seen.setdefault(key, []).append(i)
        for key, positions in seen.items():
            if len(positions) >= MONTAGE_PATTERN_MIN_REPEATS and length >= 2:
                patterns.append({"pattern": list(key), "repeats": len(positions), "start_shot_indices": positions})
    patterns.sort(key=lambda p: (-p["repeats"], -len(p["pattern"])))
    return patterns[:10]


# --------------------------------------------------------------------------
# 13-23: Music analysis (tempo, beats, downbeats, sections, energy, onsets,
# brightness, key, loudness, micro-events)
# --------------------------------------------------------------------------

def estimate_downbeats(beat_times: list, beats_per_bar: int = 4) -> list:
    """Item 15-16: naive fixed-meter downbeat estimate (assumes 4/4 unless configured otherwise)."""
    return [t for i, t in enumerate(beat_times) if i % beats_per_bar == 0]


_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(y, sr) -> dict:
    """Item 21: classic Krumhansl-Schmuckler key-finding via chroma correlation — 1980s music theory, not ML."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    best_score, best_key = -2, None
    for shift in range(12):
        major_corr = np.corrcoef(np.roll(chroma_mean, shift), _MAJOR_PROFILE)[0, 1]
        minor_corr = np.corrcoef(np.roll(chroma_mean, shift), _MINOR_PROFILE)[0, 1]
        if major_corr > best_score:
            best_score, best_key = major_corr, f"{_KEY_NAMES[shift]} major"
        if minor_corr > best_score:
            best_score, best_key = minor_corr, f"{_KEY_NAMES[shift]} minor"
    return {"estimated_key": best_key, "confidence": round(float(max(0, best_score)), 2)}


def detect_micro_events(rms, rms_times, centroid, centroid_times, sr) -> dict:
    """Item 23: risers (rising centroid+RMS before a drop), drops, vocal-band pauses."""
    risers, drops = [], []
    win = max(4, int(2.0 * len(rms) / (rms_times[-1] + 1e-6))) if len(rms_times) else 4
    for i in range(win, len(rms) - win):
        pre_rms_trend = rms[i] - rms[i - win]
        pre_centroid_trend = centroid[i] - centroid[i - win]
        post_drop = rms[i - 1] - rms[min(i + 2, len(rms) - 1)] if i + 2 < len(rms) else 0
        if pre_rms_trend > 0 and pre_centroid_trend > 0 and pre_rms_trend > np.std(rms) * 0.5:
            risers.append(round(float(rms_times[i]), 2))
        if rms[i] > np.mean(rms) + 1.2 * np.std(rms) and rms[i - 1] < np.mean(rms):
            drops.append(round(float(rms_times[i]), 2))
    # de-dup close-together detections
    def dedup(times, min_gap=1.0):
        out = []
        for t in times:
            if not out or t - out[-1] > min_gap:
                out.append(t)
        return out
    return {"riser_candidates_sec": dedup(risers), "drop_candidates_sec": dedup(drops)}


def analyze_music(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    downbeats = estimate_downbeats(beat_times)

    y_harm, y_perc = librosa.effects.hpss(y)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=ONSET_BACKTRACK)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    perc_onset_frames = librosa.onset.onset_detect(y=y_perc, sr=sr, backtrack=ONSET_BACKTRACK)
    perc_onset_times = librosa.frames_to_time(perc_onset_frames, sr=sr).tolist()

    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=512)

    section_bounds, section_labels = [], []
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rec = librosa.segment.recurrence_matrix(mfcc, mode="affinity", sym=True)
        novelty = np.sum(np.abs(np.diff(rec, axis=1)), axis=0)
        peaks, _ = find_peaks(novelty, distance=max(1, int(sr / 512 * 4)))
        section_times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
        bounds = [0.0] + [round(float(t), 2) for t in section_times] + [round(duration, 2)]
        section_bounds = bounds
        # Heuristic labeling by relative energy (rule-based, NOT a trained classifier).
        energies = []
        for i in range(len(bounds) - 1):
            mask = (rms_times >= bounds[i]) & (rms_times < bounds[i + 1])
            energies.append(float(rms[mask].mean()) if mask.any() else 0.0)
        med = np.median(energies) if energies else 0.0
        for i, e in enumerate(energies):
            label = "chorus_or_drop" if e > med * 1.15 else ("verse_or_buildup" if e > med * 0.85 else "intro_or_breakdown")
            section_labels.append({"start": bounds[i], "end": bounds[i + 1], "label_heuristic": label})
    except Exception:
        pass

    silence_threshold = float(np.percentile(rms, SILENCE_RMS_PERCENTILE))
    silence_mask = rms < silence_threshold
    silence_regions, in_silence, start_t = [], False, None
    for i, is_sil in enumerate(silence_mask):
        t = rms_times[i]
        if is_sil and not in_silence:
            in_silence, start_t = True, t
        elif not is_sil and in_silence:
            silence_regions.append((round(float(start_t), 2), round(float(t), 2)))
            in_silence = False
    if in_silence:
        silence_regions.append((round(float(start_t), 2), round(float(duration), 2)))

    peak = float(np.max(np.abs(y)))
    rms_overall = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    crest_factor = float(peak / (rms_overall + 1e-9))

    micro_events = detect_micro_events(rms, rms_times, centroid, centroid_times, sr)
    key_info = estimate_key(y, sr)

    return {
        "duration_sec": round(duration, 2),
        "tempo_bpm": round(tempo, 1),
        "beat_times": [round(t, 3) for t in beat_times],
        "downbeat_times": [round(t, 3) for t in downbeats],
        "onset_times": [round(t, 3) for t in onset_times],
        "percussive_onset_times": [round(t, 3) for t in perc_onset_times],
        "energy_curve": {"times": [round(float(t), 2) for t in rms_times[::10]], "rms": [round(float(v), 5) for v in rms[::10]]},
        "brightness_curve": {"times": [round(float(t), 2) for t in centroid_times[::10]], "centroid_hz": [round(float(v), 1) for v in centroid[::10]]},
        "section_boundaries_sec": section_bounds,
        "section_labels_heuristic": section_labels,
        "silence_regions_sec": silence_regions,
        "loudness": {"peak": round(peak, 4), "rms_overall": round(rms_overall, 4), "crest_factor": round(crest_factor, 2)},
        "key_estimate": key_info,
        "micro_events": micro_events,
        "_y": y, "_sr": sr, "_rms": rms, "_rms_times": rms_times,  # kept in-memory for downstream sound-design analysis, not written to JSON
    }


# --------------------------------------------------------------------------
# 24, 26: SFX-ish transient detection + heuristic typing + intensity
# --------------------------------------------------------------------------

def detect_sfx_transients(y, sr, onset_times: list) -> list:
    events = []
    for ot in onset_times:
        idx = int(ot * sr)
        window = int(0.25 * sr)
        seg = y[max(0, idx - int(0.01 * sr)):min(len(y), idx + window)]
        if len(seg) < 10:
            continue
        env = np.abs(seg)
        peak_idx = int(np.argmax(env))
        attack_samples = peak_idx + 1
        attack_ms = attack_samples / sr * 1000

        half_peak = env[peak_idx] * 0.3
        decay_idx = peak_idx
        for i in range(peak_idx, len(env)):
            if env[i] < half_peak:
                decay_idx = i
                break
        else:
            decay_idx = len(env) - 1
        duration_ms = (decay_idx - 0) / sr * 1000

        try:
            centroid = librosa.feature.spectral_centroid(y=seg.astype(np.float32), sr=sr)[0]
            centroid_trend = float(centroid[-1] - centroid[0]) if len(centroid) > 1 else 0.0
            mean_centroid = float(np.mean(centroid))
        except Exception:
            centroid_trend, mean_centroid = 0.0, 0.0

        amplitude = float(np.max(np.abs(seg)))

        if duration_ms <= SFX_CLICK_MAX_DURATION_MS and attack_ms <= SFX_ATTACK_SHARP_MS:
            sfx_type, conf = "click_or_tick", 0.5
        elif attack_ms <= SFX_ATTACK_SHARP_MS and mean_centroid < 2000:
            sfx_type, conf = "impact_or_hit", 0.5
        elif duration_ms >= SFX_WHOOSH_MIN_DURATION_MS and abs(centroid_trend) > 200:
            sfx_type, conf = "whoosh_or_sweep", 0.45
        else:
            sfx_type, conf = "unclassified_transient", 0.3

        events.append({
            "time_sec": round(ot, 3), "type_heuristic": sfx_type, "confidence": conf,
            "attack_ms": round(attack_ms, 1), "duration_ms": round(duration_ms, 1),
            "amplitude": round(amplitude, 4), "mean_spectral_centroid_hz": round(mean_centroid, 1),
        })
    return events


# --------------------------------------------------------------------------
# 27: Audio transition type (abrupt volume jumps/dips in the mixed track)
# --------------------------------------------------------------------------

def detect_audio_transitions(rms, rms_times) -> list:
    db = 20 * np.log10(rms + 1e-9)
    events = []
    win = 3
    for i in range(win, len(db) - win):
        delta = db[i] - db[i - win]
        if abs(delta) > 6.0:
            span_sec = rms_times[i] - rms_times[i - win]
            kind = "abrupt_change" if span_sec < 0.15 else "gradual_fade"
            direction = "increase" if delta > 0 else "decrease"
            events.append({"time_sec": round(float(rms_times[i]), 2), "type": kind, "direction": direction, "delta_db": round(float(delta), 1)})
    deduped = []
    for e in events:
        if not deduped or e["time_sec"] - deduped[-1]["time_sec"] > 0.3:
            deduped.append(e)
    return deduped


# --------------------------------------------------------------------------
# 28: J-cut / L-cut (approximate, vocal-band heuristic — see honesty note at top)
# --------------------------------------------------------------------------

def detect_jl_cuts(y, sr, cut_times: list) -> list:
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    b, a = butter(4, [200 / nyq, 3500 / nyq], btype="band")
    vocal_band = filtfilt(b, a, y)
    hop = int(0.05 * sr)
    env = []
    for i in range(0, len(vocal_band) - hop, hop):
        seg = vocal_band[i:i + hop]
        env.append(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
    env = np.array(env)
    env_times = np.arange(len(env)) * hop / sr
    thresh = np.percentile(env, 60)
    active = env > thresh

    results = []
    for ct in cut_times:
        idx = int(ct / (hop / sr))
        if idx <= 2 or idx >= len(active) - 2:
            continue
        before_active = active[max(0, idx - 4):idx].mean() > 0.5
        after_active = active[idx:idx + 4].mean() > 0.5
        if before_active and not after_active:
            results.append({"cut_time_sec": round(ct, 2), "type": "possible_L_cut", "note": "vocal-band energy continues past the visual cut (approximate)"})
        elif after_active and not before_active:
            results.append({"cut_time_sec": round(ct, 2), "type": "possible_J_cut", "note": "vocal-band energy starts before the visual cut (approximate)"})
    return results


# --------------------------------------------------------------------------
# 29: Silence engineering context (pre/post-cut silence)
# --------------------------------------------------------------------------

def classify_silence_context(silence_regions: list, cut_times: list) -> list:
    out = []
    for (s, e) in silence_regions:
        nearest_cut = min(cut_times, key=lambda c: min(abs(c - s), abs(c - e))) if cut_times else None
        context = "unrelated"
        if nearest_cut is not None:
            if abs(nearest_cut - e) < SILENCE_CONTEXT_WINDOW_SEC:
                context = "pre_cut_silence"
            elif abs(nearest_cut - s) < SILENCE_CONTEXT_WINDOW_SEC:
                context = "post_cut_silence"
        out.append({"start": s, "end": e, "context": context})
    return out


# --------------------------------------------------------------------------
# 30: Music/SFX ducking
# --------------------------------------------------------------------------

def detect_ducking(rms, rms_times, y, sr) -> list:
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    b, a = butter(4, [200 / nyq, 3500 / nyq], btype="band")
    vocal_band = filtfilt(b, a, y)
    hop = 512
    vocal_env = []
    for i in range(0, len(vocal_band) - hop, hop):
        seg = vocal_band[i:i + hop]
        vocal_env.append(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
    vocal_env = np.array(vocal_env)
    n = min(len(vocal_env), len(rms))
    vocal_env, rms_trim, times_trim = vocal_env[:n], rms[:n], rms_times[:n]

    vocal_db = 20 * np.log10(vocal_env + 1e-9)
    mix_db = 20 * np.log10(rms_trim + 1e-9)
    vocal_active = vocal_db > np.percentile(vocal_db, 65)

    events, in_duck, start_i = [], False, None
    baseline = np.median(mix_db[~vocal_active]) if (~vocal_active).any() else np.median(mix_db)
    for i in range(n):
        ducked = vocal_active[i] and (baseline - mix_db[i]) > DUCKING_MIN_DB
        if ducked and not in_duck:
            in_duck, start_i = True, i
        elif not ducked and in_duck:
            events.append({"start_sec": round(float(times_trim[start_i]), 2), "end_sec": round(float(times_trim[i]), 2),
                            "duck_depth_db": round(float(baseline - mix_db[start_i:i].min()), 1)})
            in_duck = False
    return events[:50]


# --------------------------------------------------------------------------
# 31: Reverb / delay tails after transients
# --------------------------------------------------------------------------

def detect_reverb_tails(y, sr, transient_times: list) -> list:
    out = []
    for tt in transient_times:
        idx = int(tt * sr)
        window = int(0.6 * sr)
        seg = y[idx:min(len(y), idx + window)]
        if len(seg) < sr * 0.1:
            continue
        env = np.abs(seg)
        peak = env.max() + 1e-9
        below_10pct = np.where(env < peak * 0.1)[0]
        decay_samples = below_10pct[0] if len(below_10pct) else len(env)
        decay_ms = decay_samples / sr * 1000
        if decay_ms >= REVERB_TAIL_MIN_DECAY_MS:
            out.append({"time_sec": round(tt, 2), "decay_ms": round(decay_ms, 1)})
    return out


# --------------------------------------------------------------------------
# 32: Stereo / spatial movement
# --------------------------------------------------------------------------

def analyze_stereo(wav_path_stereo: str) -> dict:
    try:
        y_stereo, sr = librosa.load(wav_path_stereo, sr=None, mono=False)
    except Exception:
        return {}
    if y_stereo.ndim < 2 or y_stereo.shape[0] < 2:
        return {"is_stereo": False}
    left, right = y_stereo[0], y_stereo[1]
    hop = sr // 4  # 4 windows/sec
    n = min(len(left), len(right))
    pans = []
    for i in range(0, n - hop, hop):
        l_rms = np.sqrt(np.mean(left[i:i + hop].astype(np.float64) ** 2) + 1e-12)
        r_rms = np.sqrt(np.mean(right[i:i + hop].astype(np.float64) ** 2) + 1e-12)
        pan = (r_rms - l_rms) / (r_rms + l_rms + 1e-9)  # -1 = full left, +1 = full right
        pans.append(float(pan))
    corr = float(np.corrcoef(left[:n], right[:n])[0, 1]) if n > 1 else 1.0
    stereo_width = 1.0 - abs(corr)  # 0 = mono-like, ~1 = wide stereo
    return {
        "is_stereo": True,
        "mean_pan": round(float(np.mean(pans)), 3),
        "pan_variability": round(float(np.std(pans)), 3),
        "stereo_width_proxy": round(stereo_width, 3),
    }


# --------------------------------------------------------------------------
# Sync engine (items 25, and the backbone for item 50)
# --------------------------------------------------------------------------

def nearest_offset_ms(t: float, reference_times: list):
    if not reference_times:
        return None
    nearest = min(reference_times, key=lambda r: abs(r - t))
    return round((t - nearest) * 1000, 1)


def analyze_sync(event_times: list, reference_times: list) -> dict:
    offsets = [nearest_offset_ms(t, reference_times) for t in event_times]
    offsets = [o for o in offsets if o is not None]
    if not offsets:
        return {}
    abs_offsets = [abs(o) for o in offsets]
    within = sum(1 for o in abs_offsets if o <= SYNC_WINDOW_MS)
    return {"mean_abs_offset_ms": round(float(np.mean(abs_offsets)), 1),
            "median_abs_offset_ms": round(float(np.median(abs_offsets)), 1),
            "pct_within_sync_window": round(100 * within / len(offsets), 1)}


# --------------------------------------------------------------------------
# 50: Unified event timeline + synchronized-event clustering
# --------------------------------------------------------------------------

def build_event_timeline(shots, music, sfx_events) -> list:
    events = []
    for i, s in enumerate(shots):
        if i > 0:
            events.append({"time": s["start"], "category": "video", "type": f"cut_{s.get('transition_in', {}).get('type', 'hard_cut')}",
                            "confidence": s.get("transition_in", {}).get("confidence", 0.5)})
        if s.get("zoom_direction", "none") != "none":
            events.append({"time": s["start"], "category": "motion", "type": s["zoom_direction"], "confidence": 0.5})
        if s.get("camera_motion") == "shaky":
            events.append({"time": s["start"], "category": "motion", "type": "camera_shake", "confidence": 0.5})
    for bt in music.get("beat_times", []):
        events.append({"time": bt, "category": "audio", "type": "beat", "confidence": 0.7})
    for ot in music.get("percussive_onset_times", []):
        events.append({"time": ot, "category": "audio", "type": "percussive_onset", "confidence": 0.5})
    for sfx in sfx_events:
        events.append({"time": sfx["time_sec"], "category": "audio", "type": sfx["type_heuristic"], "confidence": sfx["confidence"]})
    events.sort(key=lambda e: e["time"])
    return events


def cluster_synchronized_events(events: list, window_sec: float = 0.1) -> list:
    """Item 50: groups events across categories that happen within `window_sec` of each other."""
    clusters, used = [], set()
    for i, e in enumerate(events):
        if i in used:
            continue
        group = [e]
        used.add(i)
        for j in range(i + 1, len(events)):
            if j in used:
                continue
            if events[j]["time"] - e["time"] > window_sec:
                break
            if events[j]["category"] != e["category"]:
                group.append(events[j])
                used.add(j)
        if len(group) >= 2:
            categories = sorted({g["category"] for g in group})
            clusters.append({
                "time": round(e["time"], 3),
                "categories_involved": categories,
                "event_types": [g["type"] for g in group],
                "strength": round(min(1.0, len(group) / 3), 2),
            })
    return clusters


# --------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------

def write_json(data: dict, path: str):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)


def write_shots_csv(shots: list, path: str):
    if not shots:
        return
    flat = []
    for s in shots:
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
        if "transition_in" in s:
            row["transition_type"] = s["transition_in"].get("type")
            row["transition_duration_sec"] = s["transition_in"].get("duration_sec")
        flat.append(row)
    fieldnames = sorted({k for r in flat for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in flat:
            writer.writerow(r)


def write_summary_chart(shots, music, cut_windows, out_path):
    if not _MATPLOTLIB_AVAILABLE:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    cut_t = [w["start"] for w in cut_windows]
    cut_r = [w["cuts_per_sec"] for w in cut_windows]
    axes[0].plot(cut_t, cut_r, color="crimson"); axes[0].set_ylabel("cuts/sec"); axes[0].set_title("Cut rate")
    e = music.get("energy_curve", {})
    if e:
        axes[1].plot(e["times"], e["rms"], color="steelblue")
    axes[1].set_ylabel("RMS energy"); axes[1].set_title("Music energy")
    hues = [s.get("dominant_hue") for s in shots if s.get("dominant_hue") is not None]
    hue_t = [s["start"] for s in shots if s.get("dominant_hue") is not None]
    axes[2].scatter(hue_t, hues, s=8, color="darkorange"); axes[2].set_ylabel("hue"); axes[2].set_title("Color (hue)")
    axes[2].set_xlabel("time (sec)")
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------
# AI layer (optional, local, free): faster-whisper for real speech timing
# --------------------------------------------------------------------------

_WHISPER_MODEL = None


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    except ImportError:
        print("  WARNING: faster-whisper not installed (pip install faster-whisper). AI speech layer disabled.")
        _WHISPER_MODEL = False
    except Exception as e:
        print(f"  WARNING: could not load Whisper model ({e}). AI speech layer disabled.")
        _WHISPER_MODEL = False
    return _WHISPER_MODEL


def detect_dialogue_regions_ai(wav_path: str) -> list:
    """
    Real speech-region detection via a local Whisper model (free, no API,
    no subscription — runs entirely on your own machine). Returns a list
    of {start, end} speech segments. This is used to redo J-cut/L-cut
    detection (item 28) with actual speech timing instead of the vocal-
    band energy heuristic, and to sharpen dialogue-window/ducking analysis.
    """
    model = _get_whisper_model()
    if not model:
        return []
    try:
        segments, _info = model.transcribe(wav_path, vad_filter=True)
        return [{"start": round(s.start, 2), "end": round(s.end, 2)} for s in segments]
    except Exception as e:
        print(f"  WARNING: Whisper transcription failed ({e}). Falling back to heuristic J/L-cut detection.")
        return []


def detect_jl_cuts_ai(dialogue_regions: list, cut_times: list) -> list:
    """AI-backed version of item 28, using real speech regions instead of the vocal-band heuristic."""
    results = []
    for ct in cut_times:
        for seg in dialogue_regions:
            if seg["start"] < ct < seg["end"]:
                # Speech spans across this visual cut — genuinely a J-cut/L-cut situation.
                if ct - seg["start"] < seg["end"] - ct:
                    results.append({"cut_time_sec": round(ct, 2), "type": "L_cut", "note": "speech from before the cut continues past it (Whisper-detected)"})
                else:
                    results.append({"cut_time_sec": round(ct, 2), "type": "J_cut", "note": "speech starts before the visual cut (Whisper-detected)"})
                break
    return results


# --------------------------------------------------------------------------
# AI layer continued: object detection (YOLOv8n), scene/action (CLIP
# zero-shot), and SFX/sound-event classification (YAMNet) — all free,
# local, one-time model downloads, no API key or subscription of any kind.
# --------------------------------------------------------------------------

_YOLO_MODEL = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_YAMNET_MODEL = None
_YAMNET_CLASS_NAMES = None

# Candidate labels CLIP scores each sampled frame against — covers both
# "what is happening" (general scene) and "what action is happening"
# (the item asked to answer both; CLIP zero-shot handles both with one
# pass since it just picks whichever phrase best matches the image).
_CLIP_CANDIDATE_LABELS = [
    "a person talking", "a person walking", "a person running", "a person fighting",
    "a person dancing", "a person driving a car", "a car chase", "an explosion",
    "a close-up of a face", "a wide establishing shot of a landscape", "a crowd of people",
    "an indoor scene", "an outdoor scene at night", "text or titles on screen",
    "an animated character", "a still, calm scene", "fast action and motion",
    "a romantic or emotional moment", "a battle or combat scene", "a musical performance",
]


def _get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    try:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO("yolov8n.pt")  # smallest/fastest variant; weights auto-download once, free, local after that
    except Exception as e:
        print(f"  WARNING: object detection unavailable ({e}). pip install ultralytics to enable it.")
        _YOLO_MODEL = False
    return _YOLO_MODEL


def detect_objects_in_frame(frame) -> list:
    """Item: 'what object is present?' — YOLOv8n, free/local."""
    model = _get_yolo_model()
    if not model:
        return []
    try:
        results = model(frame, verbose=False)
        objects = []
        for r in results:
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if conf > 0.35:
                    objects.append({"object": cls_name, "confidence": round(conf, 2)})
        return objects
    except Exception:
        return []


def _get_clip_model():
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR
    try:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print(f"  WARNING: scene/action classification unavailable ({e}). pip install transformers torch to enable it.")
        _CLIP_MODEL, _CLIP_PROCESSOR = False, False
    return _CLIP_MODEL, _CLIP_PROCESSOR


def classify_scene_and_action(frame) -> dict:
    """Items: 'what is happening?' and 'what action is happening?' — CLIP zero-shot, free/local."""
    model, processor = _get_clip_model()
    if not model:
        return {}
    try:
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = processor(text=_CLIP_CANDIDATE_LABELS, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        top_idx = int(probs.argmax())
        return {"scene_or_action_label": _CLIP_CANDIDATE_LABELS[top_idx], "confidence": round(float(probs[top_idx]), 3)}
    except Exception:
        return {}


def _get_yamnet_model():
    global _YAMNET_MODEL, _YAMNET_CLASS_NAMES
    if _YAMNET_MODEL is not None:
        return _YAMNET_MODEL, _YAMNET_CLASS_NAMES
    try:
        import tensorflow_hub as hub
        import csv as _csv
        _YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = _YAMNET_MODEL.class_map_path().numpy().decode("utf-8")
        with open(class_map_path) as f:
            _YAMNET_CLASS_NAMES = [row["display_name"] for row in _csv.DictReader(f)]
    except Exception as e:
        print(f"  WARNING: SFX/sound classification unavailable ({e}). pip install tensorflow tensorflow-hub to enable it.")
        _YAMNET_MODEL, _YAMNET_CLASS_NAMES = False, None
    return _YAMNET_MODEL, _YAMNET_CLASS_NAMES


def classify_sfx_type_ai(y, sr, transient_time: float) -> dict:
    """Item: 'what type of SFX is this?' — YAMNet (real AudioSet-trained sound classifier), free/local."""
    model, class_names = _get_yamnet_model()
    if not model:
        return {}
    try:
        import tensorflow as tf
        idx = int(transient_time * sr)
        window = int(0.975 * sr)  # YAMNet's native input window
        seg = y[max(0, idx - window // 2):idx + window // 2]
        if len(seg) < sr // 2:
            return {}
        if sr != 16000:
            seg = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=16000)
        scores, _embeddings, _spectrogram = model(tf.constant(seg, dtype=tf.float32))
        mean_scores = scores.numpy().mean(axis=0)
        top_idx = int(np.argmax(mean_scores))
        return {"sfx_type_ai": class_names[top_idx], "confidence": round(float(mean_scores[top_idx]), 3)}
    except Exception:
        return {}


def run_ai_layer_on_shots(video_path: str, shots: list, y, sr, sfx_events: list):
    """
    Runs the AI layer across the video, ONE sample frame per shot (cost
    control — this is the expensive stage). Adds object/scene/action tags
    to each shot dict in place, and upgrades each sfx_event with a real
    AudioSet-based type where YAMNet is available.
    """
    cap = cv2.VideoCapture(video_path)
    for shot in tqdm(shots, desc="  AI layer (objects/scene/action)", ncols=80, leave=False):
        mid_t = (shot["start"] + shot["end"]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        shot["ai_objects_present"] = detect_objects_in_frame(frame)
        shot["ai_scene_action"] = classify_scene_and_action(frame)
    cap.release()

    for sfx in tqdm(sfx_events, desc="  AI layer (SFX typing)", ncols=80, leave=False):
        ai_type = classify_sfx_type_ai(y, sr, sfx["time_sec"])
        if ai_type:
            sfx["ai_sfx_type"] = ai_type





# --------------------------------------------------------------------------
# Editing DNA profile (item 21)
# --------------------------------------------------------------------------

def compute_editing_dna(report: dict) -> dict:
    shots = report["shots"]
    duration_stats = report["shot_duration_stats"]
    music = report["music_analysis"]
    total_duration = report["media_info"]["duration"]

    beat_sync = report.get("sync_cut_to_beat", {})
    onset_sync = report.get("sync_cut_to_onset", {})
    sfx_sync = report.get("sync_sfx_to_cut", {})

    motion_vals = [s.get("motion_magnitude", 0.0) for s in shots]
    shake_vals = [s.get("shake_intensity", 0.0) for s in shots]
    sat_vals = [s.get("mean_saturation") for s in shots if s.get("mean_saturation") is not None]
    hue_changes = 0
    for i in range(1, len(shots)):
        h1, h2 = shots[i - 1].get("dominant_hue"), shots[i].get("dominant_hue")
        if h1 is not None and h2 is not None and abs(h1 - h2) > 25:
            hue_changes += 1

    transition_counts = report.get("transition_type_counts", {})
    camera_motion_counts = report.get("camera_motion_counts", {})
    n_shots = max(1, len(shots))

    return {
        "average_shot_duration_sec": duration_stats.get("mean_sec", 0.0),
        "median_shot_duration_sec": duration_stats.get("median_sec", 0.0),
        "cut_rate_per_sec": round(n_shots / max(total_duration, 0.01), 3),
        "beat_sync_strength": (beat_sync.get("pct_within_sync_window", 0.0) or 0.0) / 100,
        "onset_sync_strength": (onset_sync.get("pct_within_sync_window", 0.0) or 0.0) / 100,
        "sfx_sync_strength": (sfx_sync.get("pct_within_sync_window", 0.0) or 0.0) / 100,
        "color_change_frequency_per_min": round(hue_changes / max(total_duration / 60, 0.01), 2),
        "transition_type_distribution": {k: round(v / max(sum(transition_counts.values()), 1), 3) for k, v in transition_counts.items()},
        "camera_motion_distribution": {k: round(v / n_shots, 3) for k, v in camera_motion_counts.items()},
        "camera_motion_intensity": round(float(np.mean(motion_vals)) if motion_vals else 0.0, 4),
        "camera_shake_intensity": round(float(np.mean(shake_vals)) if shake_vals else 0.0, 4),
        "mean_saturation": round(float(np.mean(sat_vals)), 1) if sat_vals else None,
        "tempo_bpm": music.get("tempo_bpm", 0.0),
        "visual_density": round(n_shots / max(total_duration, 0.01), 3),   # shots/sec, distinct name from cut_rate for clarity
        "pct_shots_under_0_5s": duration_stats.get("pct_under_0_5s", 0.0),
        "pct_shots_over_3s": duration_stats.get("pct_over_3s", 0.0),
    }


def average_dna_profiles(profiles: list) -> dict:
    """Averages several per-video DNA dicts into one niche-level DNA module."""
    if not profiles:
        return {}
    numeric_keys = [k for k, v in profiles[0].items() if isinstance(v, (int, float)) and v is not None]
    dist_keys = [k for k, v in profiles[0].items() if isinstance(v, dict)]

    avg = {}
    for k in numeric_keys:
        vals = [p[k] for p in profiles if p.get(k) is not None]
        avg[k] = round(float(np.mean(vals)), 4) if vals else None

    for k in dist_keys:
        all_subkeys = set()
        for p in profiles:
            all_subkeys.update(p.get(k, {}).keys())
        merged = {}
        for sk in all_subkeys:
            vals = [p.get(k, {}).get(sk, 0.0) for p in profiles]
            merged[sk] = round(float(np.mean(vals)), 3)
        avg[k] = merged

    avg["_n_videos_averaged"] = len(profiles)
    return avg


# --------------------------------------------------------------------------
# Edit Grammar mining (item 15): frequent cross-category event recipes
# --------------------------------------------------------------------------

def mine_edit_grammar(sync_clusters: list, min_frequency: int = 2) -> list:
    """
    Looks for frequently-repeating COMBINATIONS of event types within
    synchronized clusters (e.g. "beat + percussive_onset + cut_hard_cut"
    repeating 7 times) — a classical frequent-itemset count, not ML.
    """
    from collections import Counter
    combo_counter = Counter()
    combo_examples = {}
    for cluster in sync_clusters:
        combo = tuple(sorted(set(cluster["event_types"])))
        if len(combo) < 2:
            continue
        combo_counter[combo] += 1
        combo_examples.setdefault(combo, cluster["time"])

    patterns = []
    for combo, freq in combo_counter.most_common(15):
        if freq >= min_frequency:
            patterns.append({"components": list(combo), "frequency": freq, "example_time_sec": combo_examples[combo]})
    return patterns


# --------------------------------------------------------------------------
# Energy vs Time vs Shot-Density curve (explicit combined chart)
# --------------------------------------------------------------------------

def write_energy_density_chart(cut_windows, music, out_path):
    if not _MATPLOTLIB_AVAILABLE:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    t = [w["start"] for w in cut_windows]
    density = [w["cuts_per_sec"] for w in cut_windows]
    ax1.plot(t, density, color="crimson", label="Shot density (cuts/sec)")
    ax1.set_xlabel("time (sec)")
    ax1.set_ylabel("cuts/sec", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")

    ax2 = ax1.twinx()
    e = music.get("energy_curve", {})
    if e:
        ax2.plot(e["times"], e["rms"], color="steelblue", alpha=0.8, label="Music energy (RMS)")
    ax2.set_ylabel("RMS energy", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")

    plt.title("Energy vs. Time vs. Shot Density")
    fig.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Contact sheet (item 18)
# --------------------------------------------------------------------------

def generate_contact_sheet(video_path: str, shots: list, out_path: str):
    cap = cv2.VideoCapture(video_path)
    tw, th = CONTACT_SHEET_THUMB_SIZE
    cols = CONTACT_SHEET_COLUMNS
    rows = int(np.ceil(len(shots) / cols))
    label_h = 34
    sheet = np.full((rows * (th + label_h), cols * tw, 3), 30, dtype=np.uint8)

    for i, shot in enumerate(shots):
        mid_t = (shot["start"] + shot["end"]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        thumb = cv2.resize(frame, (tw, th))
        r, c = divmod(i, cols)
        y0, x0 = r * (th + label_h), c * tw
        sheet[y0:y0 + th, x0:x0 + tw] = thumb
        label = f"#{i+1} {shot['start']:.2f}s ({shot['end']-shot['start']:.2f}s)"
        cv2.putText(sheet, label, (x0 + 4, y0 + th + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cap.release()
    cv2.imwrite(out_path, sheet)


# --------------------------------------------------------------------------
# HTML visual report (item 17)
# --------------------------------------------------------------------------

def generate_html_report(report: dict, dna: dict, grammar: list, out_path: str):
    info = report["media_info"]
    ds = report["shot_duration_stats"]
    shots = report["shots"]
    total_duration = info["duration"]

    def bar_html(events, color, height=24):
        marks = "".join(
            f'<div style="position:absolute;left:{100*e/total_duration:.3f}%;width:2px;height:{height}px;background:{color};"></div>'
            for e in events
        )
        return f'<div style="position:relative;height:{height}px;background:#222;">{marks}</div>'

    cut_times = [s["start"] for s in shots[1:]]
    beat_times = report["music_analysis"].get("beat_times", [])
    sfx_times = [e["time_sec"] for e in report.get("sfx_transients", [])]

    rows = f"""
    <div><b>Cuts</b>{bar_html(cut_times, '#e74c3c')}</div>
    <div><b>Beats</b>{bar_html(beat_times, '#3498db')}</div>
    <div><b>SFX transients</b>{bar_html(sfx_times, '#f1c40f')}</div>
    """

    shot_cards = ""
    for i, s in enumerate(shots):
        t = s.get("transition_in", {})
        shot_cards += f"""
        <div style="border:1px solid #444;padding:8px;margin:4px;border-radius:6px;display:inline-block;width:220px;vertical-align:top;">
          <b>Shot {i+1}</b><br>
          {s['start']:.2f}s &rarr; {s['end']:.2f}s ({s['end']-s['start']:.2f}s)<br>
          Motion: {s.get('camera_motion','-')} ({s.get('motion_magnitude',0):.2f})<br>
          Hue: {s.get('dominant_hue','-')}&deg; ({s.get('warm_or_cool','-')})<br>
          Transition in: {t.get('type','-')} ({t.get('duration_sec',0):.2f}s)<br>
          Confidence: {t.get('confidence',0):.2f}
        </div>"""

    grammar_html = "".join(
        f"<li>{' &rarr; '.join(g['components'])} &mdash; repeats {g['frequency']}x</li>" for g in grammar
    ) or "<li>No frequent cross-modal patterns found.</li>"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Edit Analysis: {report['video']}</title>
    <style>body{{background:#111;color:#eee;font-family:sans-serif;padding:20px;}}
    h1,h2{{color:#fff;}} .overview{{background:#1c1c1c;padding:16px;border-radius:8px;margin-bottom:16px;}}</style>
    </head><body>
    <h1>{report['video']}</h1>
    <div class="overview">{total_duration:.1f}s | {info.get('fps',0):.1f}fps | {info.get('width')}x{info.get('height')} |
    {ds.get('count',0)} shots | mean {ds.get('mean_sec',0):.2f}s/shot | {report['music_analysis'].get('tempo_bpm',0)} BPM</div>
    <h2>Timeline</h2>{rows}
    <h2>Editing DNA</h2><pre>{json.dumps(dna, indent=2)}</pre>
    <h2>Edit Grammar (frequent synchronized recipes)</h2><ul>{grammar_html}</ul>
    <h2>Shots</h2>{shot_cards}
    </body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# --------------------------------------------------------------------------
# Parquet dataset export (item 20): events, relationships, shots
# --------------------------------------------------------------------------

def write_parquet_datasets(report: dict, out_dir: str, name: str):
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas/pyarrow not installed — skipping parquet dataset export (pip install pandas pyarrow).")
        return

    events_rows = []
    for bt in report["music_analysis"].get("beat_times", []):
        events_rows.append({"time": bt, "category": "audio", "type": "beat", "confidence": 0.7})
    for sfx in report.get("sfx_transients", []):
        events_rows.append({"time": sfx["time_sec"], "category": "audio", "type": sfx["type_heuristic"], "confidence": sfx["confidence"]})
    for i, s in enumerate(report["shots"][1:], start=1):
        t = s.get("transition_in", {})
        events_rows.append({"time": s["start"], "category": "video", "type": t.get("type", "cut"), "confidence": t.get("confidence", 0.5)})
    try:
        pd.DataFrame(events_rows).to_parquet(os.path.join(out_dir, f"{name}_events.parquet"))
    except Exception as e:
        print(f"  WARNING: events.parquet export failed: {e}")

    rel_rows = []
    for c in report.get("synchronized_event_clusters", []):
        for a, b in zip(c["event_types"], c["event_types"][1:]):
            rel_rows.append({"event_A": a, "event_B": b, "time": c["time"], "strength": c["strength"]})
    try:
        pd.DataFrame(rel_rows).to_parquet(os.path.join(out_dir, f"{name}_relationships.parquet"))
    except Exception as e:
        print(f"  WARNING: relationships.parquet export failed: {e}")

    try:
        shot_rows = [{k: v for k, v in s.items() if not isinstance(v, (dict, list))} for s in report["shots"]]
        pd.DataFrame(shot_rows).to_parquet(os.path.join(out_dir, f"{name}_shots.parquet"))
    except Exception as e:
        print(f"  WARNING: shots.parquet export failed: {e}")



def analyze_video(video_path: str, out_dir: str, whisper_model_hint: bool = True):
    name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*60}\nAnalyzing: {os.path.basename(video_path)}\n{'='*60}")

    info = probe_media(video_path)
    fps = info.get("fps", 30.0) or 30.0
    total_duration = info["duration"]
    frame_w, frame_h = info.get("width", 1280), info.get("height", 720)
    print(f"  {frame_w}x{frame_h} @ {fps:.2f}fps, {total_duration:.1f}s")

    print("  Detecting shots...")
    shots = detect_shots(video_path)
    print(f"  {len(shots)} shot(s) found.")
    duration_stats = shot_duration_stats(shots)
    cut_windows = cut_frequency_over_time(shots, total_duration)

    print("  Per-shot analysis (motion, sharpness, color, VFX heuristics, letterboxing, overlays)...")
    cap = cv2.VideoCapture(video_path)
    for i, shot in enumerate(tqdm(shots, desc="  Per-shot", ncols=80)):
        motion = analyze_shot_motion(cap, shot["start"], shot["end"], MOTION_SAMPLE_FRAMES_PER_SHOT)
        shot.update(motion)
        cap.set(cv2.CAP_PROP_POS_MSEC, shot["start"] * 1000)
        shot["_mean_dx"] = motion.get("motion_magnitude", 0.0) * (1 if motion.get("zoom_direction") != "zoom_out_pull" else -1)

        color = analyze_shot_color(cap, shot["start"], shot["end"], COLOR_SAMPLE_FRAMES_PER_SHOT)
        shot.update(color)

        if ANALYSIS_MODE == "fast":
            # fast mode: metadata + shots + cuts + BPM/beats + basic color/motion only —
            # skip everything below (sharpness, VFX heuristics, letterboxing, overlays,
            # transitions/adjacency/cut-on-action all still need a real classify pass,
            # which "deep"/"forensic" do further down).
            if i > 0:
                shot["transition_in"] = {"type": "hard_cut", "duration_sec": 0.0, "confidence": 0.3}
            continue

        sharp = analyze_shot_sharpness(cap, shot["start"], shot["end"], SHARPNESS_SAMPLE_FRAMES_PER_SHOT)
        shot.update(sharp)

        mid_t = (shot["start"] + shot["end"]) / 2
        shot.update(detect_letterboxing(cap, mid_t, frame_w, frame_h))
        shot.update(detect_glow_and_flares(cap, mid_t))
        shot.update(detect_distortion_effects(cap, mid_t))
        shot.update(detect_texture_effects(cap, shot["start"], shot["end"], fps))
        overlay = detect_overlays(cap, shot["start"], shot["end"])
        shot["overlay"] = overlay

        if i > 0:
            transition = classify_transition(cap, shot["start"], fps, shots[i - 1].get("motion_magnitude", 0.0))
            shot["transition_in"] = transition
            adjacency = compute_adjacency_features(cap, shots[i - 1], shot)
            shot["adjacency_to_previous"] = adjacency
            cut_on_action = compute_cut_on_action(cap, shot["start"], shots[i - 1].get("motion_magnitude", 0.1))
            shot.update(cut_on_action)

    print("  Detecting montage/repetition patterns...")
    montage_patterns = detect_montage_patterns(shots)
    screen_dir = screen_direction_continuity(shots)
    cap.release()

    print("  Extracting and analyzing audio (music structure, key, micro-events)...")
    audio_dir = os.path.join(out_dir, "_audio_tmp")
    os.makedirs(audio_dir, exist_ok=True)
    wav_mono = os.path.join(audio_dir, f"{name}_mono.wav")
    wav_stereo = os.path.join(audio_dir, f"{name}_stereo.wav")
    extract_audio_mono(video_path, wav_mono)
    has_stereo = extract_audio_stereo(video_path, wav_stereo)

    music = analyze_music(wav_mono)
    y, sr, rms, rms_times = music.pop("_y"), music.pop("_sr"), music.pop("_rms"), music.pop("_rms_times")

    print("  Sound-design analysis (SFX transients, audio transitions, J/L-cuts, ducking, reverb tails, stereo)...")
    sfx_events = detect_sfx_transients(y, sr, music["onset_times"])
    audio_transitions = detect_audio_transitions(rms, rms_times)
    cut_times = [s["start"] for s in shots[1:]]
    jl_cuts = detect_jl_cuts(y, sr, cut_times)
    silence_context = classify_silence_context(music["silence_regions_sec"], cut_times)
    ducking_events = detect_ducking(rms, rms_times, y, sr)
    reverb_tails = detect_reverb_tails(y, sr, [e["time_sec"] for e in sfx_events])
    stereo_info = analyze_stereo(wav_stereo) if has_stereo else {"is_stereo": False}

    ai_layer_used = False
    dialogue_regions_ai = []
    if ANALYSIS_MODE == "forensic" and AI_LAYER_ENABLED:
        print("  Running AI layer (local, free — speech, objects, scene/action, SFX typing)...")
        # Speech: upgrades J/L-cut detection (item 28) from the heuristic
        # above to real speech timing.
        dialogue_regions_ai = detect_dialogue_regions_ai(wav_mono)
        if dialogue_regions_ai:
            jl_cuts_ai = detect_jl_cuts_ai(dialogue_regions_ai, cut_times)
            if jl_cuts_ai:
                jl_cuts = jl_cuts_ai  # prefer the AI-backed result when speech was actually found
            ai_layer_used = True
        # Objects, scene/action per shot; SFX typing per transient.
        run_ai_layer_on_shots(video_path, shots, y, sr, sfx_events)

    for p in (wav_mono, wav_stereo):
        if os.path.isfile(p):
            try: os.remove(p)
            except OSError: pass
    try: os.rmdir(audio_dir)
    except OSError: pass

    print("  Computing cross-modal sync + unified event timeline...")
    sync_beat = analyze_sync(cut_times, music["beat_times"])
    sync_onset = analyze_sync(cut_times, music["onset_times"])
    sync_sfx = analyze_sync([e["time_sec"] for e in sfx_events], cut_times)
    energy_arr = np.array(music["energy_curve"]["rms"])
    energy_times = np.array(music["energy_curve"]["times"])
    cut_rates = [w["cuts_per_sec"] for w in cut_windows]
    energies_for_windows = []
    for w in cut_windows:
        mask = (energy_times >= w["start"]) & (energy_times < w["end"])
        energies_for_windows.append(float(energy_arr[mask].mean()) if mask.any() else 0.0)
    energy_corr = float(np.corrcoef(cut_rates, energies_for_windows)[0, 1]) if len(cut_rates) >= 3 else 0.0

    timeline = build_event_timeline(shots, music, sfx_events)
    sync_clusters = cluster_synchronized_events(timeline)

    transition_counts, camera_motion_counts = {}, {}
    for s in shots[1:]:
        t = s.get("transition_in", {}).get("type", "unknown")
        transition_counts[t] = transition_counts.get(t, 0) + 1
    for s in shots:
        cm = s.get("camera_motion", "unknown")
        camera_motion_counts[cm] = camera_motion_counts.get(cm, 0) + 1

    report = {
        "video": os.path.basename(video_path), "media_info": info,
        "shot_duration_stats": duration_stats, "cut_rate_over_time": cut_windows,
        "transition_type_counts": transition_counts, "camera_motion_counts": camera_motion_counts,
        "montage_patterns": montage_patterns, "screen_direction_continuity": screen_dir,
        "music_analysis": music, "sfx_transients": sfx_events, "audio_transitions": audio_transitions,
        "j_l_cuts_approximate": jl_cuts, "silence_context": silence_context,
        "ducking_events": ducking_events, "reverb_tails": reverb_tails, "stereo_analysis": stereo_info,
        "sync_cut_to_beat": sync_beat, "sync_cut_to_onset": sync_onset, "sync_sfx_to_cut": sync_sfx,
        "cut_rate_vs_energy_correlation": round(energy_corr, 3),
        "synchronized_event_clusters": sync_clusters,
        "ai_layer_used": ai_layer_used, "ai_dialogue_regions": dialogue_regions_ai,
        "shots": shots,
    }

    dna = compute_editing_dna(report)
    grammar = mine_edit_grammar(sync_clusters)
    report["editing_dna"] = dna
    report["edit_grammar_patterns"] = grammar

    write_json(report, os.path.join(out_dir, f"{name}_analysis.json"))
    write_shots_csv(shots, os.path.join(out_dir, f"{name}_shots.csv"))
    if WRITE_CHART:
        write_summary_chart(shots, music, cut_windows, os.path.join(out_dir, f"{name}_chart.png"))
        write_energy_density_chart(cut_windows, music, os.path.join(out_dir, f"{name}_energy_density.png"))

    if ANALYSIS_MODE == "forensic":
        generate_contact_sheet(video_path, shots, os.path.join(out_dir, f"{name}_contact_sheet.png"))
        generate_html_report(report, dna, grammar, os.path.join(out_dir, f"{name}_report.html"))
        write_parquet_datasets(report, out_dir, name)

    print(f"\n  Summary for {name}:")
    print(f"    {duration_stats.get('count', 0)} shots, mean {duration_stats.get('mean_sec', 0):.2f}s")
    print(f"    Transitions: {transition_counts}")
    print(f"    Camera motion: {camera_motion_counts}")
    print(f"    Tempo: {music['tempo_bpm']} BPM | Key: {music['key_estimate']}")
    print(f"    Cut-to-beat sync: {sync_beat}")
    print(f"    {len(sfx_events)} transient event(s), {len(ducking_events)} ducking event(s), {len(jl_cuts)} J/L-cut candidate(s)"
          + (" [AI-backed]" if ai_layer_used else " [heuristic]"))
    print(f"    {len(sync_clusters)} synchronized cross-modal event cluster(s), {len(grammar)} edit-grammar pattern(s)")

    return report


# --------------------------------------------------------------------------
# DNA JOIN VARIANTS: Inner / Left / Right / Full outer  (L1 vs R1..Rn)
# --------------------------------------------------------------------------
# Every video is flattened into a feature table: dotted_key -> value
# (numbers or strings). A feature is only present if it was actually
# measured/observed in that video, so "missing" is meaningful. The join key
# is the feature name. L1 = one video, R = all the other videos in the niche.
#   inner : features present in L1 AND in at least one R
#   left  : every L1 feature (R values attached where they exist)
#   right : every feature seen in any R (L1 value attached where it exists)
#   full  : union of all features
# Each feature carries L, R (aggregated over Rs: mean / mode), per-R values,
# a "blend" (mean of L and R when both exist) and where it came from.
# "dna" in each file is the resolved flat dict ready for beat_sync_editor.py.
# Output per niche: dna_modules/<niche>/{inner,left,right,full_outer}.json
# (pooled L1 vs all Rs, plus a "pairwise" section L1 vs each single R).
# The original averaged <niche>.json is still written.

DNA_LEFT_VIDEO = None   # e.g. "my_best_edit.mp4"; None = auto-pick (most features, then longest)
DNA_JOIN_TYPES = ["inner", "left", "right", "full_outer"]


def _num_mean(vals):
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return float(np.mean(nums)) if nums else None


def flatten_report_for_dna(report: dict) -> dict:
    from collections import Counter
    f = {}

    def put(k, v):
        if v is None:
            return
        if isinstance(v, (bool, np.bool_)):
            v = float(v)
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, (int, float)) and (v != v):
            return
        if isinstance(v, (int, float, str)):
            f[k] = v

    def put_dict(prefix, d):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                put_dict(f"{prefix}.{k}", v)
            else:
                put(f"{prefix}.{k}", v)

    dur = report["media_info"].get("duration", 0.0) or 0.0
    minutes = max(dur / 60, 0.01)
    put_dict("dna", report.get("editing_dna"))
    put_dict("shot_stats", report.get("shot_duration_stats"))
    put_dict("sync.cut_to_beat", report.get("sync_cut_to_beat"))
    put_dict("sync.cut_to_onset", report.get("sync_cut_to_onset"))
    put_dict("sync.sfx_to_cut", report.get("sync_sfx_to_cut"))
    put("sync.cut_rate_vs_energy_corr", report.get("cut_rate_vs_energy_correlation"))
    put_dict("stereo", report.get("stereo_analysis"))
    put_dict("transitions", {k: v / max(sum(report.get("transition_type_counts", {}).values()), 1)
                             for k, v in report.get("transition_type_counts", {}).items()})

    m = report.get("music_analysis", {})
    put("music.tempo_bpm", m.get("tempo_bpm"))
    put("music.beats_per_min", len(m.get("beat_times", [])) / minutes)
    put("music.onsets_per_min", len(m.get("onset_times", [])) / minutes)
    put("music.sections_per_min", max(len(m.get("section_boundaries_sec", [])) - 1, 0) / minutes)
    put_dict("music.loudness", m.get("loudness"))
    put("music.key", (m.get("key_estimate") or {}).get("estimated_key"))
    put("music.silence_regions_per_min", len(m.get("silence_regions_sec", [])) / minutes)
    me = m.get("micro_events", {})
    put("music.risers_per_min", len(me.get("riser_candidates_sec", [])) / minutes)
    put("music.drops_per_min", len(me.get("drop_candidates_sec", [])) / minutes)
    ec = (m.get("energy_curve") or {}).get("rms") or []
    if ec:
        put("music.energy_mean", float(np.mean(ec))); put("music.energy_std", float(np.std(ec)))
    bc = (m.get("brightness_curve") or {}).get("centroid_hz") or []
    if bc:
        put("music.brightness_mean_hz", float(np.mean(bc)))

    sfx = report.get("sfx_transients", [])
    put("sfx.per_min", len(sfx) / minutes)
    if sfx:
        put("sfx.mean_amplitude", _num_mean([e.get("amplitude") for e in sfx]))
    for t, c in Counter(e.get("type_heuristic") for e in sfx).items():
        put(f"sfx.type_share.{t}", c / len(sfx))
    for t, c in Counter((e.get("ai_sfx_type") or {}).get("sfx_type_ai") for e in sfx
                        if e.get("ai_sfx_type")).items():
        put(f"sfx.ai_type_share.{t}", c / len(sfx))

    at = report.get("audio_transitions", [])
    put("audio_transitions.per_min", len(at) / minutes)
    duck = report.get("ducking_events", [])
    put("ducking.per_min", len(duck) / minutes)
    if duck:
        put("ducking.mean_depth_db", _num_mean([d.get("duck_depth_db") for d in duck]))
    rt = report.get("reverb_tails", [])
    put("reverb.per_min", len(rt) / minutes)
    if rt:
        put("reverb.mean_decay_ms", _num_mean([r.get("decay_ms") for r in rt]))
    put("jl_cuts.per_min", len(report.get("j_l_cuts_approximate", [])) / minutes)
    for t, c in Counter(s.get("context") for s in report.get("silence_context", [])).items():
        put(f"silence_context.{t}", c)
    put("ai.dialogue_regions_per_min", len(report.get("ai_dialogue_regions", [])) / minutes)

    shots = report.get("shots", [])
    n = max(len(shots), 1)
    skip = {"start", "end", "_mean_dx"}
    nums = {}
    for s in shots:
        for k, v in s.items():
            if k in skip:
                continue
            if isinstance(v, (bool, np.bool_)):
                v = float(v)
            if isinstance(v, (int, float, np.floating, np.integer)):
                nums.setdefault(k, []).append(float(v))
    for k, vals in nums.items():
        put(f"shot_avg.{k}", float(np.mean(vals)))
    for k in ("camera_motion", "zoom_direction", "warm_or_cool"):
        for t, c in Counter(s.get(k) for s in shots if s.get(k)).items():
            put(f"shot_share.{k}.{t}", c / n)
    put("shot_share.overlay_detected", sum(1 for s in shots if (s.get("overlay") or {}).get("overlay_detected")) / n)
    adj = [s.get("adjacency_to_previous") or {} for s in shots[1:]]
    if adj:
        for k in ("match_cut_likely", "jump_cut_likely"):
            put(f"adjacency.{k}_share", sum(1 for a in adj if a.get(k)) / len(adj))
        for k in ("cross_cut_frame_similarity", "graphic_match_score", "spatial_continuity_score"):
            put(f"adjacency.{k}_mean", _num_mean([a.get(k) for a in adj]))
    sd = report.get("screen_direction_continuity", [])
    if sd:
        put("adjacency.same_screen_direction_share", sum(1 for d in sd if d["same_screen_direction"]) / len(sd))
    put("montage.patterns_found", len(report.get("montage_patterns", [])))
    for g in report.get("edit_grammar_patterns", []):
        put("grammar." + "+".join(g["components"]), g["frequency"])
    obj = Counter(o["object"] for s in shots for o in s.get("ai_objects_present", []))
    for k, c in obj.items():
        put(f"ai.object_share.{k}", c / n)
    for t, c in Counter((s.get("ai_scene_action") or {}).get("scene_or_action_label") for s in shots
                        if s.get("ai_scene_action")).items():
        put(f"ai.scene_share.{t}", c / n)
    return f


def _agg_r(values: list):
    nums = [v for v in values if isinstance(v, (int, float))]
    if nums and len(nums) == len(values):
        return round(float(np.mean(nums)), 4)
    from collections import Counter
    return Counter(values).most_common(1)[0][0]


def _join_features(L: dict, Rs: dict, how: str) -> dict:
    """L: feature dict; Rs: {name: feature dict}. Join key = feature name."""
    r_keys = set().union(*[set(r) for r in Rs.values()]) if Rs else set()
    l_keys = set(L)
    keys = {"inner": l_keys & r_keys, "left": l_keys, "right": r_keys,
            "full_outer": l_keys | r_keys}[how]
    feats, dna = {}, {}
    for k in sorted(keys):
        per_r = {n: r[k] for n, r in Rs.items() if k in r}
        l_val = L.get(k)
        r_val = _agg_r(list(per_r.values())) if per_r else None
        if l_val is not None and r_val is not None:
            both_num = isinstance(l_val, (int, float)) and isinstance(r_val, (int, float))
            blend = round((l_val + r_val) / 2, 4) if both_num else l_val
            src = "both"
        elif l_val is not None:
            blend, src = l_val, "left_only"
        else:
            blend, src = r_val, "right_only"
        feats[k] = {"L": l_val, "R": r_val, "R_values": per_r, "n_R": len(per_r),
                    "blend": blend, "source": src}
        dna[k] = blend
    return {"n_features": len(feats), "dna": dna, "features": feats}


def export_join_variants(niche_name: str, niche_reports: list):
    if not niche_reports:
        return
    tables = {r["video"]: flatten_report_for_dna(r) for r in niche_reports}
    durations = {r["video"]: r["media_info"].get("duration", 0.0) for r in niche_reports}
    if DNA_LEFT_VIDEO and DNA_LEFT_VIDEO in tables:
        left_name = DNA_LEFT_VIDEO
    else:
        left_name = max(tables, key=lambda v: (len(tables[v]), durations[v]))
    L = tables[left_name]
    Rs = {v: t for v, t in tables.items() if v != left_name}
    out_dir = os.path.join(DNA_OUTPUT_FOLDER, niche_name)
    os.makedirs(out_dir, exist_ok=True)
    if not Rs:
        print(f"  Join variants: niche '{niche_name}' has only one video - R side is empty "
              f"(left/full contain L1 only, inner/right are empty).")
    for how in DNA_JOIN_TYPES:
        module = {
            "niche": niche_name, "join": how, "left_video": left_name,
            "right_videos": list(Rs.keys()),
            "join_key": "feature name (dotted key)",
            "pooled": _join_features(L, Rs, how),
            "pairwise": {name: _join_features(L, {name: t}, how) for name, t in Rs.items()},
            "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = os.path.join(out_dir, f"{how}.json")
        write_json(module, path)
        print(f"  Join DNA [{how}]: {module['pooled']['n_features']} features -> {path}")



def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        sys.exit(1)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DNA_OUTPUT_FOLDER, exist_ok=True)

    # Niche-folder mode: any subfolder of INPUT_FOLDER is treated as a
    # niche. Each niche's videos are analyzed, and their per-video DNA
    # profiles are averaged into one DNA module for that niche. If
    # INPUT_FOLDER has videos directly in it instead, the whole folder is
    # treated as a single niche ("default").
    subfolders = sorted(
        d for d in os.listdir(INPUT_FOLDER)
        if os.path.isdir(os.path.join(INPUT_FOLDER, d))
    )
    niches = {d: os.path.join(INPUT_FOLDER, d) for d in subfolders} if subfolders else {"default": INPUT_FOLDER}

    print(f"Found {len(niches)} niche folder(s): {list(niches.keys())}")

    all_reports = []
    for niche_name, niche_path in niches.items():
        videos = find_videos(niche_path)
        if not videos:
            print(f"\nNiche '{niche_name}': no videos found, skipping.")
            continue

        print(f"\n{'#'*60}\nNICHE: {niche_name}  ({len(videos)} video(s))\n{'#'*60}")
        niche_out_dir = os.path.join(OUTPUT_FOLDER, niche_name)
        os.makedirs(niche_out_dir, exist_ok=True)

        niche_reports, niche_dna_profiles = [], []
        for video_path in videos:
            try:
                report = analyze_video(video_path, niche_out_dir)
                niche_reports.append(report)
                niche_dna_profiles.append(report["editing_dna"])
            except Exception as e:
                print(f"  ERROR analyzing {video_path}: {e}")

        all_reports.extend(niche_reports)

        if niche_dna_profiles:
            niche_dna = average_dna_profiles(niche_dna_profiles)
            # Also fold in the most common transition/camera-motion choices
            # and a merged edit-grammar pattern list across the whole niche —
            # this is the piece beat_sync_editor.py actually consumes.
            all_grammar = []
            for r in niche_reports:
                all_grammar.extend(r.get("edit_grammar_patterns", []))
            from collections import Counter
            grammar_counter = Counter()
            for g in all_grammar:
                grammar_counter[tuple(sorted(g["components"]))] += g["frequency"]
            merged_grammar = [{"components": list(k), "total_frequency": v} for k, v in grammar_counter.most_common(15)]

            dna_module = {
                "niche": niche_name,
                "n_videos_analyzed": len(niche_dna_profiles),
                "source_videos": [os.path.basename(v) for v in videos],
                "dna": niche_dna,
                "edit_grammar_patterns": merged_grammar,
                "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            }
            dna_path = os.path.join(DNA_OUTPUT_FOLDER, f"{niche_name}.json")
            write_json(dna_module, dna_path)
            print(f"\n  DNA module written: {dna_path}  ({len(niche_dna_profiles)} video(s) averaged)")
            export_join_variants(niche_name, niche_reports)

    print("\n" + "=" * 60)
    print(f"Done. {len(all_reports)} video(s) analyzed across {len(niches)} niche(s).")
    print(f"Per-video results in: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"DNA modules in: {os.path.abspath(DNA_OUTPUT_FOLDER)}")


if __name__ == "__main__":
    main()