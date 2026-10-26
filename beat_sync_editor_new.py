"""
beat_sync_editor.py

Automatically edits a long source video (or set of source videos) into a
music-video-style cut, synced to the beat of a track you provide — one
edit per music track found, all reusing the same shared shot pool.

--------------------------------------------------------------------------
PIPELINE ORDER (this is the order things actually run in, per track)
--------------------------------------------------------------------------
  1. ANALYZE the track: beat timeline + per-beat energy.
  2. LOAD the revised story-arc script for this track (a .txt in
     REMIX_FOLDER, matched by the track's filename) — see STORY-ARC
     SCRIPT SYSTEM below. If none is found, falls back to plain
     beat-driven cutting across the whole track with no text-guided
     shot selection (logged clearly either way).
  3. CUT video segments: within each story-arc stage's time window, cuts
     are still placed on the beat (reusing the same beat-driven timeline
     logic as before), but WHICH SHOT fills each cut is chosen by fuzzy-
     matching that stage's script text against each candidate shot's own
     transcript (from transcribe.py) — not randomly. Segments are saved
     to video_cutaways/<track_name>/.
  4. BUILD seamless transitions between the cut segments (xfade/acrossfade).
  5. DUCKING (no script needed, no singing detection): every VO_WINDOW_SEC
     (5s) a voice-over window opens, then VO_GAP_SEC (10s) of music only,
     repeating. In a window the music ducks and the edit's OWN audio (the
     dialogue/voice-over inside the source clips) is brought up; outside
     it the clips' audio is muted. Exported as outputs/<track>_ducking.mp3,
     <track>_mix.mp3, a cue sheet (_vo_cues.txt) and plan (_ducking_plan.json).
  6. MIX: FINAL_AUDIO_MODE = "mix" (default) puts that music + clip-audio
     mix into the final video; "plain"/"ducked"/"blend" are still available.

--------------------------------------------------------------------------
STORY-ARC SCRIPT SYSTEM (NEW — replaces the old lyrics-line cut planning
and the old dialogue-interjection/ducking-into-the-edit system entirely)
--------------------------------------------------------------------------
Drop a revised .txt script into REMIX_FOLDER, named after the music track
it belongs to (e.g. music/mytrack.mp3 -> remix/mytrack.txt). The script is
divided into labeled sections following the story arc:

    Hook -> Rising Action -> Conflict -> Comeback -> Rising Action -> Payoff

Section headers are matched case-insensitively and can look like any of:
    Hook
    ## Hook
    [HOOK]
    HOOK:
The cycle can repeat multiple times across the script (every ~30s of the
edit, per your own pacing) — each new "Hook" header starts a new cycle.

How the script drives the edit:
  1. The whole music track is divided into N cycles, where N = however
     many "Hook" sections your script actually contains (NOT assumed —
     read from what you wrote). Cycles are laid out evenly across the
     track's duration by default (STORY_ARC_CYCLE_SEC is only used as a
     fallback if a script has just one un-cycled pass through the arc).
  2. Within each cycle, each stage gets a time window sized proportional
     to how much text that stage has in your script (a stage with more
     written content gets more screen time), THEN each internal boundary
     is snapped to the nearest real, sharp change in the music's own RMS
     energy nearby (see snap_boundary_to_energy_shift) — so stage changes
     land on actual musical transitions instead of arbitrary math.
  3. Cuts within a stage window are still placed on the beat (same
     beat-driven timeline logic as before, restricted to that window).
  4. For each cut, candidate shots are scored primarily by how well their
     OWN transcript text (from TRANSCRIPT_MAP, produced by transcribe.py)
     matches that stage's script text — content-guided selection instead
     of randomness — blended with the existing motion/face/energy scoring
     so it still respects visual pacing.
  5. Each stage has a configured music ducking TARGET LEVEL
     (STAGE_DUCKING_LEVEL) — used when a script exists. Without a script
     the built-in ducking planner takes over (see the ducking engine).

If no arc script is found for a track, the whole pipeline falls back to
the old whole-song beat-driven cutting with purely motion/face-based shot
scoring (no text guidance; ducking still runs via the built-in planner).

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
pip install librosa soundfile scenedetect[opencv] opencv-python tqdm numpy scipy mediapipe
ffmpeg/ffprobe must be installed and on PATH.
(clean_audio.py is NO LONGER used by this script — VO/dialogue integration
has been removed entirely, per your request.)
--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python beat_sync_editor.py                       # uses CONFIG below
    python beat_sync_editor.py --duration 60         # 60-second edit
    python beat_sync_editor.py --dna AMV-EDITS       # use dna_modules/AMV-EDITS.json
    python beat_sync_editor.py --dna A,B             # blend two DNA modules
    python beat_sync_editor.py --dna ""              # no DNA (old behaviour)
    python beat_sync_editor.py --dna AMV-EDITS --dna-join inner        # one variant of a niche
    python beat_sync_editor.py --dna AMV-EDITS --dna-join all          # one edit per join variant
    python beat_sync_editor.py --dna AMV-EDITS/left                    # explicit variant file
    python beat_sync_editor.py --duration 45 --start 30   # 45s edit from 0:30 of the track

EDITING DNA (NEW): pick one or more dna_modules/*.json (made by
edit_analyzer_full.py) and the editor imitates their measured editing style:
  - cut pacing        <- median_shot_duration_sec (+ track tempo)  -> beats per cut
  - beat/onset sync   <- beat_sync_strength / onset_sync_strength  -> where cuts land
  - transitions       <- transition_type_distribution              -> xfade choice + length
  - colour changes    <- color_change_frequency_per_min            -> how long a grade is held
  - camera behaviour  <- camera_motion_distribution                -> shot motion target, shake/zoom FX
Choose modules in CONFIG (DNA_MODULES / DNA_PER_TRACK) or with --dna NAME[,NAME2].
DNA_INFLUENCE (0..1 per aspect) blends DNA vs. the editor's own defaults.

GL TRANSITIONS: the ~120 shaders of gl-transitions.com are part of every transition set (needs
`pip install moderngl`; downloaded + test-rendered once, cached in transition_library/gl/; your own .glsl
files dropped there are used too). Share per set: TRANSITION_GL_MAX_SHARE.

TRANSITIONS (NEW): the transition families come from the chosen DNA module(s)
(transition_type_distribution). Fade-to-black is banned. Every edit gets its OWN set of
200+ unique recipes (base transition x duration x post effect) from transition_library/;
a ledger guarantees no recipe is reused by another edit (30 planned sets, then it keeps
chaining). If the DNA's families run dry, the engine remixes and spills into other
families, and tops up from an online library (xfade-easing, MIT) when reachable.
    python beat_sync_editor_new.py --build-transition-sets 30   # pre-build 30 sets for the chosen DNA
    python beat_sync_editor.py --transition-report          # show the ledger
    python beat_sync_editor_new.py --online-refresh             # re-download the online library

EDIT LENGTH CONTROL: set TARGET_DURATION_SEC (or pass --duration) and the
editor only uses that much of the track, so you do NOT need to pre-trim your
music or rewrite your script. The script's Hook/Rising/Conflict/... cycles are
stretched across the chosen length (each "Hook" = one cycle, spread evenly).
START_OFFSET_SEC / --start picks WHERE in the track the edit begins.
--------------------------------------------------------------------------
"""

import os
import re
import sys
import json
import time
import random
import shutil
import logging
import difflib
import itertools
import subprocess
import traceback
import hashlib
import glob
import tempfile
import importlib.util
import concurrent.futures
from collections import defaultdict

import numpy as np
import cv2
import librosa
from tqdm import tqdm
from scenedetect import detect, ContentDetector

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# ============================== CONFIG ===================================
SOURCE_VIDEO_FOLDER = "source_footage"
MUSIC_FOLDER = "music"
MUSIC_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
SOURCE_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

# --- Edit length control (NEW) ---
# None = use the full track. Otherwise the edit is exactly this many seconds
# (or shorter if the track doesn't have that much left after START_OFFSET_SEC).
# Can also be overridden per run:  python beat_sync_editor.py --duration 60
TARGET_DURATION_SEC = None
START_OFFSET_SEC = 0.0        # where in the track the edit starts (snapped to nearest beat)
TRIM_FADE_OUT_SEC = 2.0       # audio fade-out at the end when the track is trimmed (0 = none)

# --- Editing DNA (NEW) ---
DNA_FOLDER = "dna_modules"
DNA_MODULES = ["AMV-EDITS"]        # module names (file stem) used for every track; [] = no DNA
DNA_PER_TRACK = {}                 # per-track override, e.g. {"mytrack.mp3": ["AMV-EDITS", "OTHER"]}
# Join variants (made by edit_analyzer_full.py into dna_modules/<niche>/{inner,left,right,full_outer}.json):
#   "full_outer" | "inner" | "left" | "right"  -> use that one variant
#   "all"  -> make one separate edit per variant found (output names carry the variant)
#   a list, e.g. ["inner", "left"] -> one edit per listed variant
# Applies to DNA_MODULES entries that are niche folders. Pick one explicitly with "NICHE/inner" too.
DNA_JOIN_MODE = "full_outer"
DNA_BLEND_WEIGHTS = None           # when blending several modules, e.g. {"AMV-EDITS": 0.7, "OTHER": 0.3}; None = equal
# 0 = ignore DNA for that aspect (editor's own defaults), 1 = follow DNA fully.
# "transitions" = how strongly the DNA's transition mix decides the families in each edit's set
# (below 1.0 blends in an even mix of all families).
DNA_INFLUENCE = {"pacing": 1.0, "sync": 1.0, "transitions": 1.0, "motion": 1.0, "color": 1.0}
DNA_SNAP_TOLERANCE_SEC = 0.05      # max distance a cut may move to land on a musical onset
DNA_MIN_CUT_SEC = 0.30             # snapping never creates a cut shorter than this

# --- Transition engine (NEW) ---
# Transitions are no longer hard-coded: the DNA module's transition_type_distribution picks the
# FAMILIES, and each edit draws from its own set of unique recipes. See "Transition engine" below.
TRANSITION_SETS_ENABLED = True
TRANSITION_LIBRARY_FOLDER = "transition_library"   # sets, ledger, usage log, label map, online cache
TRANSITION_SET_SIZE = 1000                          # minimum unique recipes in each edit's set
TRANSITION_SETS_PLANNED = 40                       # planned cycles; set 31+ keeps chaining automatically
TRANSITION_BANNED_BASES = {"fadeblack"}            # anything containing "black" is banned too
TRANSITION_BANNED_LABELS = {"fade_to_black"}       # DNA labels dropped (their share is redistributed)
TRANSITION_REPEAT_EXEMPT_FAMILIES = {"hard_cut"}   # a hard cut has no look to repeat
TRANSITION_USE_CUSTOM = True                       # procedural + online expression transitions
# Expression transitions render ~4x slower than built-in ones at 1440p. This caps how many of a
# set's recipes may use them (0.0 = built-ins only, 1.0 = no cap).
TRANSITION_CUSTOM_MAX_SHARE = 0.9
TRANSITION_ONLINE_DISCOVERY = True                 # download extra transitions + easings (cached)
TRANSITION_ONLINE_RECHECK_DAYS = 1
# gl-transitions (https://gl-transitions.com): ~120 GLSL shader transitions, rendered with moderngl.
TRANSITION_GL_ENABLED = True
TRANSITION_GL_FOLDER = "gl"                        # inside TRANSITION_LIBRARY_FOLDER; your own .glsl files work too
TRANSITION_GL_ZIP_URL = "https://github.com/gl-transitions/gl-transitions/archive/refs/heads/master.zip"
TRANSITION_GL_ZIP_URL_ALT = "https://codeload.github.com/gl-transitions/gl-transitions/zip/refs/heads/master"
TRANSITION_GL_MAX_SHARE = 0.7                     # max share of a set's recipes that may be GL shaders
TRANSITION_ONLINE_BASE_URL = "https://raw.githubusercontent.com/scriptituk/xfade-easing/main/expr/"
TRANSITION_ONLINE_FILES = {"transitions": "eased-transitions-yuv420p-inline.txt",
                           "easings": "xfade-easings-inline.txt"}
TRANSITION_FAMILY_DURATIONS = {   # (min, max) seconds per family
    "dissolve": (0.35, 0.70), "blur": (0.25, 0.55), "shape_mask": (0.20, 0.40), "circle": (0.20, 0.40),
    "wipe": (0.12, 0.30), "slide": (0.10, 0.22), "whip": (0.10, 0.22), "glitch": (0.08, 0.20),
    "zoom": (0.15, 0.35), "misc": (0.15, 0.40), "hard_cut": (0.034, 0.067),
}

# --- Story-arc script system ---
REMIX_FOLDER = "remix"              # revised story-arc .txt scripts, matched by music track basename
STORY_ARC_CYCLE_SEC = 30.0          # fallback cycle length ONLY used if the script has a single un-cycled pass
STORY_ARC_STAGE_NAMES = ["hook", "rising action", "conflict", "comeback", "payoff"]  # canonical stage keys
MIN_STAGE_WINDOW_SEC = 1.5          # a stage window shorter than this gets merged into its neighbor
ENERGY_SNAP_SEARCH_RADIUS_SEC = 2.5 # how far to search for a natural energy transition near a proportional boundary
SCRIPT_MATCH_WEIGHT = 0.6           # how much a shot's transcript-text match to the stage script influences selection
MULTI_TRANSCRIPT_MERGE = "any"      # "any": a source qualifies if it has speech overlapping the shot at all

# --- Audio ducking (fixed voice-over rhythm — no singing detection) ---
DUCKING_ENABLED = True
DUCK_MODE = "periodic"        # "periodic": 5s voice-over / 10s music only | "script": story-arc stage levels | "off"
# "mix"   : music (ducked) + the edit's OWN audio (dialogue/VO inside the source clips) in the windows  <- default
# "plain" : music only, never ducked | "ducked": ducked music only | "blend": music + a separate voiceover/<track> file
FINAL_AUDIO_MODE = "plain"
DUCKING_OUTPUT_SUFFIX = "_ducking.mp3"

# The rule: VO_WINDOW_SEC of voice-over, then VO_GAP_SEC of music only, repeating.
VO_WINDOW_SEC = 5.0
VO_GAP_SEC = 7.0
VO_FIRST_START_SEC = 0.0           # music plays alone for this long, then the first window opens (0 = start with a VO window)
VO_MIN_WINDOW_SEC = 1.5             # a window cut short by the end of the edit is dropped below this length
PERIODIC_DUCK_LEVEL = 0.4        # music level inside a window (1.0 = untouched, lower = more room for the voice)

# The edit's own audio (existing voice-overs / dialogue inside the source clips)
VIDEO_AUDIO_ENABLED = True
VIDEO_AUDIO_GAIN_DB = 0.35           # boost/cut the clips' own audio in the mix
VIDEO_AUDIO_GATE_FADE_SEC = 0.12    # fade in/out of the clips' audio at the window edges (it is muted outside windows)

# Script mode (DUCK_MODE = "script"): music level during each story-arc stage.
STAGE_DUCKING_LEVEL = {
    "hook": 0.35,
    "rising action": 0.45,
    "conflict": 0.70,
    "comeback": 0.40,
    "payoff": 1.00,
}
DUCKING_DEFAULT_LEVEL = 0.5

# Blend quality
DUCK_ATTACK_SEC = 0.35              # music is already ducked when the window starts (pre-duck ramp)
DUCK_RELEASE_SEC = 0.90             # music swells back this slowly after the window
DUCK_EQ_POCKET_DB = -5.0            # "voice pocket": dip around 2.8 kHz that deepens with the duck (0 = off)
DUCK_EQ_POCKET_HZ = 2800
DUCK_EQ_POCKET_Q = 0.8

# Separate voice-over file (optional): put it in voiceover/<track name>.wav|mp3|... 
VO_FOLDER = "voiceover"
VO_START_SEC = 0.0                  # where the VO file begins on the edit timeline
VO_DUCK_LEVEL = 0.30                # music level while the voice speaks
VO_ATTACK_SEC = 0.12
VO_RELEASE_SEC = 0.70
VO_GAIN_DB = 0.0
VO_BRIDGE_PAUSE_SEC = 0.35          # pauses shorter than this between words don't let the music pump back up
VO_MIN_SEGMENT_SEC = 0.15
BLEND_LOUDNORM = True               # loudness-normalise the blend (-14 LUFS) with a limiter

# --- Multi-track output ---
PROCESS_ALL_TRACKS = True
OUTPUT_FOLDER = "outputs"
OUTPUT_PATH_SINGLE = "beat_synced_edit.mp4"
CONTINUE_ON_TRACK_FAILURE = True
WRITE_MANIFEST = True

SCENE_DETECTOR_THRESHOLD = 27.0
MIN_SHOT_DURATION_SEC = 0.6

BEATS_PER_CUT_HIGH_ENERGY = 0.3
BEATS_PER_CUT_MED_ENERGY = 0.7
BEATS_PER_CUT_LOW_ENERGY = 1

# A "low energy" cut can span several beats at once, and on a slow outro
# that can add up to a genuinely long continuous span. If no single shot
# is actually that long, ffmpeg silently returns a SHORTER clip than
# requested (no error). MAX_CUT_DURATION_SEC caps any single cut's
# length; anything longer gets split into several same-energy sub-cuts.
MAX_CUT_DURATION_SEC = 1.3

AVOID_REPEAT_WINDOW = 50
AVOID_REPEAT_WINDOW_VFX = 50
AVOID_REPEAT_WINDOW_TRANSITION = 50

# --- Content-aware shot scoring ---
CONTENT_SCORING_ENABLED = True
MOTION_SAMPLE_FRAMES = 6
PREFER_FACES = True
FACE_DETECTION_SAMPLE_FRAMES = 3
CONTENT_SCORE_WEIGHT = 0.65
FACE_MODEL_SELECTION = 1
FACE_MIN_CONFIDENCE = 0.4

# --- Transcripts of the SOURCE VIDEOS (from transcribe.py) — used to match
# shots against the story-arc script's text for each stage. ---
TRANSCRIPT_GUIDED_ENABLED = True
TRANSCRIPT_MAP = {
    # "101 Dalmatians.mp4": "101 Dalmatians.txt",
}

# --- Shot-detection cache ---
SHOT_CACHE_ENABLED = True
SHOT_CACHE_FOLDER = "shot_cache"
SCORING_CHECKPOINT_EVERY = 50

# --- RNG ---
RANDOM_SEED = 100
PER_TRACK_SEEDING = True

VIDEO_CODEC = "libx264"
VIDEO_CRF = "18"
VIDEO_PRESET = "veryfast"
TARGET_FPS = 30
TARGET_RESOLUTION = "1920x1080"  #"2560x1440"
TARGET_WIDTH, TARGET_HEIGHT = (int(v) for v in TARGET_RESOLUTION.split("x"))

# --- VFX ---
VFX_ENABLED = True
VFX_LIBRARY_SIZE = 1000
VFX_LIBRARY_SEED = 7

# --- Transitions ---
TRANSITIONS_ENABLED = True
FAST_TRANSITION_DURATION_RANGE = (0.10, 0.22)
SLOW_TRANSITION_DURATION_RANGE = (0.35, 0.70)
FAST_TRANSITIONS = [
    "circleopen", "circleclose", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown", "zoomin", "pixelize",
    "hlslice", "hrslice", "vuslice", "vdslice",
]
SLOW_TRANSITIONS = [
    "fade", "fadewhite", "dissolve", "smoothleft", "smoothright",
    "smoothup", "smoothdown", "hblur", "distance", "diagtl", "diagbr",
]
AUDIO_CROSSFADE_CURVES = ["tri", "qsin", "hsin", "esin", "log", "ipar"]

# --- Logging ---
LOG_TO_FILE = True
LOG_FILE_PATH = "beat_sync_editor.log"

# --- Checkpointing / resume ---
CHECKPOINT_FOLDER = "checkpoints"
RESUME_FROM_CHECKPOINT = True
CLEAN_CHECKPOINT_ON_SUCCESS = False
# ===========================================================================


# --------------------------------------------------------------------------
# Logging + preflight
# --------------------------------------------------------------------------

def setup_logging():
    handlers = [logging.StreamHandler(sys.stdout)]
    if LOG_TO_FILE:
        handlers.append(logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                         datefmt="%H:%M:%S", handlers=handlers)
    logging.getLogger("PIL").setLevel(logging.WARNING)


log = logging.getLogger("beat_sync_editor")


def check_binary_on_path(name: str) -> bool:
    return shutil.which(name) is not None


def run_preflight_checks():
    problems = []
    if not check_binary_on_path("ffmpeg"):
        problems.append("ffmpeg not found on PATH.")
    if not check_binary_on_path("ffprobe"):
        problems.append("ffprobe not found on PATH.")
    if problems:
        for p in problems:
            log.error(f"PREFLIGHT FAILED: {p}")
        sys.exit(1)
    log.info("Preflight checks passed.")


# --------------------------------------------------------------------------
# Shot detection helpers
# --------------------------------------------------------------------------

def get_video_duration_sec(video_path: str) -> float:
    """Length of the VIDEO stream (the container length can be longer when the audio overruns)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
           "-of", "default=nw=1:nk=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        if result.returncode == 0:
            return float(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        pass
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(json.loads(result.stdout)["format"]["duration"])


def find_source_videos(folder: str):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in SOURCE_VIDEO_EXTENSIONS:
            files.append(path)
    return files


def find_music_tracks(folder: str):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in MUSIC_EXTENSIONS:
            files.append(path)
    return files


def score_shot_motion(video_path: str, start: float, end: float, sample_frames: int = 6) -> float:
    cap = cv2.VideoCapture(video_path)
    duration = max(0.05, end - start)
    sample_count = max(2, sample_frames)
    times = [start + duration * (i / sample_count) for i in range(sample_count)]
    prev_gray, scores = None, []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            scores.append(float(magnitude.mean()))
        prev_gray = gray
    cap.release()
    return sum(scores) / len(scores) if scores else 0.0


# --- Face detection: mediapipe ---
_MP_FACE_DETECTOR = None
_MP_FACE_BACKEND = None
_MP_FACE_DETECTOR_UNAVAILABLE = False


def _get_face_detector():
    global _MP_FACE_DETECTOR, _MP_FACE_BACKEND, _MP_FACE_DETECTOR_UNAVAILABLE
    if _MP_FACE_DETECTOR is not None or _MP_FACE_DETECTOR_UNAVAILABLE:
        return _MP_FACE_DETECTOR
    if not _MEDIAPIPE_AVAILABLE:
        _MP_FACE_DETECTOR_UNAVAILABLE = True
        log.warning("  WARNING: mediapipe not installed (pip install mediapipe). Face detection disabled.")
        return None

    try:
        _MP_FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=FACE_MODEL_SELECTION, min_detection_confidence=FACE_MIN_CONFIDENCE
        )
        _MP_FACE_BACKEND = "solutions"
        return _MP_FACE_DETECTOR
    except AttributeError:
        pass
    except Exception as e:
        log.warning(f"  WARNING: mediapipe 'solutions' API failed to initialize ({e}); trying the 'tasks' API instead.")

    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        model_path = os.path.join(SHOT_CACHE_FOLDER, "blaze_face_short_range.tflite")
        if not os.path.isfile(model_path):
            import urllib.request
            os.makedirs(SHOT_CACHE_FOLDER, exist_ok=True)
            model_url = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                         "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
            log.info("  Downloading mediapipe face detection model (one-time)...")
            urllib.request.urlretrieve(model_url, model_path)

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=FACE_MIN_CONFIDENCE)
        _MP_FACE_DETECTOR = mp_vision.FaceDetector.create_from_options(options)
        _MP_FACE_BACKEND = "tasks"
        log.info("  Face detection using mediapipe 'tasks' API (solutions API unavailable in this install).")
        return _MP_FACE_DETECTOR
    except Exception as e:
        _MP_FACE_DETECTOR_UNAVAILABLE = True
        log.warning(f"  WARNING: face detection unavailable on both mediapipe APIs ({e}). Continuing with it disabled.")
        _MP_FACE_DETECTOR = None
        return None


def score_shot_has_face(video_path: str, start: float, end: float, sample_frames: int = 3) -> bool:
    detector = _get_face_detector()
    if detector is None:
        return False
    cap = cv2.VideoCapture(video_path)
    duration = max(0.05, end - start)
    sample_count = max(1, sample_frames)
    times = [start + duration * (i / sample_count) for i in range(sample_count)]
    found = False
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if _MP_FACE_BACKEND == "solutions":
            result = detector.process(rgb)
            if result.detections:
                found = True
                break
        elif _MP_FACE_BACKEND == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if result.detections:
                found = True
                break
    cap.release()
    return found


# --------------------------------------------------------------------------
# Source-video transcripts (used for story-arc text matching)
# --------------------------------------------------------------------------

_TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\]\s*(?:Speaker\s*\d+:\s*)?(.*)$"
)


def parse_transcript(path: str):
    """Returns a list of (start_sec, end_sec, text) — text is kept this time, since we need it for matching."""
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _TRANSCRIPT_LINE_RE.match(line.strip())
            if not m:
                continue
            h1, m1, s1, ms1, h2, m2, s2, ms2, text = m.groups()
            start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
            end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
            text = text.strip()
            if not text or text == "[MUSIC / NON-SPEECH]":
                continue
            segments.append((start, end, text))
    return segments


def load_transcripts(video_paths):
    transcripts = {}
    if not TRANSCRIPT_GUIDED_ENABLED:
        return transcripts
    for v in video_paths:
        name = os.path.basename(v)
        transcript_path = TRANSCRIPT_MAP.get(name)
        if transcript_path and os.path.isfile(transcript_path):
            transcripts[v] = parse_transcript(transcript_path)
            log.info(f"  Loaded transcript for {name}: {len(transcripts[v])} speech segment(s).")
        elif transcript_path:
            log.warning(f"  WARNING: transcript listed for {name} but file not found: {transcript_path}")
    return transcripts


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_shot_transcript_text(source: str, start: float, end: float, transcripts: dict) -> str:
    """Concatenates whatever transcript text overlaps this shot's time span."""
    segs = transcripts.get(source)
    if not segs:
        return ""
    pieces = [text for (s, e, text) in segs if e > start and s < end]
    return " ".join(pieces)


def text_similarity(a: str, b: str) -> float:
    a, b = _normalize_text(a), _normalize_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# --- Shot-detection cache (resumable scoring) ---

def _shot_cache_path(video_path: str) -> str:
    os.makedirs(SHOT_CACHE_FOLDER, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(SHOT_CACHE_FOLDER, base + ".json")


def _video_fingerprint(video_path: str) -> dict:
    stat = os.stat(video_path)
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def load_cached_shots(video_path: str):
    if not SHOT_CACHE_ENABLED:
        return None, False
    cache_path = _shot_cache_path(video_path)
    if not os.path.exists(cache_path):
        return None, False
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, False
    if data.get("fingerprint") != _video_fingerprint(video_path):
        return None, False
    shots = data.get("shots")
    if not shots:
        return None, False
    return shots, not data.get("partial", False)


def save_cached_shots(video_path: str, shots: list, partial: bool = False):
    if not SHOT_CACHE_ENABLED:
        return
    cache_path = _shot_cache_path(video_path)
    payload = {"fingerprint": _video_fingerprint(video_path), "shots": shots, "partial": partial}
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, cache_path)


def detect_shots(video_paths):
    all_shots = []
    with tqdm(total=len(video_paths), desc="Detecting shots", unit="video", ncols=80) as pbar:
        for video_path in video_paths:
            cached_shots, is_complete = load_cached_shots(video_path)

            if cached_shots is not None and is_complete:
                tqdm.write(f"  Using cached shots for: {os.path.basename(video_path)} ({len(cached_shots)} shot(s))")
                all_shots.extend(cached_shots)
                pbar.update(1)
                continue

            if cached_shots is not None:
                shots = cached_shots
                already_scored = sum(1 for s in shots if "motion_score" in s)
                tqdm.write(f"  Resuming scoring for: {os.path.basename(video_path)} "
                           f"({already_scored}/{len(shots)} shot(s) already scored)")
            else:
                tqdm.write(f"  Scanning: {os.path.basename(video_path)}...")
                try:
                    scene_list = detect(video_path, ContentDetector(threshold=SCENE_DETECTOR_THRESHOLD))
                except Exception as e:
                    tqdm.write(f"  Shot detection failed for {os.path.basename(video_path)}: {e}. Skipping this video.")
                    pbar.update(1)
                    continue
                shots = [{"source": video_path, "start": s.seconds, "end": e.seconds} for s, e in scene_list]
                shots = [s for s in shots if (s["end"] - s["start"]) >= MIN_SHOT_DURATION_SEC]
                tqdm.write(f"    {len(shots)} usable shot(s) found.")

            if CONTENT_SCORING_ENABLED and shots:
                to_score = [i for i, s in enumerate(shots) if "motion_score" not in s]
                if to_score:
                    tqdm.write("    Scoring motion/faces...")
                for count, idx in enumerate(tqdm(to_score, desc="    scoring", unit="shot", ncols=80, leave=False)):
                    shot = shots[idx]
                    try:
                        shot["motion_score"] = score_shot_motion(shot["source"], shot["start"], shot["end"], MOTION_SAMPLE_FRAMES)
                    except Exception as e:
                        tqdm.write(f"    motion scoring failed, using 0.0: {e}")
                        shot["motion_score"] = 0.0
                    try:
                        shot["has_face"] = score_shot_has_face(shot["source"], shot["start"], shot["end"], FACE_DETECTION_SAMPLE_FRAMES) if PREFER_FACES else False
                    except Exception as e:
                        tqdm.write(f"    face scoring failed, using False: {e}")
                        shot["has_face"] = False
                    if SHOT_CACHE_ENABLED and (count + 1) % SCORING_CHECKPOINT_EVERY == 0:
                        save_cached_shots(video_path, shots, partial=True)

                motion_values = [s["motion_score"] for s in shots if "motion_score" in s]
                max_motion = max(motion_values) if motion_values else 1.0
                for shot in shots:
                    shot["motion_score_norm"] = shot.get("motion_score", 0.0) / max_motion if max_motion > 0 else 0.0
            else:
                for shot in shots:
                    shot.setdefault("motion_score", 0.0)
                    shot.setdefault("motion_score_norm", 0.0)
                    shot.setdefault("has_face", False)

            save_cached_shots(video_path, shots, partial=False)
            all_shots.extend(shots)
            pbar.update(1)

    log.info(f"Total shot pool across {len(video_paths)} source video(s): {len(all_shots)} shot(s).")
    face_count = sum(1 for s in all_shots if s.get("has_face"))
    log.info(f"  {face_count}/{len(all_shots)} shot(s) have a visible face.")
    return all_shots


# --------------------------------------------------------------------------
# Music analysis
# --------------------------------------------------------------------------

def analyze_music(music_path: str):
    log.info(f"Analyzing music: {music_path}")
    y, sr = librosa.load(music_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = np.asarray(tempo).reshape(-1)
    tempo = float(tempo[0]) if tempo.size else 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    beat_energy = np.interp(beat_times, rms_times, rms)
    low_thresh, high_thresh = np.percentile(beat_energy, [33, 66])
    labels = ["high" if e >= high_thresh else ("medium" if e >= low_thresh else "low") for e in beat_energy]
    duration = len(y) / sr
    log.info(f"  Tempo ~{tempo:.1f} BPM, {len(beat_times)} beats, {duration:.1f}s total.")
    return beat_times, labels, duration, tempo, rms, rms_times, onset_times


def is_trimmed() -> bool:
    return bool(TARGET_DURATION_SEC) or START_OFFSET_SEC > 0


def apply_duration_window(beat_times, beat_labels, music_duration, rms, rms_times, onset_times):
    """
    Restricts analysis to [ws, ws + TARGET_DURATION_SEC] and REBASES all times
    to start at 0, so every downstream step (arc windows, cuts, transitions,
    padding) just sees a shorter 'track'. Returns the rebased data plus ws
    (the real offset into the music file, used for audio extraction).
    """
    if not is_trimmed():
        return beat_times, beat_labels, music_duration, rms, rms_times, onset_times, 0.0

    ws = float(START_OFFSET_SEC)
    if ws > 0 and len(beat_times):
        ws = float(beat_times[int(np.argmin(np.abs(beat_times - ws)))])
    if ws >= music_duration - 1.0:
        log.warning(f"  START_OFFSET_SEC ({START_OFFSET_SEC}s) is past the end of this "
                     f"{music_duration:.1f}s track — starting from 0 instead.")
        ws = 0.0

    available = music_duration - ws
    edit_dur = min(float(TARGET_DURATION_SEC), available) if TARGET_DURATION_SEC else available
    if TARGET_DURATION_SEC and available < TARGET_DURATION_SEC:
        log.warning(f"  Track only has {available:.1f}s after the start point — edit will be "
                     f"{edit_dur:.1f}s, not the requested {TARGET_DURATION_SEC:.0f}s.")
    we = ws + edit_dur

    bmask = (beat_times >= ws) & (beat_times < we)
    beats = beat_times[bmask] - ws
    labels = [l for l, m in zip(beat_labels, bmask) if m]
    rmask = (rms_times >= ws) & (rms_times <= we)
    log.info(f"  Edit window: {ws:.1f}s -> {we:.1f}s of the track ({edit_dur:.1f}s, {len(beats)} beats).")
    onsets = onset_times[(onset_times >= ws) & (onset_times < we)] - ws
    return beats, labels, edit_dur, rms[rmask], rms_times[rmask] - ws, onsets, ws


# --------------------------------------------------------------------------
# Editing DNA (NEW): load/blend modules, then turn them into editor behaviour
# --------------------------------------------------------------------------

ACTIVE_BEATS_PER_CUT = {"high": BEATS_PER_CUT_HIGH_ENERGY, "medium": BEATS_PER_CUT_MED_ENERGY,
                        "low": BEATS_PER_CUT_LOW_ENERGY}
_DNA_CACHE = {}


JOIN_VARIANTS = ["full_outer", "inner", "left", "right"]


def _strip_json(n: str) -> str:
    n = n.replace("\\", "/")
    return n[:-5] if n.endswith(".json") else n


def dna_runs_for_track(music_path: str):
    """Returns a list of DNA name-lists; each one becomes its own edit of this track.
    Entries: 'NICHE' (a niche folder -> join variants per DNA_JOIN_MODE, or a legacy NICHE.json),
    'NICHE/inner' (one explicit variant). Several niches are blended using the same variant."""
    names = DNA_PER_TRACK.get(os.path.basename(music_path), DNA_MODULES) or []
    if isinstance(names, str):
        names = [names]
    names = [_strip_json(n) for n in names]
    if not names:
        return [[]]
    mode = DNA_JOIN_MODE
    wanted = list(JOIN_VARIANTS) if mode == "all" else ([mode] if isinstance(mode, str) else list(mode or []))
    fixed, niches = [], []
    for n in names:
        if "/" in n or os.path.isfile(os.path.join(DNA_FOLDER, n + ".json")):
            fixed.append(n)                       # explicit file or legacy averaged module
        elif os.path.isdir(os.path.join(DNA_FOLDER, n)):
            niches.append(n)
        else:
            fixed.append(n)                       # let the loader raise a clear "not found"
    if not niches:
        return [fixed]
    runs = []
    for v in wanted:
        have = [n for n in niches if os.path.isfile(os.path.join(DNA_FOLDER, n, v + ".json"))]
        if len(have) < len(niches):
            missing = [n for n in niches if n not in have]
            log.warning(f"  DNA variant '{v}' missing for {missing} - skipping that variant.")
            if not have:
                continue
        runs.append(fixed + [f"{n}/{v}" for n in have])
    if not runs:
        raise FileNotFoundError(f"No DNA join variants {wanted} found for {niches} in {DNA_FOLDER}/")
    return runs


def dna_names_for_track(music_path: str):
    return dna_runs_for_track(music_path)[0]


def _unflatten_join_dna(flat: dict):
    """Join modules store flat dotted features; rebuild the nested dict the editor reads
    ('dna.median_shot_duration_sec', 'dna.transition_type_distribution.cut' ...)."""
    out, grammar = {}, []
    for k, v in flat.items():
        if k.startswith("dna.") and isinstance(v, (int, float)):
            rest = k[4:]
            for top in ("transition_type_distribution", "camera_motion_distribution"):
                if rest.startswith(top + "."):
                    out.setdefault(top, {})[rest[len(top) + 1:]] = v
                    break
            else:
                out[rest] = v
        elif k.startswith("grammar.") and isinstance(v, (int, float)):
            grammar.append({"components": k[8:].split("+"), "frequency": v})
    return out, grammar


def _infl(aspect: str) -> float:
    return max(0.0, min(1.0, float(DNA_INFLUENCE.get(aspect, 1.0))))


def _lerp(a, b, t):
    return a + (b - a) * t


def load_dna_profile(names):
    """Loads one or more dna_modules/<name>.json and blends them into one {'dna', 'grammar', 'names'}."""
    if not names:
        return None
    key = tuple(names)
    if key in _DNA_CACHE:
        return _DNA_CACHE[key]
    mods = []
    for n in names:
        path = os.path.join(DNA_FOLDER, *n.split("/")) + ".json"
        if not os.path.isfile(path):
            avail = sorted(os.path.relpath(os.path.join(r, f), DNA_FOLDER)[:-5].replace(os.sep, "/")
                           for r, _, fs in os.walk(DNA_FOLDER) for f in fs if f.endswith(".json")) if os.path.isdir(DNA_FOLDER) else []
            raise FileNotFoundError(f"DNA module '{n}' not found at {path}. Available: {avail or 'none'}")
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if "pooled" in m:                       # join-variant module from edit_analyzer_full.py
            d, g = _unflatten_join_dna((m["pooled"] or {}).get("dna", {}))
            m = {"dna": d, "edit_grammar_patterns": g}
        mods.append((n, m))
    w = np.array([float((DNA_BLEND_WEIGHTS or {}).get(n, 1.0)) for n, _ in mods])
    w = w / w.sum()

    merged = {}
    all_keys = {k for _, m in mods for k in m["dna"]}
    for k in all_keys:
        vals = [(wi, m["dna"][k]) for wi, (_, m) in zip(w, mods) if k in m["dna"] and m["dna"][k] is not None]
        if not vals:
            continue
        if isinstance(vals[0][1], dict):
            sub = {}
            for sk in {x for _, d in vals for x in d}:
                sub[sk] = sum(wi * d.get(sk, 0.0) for wi, d in vals) / sum(wi for wi, _ in vals)
            merged[k] = sub
        elif isinstance(vals[0][1], (int, float)):
            merged[k] = sum(wi * v for wi, v in vals) / sum(wi for wi, _ in vals)
    grammar = [g for _, m in mods for g in m.get("edit_grammar_patterns", [])]
    profile = {"names": list(names), "dna": merged, "grammar": grammar}
    _DNA_CACHE[key] = profile
    return profile


def derive_beats_per_cut(dna, tempo: float) -> dict:
    """DNA median shot length + this track's tempo -> beats per cut for each energy level."""
    defaults = {"high": BEATS_PER_CUT_HIGH_ENERGY, "medium": BEATS_PER_CUT_MED_ENERGY, "low": BEATS_PER_CUT_LOW_ENERGY}
    median = (dna or {}).get("median_shot_duration_sec")
    t = _infl("pacing")
    if not median or tempo <= 0 or t <= 0:
        return defaults
    target_beats = median / (60.0 / tempo)
    dna_vals = {"high": target_beats * 0.6, "medium": target_beats, "low": target_beats * 1.6}
    out = {k: max(1, int(round(_lerp(defaults[k], dna_vals[k], t)))) for k in defaults}
    out["medium"] = max(out["medium"], out["high"])
    out["low"] = max(out["low"], out["medium"])
    return out


def apply_dna_sync(cuts: list, onset_times, dna) -> list:
    """
    Reproduces the reference's sync habits. Each internal cut point stays on
    its beat with probability beat_sync_strength; otherwise it may snap to the
    nearest musical onset (probability from onset_sync_strength). Stage
    boundaries from the story-arc script are never moved.
    """
    t = _infl("sync")
    if not dna or t <= 0 or onset_times is None or len(onset_times) == 0:
        return cuts
    beat_p = _lerp(1.0, float(dna.get("beat_sync_strength", 1.0)), t)
    onset_p = float(dna.get("onset_sync_strength", 0.0)) * t
    off_beat_snap_p = min(1.0, onset_p / max(1.0 - beat_p, 0.05))
    moved = 0
    for i in range(1, len(cuts)):
        prev, cur = cuts[i - 1], cuts[i]
        if prev.get("stage") != cur.get("stage"):
            continue
        if random.random() < beat_p:
            continue
        if random.random() > off_beat_snap_p:
            continue
        tcut = cur["start"]
        new = float(onset_times[int(np.argmin(np.abs(onset_times - tcut)))])
        if (abs(new - tcut) <= DNA_SNAP_TOLERANCE_SEC and new - prev["start"] >= DNA_MIN_CUT_SEC
                and cur["end"] - new >= DNA_MIN_CUT_SEC):
            cur["start"] = new
            prev["end"] = new
            moved += 1
    log.info(f"  DNA sync: {moved} cut point(s) moved off-beat onto onsets "
             f"(beat-lock p={beat_p:.2f}, onset snap p={off_beat_snap_p:.2f}).")
    return cuts


def dna_low_motion_target(dna) -> float:
    """Few static shots in the reference -> don't pick near-still shots on low-energy cuts."""
    t = _infl("motion")
    if not dna or t <= 0:
        return 0.15
    static = float((dna.get("camera_motion_distribution") or {}).get("static", 0.15))
    return 0.15 + 0.35 * max(0.0, min(1.0, (0.15 - static) / 0.15)) * t


def choose_vfx_with_dna(vfx_library, energy_label, recent_names, dna, grade_state: dict, cut_start: float):
    """
    Colour: one colour grade is HELD for 60/color_change_frequency_per_min
    seconds (so grade changes happen as often as in the reference), with the
    per-cut effect chosen separately. Camera: a high shake share in the DNA
    raises the odds of shake/zoom-punch effects on high-energy cuts.
    """
    fx_lib = [p for p in vfx_library if p["family"] != "color_grade"] or vfx_library
    parts, names = [], []

    ct = _infl("color")
    freq = float(dna.get("color_change_frequency_per_min", 0.0)) if dna else 0.0
    if ct > 0 and freq > 0:
        if cut_start >= grade_state.get("until", -1.0):
            grades = [p for p in vfx_library if p["family"] == "color_grade"]
            pool = [g for g in grades if energy_label in g["energy_tags"]] or grades
            if pool:
                grade_state["preset"] = random.choice(pool)
                grade_state["until"] = cut_start + 60.0 / freq
        if grade_state.get("preset"):
            parts.append(grade_state["preset"]["filter"])
            names.append(grade_state["preset"]["name"])

    mt = _infl("motion")
    shake_share = float((dna.get("camera_motion_distribution") or {}).get("shaky", 0.0)) if dna else 0.0
    pool = fx_lib
    if energy_label == "high" and random.random() < min(0.6, shake_share * 3 * mt):
        pool = [p for p in fx_lib if p["family"] in ("shake", "zoom_punch")] or fx_lib
    fx = choose_vfx(pool, energy_label, recent_names)
    parts.append(fx["filter"])
    names.append(fx["name"])
    return ",".join(parts), fx["name"]


def _split_long_cuts(cuts):
    split_cuts = []
    for cut in cuts:
        span_len = cut["end"] - cut["start"]
        if span_len <= MAX_CUT_DURATION_SEC:
            split_cuts.append(cut)
            continue
        n_parts = max(1, int(np.ceil(span_len / MAX_CUT_DURATION_SEC)))
        part_len = span_len / n_parts
        for p in range(n_parts):
            split_cuts.append({
                "start": cut["start"] + p * part_len,
                "end": cut["start"] + (p + 1) * part_len,
                "energy": cut["energy"],
                "stage": cut.get("stage"), "stage_text": cut.get("stage_text"),
            })
    return split_cuts


def generate_beat_cuts_in_range(beat_times, beat_labels, range_start: float, range_end: float):
    idxs = [i for i, t in enumerate(beat_times) if range_start <= t < range_end]
    if not idxs:
        return [{"start": range_start, "end": range_end, "energy": "low"}] if range_end > range_start else []

    cuts = []
    i, n = idxs[0], idxs[-1] + 1
    while i < n:
        label = beat_labels[i]
        span = ACTIVE_BEATS_PER_CUT.get(label, ACTIVE_BEATS_PER_CUT["low"])
        start_time = beat_times[i]
        end_idx = min(i + span, n - 1)
        end_time = beat_times[end_idx] if end_idx > i else min(range_end, start_time + 0.5)
        cuts.append({"start": start_time, "end": min(end_time, range_end), "energy": label})
        i = end_idx if end_idx > i else i + 1
    if cuts and cuts[-1]["end"] < range_end:
        cuts[-1]["end"] = range_end
    if cuts and cuts[0]["start"] > range_start:
        cuts[0]["start"] = range_start
    return cuts


def build_cut_timeline(beat_times, beat_labels, music_duration: float):
    """Whole-song beat-driven fallback timeline (used when no arc script is found for this track)."""
    cuts = generate_beat_cuts_in_range(beat_times, beat_labels, 0.0, music_duration)
    return _split_long_cuts(cuts)


# --------------------------------------------------------------------------
# Story-arc script system (NEW)
# --------------------------------------------------------------------------

_STAGE_HEADER_RE = re.compile(r"^[#\*\[\s]*([A-Za-z][A-Za-z\s]*?)[:\]\*\s]*$")


def find_arc_script(music_path: str):
    base = os.path.splitext(os.path.basename(music_path))[0]
    if not os.path.isdir(REMIX_FOLDER):
        return None
    candidate = os.path.join(REMIX_FOLDER, base + ".txt")
    return candidate if os.path.isfile(candidate) else None


def _match_stage_name(header_text: str):
    cleaned = _normalize_text(header_text)
    if not cleaned:
        return None
    for stage in STORY_ARC_STAGE_NAMES:
        if cleaned == stage or cleaned.startswith(stage):
            return stage
    return None


def parse_story_arc_script(path: str):
    """
    Parses the script into an ordered list of {"stage", "cycle", "text"}
    sections. A new "hook" header starts a new cycle. Returns None if no
    recognizable stage headers were found at all.
    """
    sections = []
    current_stage, current_lines = None, []
    cycle = 0

    def flush():
        if current_stage is not None:
            text = " ".join(current_lines).strip()
            if text:
                sections.append({"stage": current_stage, "cycle": cycle, "text": text})

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            m = _STAGE_HEADER_RE.match(line)
            stage = _match_stage_name(m.group(1)) if m else None
            if stage:
                flush()
                if stage == "hook":
                    cycle += 1
                current_stage, current_lines = stage, []
            else:
                if current_stage is not None:
                    current_lines.append(line)
        flush()

    return sections if sections else None


def snap_boundary_to_energy_shift(t: float, rms, rms_times, search_radius: float, bounds: tuple) -> float:
    """
    Looks for the sharpest nearby change in RMS energy (biggest |derivative|)
    within search_radius of t, and snaps t to that point instead — so stage
    boundaries land on real musical transitions rather than arbitrary math.
    Never moves outside the given (min, max) bounds.
    """
    lo, hi = max(bounds[0], t - search_radius), min(bounds[1], t + search_radius)
    mask = (rms_times >= lo) & (rms_times <= hi)
    if mask.sum() < 3:
        return t
    local_rms, local_t = rms[mask], rms_times[mask]
    d = np.abs(np.gradient(local_rms))
    idx = int(np.argmax(d))
    return float(local_t[idx])


def build_story_arc_windows(sections: list, music_duration: float, rms, rms_times) -> list:
    """
    Lays the parsed sections out across the actual music duration: cycles
    spaced evenly, each stage's share of its cycle proportional to how
    much text that stage has, boundaries snapped to real energy shifts.
    Returns a flat, time-ordered list of
    {"stage", "cycle", "start", "end", "text"} windows covering [0, music_duration).
    """
    n_cycles = max(1, max(s["cycle"] for s in sections))
    cycle_len = music_duration / n_cycles

    windows = []
    for cycle_idx in range(1, n_cycles + 1):
        cycle_sections = [s for s in sections if s["cycle"] == cycle_idx]
        if not cycle_sections:
            continue
        cycle_start = (cycle_idx - 1) * cycle_len
        cycle_end = min(cycle_idx * cycle_len, music_duration)

        word_counts = [max(1, len(s["text"].split())) for s in cycle_sections]
        total_words = sum(word_counts)

        cursor = cycle_start
        for i, (section, wc) in enumerate(zip(cycle_sections, word_counts)):
            share = wc / total_words
            raw_end = cursor + share * (cycle_end - cycle_start)
            is_last_in_cycle = (i == len(cycle_sections) - 1)
            if is_last_in_cycle:
                stage_end = cycle_end
            else:
                stage_end = snap_boundary_to_energy_shift(
                    raw_end, rms, rms_times, ENERGY_SNAP_SEARCH_RADIUS_SEC, (cursor + 0.5, cycle_end - 0.5)
                )
            stage_end = max(stage_end, cursor + 0.1)
            windows.append({"stage": section["stage"], "cycle": cycle_idx, "start": cursor, "end": stage_end, "text": section["text"]})
            cursor = stage_end

    # Merge any window that ended up too short into its neighbor, so tiny
    # slivers don't produce unusable cut windows downstream.
    merged = []
    for w in windows:
        if merged and (w["end"] - w["start"]) < MIN_STAGE_WINDOW_SEC:
            merged[-1]["end"] = w["end"]
            merged[-1]["text"] += " " + w["text"]
        else:
            merged.append(w)

    return merged


def build_cut_timeline_from_arc(arc_windows: list, beat_times, beat_labels) -> list:
    """Beat-driven cuts within each story-arc stage window, tagged with that stage's text for shot selection."""
    all_cuts = []
    for w in arc_windows:
        stage_cuts = generate_beat_cuts_in_range(beat_times, beat_labels, w["start"], w["end"])
        for c in stage_cuts:
            c["stage"] = w["stage"]
            c["stage_text"] = w["text"]
        all_cuts.extend(stage_cuts)
    all_cuts.sort(key=lambda c: c["start"])
    return _split_long_cuts(all_cuts)


# --------------------------------------------------------------------------
#  AUDIO DUCKING ENGINE — fixed voice-over rhythm (no singing detection)
# --------------------------------------------------------------------------
#  THE RULE:  a voice-over window of VO_WINDOW_SEC (5s), then VO_GAP_SEC (10s) of music only,
#  then the next window, and so on (a 15s cycle). Inside a window:
#     - the music ducks down to PERIODIC_DUCK_LEVEL (pre-ducked ahead of the window, slow release),
#     - the edit's OWN audio (the dialogue / voice-over inside the source clips) is brought up.
#  Outside the windows the music plays alone and the clips' own audio is muted, so stray
#  background noise from the footage never leaks in.
#
#  WHAT IT PRODUCES (all on the EDIT timeline, 0:00 = first frame of the edit):
#    outputs/<edit>_ducking.mp3        the music with ducking baked in
#    outputs/<edit>_mix.mp3            music (ducked) + the edit's own audio in the windows
#                                      (this is what goes into the video when FINAL_AUDIO_MODE = "mix")
#    outputs/<edit>_ducking_plan.json  the windows and levels used
#    outputs/<edit>_vo_cues.txt        the same plan as a readable cue sheet (mm:ss.mmm)
#    outputs/<edit>_blend.mp3          ONLY if a separate voiceover/<track>.* file exists
#
#  PRIORITY:  1. separate VO file present -> music follows the real voice
#             2. DUCK_MODE = "script"     -> each story-arc stage ducks to STAGE_DUCKING_LEVEL
#             3. otherwise ("periodic")   -> the 5s-on / 10s-off rule above

# ---- small utilities --------------------------------------------------------

def _rc(x):
    """Raised-cosine ramp 0->1 over x in [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def _fmt_tc(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t - 60 * int(t // 60):06.3f}"


def _mask_to_segments(mask, hop: float):
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * hop, i * hop))
            start = None
    if start is not None:
        segs.append((start * hop, len(mask) * hop))
    return segs


def _clean_segments(segs, merge_gap: float, min_len: float):
    segs = sorted(segs)
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


# ---- planning -----------------------------------------------------------------

def _periodic_slots(duration: float):
    """VO windows: VO_WINDOW_SEC on, VO_GAP_SEC off, repeating, first window at VO_FIRST_START_SEC."""
    slots, s = [], float(VO_FIRST_START_SEC)
    period = VO_WINDOW_SEC + VO_GAP_SEC
    while s < duration - 0.05:
        e = min(s + VO_WINDOW_SEC, duration)
        if e - s >= VO_MIN_WINDOW_SEC:
            slots.append({"start": s, "end": e, "level": PERIODIC_DUCK_LEVEL,
                          "source": f"periodic: {VO_WINDOW_SEC:g}s voice-over / {VO_GAP_SEC:g}s music only"})
        s += period
    return slots


def plan_ducking(duration, arc_windows, vo_segments=None):
    """Returns {"mode", "slots", "notes"} or None when ducking is off."""
    notes = []
    if DUCK_MODE == "off":
        return None

    if vo_segments:                                              # 1. follow a separate VO file
        slots = [{"start": s, "end": e, "level": VO_DUCK_LEVEL, "source": "voice-over (follows the real VO)",
                  "attack": VO_ATTACK_SEC, "release": VO_RELEASE_SEC} for s, e in vo_segments]
        return {"mode": "vo_follow", "slots": slots, "notes": notes}

    if DUCK_MODE == "script":                                    # 2. story-arc stages (opt-in)
        if arc_windows:
            slots = [{"start": w["start"], "end": w["end"],
                      "level": STAGE_DUCKING_LEVEL.get(w["stage"], DUCKING_DEFAULT_LEVEL),
                      "source": f"script: {w['stage']}"} for w in arc_windows]
            return {"mode": "script", "slots": slots, "notes": notes}
        notes.append("DUCK_MODE='script' but no script was found — using the periodic 5s/10s rule instead.")

    return {"mode": "periodic", "slots": _periodic_slots(duration), "notes": notes}      # 3. default


def detect_vo_segments(vo_path: str):
    """Speech activity of a voice-over file (VO-file time): gated RMS, pauses bridged so the music doesn't pump."""
    y, sr = librosa.load(vo_path, sr=22050, mono=True)
    hop = int(0.02 * sr)
    rms = librosa.feature.rms(y=y, frame_length=int(0.04 * sr), hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    thr = max(np.percentile(db, 10) + 10.0, np.percentile(db, 98) - 40.0)
    segs = _clean_segments(_mask_to_segments(db > thr, 0.02), VO_BRIDGE_PAUSE_SEC, VO_MIN_SEGMENT_SEC)
    return segs, len(y) / sr


def find_vo_file(music_path: str):
    if not os.path.isdir(VO_FOLDER):
        return None
    base = os.path.splitext(os.path.basename(music_path))[0].lower()
    for name in sorted(os.listdir(VO_FOLDER)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in MUSIC_EXTENSIONS and stem.lower() == base:
            return os.path.join(VO_FOLDER, name)
    return None


# ---- rendering ----------------------------------------------------------------

def build_gain_curve(duration, slots, ctrl_hz: int = 200):
    """Control-rate MUSIC gain (0..1): min over slots of raised-cosine pre-duck / hold / release."""
    t = np.arange(0.0, duration + 1.0 / ctrl_hz, 1.0 / ctrl_hz)
    gain = np.ones_like(t)
    for s in slots:
        a, r = max(float(s.get("attack", DUCK_ATTACK_SEC)), 1e-3), max(float(s.get("release", DUCK_RELEASE_SEC)), 1e-3)
        w = _rc((t - (s["start"] - a)) / a) * (1.0 - _rc((t - s["end"]) / r))
        gain = np.minimum(gain, 1.0 - (1.0 - s["level"]) * w)
    return t, gain


def build_voice_gate(duration, slots, fade: float, ctrl_hz: int = 200):
    """Control-rate gate (0..1) for the edit's own audio: open inside the slots, closed outside."""
    t = np.arange(0.0, duration + 1.0 / ctrl_hz, 1.0 / ctrl_hz)
    gate = np.zeros_like(t)
    f = max(fade, 1e-3)
    for s in slots:
        w = _rc((t - s["start"]) / f) * (1.0 - _rc((t - (s["end"] - f)) / f))
        gate = np.maximum(gate, w)
    return t, gate


def _peaking_eq(x, sr, f0, gain_db, q):
    from scipy.signal import lfilter
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * min(f0, sr * 0.45) / sr
    alpha, c = np.sin(w0) / (2 * q), np.cos(w0)
    b = np.array([1 + alpha * A, -2 * c, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * c, 1 - alpha / A])
    return lfilter(b / a[0], a / a[0], x, axis=-1)


def render_ducked_audio(music_path, window_start, duration, plan, out_wav, vo_path=None,
                        vo_offset=0.0, vo_gain_db=0.0, gate_slots=None):
    """
    Returns (out_wav, has_vo). Applies gain curve + voice-pocket EQ to the music, then (optionally)
    mixes a voice track on top. gate_slots: when given, the voice track is only audible inside those
    slots (used for the edit's own audio, so footage noise between windows is muted).
    """
    import soundfile as sf
    y, sr = librosa.load(music_path, sr=None, mono=False, offset=window_start, duration=duration)
    if y.ndim == 1:
        y = y[None, :]
    y = y.astype(np.float32)
    n = y.shape[1]
    t_c, g_c = build_gain_curve(duration, plan["slots"])
    g = np.interp(np.arange(n) / sr, t_c, g_c).astype(np.float32)

    if DUCK_EQ_POCKET_DB < 0:
        e = np.clip((1.0 - g) / 0.6, 0.0, 1.0).astype(np.float32)
        y_eq = _peaking_eq(y, sr, DUCK_EQ_POCKET_HZ, DUCK_EQ_POCKET_DB, DUCK_EQ_POCKET_Q).astype(np.float32)
        y = (1.0 - e) * y + e * y_eq
    y = y * g

    if is_trimmed() and TRIM_FADE_OUT_SEC > 0 and duration > TRIM_FADE_OUT_SEC * 2:
        k = int(TRIM_FADE_OUT_SEC * sr)
        y[:, -k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)

    has_vo = False
    if vo_path:
        v, _ = librosa.load(vo_path, sr=sr, mono=False)
        if v.ndim == 1:
            v = v[None, :]
        if v.shape[0] == 1 and y.shape[0] == 2:
            v = np.repeat(v, 2, axis=0)
        v = v[: y.shape[0]].astype(np.float32)
        off = int(vo_offset * sr)
        end = min(n, off + v.shape[1])
        if end > off:
            seg_v = v[:, : end - off] * (10 ** (vo_gain_db / 20.0))
            if gate_slots is not None:
                tg, gg = build_voice_gate(duration, gate_slots, VIDEO_AUDIO_GATE_FADE_SEC)
                seg_v = seg_v * np.interp((np.arange(off, end)) / sr, tg, gg).astype(np.float32)
            y[:seg_v.shape[0], off:end] += seg_v
            has_vo = True
    sf.write(out_wav, y.T, sr, subtype="PCM_24")
    return out_wav, has_vo


def _encode_mp3(wav_path, out_path, loudnorm: bool = False):
    cmd = ["ffmpeg", "-y", "-i", wav_path]
    if loudnorm:
        cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.94", "-ar", "44100"]
    cmd += ["-c:a", "libmp3lame", "-q:a", "2", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Audio encode failed: {r.stderr[-500:]}")


def _extract_video_audio(video_path: str, out_wav: str, duration: float) -> bool:
    """Pulls the edit's own audio (the source clips' dialogue / VO) out of the edited video."""
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-vn", "-t", f"{duration:.3f}",
                        "-ac", "2", "-ar", "44100", out_wav], capture_output=True, text=True)
    return r.returncode == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 1024


def _write_plan_files(edit_name, plan, duration, base):
    slots = [{**s, "start": round(s["start"], 3), "end": round(s["end"], 3)} for s in plan["slots"]]
    _write_json_atomic(base + "_ducking_plan.json", {
        "edit": edit_name, "mode": plan["mode"], "duration_sec": round(duration, 3), "notes": plan["notes"],
        "vo_slots": slots,
        "params": {"vo_window_sec": VO_WINDOW_SEC, "vo_gap_sec": VO_GAP_SEC, "first_start_sec": VO_FIRST_START_SEC,
                   "music_level_in_window": PERIODIC_DUCK_LEVEL, "attack_sec": DUCK_ATTACK_SEC,
                   "release_sec": DUCK_RELEASE_SEC, "eq_pocket_db": DUCK_EQ_POCKET_DB}})
    lines = [f"DUCKING CUE SHEET — {edit_name}",
             "Times are on the EDIT timeline (00:00.000 = first frame of the edit).",
             f"Mode: {plan['mode']}"] + [f"Note: {n}" for n in plan["notes"]] + ["", "VO SLOTS (music is ducked here):"]
    for i, s in enumerate(slots, 1):
        lines.append(f"  {i:02d}  {_fmt_tc(s['start'])} -> {_fmt_tc(s['end'])}  ({s['end'] - s['start']:.1f}s)  "
                      f"music at {s['level'] * 100:.0f}%  [{s['source']}]")
    if not slots:
        lines.append("  (none)")
    with open(base + "_vo_cues.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_ducking_pipeline(music_path, safe_name, work_dir, window_start, duration, arc_windows, beats, tempo,
                         edited_video_path=None):
    """Plan + render the ducked track, the music + edit-audio mix, and the VO-file blend when one exists."""
    if not DUCKING_ENABLED or DUCK_MODE == "off":
        return None
    vo_path = find_vo_file(music_path)
    vo_segments = None
    if vo_path:
        segs, vo_len = detect_vo_segments(vo_path)
        vo_segments = [(s + VO_START_SEC, e + VO_START_SEC) for s, e in segs if s + VO_START_SEC < duration]
        vo_segments = [(s, min(e, duration)) for s, e in vo_segments]
        log.info(f"  Voice-over found ({os.path.basename(vo_path)}, {vo_len:.1f}s): {len(vo_segments)} speech segment(s) — "
                  f"the music will follow the real voice.")

    plan = plan_ducking(duration, arc_windows, vo_segments)
    if plan is None:
        return None
    for n in plan["notes"]:
        log.info(f"  Ducking note: {n}")
    log.info(f"  Ducking mode: {plan['mode']} — {len(plan['slots'])} VO slot(s).")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    base = os.path.join(OUTPUT_FOLDER, safe_name)
    _write_plan_files(safe_name, plan, duration, base)

    out = {"mode": plan["mode"], "slots": len(plan["slots"]), "ducked_path": None, "blend_path": None,
           "mix_path": None, "plan": base + "_ducking_plan.json", "cues": base + "_vo_cues.txt"}
    temp_files = []

    wav = os.path.join(work_dir, "ducked_music.wav")
    temp_files.append(wav)
    render_ducked_audio(music_path, window_start, duration, plan, wav)
    out["ducked_path"] = base + DUCKING_OUTPUT_SUFFIX
    _encode_mp3(wav, out["ducked_path"])
    log.info(f"  Ducked music exported -> {out['ducked_path']}")

    # The edit's own audio (existing voice-overs inside the source clips) over the ducked music.
    if edited_video_path and VIDEO_AUDIO_ENABLED:
        va_wav, mix_wav = os.path.join(work_dir, "edit_audio.wav"), os.path.join(work_dir, "mix.wav")
        temp_files += [va_wav, mix_wav]
        if _extract_video_audio(edited_video_path, va_wav, duration):
            _, has_va = render_ducked_audio(music_path, window_start, duration, plan, mix_wav, vo_path=va_wav,
                                            vo_offset=0.0, vo_gain_db=VIDEO_AUDIO_GAIN_DB, gate_slots=plan["slots"])
            if has_va:
                out["mix_path"] = base + "_mix.mp3"
                _encode_mp3(mix_wav, out["mix_path"], loudnorm=BLEND_LOUDNORM)
                log.info(f"  Music + the edit's own audio mixed -> {out['mix_path']}")
        else:
            log.warning("  The edited video has no readable audio track — the mix will be music only.")

    if vo_path:
        blend_wav = os.path.join(work_dir, "blend.wav")
        temp_files.append(blend_wav)
        _, has_vo = render_ducked_audio(music_path, window_start, duration, plan, blend_wav,
                                        vo_path=vo_path, vo_offset=VO_START_SEC, vo_gain_db=VO_GAIN_DB)
        if has_vo:
            out["blend_path"] = base + "_blend.mp3"
            _encode_mp3(blend_wav, out["blend_path"], loudnorm=BLEND_LOUDNORM)
            log.info(f"  Music + voice-over blend exported -> {out['blend_path']}")

    for p in temp_files:
        try:
            os.remove(p)
        except OSError:
            pass
    return out



def build_audio_from_file(src_path: str, total_duration: float, out_path: str):
    """Converts an already-ducked/trimmed audio file to the aac used in the final mux."""
    cmd = ["ffmpeg", "-y", "-i", src_path, "-t", f"{total_duration:.3f}", "-c:a", "aac", "-ac", "2", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Audio conversion failed: {r.stderr[-500:]}")


# --------------------------------------------------------------------------
# Shot selection — now guided by story-arc stage text, not randomness
# --------------------------------------------------------------------------

_SRC_DUR = {}


def _source_duration(path: str) -> float:
    if path not in _SRC_DUR:
        _SRC_DUR[path] = get_video_duration_sec(path)
    return _SRC_DUR[path]


def choose_shot_for_cut(shots, cut_duration: float, recent_indices: list, energy_label: str = "medium",
                         stage_text: str = None, transcripts: dict = None, low_motion_target: float = 0.15):
    candidates = [i for i, s in enumerate(shots) if (s["end"] - s["start"]) >= min(cut_duration, MIN_SHOT_DURATION_SEC)]
    if not candidates:
        candidates = list(range(len(shots)))
    fresh = [i for i in candidates if i not in recent_indices]
    pool = fresh if fresh else candidates

    if CONTENT_SCORING_ENABLED and len(pool) > 1:
        target_motion = {"high": 1.0, "medium": 0.5, "low": low_motion_target}.get(energy_label, 0.5)
        weights = []
        for i in pool:
            shot = shots[i]
            motion_fit = 1.0 - abs(shot.get("motion_score_norm", 0.0) - target_motion)
            face_bonus = 0.25 if (PREFER_FACES and shot.get("has_face")) else 0.0
            script_bonus = 0.0
            if stage_text and TRANSCRIPT_GUIDED_ENABLED and transcripts:
                shot_text = get_shot_transcript_text(shot["source"], shot["start"], shot["end"], transcripts)
                if shot_text:
                    script_bonus = SCRIPT_MATCH_WEIGHT * text_similarity(shot_text, stage_text)
            content_score = max(0.0, motion_fit + face_bonus + script_bonus)
            weight = (1 - CONTENT_SCORE_WEIGHT) + CONTENT_SCORE_WEIGHT * content_score
            weights.append(max(0.01, weight))
        idx = random.choices(pool, weights=weights, k=1)[0]
    else:
        idx = random.choice(pool)

    shot = shots[idx]
    shot_len = shot["end"] - shot["start"]
    offset = random.uniform(0, shot_len - cut_duration) if shot_len > cut_duration else 0.0
    clip_start = shot["start"] + offset
    # never start so late that the clip runs off the end of the file (that made short segments)
    try:
        src_len = _source_duration(shot["source"])
        clip_start = max(0.0, min(clip_start, src_len - cut_duration - 0.05))
    except Exception:
        pass
    return idx, shot["source"], clip_start


# --------------------------------------------------------------------------
# VFX library
# --------------------------------------------------------------------------

def generate_vfx_library(size: int = 200, seed: int = 7):
    rng = random.Random(seed)
    presets = []

    def add(name, family, energy_tags, filter_str):
        presets.append({"name": name, "family": family, "energy_tags": energy_tags, "filter": filter_str})

    for tone, (r, g, b), tags in [
        ("warm", (1.08, 1.0, 0.92), ["calm", "high"]),
        ("cool", (0.92, 1.0, 1.08), ["low", "medium"]),
        ("desat", (1.0, 1.0, 1.0), ["low"]),
        ("hicontrast", (1.0, 1.0, 1.0), ["high"]),
    ]:
        for level in range(1, 9):
            sat = max(0.0, 1.0 - level * 0.06) if tone == "desat" else 1.0 + level * 0.02
            contrast = 1.0 + level * 0.03 if tone == "hicontrast" else 1.0
            gamma_r, gamma_g, gamma_b = [1.0 + (v - 1.0) * (level / 8) for v in (r, g, b)]
            f = f"eq=contrast={contrast:.2f}:saturation={sat:.2f}:gamma_r={gamma_r:.3f}:gamma_g={gamma_g:.3f}:gamma_b={gamma_b:.3f}"
            add(f"grade_{tone}_{level}", "color_grade", tags, f)

    for i in range(10):
        angle = round(rng.uniform(0.3, 1.2), 2)
        add(f"vignette_{i}", "vignette", ["calm", "medium", "low"], f"vignette=angle={angle}")

    for i in range(12):
        rx, bx = rng.randint(-6, 6), rng.randint(-6, 6)
        add(f"chroma_ab_{i}", "chromatic_aberration", ["high", "medium"], f"rgbashift=rh={rx}:bh={bx}")

    for i in range(10):
        strength = rng.randint(8, 40)
        add(f"grain_{i}", "grain", ["calm", "low", "medium"], f"noise=alls={strength}:allf=t")

    for i in range(14):
        rx, bx, hue_deg = rng.randint(-20, 20), rng.randint(-20, 20), rng.randint(-15, 15)
        add(f"glitch_{i}", "glitch", ["high"], f"rgbashift=rh={rx}:bh={bx},hue=h={hue_deg}")

    for i in range(10):
        gap = rng.choice([3, 4, 5, 6])
        add(f"scanlines_{i}", "scan_lines", ["medium", "high"], f"drawgrid=w=iw:h={gap}:t=1:c=black@0.15")

    for i in range(12):
        warmth = round(rng.uniform(0.15, 0.5), 2)
        add(f"lightleak_{i}", "light_leak", ["calm", "high"], f"eq=brightness={warmth*0.15:.3f}:saturation={1+warmth:.2f}")

    for i in range(10):
        zoom_amt = round(rng.uniform(1.02, 1.15), 3)
        add(f"zoompunch_{i}", "zoom_punch", ["high"], f"scale=iw*{zoom_amt}:ih*{zoom_amt},crop=iw/{zoom_amt}:ih/{zoom_amt}")

    for i in range(10):
        amp, freq = rng.randint(4, 14), round(rng.uniform(4, 10), 1)
        add(f"shake_{i}", "shake", ["high", "medium"],
            f"crop=iw-{amp*2}:ih-{amp*2}:{amp}+{amp}*sin(2*PI*t*{freq}):{amp}+{amp}*cos(2*PI*t*{freq})")

    for i in range(8):
        mode = rng.choice(["colormix", "wires"])
        add(f"sketch_{i}", "sketch_edge", ["calm", "low"], f"edgedetect=mode={mode}")

    duotone_pairs = [(1, 0, 0.3), (0.2, 0.6, 1), (1, 0.4, 0), (0.6, 0, 1), (0, 0.8, 0.6), (1, 1, 0.2)]
    for i in range(14):
        r, g, b = duotone_pairs[i % len(duotone_pairs)]
        jitter = round(rng.uniform(0.85, 1.15), 2)
        add(f"duotone_{i}", "duotone", ["medium", "high"],
            f"colorchannelmixer=rr={r*jitter:.2f}:gg={g*jitter:.2f}:bb={b*jitter:.2f}")

    for i in range(8):
        frames = rng.choice([2, 3, 4])
        add(f"streak_{i}", "motion_streak", ["high"], f"tmix=frames={frames}")

    for i in range(8):
        boost = round(rng.uniform(0.2, 0.6), 2)
        add(f"flash_{i}", "flash", ["high"], f"eq=brightness={boost:.2f}")

    for i in range(6):
        bar = rng.choice([40, 60, 80])
        add(f"letterbox_{i}", "letterbox", ["calm", "low"], f"pad=iw:ih+{bar*2}:0:{bar}:black")

    if len(presets) < size:
        base_cycle = itertools.cycle(presets)
        for _ in range(size - len(presets)):
            base = next(base_cycle)
            presets.append({**base, "name": base["name"] + "_alt"})
    return presets[:size]


def choose_vfx(vfx_library, energy_label: str, recent_names: list):
    candidates = [p for p in vfx_library if energy_label in p["energy_tags"]]
    if not candidates:
        candidates = vfx_library
    fresh = [p for p in candidates if p["name"] not in recent_names]
    pool = fresh if fresh else candidates
    return random.choice(pool)


def choose_transition(energy_label: str, recent_names: list, dna=None):
    """LEGACY chooser, only used when TRANSITION_SETS_ENABLED is False. Fade-to-black is excluded."""
    is_fast = energy_label == "high"
    pool_names = [t for t in (FAST_TRANSITIONS if is_fast else SLOW_TRANSITIONS) if not _is_banned_base(t)]
    fresh = [t for t in pool_names if t not in recent_names]
    chosen = random.choice(fresh if fresh else pool_names)
    dur_range = FAST_TRANSITION_DURATION_RANGE if is_fast else SLOW_TRANSITION_DURATION_RANGE
    return chosen, round(random.uniform(*dur_range), 3)


# --------------------------------------------------------------------------
# Cutting individual segments (with VFX baked in)
# --------------------------------------------------------------------------

def cut_segment(source_path: str, start: float, duration: float, out_path: str, vfx_filter: str = None):
    base_vf = f"scale={TARGET_RESOLUTION}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    renormalize = f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    vf_chain = base_vf if not vfx_filter else f"{base_vf},{vfx_filter},{renormalize}"

    seek_buffer = 1.0
    coarse_start = max(0.0, start - seek_buffer)
    fine_offset = start - coarse_start

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{coarse_start:.3f}",
        "-i", source_path,
        "-ss", f"{fine_offset:.3f}",
        "-t", f"{duration:.3f}",
        "-vf", vf_chain,
        "-r", str(TARGET_FPS),
        "-pix_fmt", "yuv420p",
        "-c:v", VIDEO_CODEC, "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET,
        "-c:a", "aac", "-ac", "2",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


# --------------------------------------------------------------------------
# Transition engine (NEW)
# --------------------------------------------------------------------------
#  1. FAMILIES come from the selected DNA module(s): every label in
#     transition_type_distribution is resolved to transition families through
#     transition_library/label_map.json (auto-created and editable; labels the
#     analyzer invents later are classified by keyword and saved there). No
#     per-label transition lists are hard-coded in this script any more.
#  2. A REGISTRY of building blocks is probed against YOUR ffmpeg and only
#     blocks that really render are kept:
#        - ffmpeg's own xfade transitions (discovered from the local build),
#        - ~140 procedurally generated "custom expression" transitions,
#        - ONLINE: 100+ extra transitions x 40+ easings from the open-source
#          xfade-easing project (MIT, github.com/scriptituk/xfade-easing),
#          downloaded once and cached in transition_library/online/,
#        - time-gated post effects (flash, chroma split, hue/blur pulses...).
#  3. A RECIPE = base transition x duration x post effect. Each edit gets its
#     own SET of 200+ recipes split across families by the DNA mix. A ledger
#     (transition_library/transition_ledger.json) records every recipe ever
#     issued, so a recipe is never reused by another edit. Sets 1..30 are the
#     planned cycles; set 31+ simply keeps chaining.
#  4. FADE-TO-BLACK is banned at every level (labels, bases, recipes).

TRANSITION_FALLBACK_FAMILIES = ["dissolve", "blur", "shape_mask", "circle", "wipe", "slide", "whip", "glitch", "zoom"]

DEFAULT_LABEL_MAP = {
    "hard_cut": ["hard_cut"],
    "dissolve": ["dissolve", "blur"],
    "fade_to_white": ["dissolve"],
    "shape_mask_reveal": ["shape_mask"],
    "circular_reveal": ["circle"],
    "wipe": ["wipe"],
    "whip_pan": ["whip", "slide"],
    "glitch_transition": ["glitch"],
    "zoom_transition": ["zoom"],
    "fade_to_black": "BANNED",
}

_LABEL_KEYWORDS = [
    (r"glitch|slice|pixel|static|noise", ["glitch"]),
    (r"circle|circular|round", ["circle"]),
    (r"mask|reveal|shape|iris", ["shape_mask"]),
    (r"wipe", ["wipe"]),
    (r"whip|slide|pan|push", ["whip", "slide"]),
    (r"zoom|pull|scale|swirl|spin|rotate|twist|warp", ["zoom"]),
    (r"blur", ["blur"]),
    (r"dissolve|fade|cross|mix|blend", ["dissolve"]),
    (r"cut", ["hard_cut"]),
]

_FAMILY_RULES = [
    ("circle", r"circle|radial|iris|polar|clock"),
    ("shape_mask", r"(open|close)$|^rectcrop|^squeeze|circle|radial|iris|bowtie|heart|diamond|polar|clock|window|curtain|^zoomin"),
    ("wipe", r"wipe|^diag"),
    ("slide", r"^slide|^smooth|^cover|^reveal|swipe|push|swap|cube|flip|page|curl"),
    ("whip", r"^slide|^smooth|swipe|swish|warp"),
    ("glitch", r"glitch|mosaic|noise|static|squares|bars|stripe|slice|wind|^pixelize|checker|dream"),
    ("dissolve", r"^fade|dissolve|^distance|crossfade|blend|morph|luma|burn|flash"),
    ("blur", r"blur|^distance|^pixelize"),
    ("zoom", r"zoom|^squeeze|swirl|spin|rotate|kaleido|bounce|pinwheel|cross"),
]

_POST_KINDS = {
    "flash": [0.15, 0.30, 0.45, 0.60],
    "chroma": [3, 6, 10, 16],
    "hue": [25, 60, 120],
    "blur": [3, 6, 10, 16],
    "invert": [0.03, 0.05],
    "grain": [20, 40, 70],
    "sat": [1.6, 2.2, 3.0],
    "contrast": [1.4, 1.8, 2.4],
}
_POST_COMBOS = ["flash_1+chroma_2", "flash_2+hue_1", "blur_2+chroma_1", "grain_2+sat_2", "invert_1+chroma_3"]

# Online expressions are untrusted text that ends up inside an ffmpeg filtergraph:
# only plain expression characters are allowed (no quotes, brackets or colons).
_SAFE_EXPR_RE = re.compile(r"^[A-Za-z0-9_\s.,;()+\-*/^<>=!|&%]+$")

_TRANSITION_REGISTRY = None


# ---- small helpers --------------------------------------------------------

def _tl_path(*parts) -> str:
    os.makedirs(TRANSITION_LIBRARY_FOLDER, exist_ok=True)
    return os.path.join(TRANSITION_LIBRARY_FOLDER, *parts)


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _is_banned_base(name: str) -> bool:
    n = str(name).lower()
    return n in {b.lower() for b in TRANSITION_BANNED_BASES} or "black" in n


def _is_banned_label(label: str) -> bool:
    l = str(label).lower()
    return l in {b.lower() for b in TRANSITION_BANNED_LABELS} or "black" in l


def classify_base_name(name: str):
    n = name.lower()
    fams = [fam for fam, rx in _FAMILY_RULES if re.search(rx, n)]
    return fams or ["misc"]


def load_label_map() -> dict:
    path = _tl_path("label_map.json")
    user = _read_json(path, {})
    merged = {**DEFAULT_LABEL_MAP, **{k.lower(): v for k, v in user.items()}}
    if not os.path.isfile(path):
        _write_json_atomic(path, merged)
    return merged


def resolve_label_families(label: str, label_map: dict):
    """DNA transition label -> list of families, or None when the label is banned (fade-to-black)."""
    l = label.lower()
    if _is_banned_label(l) or label_map.get(l) == "BANNED":
        return None
    if label_map.get(l):
        return list(label_map[l])
    chosen = list(TRANSITION_FALLBACK_FAMILIES)   # unrecognised label: spread over every family
    for rx, fams in _LABEL_KEYWORDS:
        if re.search(rx, l):
            chosen = fams
            break
    label_map[l] = chosen
    _write_json_atomic(_tl_path("label_map.json"), label_map)
    log.info(f"  New transition label '{label}' from the DNA -> families {chosen} "
              f"(saved to {_tl_path('label_map.json')}; edit it to change).")
    return chosen


def dna_family_weights(dna) -> dict:
    """Family mix for this edit's transition set, from the DNA's transition_type_distribution."""
    t = _infl("transitions")
    label_map = load_label_map()
    dist = (dna or {}).get("transition_type_distribution") or {}
    dna_w, dropped = defaultdict(float), 0.0
    for label, w in dist.items():
        if not w or w <= 0:
            continue
        fams = resolve_label_families(label, label_map)
        if fams is None:
            dropped += w
            continue
        for f in fams:
            dna_w[f] += w / len(fams)
    if dropped > 0:
        log.info(f"  Fade-to-black banned: its {dropped * 100:.0f}% share of the DNA's transitions was dropped "
                  f"and the rest renormalized.")
    s = sum(dna_w.values())
    uni = {f: 1.0 / len(TRANSITION_FALLBACK_FAMILIES) for f in TRANSITION_FALLBACK_FAMILIES}
    if s <= 0:
        return uni
    dna_w = {f: v / s for f, v in dna_w.items()}
    out = {f: t * dna_w.get(f, 0.0) + (1 - t) * uni.get(f, 0.0) for f in set(dna_w) | set(uni)}
    tot = sum(out.values())
    return {f: v / tot for f, v in out.items() if v > 0}


# ---- building blocks ------------------------------------------------------

def generate_custom_bases():
    """
    Procedural xfade 'custom' expressions. During a transition P runs 1 -> 0,
    so q = 1-P is the progress; A/B are the outgoing/incoming pixels.
    """
    out = []

    def add(cid, expr, fams):
        out.append({"key": "custom:" + cid, "kind": "custom", "name": None, "expr": expr, "families": fams})

    def mix(m):
        return f"A*(1-({m}))+B*({m})"

    q = "(1-P)"
    for k in (1.5, 2, 3, 4):                                   # eased cross-fades (12)
        add(f"ez_in_{k:g}", mix(f"pow({q},{k:g})"), ["dissolve"])
        add(f"ez_out_{k:g}", mix(f"1-pow(P,{k:g})"), ["dissolve"])
        add(f"ez_io_{k:g}", mix(f"if(lt({q},0.5),0.5*pow(2*{q},{k:g}),1-0.5*pow(2*P,{k:g}))"), ["dissolve"])

    dirs = {"L": "X/W", "R": "(1-X/W)", "T": "Y/H", "B": "(1-Y/H)",
            "TL": "((X/W+Y/H)/2)", "TR": "(((1-X/W)+Y/H)/2)"}
    for dname, u in dirs.items():                               # feathered wipes (24)
        for f in (0.04, 0.10, 0.20, 0.35):
            fams = ["wipe"] + (["whip"] if dname in ("L", "R") and f <= 0.10 else [])
            add(f"fw_{dname}_{int(f * 100)}", mix(f"clip(({q}*(1+{f})-{u})/{f},0,1)"), fams)

    for dname, along, across in (("L", "X/W", "Y/H"), ("R", "(1-X/W)", "Y/H"), ("T", "Y/H", "X/W")):
        for a in (0.04, 0.08, 0.15):                            # wavy wipes (27)
            for kf in (2, 4, 7):
                u = f"({along}+{a}*sin({across}*6.2832*{kf}))"
                add(f"fv_{dname}_{int(a * 100)}_{kf}",
                    mix(f"clip(({q}*(1+0.08+2*{a})-({u}+{a}))/0.08,0,1)"), ["wipe"])

    centers = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for ci, (cx, cy) in enumerate(centers):                     # feathered radial open/close (30)
        mx, my = max(cx, 1 - cx), max(cy, 1 - cy)
        d = f"(hypot(X-W*{cx},Y-H*{cy})/hypot(W*{mx},H*{my}))"
        for f in (0.08, 0.18, 0.35):
            add(f"rad_o_{int(f * 100)}_c{ci}", mix(f"clip(({q}*(1+{f})-{d})/{f},0,1)"), ["circle", "shape_mask"])
            add(f"rad_c_{int(f * 100)}_c{ci}", mix(f"clip(({q}*(1+{f})-(1-{d}))/{f},0,1)"), ["circle", "shape_mask"])

    shapes = {"rect": "max(abs(X-W/2)/(W/2),abs(Y-H/2)/(H/2))",
              "dia": "((abs(X-W/2)/(W/2)+abs(Y-H/2)/(H/2))/2)"}
    for sname, d in shapes.items():                             # rectangular / diamond iris (12)
        for f in (0.06, 0.15, 0.30):
            add(f"iris_{sname}_o_{int(f * 100)}", mix(f"clip(({q}*(1+{f})-({d}))/{f},0,1)"), ["shape_mask"])
            add(f"iris_{sname}_c_{int(f * 100)}", mix(f"clip(({q}*(1+{f})-(1-({d})))/{f},0,1)"), ["shape_mask"])

    consts = [(0.618034, 0.381966), (0.7071, 0.2929), (0.4142, 0.5858)]
    for n in (6, 10, 16, 24):                                   # block / row / column scatter (36)
        for ci, (c1, c2) in enumerate(consts):
            thr = {"blk": f"mod(floor(X*{n}/W)*{c1}+floor(Y*{n}/W)*{c2},1)",
                   "row": f"mod(floor(Y*{n}/H)*{c1},1)",
                   "col": f"mod(floor(X*{n}/W)*{c1},1)"}
            for mname, t_ in thr.items():
                add(f"{mname}_{n}_{ci}", f"if(gt({q},{t_}),B,A)", ["glitch"])
    return out


def all_post_ids():
    ids = ["none"]
    for kind, vals in _POST_KINDS.items():
        ids += [f"{kind}_{i + 1}" for i in range(len(vals))]
    return ids + _POST_COMBOS


def build_post_filter(post_id: str, off: float, dur: float):
    """Time-gated effect applied to the merged output only around the transition window."""
    if post_id == "none":
        return None
    c = off + dur / 2.0
    win = f"between(t,{off:.3f},{off + dur:.3f})"

    def blip(half):
        return f"between(t,{c - half:.3f},{c + half:.3f})"

    parts = []
    for token in post_id.split("+"):
        kind, lvl = token.rsplit("_", 1)
        v = _POST_KINDS[kind][int(lvl) - 1]
        if kind == "flash":
            parts.append(f"eq=brightness={v}:contrast=1.3:enable='{blip(0.05)}'")
        elif kind == "chroma":
            parts.append(f"rgbashift=rh={v}:bh=-{v}:gv={v // 2}:enable='{win}'")
        elif kind == "hue":
            parts.append(f"hue=h={v}:s=1.3:enable='{win}'")
        elif kind == "blur":
            parts.append(f"gblur=sigma={v}:enable='{win}'")
        elif kind == "invert":
            parts.append(f"negate=enable='{blip(v)}'")
        elif kind == "grain":
            parts.append(f"noise=alls={v}:allf=t:enable='{win}'")
        elif kind == "sat":
            parts.append(f"eq=saturation={v}:enable='{win}'")
        elif kind == "contrast":
            parts.append(f"eq=contrast={v}:enable='{win}'")
    return ",".join(parts)


# ---- probing against the local ffmpeg ---------------------------------------

def _ffmpeg_version() -> str:
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return (r.stdout.splitlines() or ["unknown"])[0]
    except OSError:
        return "unknown"


def _local_xfade_names():
    r = subprocess.run(["ffmpeg", "-hide_banner", "-h", "filter=xfade"], capture_output=True, text=True)
    names = []
    for line in (r.stdout + r.stderr).splitlines():
        m = re.match(r"^\s+([a-z0-9]+)\s+-?\d+\s+\.\.", line)
        if m:
            names.append(m.group(1))
    return names


def _probe_transition(name: str = None, expr: str = None) -> bool:
    spec = f"custom:expr='{expr}'" if expr else name
    fc = (f"[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];"
          f"[a][b]xfade=transition={spec}:duration=0.4:offset=0.2[v]")
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=10:duration=1",
           "-f", "lavfi", "-i", "color=c=red:size=64x36:rate=10:duration=1",
           "-filter_complex", fc, "-map", "[v]", "-frames:v", "4", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _probe_post(post_id: str) -> bool:
    flt = build_post_filter(post_id, 0.1, 0.3)
    if flt is None:
        return True
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=10:duration=1",
           "-vf", f"format=yuv420p,{flt}", "-frames:v", "5", "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:6]


def _parse_online_exprs(text: str) -> dict:
    out, name = {}, None
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            name = None
            continue
        if line.startswith("#"):
            continue
        if name is None and re.fullmatch(r"[A-Z0-9_\-]+", line.strip()):
            name = line.strip()
        elif name:
            out[name] = line.strip()
            name = None
    return out


def fetch_online_exprs(force: bool = False):
    """(transitions, easings, status). Cached on disk; refreshed every TRANSITION_ONLINE_RECHECK_DAYS."""
    folder = _tl_path("online")
    os.makedirs(folder, exist_ok=True)
    meta_path = os.path.join(folder, "meta.json")
    meta = _read_json(meta_path, {})
    paths = {k: os.path.join(folder, fn) for k, fn in TRANSITION_ONLINE_FILES.items()}
    have_cache = all(os.path.isfile(p) for p in paths.values())
    fresh = bool(meta.get("fetched_at")) and (time.time() - meta["fetched_at"]) < TRANSITION_ONLINE_RECHECK_DAYS * 86400
    status = "cache" if have_cache else "none"

    if TRANSITION_ONLINE_DISCOVERY and (force or not (fresh and have_cache)):
        try:
            import urllib.request
            for k, fn in TRANSITION_ONLINE_FILES.items():
                req = urllib.request.Request(TRANSITION_ONLINE_BASE_URL + fn, headers={"User-Agent": "beat_sync_editor"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read().decode("utf-8")
                with open(paths[k] + ".tmp", "w", encoding="utf-8") as f:
                    f.write(data)
                os.replace(paths[k] + ".tmp", paths[k])
            meta["fetched_at"] = time.time()
            _write_json_atomic(meta_path, meta)
            have_cache, status = True, "downloaded"
        except Exception as e:
            log.warning(f"  Transition library: online fetch failed ({e}). "
                         f"{'Using the cached copy.' if have_cache else 'Continuing without online transitions.'}")
            status = "cache (offline)" if have_cache else "offline"
    if not have_cache:
        return {}, {}, status

    with open(paths["transitions"], "r", encoding="utf-8") as f:
        trans = _parse_online_exprs(f.read())
    with open(paths["easings"], "r", encoding="utf-8") as f:
        eas = _parse_online_exprs(f.read())
    trans = {n: e for n, e in trans.items() if not _is_banned_base(n) and _SAFE_EXPR_RE.match(e)}
    eas = {n: e for n, e in eas.items() if _SAFE_EXPR_RE.match(e)}
    return trans, eas, status


# ---- gl-transitions (https://gl-transitions.com, MIT): GLSL shaders rendered with moderngl ----------------
# ffmpeg cannot run GLSL shaders, so for these transitions only the overlap window (a few frames) is
# rendered on the GPU: the outgoing and incoming frames go through the shader, the result is encoded as a
# tiny clip and spliced between the untouched start of the running edit and the rest of the next segment.
# Needs:  pip install moderngl   (any OpenGL 3.3 GPU/driver). Without it these transitions are skipped.

_GL = {"ctx": None, "tried": False, "progs": {}, "vao": {}, "tex": {}, "warned": False}

_GL_VERT = """#version 330
in vec2 in_pos; out vec2 _uv;
void main(){ _uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }"""
_GL_FRAG_HEAD = """#version 330
uniform sampler2D tex_from; uniform sampler2D tex_to; uniform float progress; uniform float ratio;
in vec2 _uv; out vec4 fragColor;
vec4 getFromColor(vec2 uv){ return texture(tex_from, uv); }
vec4 getToColor(vec2 uv){ return texture(tex_to, uv); }
"""
_GL_FRAG_TAIL = "\nvoid main(){ fragColor = transition(_uv); }\n"
_GL_UNIFORM_RE = re.compile(
    r"^\s*uniform\s+(\w+)\s+(\w+)\s*(?:;\s*//\s*=\s*(.*?)\s*$|/\*\s*=\s*(.*?)\s*\*/\s*;)", re.M)


def _gl_context():
    if _GL["tried"]:
        return _GL["ctx"]
    _GL["tried"] = True
    try:
        import moderngl
    except ImportError:
        log.warning("  gl-transitions: moderngl is not installed (pip install moderngl) - GL transitions are off.")
        return None
    for backend in (None, "egl"):
        try:
            ctx = moderngl.create_standalone_context(backend=backend) if backend else moderngl.create_standalone_context()
            _GL["ctx"] = ctx
            return ctx
        except Exception as e:
            err = e
    log.warning(f"  gl-transitions: no OpenGL context available ({err}) - GL transitions are off.")
    return None


def _gl_prepare_source(src: str):
    """Turns a gl-transitions shader into a full fragment shader. Uniform parameters become constants set to
    their documented defaults. Shaders needing extra textures or parameters without a default are skipped."""
    if re.search(r"uniform\s+sampler2D", src):
        return None
    body = _GL_UNIFORM_RE.sub(lambda m: f"const {m.group(1)} {m.group(2)} = {m.group(3) or m.group(4)};", src)
    if re.search(r"^\s*uniform\s", body, re.M):
        return None
    return _GL_FRAG_HEAD + body + _GL_FRAG_TAIL


def gl_load_shaders():
    """{name: glsl source}: cached in transition_library/gl/, refreshed from the project every few days.
    You can also drop your own .glsl files into that folder."""
    folder = _tl_path(TRANSITION_GL_FOLDER)
    os.makedirs(folder, exist_ok=True)
    meta_path = os.path.join(folder, "meta.json")
    meta = _read_json(meta_path, {})
    have = [f for f in os.listdir(folder) if f.endswith(".glsl")]
    fresh = bool(meta.get("fetched_at")) and (time.time() - meta["fetched_at"]) < TRANSITION_ONLINE_RECHECK_DAYS * 86400
    if TRANSITION_ONLINE_DISCOVERY and not (fresh and have):
        try:
            import urllib.request, zipfile, io
            data, last_err = None, None
            for url in (TRANSITION_GL_ZIP_URL, TRANSITION_GL_ZIP_URL_ALT):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "beat_sync_editor"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        data = resp.read()
                    break
                except Exception as e:
                    last_err = e
            if data is None:
                raise last_err
            n = 0
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for zi in z.namelist():
                    parts = zi.split("/")
                    if len(parts) >= 3 and parts[-2] == "transitions" and parts[-1].endswith(".glsl"):
                        with open(os.path.join(folder, parts[-1]), "wb") as f:
                            f.write(z.read(zi))
                        n += 1
            meta["fetched_at"] = time.time()
            _write_json_atomic(meta_path, meta)
            log.info(f"  gl-transitions: downloaded {n} shader(s) into {folder}.")
        except Exception as e:
            log.warning(f"  gl-transitions: download failed ({e}). "
                         f"{'Using the cached shaders.' if have else 'Drop .glsl files into ' + folder + ' to use them offline.'}")
    out = {}
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".glsl"):
            try:
                with open(os.path.join(folder, fn), "r", encoding="utf-8", errors="replace") as f:
                    out[os.path.splitext(fn)[0]] = f.read()
            except OSError:
                pass
    return out


def _gl_program(name: str, src: str):
    if name in _GL["progs"]:
        return _GL["progs"][name]
    ctx = _gl_context()
    full = _gl_prepare_source(src)
    if ctx is None or full is None:
        raise RuntimeError("shader unavailable")
    prog = ctx.program(vertex_shader=_GL_VERT, fragment_shader=full)
    import moderngl
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
    _GL["vao"][name] = ctx.simple_vertex_array(prog, vbo, "in_pos")
    _GL["progs"][name] = prog
    return prog


def _gl_targets(w: int, h: int):
    key = (w, h)
    if key not in _GL["tex"]:
        ctx = _gl_context()
        t_from, t_to = ctx.texture((w, h), 3), ctx.texture((w, h), 3)
        out = ctx.texture((w, h), 3)
        _GL["tex"][key] = (t_from, t_to, ctx.framebuffer(color_attachments=[out]))
    return _GL["tex"][key]


def gl_render_frame(name: str, src: str, from_rgb: bytes, to_rgb: bytes, w: int, h: int, progress: float) -> bytes:
    prog = _gl_program(name, src)
    t_from, t_to, fbo = _gl_targets(w, h)
    # gl-transitions uses y-up uv (0,0 = bottom-left), so upload rows bottom-first
    for tex, raw in ((t_from, from_rgb), (t_to, to_rgb)):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)[::-1]
        tex.write(np.ascontiguousarray(arr).tobytes(), alignment=1)
    t_from.use(0)
    t_to.use(1)
    for uname, val in (("tex_from", 0), ("tex_to", 1), ("progress", float(progress)), ("ratio", w / float(h))):
        if uname in prog:                      # unused uniforms are optimised away by the GLSL compiler
            prog[uname].value = val
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    _GL["vao"][name].render(mode=_GL_TRIANGLE_STRIP)
    data = np.frombuffer(fbo.read(components=3, alignment=1), dtype=np.uint8).reshape(h, w, 3)[::-1]
    return np.ascontiguousarray(data).tobytes()


try:
    import moderngl as _mgl
    _GL_TRIANGLE_STRIP = _mgl.TRIANGLE_STRIP
except Exception:
    _GL_TRIANGLE_STRIP = 5


def _probe_gl(name: str, src: str) -> bool:
    try:
        w, h = 64, 36
        a = bytes(random.Random(1).randrange(256) for _ in range(w * h * 3))
        b = bytes(255 - x for x in a)
        for p in (0.25, 0.75):
            out = gl_render_frame(name, src, a, b, w, h, p)
            if len(out) != w * h * 3:
                return False
        return True
    except Exception:
        return False


def gl_shader_tags(src: str) -> str:
    m = re.search(r"^//\s*Tags:\s*(.*)$", src, re.M | re.I)
    return (m.group(1) if m else "").lower()


def _merge_gl(running_path, next_path, out_path, offset, d, audio_curve, shader, post_id):
    """Beat-locked GL merge: [running up to offset] + [GL-rendered overlap] + [rest of next]."""
    src = gl_load_shaders().get(shader)
    if src is None:
        raise RuntimeError(f"GL shader '{shader}' not found")
    fps, w, h = TARGET_FPS, TARGET_WIDTH, TARGET_HEIGHT
    n = max(2, int(round(d * fps)))
    K = max(0, int(round(offset * fps)))
    t0 = K / fps
    fsize = w * h * 3
    scale = (f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
             f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    def decoder(path, start):
        return subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{start:.5f}", "-i", path, "-an", "-frames:v", str(n),
             "-vf", scale, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read_frame(proc, last):
        buf = b""
        while len(buf) < fsize:
            chunk = proc.stdout.read(fsize - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf if len(buf) == fsize else last

    mid = out_path + ".mid.mp4"
    dec_a, dec_b = decoder(running_path, t0), decoder(next_path, 0.0)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-c:v", VIDEO_CODEC, "-crf", "14", "-preset", "ultrafast", "-pix_fmt", "yuv420p", mid],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        last_a = last_b = None
        for k in range(n):
            fa, fb = read_frame(dec_a, last_a), read_frame(dec_b, last_b)
            if fa is None or fb is None:
                raise RuntimeError("could not decode frames for the GL overlap")
            last_a, last_b = fa, fb
            enc.stdin.write(gl_render_frame(shader, src, fa, fb, w, h, (k + 1) / (n + 1)))
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        if enc.wait() != 0:
            raise RuntimeError(f"GL overlap encode failed: {err[-800:]}")
    finally:
        for p in (dec_a, dec_b):
            try:
                p.stdout.close()
                p.kill()
            except Exception:
                pass
        if enc.poll() is None:
            enc.kill()

    norm = f"fps={fps},format=yuv420p,setsar=1"
    parts = []
    if K > 0:
        parts.append(f"[0:v]{norm},trim=end_frame={K},setpts=PTS-STARTPTS[va]")
    parts.append(f"[2:v]{norm},setpts=PTS-STARTPTS[vm]")
    parts.append(f"[1:v]{norm},trim=start_frame={n},setpts=PTS-STARTPTS[vc]")
    labels = ("[va]" if K > 0 else "") + "[vm][vc]"
    parts.append(f"{labels}concat=n={3 if K > 0 else 2}:v=1:a=0[vx]")
    post = build_post_filter(post_id, offset, d) if post_id else None
    if post:
        parts.append(f"[vx]{post}[v]")
        vlabel = "[v]"
    else:
        vlabel = "[vx]"
    parts.append(f"[0:a][1:a]acrossfade=d={d:.3f}:c1={audio_curve}:c2={audio_curve}[a]")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
           "-i", running_path, "-i", next_path, "-i", mid, "-filter_complex", ";".join(parts),
           "-map", vlabel, "-map", "[a]", "-pix_fmt", "yuv420p",
           "-c:v", VIDEO_CODEC, "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET, "-c:a", "aac", out_path]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        try:
            os.remove(mid)
        except OSError:
            pass


def get_transition_registry(force_online: bool = False):
    global _TRANSITION_REGISTRY
    if _TRANSITION_REGISTRY is not None and not force_online:
        return _TRANSITION_REGISTRY

    cache_path = _tl_path("capabilities.json")
    ver = _ffmpeg_version()
    cache = _read_json(cache_path, {})
    if cache.get("ffmpeg") != ver:
        cache = {"ffmpeg": ver, "probe": {}}
    probe = cache.setdefault("probe", {})

    local = _local_xfade_names()
    custom_ok = "custom" in local and TRANSITION_USE_CUSTOM
    builtin = [n for n in local if n != "custom" and not _is_banned_base(n)]
    custom = generate_custom_bases() if custom_ok else []
    on_trans, on_eas, on_status = fetch_online_exprs(force_online) if custom_ok else ({}, {}, "disabled")
    posts_all = all_post_ids()

    jobs = []  # (cache_key, callable)
    for n in builtin:
        jobs.append((f"b:{n}", lambda n=n: _probe_transition(name=n)))
    for c in custom:
        jobs.append((f"c:{c['key']}:{_md5(c['expr'])}", lambda e=c["expr"]: _probe_transition(expr=e)))
    for n, e in on_trans.items():
        jobs.append((f"o:{n}:{_md5(e)}", lambda e=e: _probe_transition(expr="st(0,P);" + e)))
    for n, e in on_eas.items():
        jobs.append((f"e:{n}:{_md5(e)}", lambda e=e: _probe_transition(expr=e + ";" + (on_trans.get("CIRCLEOPEN") or "A"))))
    for p in posts_all:
        jobs.append((f"p:{p}", lambda p=p: _probe_post(p)))

    gl_src = {}
    if TRANSITION_GL_ENABLED and _gl_context() is not None:
        gl_src = {n: src for n, src in gl_load_shaders().items()
                  if not _is_banned_base(n) and _gl_prepare_source(src) is not None}
    todo = [(k, fn) for k, fn in jobs if k not in probe]
    if todo:
        log.info(f"  Transition registry: probing {len(todo)} building block(s) against your ffmpeg (one-time, cached)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for (k, _), ok in zip(todo, ex.map(lambda kf: kf[1](), todo)):
                probe[k] = bool(ok)
        _write_json_atomic(cache_path, cache)

    gl_todo = [n for n, src in gl_src.items() if f"g:{n}:{_md5(src)}" not in probe]
    if gl_todo:
        log.info(f"  Transition registry: test-rendering {len(gl_todo)} GL shader(s) (one-time, cached)...")
        for n in gl_todo:                                   # GL contexts are not thread-safe: run serially
            probe[f"g:{n}:{_md5(gl_src[n])}"] = _probe_gl(n, gl_src[n])
        _write_json_atomic(cache_path, cache)
    bases = []
    for n in builtin:
        if probe.get(f"b:{n}"):
            bases.append({"key": f"xfade:{n}", "kind": "xfade", "name": n, "expr": None, "families": classify_base_name(n)})
    for c in custom:
        if probe.get(f"c:{c['key']}:{_md5(c['expr'])}"):
            bases.append(c)
    n_gl = 0
    for n, src in gl_src.items():
        if probe.get(f"g:{n}:{_md5(src)}"):
            bases.append({"key": f"gl:{n}", "kind": "gl", "name": n, "expr": None,
                          "families": classify_base_name(n + " " + gl_shader_tags(src))})
            n_gl += 1
    easings_ok = {n: e for n, e in on_eas.items() if probe.get(f"e:{n}:{_md5(e)}")}
    n_online = 0
    if easings_ok:
        for n, e in on_trans.items():
            if probe.get(f"o:{n}:{_md5(e)}"):
                bases.append({"key": f"online:{n}", "kind": "online", "name": n, "expr": e,
                              "families": classify_base_name(n), "easings_ok": sorted(easings_ok)})
                n_online += 1
    posts = [p for p in posts_all if probe.get(f"p:{p}")] or ["none"]

    by_family = defaultdict(list)
    for b in bases:
        for f in b["families"]:
            by_family[f].append(b)
    fade = next((b for b in bases if b["key"] == "xfade:fade"), None)
    if fade:
        by_family["hard_cut"] = [fade]

    n_b = sum(1 for b in bases if b["kind"] == "xfade")
    n_c = sum(1 for b in bases if b["kind"] == "custom")
    log.info(f"Transition registry: {n_b} ffmpeg built-in + {n_c} procedural + {n_online} online "
              f"(x{len(easings_ok)} easings, source: {on_status}) + {n_gl} GL (gl-transitions) transition bases, "
              f"{len(posts)} post effects. "
              f"Fade-to-black banned.")
    _TRANSITION_REGISTRY = {"bases": bases, "by_family": dict(by_family), "posts": posts,
                            "easing_exprs": easings_ok, "online_status": on_status}
    return _TRANSITION_REGISTRY


# ---- recipes, sets and the ledger -------------------------------------------

def _dur_grid(family: str, density: int = 12):
    lo, hi = TRANSITION_FAMILY_DURATIONS.get(family, TRANSITION_FAMILY_DURATIONS["misc"])
    return sorted({round(lo + (hi - lo) * i / (density - 1), 3) for i in range(density)})


def _recipe_id(base_key: str, dur: float, post: str) -> str:
    return hashlib.sha1(f"{base_key}|{dur:.3f}|{post}".encode("utf-8")).hexdigest()[:10]


def _sample_family(registry, family, n, global_ids, set_ids, budget, rng, density):
    bases = registry["by_family"].get(family, [])
    if not bases or n <= 0:
        return []
    exempt = family in TRANSITION_REPEAT_EXEMPT_FAMILIES
    durs, posts = _dur_grid(family, density), registry["posts"]
    out, attempts = [], 0
    while len(out) < n and attempts < n * 60 + 300:
        attempts += 1
        base = rng.choice(bases)
        if base["kind"] in ("custom", "online") and budget[0] <= 0:
            continue
        if base["kind"] == "gl" and budget[1] <= 0:
            continue
        key, expr = base["key"], base.get("expr")
        if base["kind"] == "online":
            e = rng.choice(base["easings_ok"])
            key, expr = f"{base['key']}@{e}", registry["easing_exprs"][e] + ";" + base["expr"]
        dur, post = rng.choice(durs), rng.choice(posts)
        rid = _recipe_id(key, dur, post)
        if rid in set_ids or (not exempt and rid in global_ids):
            continue
        set_ids.add(rid)
        if not exempt:
            global_ids.add(rid)
        if base["kind"] in ("custom", "online"):
            budget[0] -= 1
        elif base["kind"] == "gl":
            budget[1] -= 1
        kind = "xfade" if base["kind"] == "xfade" else ("gl" if base["kind"] == "gl" else "custom")
        rec = {"id": rid, "base": key, "kind": kind,
               "name": base.get("name") if base["kind"] == "xfade" else None,
               "expr": expr if kind == "custom" else None,
               "dur": dur, "post": post, "family": family}
        if kind == "gl":
            rec["gl"] = base["name"]
        out.append(rec)
    return out


def generate_recipes(n, registry, fam_weights, global_ids, set_ids, budget, rng):
    """n NEW unique recipes split by fam_weights; if a family runs dry, spill into other families."""
    fams = [f for f in fam_weights if registry["by_family"].get(f)]
    if not fams:
        fams = [f for f in registry["by_family"] if f != "hard_cut"]
        fam_weights = {f: 1.0 for f in fams}
    tot = sum(fam_weights[f] for f in fams) or 1.0
    raw = {f: n * fam_weights[f] / tot for f in fams}
    quota = {f: int(raw[f]) for f in fams}
    for f in sorted(fams, key=lambda f: raw[f] - quota[f], reverse=True)[:n - sum(quota.values())]:
        quota[f] += 1

    recipes = []
    for f in fams:
        got = []
        for density in (12, 24, 48, 96):
            got += _sample_family(registry, f, quota[f] - len(got), global_ids, set_ids, budget, rng, density)
            if len(got) >= quota[f]:
                break
        recipes += got
    short = n - len(recipes)
    spill = 0
    if short > 0:
        others = [f for f in sorted(registry["by_family"], key=lambda f: -fam_weights.get(f, 0.0))
                  if f != "hard_cut" or fam_weights.get("hard_cut")]
        for density in (12, 24, 48, 96):
            for f in others:
                if short <= 0:
                    break
                got = _sample_family(registry, f, short, global_ids, set_ids, budget, rng, density)
                recipes += got
                short -= len(got)
                spill += len(got)
            if short <= 0:
                break
    if short > 0:
        log.warning(f"  Transition set is {short} recipe(s) short of the target (building blocks exhausted).")
    if spill:
        log.info(f"  {spill} recipe(s) came from spill-over families (the DNA's own families ran out of fresh combinations).")
    return recipes


def _ledger_load() -> dict:
    led = _read_json(_tl_path("transition_ledger.json"), None)
    if not isinstance(led, dict) or "sets" not in led:
        led = {"version": 1, "next_set_index": 1, "sets": {}, "all_recipe_ids": []}
    return led


def _ledger_save(led: dict):
    _write_json_atomic(_tl_path("transition_ledger.json"), led)


def _set_file(idx: int) -> str:
    return _tl_path(f"set_{idx:02d}.json")


def _new_set(led, registry, fam_weights, dna_key, size):
    idx = led["next_set_index"]
    global_ids = set(led["all_recipe_ids"])
    rng = random.Random(RANDOM_SEED * 100003 + idx * 7919 + int(_md5(dna_key), 16))
    budget = [int(size * TRANSITION_CUSTOM_MAX_SHARE) if TRANSITION_USE_CUSTOM else 0,
              int(size * TRANSITION_GL_MAX_SHARE) if TRANSITION_GL_ENABLED else 0]
    recipes = generate_recipes(size, registry, fam_weights, global_ids, set(), budget, rng)
    _write_json_atomic(_set_file(idx), {"index": idx, "dna_key": dna_key, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "ffmpeg": _ffmpeg_version(), "recipes": recipes})
    led["next_set_index"] = idx + 1
    led["all_recipe_ids"] = sorted(global_ids)
    led["sets"][str(idx)] = {"file": os.path.basename(_set_file(idx)), "dna_key": dna_key,
                             "size": len(recipes), "assigned_to": None, "used": []}
    _ledger_save(led)
    fam_counts = defaultdict(int)
    for r in recipes:
        fam_counts[r["family"]] += 1
    log.info(f"  Built transition set #{idx} ({dna_key}): {len(recipes)} unique recipes {dict(fam_counts)}.")
    return idx


class TransitionSet:
    """One edit's private pool of unique transition recipes + its usage log."""

    def __init__(self, led, idx, recipes, fam_weights, registry, edit_name, dna_key):
        self.led, self.idx, self.recipes = led, idx, recipes
        self.fam_weights, self.registry = fam_weights, registry
        self.edit_name, self.dna_key = edit_name, dna_key
        self.used_ids = list(led["sets"][str(idx)]["used"])
        self.order = []

    def _extend(self, n=50):
        rng = random.Random(RANDOM_SEED + self.idx * 31 + len(self.recipes))
        global_ids = set(self.led["all_recipe_ids"])
        set_ids = {r["id"] for r in self.recipes}
        used_custom = sum(1 for r in self.recipes if r["kind"] == "custom")
        used_gl = sum(1 for r in self.recipes if r["kind"] == "gl")
        budget = [max(0, int((len(self.recipes) + n) * TRANSITION_CUSTOM_MAX_SHARE) - used_custom) if TRANSITION_USE_CUSTOM else 0,
                  max(0, int((len(self.recipes) + n) * TRANSITION_GL_MAX_SHARE) - used_gl) if TRANSITION_GL_ENABLED else 0]
        new = generate_recipes(n, self.registry, self.fam_weights, global_ids, set_ids, budget, rng)
        self.recipes += new
        self.led["all_recipe_ids"] = sorted(global_ids)
        self.led["sets"][str(self.idx)]["size"] = len(self.recipes)
        sf = _read_json(_set_file(self.idx), {})
        sf["recipes"] = self.recipes
        _write_json_atomic(_set_file(self.idx), sf)
        log.info(f"  Transition set #{self.idx} extended by {len(new)} recipe(s) (more cuts than recipes — chain continues).")

    def pick(self, energy_label: str, cut_index: int):
        used = set(self.used_ids)
        unused = [r for r in self.recipes if r["id"] not in used]
        if not unused:
            self._extend()
            unused = [r for r in self.recipes if r["id"] not in used]
        by_fam = defaultdict(list)
        for r in unused:
            by_fam[r["family"]].append(r)
        fams = list(by_fam)
        fam = random.choices(fams, weights=[self.fam_weights.get(f, 0.0) + 0.001 for f in fams], k=1)[0]
        cands = sorted(by_fam[fam], key=lambda r: r["dur"])
        half = max(1, len(cands) // 2)
        pool = cands[:half] if energy_label == "high" else (cands[half:] if energy_label == "low" and len(cands) > 1 else cands)
        r = random.choice(pool)
        self.used_ids.append(r["id"])
        self.order.append(r["id"])
        self.led["sets"][str(self.idx)]["used"] = self.used_ids
        _ledger_save(self.led)
        try:
            with open(_tl_path("usage_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "edit": self.edit_name, "set": self.idx,
                                    "cut": cut_index, "id": r["id"], "base": r["base"], "dur": r["dur"],
                                    "post": r["post"], "family": r["family"]}) + "\n")
        except OSError:
            pass
        return r

    def summary(self) -> dict:
        by_id = {r["id"]: r for r in self.recipes}
        fam = defaultdict(int)
        for i in self.used_ids:
            if i in by_id:
                fam[by_id[i]["family"]] += 1
        return {"set_index": self.idx, "dna": self.dna_key, "recipes_in_set": len(self.recipes),
                "recipes_used": len(self.used_ids), "used_by_family": dict(fam)}

    def export_used(self, path: str):
        by_id = {r["id"]: r for r in self.recipes}
        _write_json_atomic(path, {"edit": self.edit_name, "set_index": self.idx, "dna": self.dna_key,
                                  "transitions_in_order": [by_id[i] for i in self.used_ids if i in by_id]})


def allocate_transition_set(edit_name: str, dna_profile, registry, fresh: bool = True):
    """Gives this edit its own set: the one it already owns (re-run), else a pre-built unassigned
    set for the same DNA, else a brand-new set. `fresh` clears that set's usage list for a clean re-render."""
    led = _ledger_load()
    dna_key = "+".join(dna_profile["names"]) if dna_profile else "none"
    fam_weights = dna_family_weights(dna_profile["dna"] if dna_profile else None)

    idx = next((int(k) for k, m in led["sets"].items() if (m.get("assigned_to") or {}).get("edit") == edit_name), None)
    if idx is None:
        idx = next((int(k) for k, m in sorted(led["sets"].items(), key=lambda kv: int(kv[0]))
                    if m.get("assigned_to") is None and m.get("dna_key") == dna_key
                    and os.path.isfile(_set_file(int(k)))), None)
    if idx is None:
        idx = _new_set(led, registry, fam_weights, dna_key, TRANSITION_SET_SIZE)
    meta = led["sets"][str(idx)]
    meta["assigned_to"] = {"edit": edit_name, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if fresh:
        meta["used"] = []
    _ledger_save(led)
    recipes = _read_json(_set_file(idx), {}).get("recipes", [])
    planned = f" (cycle {idx}/{TRANSITION_SETS_PLANNED})" if idx <= TRANSITION_SETS_PLANNED else f" (chained set beyond the {TRANSITION_SETS_PLANNED} planned)"
    log.info(f"  Transition set #{idx}{planned} assigned to this edit: {len(recipes)} unique recipes, DNA '{dna_key}'.")
    return TransitionSet(led, idx, recipes, fam_weights, registry, edit_name, dna_key)


def prebuild_transition_sets(n: int, dna_profile, registry):
    led = _ledger_load()
    dna_key = "+".join(dna_profile["names"]) if dna_profile else "none"
    fam_weights = dna_family_weights(dna_profile["dna"] if dna_profile else None)
    for _ in range(n):
        _new_set(led, registry, fam_weights, dna_key, TRANSITION_SET_SIZE)


def print_transition_report():
    led = _ledger_load()
    log.info("")
    log.info(f"Transition ledger: {len(led['sets'])} set(s) built ({TRANSITION_SETS_PLANNED} planned), "
              f"{len(led['all_recipe_ids'])} unique recipes issued in total.")
    for k in sorted(led["sets"], key=int):
        m = led["sets"][k]
        who = (m.get("assigned_to") or {}).get("edit", "unassigned")
        log.info(f"  set {int(k):>3}  dna={m.get('dna_key'):<16} size={m.get('size'):>4}  used={len(m.get('used', [])):>4}  -> {who}")


def handle_transition_cli(args) -> bool:
    if not (args.build_transition_sets or args.transition_report or args.online_refresh):
        return False
    if args.build_transition_sets or args.online_refresh:
        registry = get_transition_registry(force_online=args.online_refresh)
    if args.build_transition_sets:
        profile = load_dna_profile(dna_runs_for_track("")[0])
        prebuild_transition_sets(args.build_transition_sets, profile, registry)
    print_transition_report()
    return True


# --------------------------------------------------------------------------
# Chained crossfade merge (resumable per-merge)
# --------------------------------------------------------------------------

_CRASHED_BASES = None


def _crash_file() -> str:
    return _tl_path("crashed_bases.json")


def crashed_bases() -> set:
    """Transition bases that crashed YOUR ffmpeg build (e.g. 0xC0000005 access violation); never picked again."""
    global _CRASHED_BASES
    if _CRASHED_BASES is None:
        _CRASHED_BASES = set(_read_json(_crash_file(), []))
    return _CRASHED_BASES


def _record_crash(base: str):
    if base and base not in crashed_bases():
        crashed_bases().add(base)
        try:
            _write_json_atomic(_crash_file(), sorted(crashed_bases()))
        except OSError:
            pass
        log.warning(f"  Transition '{base}' crashed ffmpeg - it is now excluded from future picks "
                     f"({_crash_file()}).")


SAFE_FALLBACK_TRANSITIONS = ["fade", "dissolve", "wipeleft", "slideleft"]


def merge_with_transition(running_path: str, next_path: str, out_path: str,
                           running_duration: float, transition_name: str,
                           transition_duration: float, audio_curve: str, recipe: dict = None, offset: float = None):
    """Transition merge with a crash-proof fallback ladder. Timing (duration/offset) never changes, so the
    beat lock holds whichever step succeeds:
      xfade/expression recipes: 1 as planned -> 2 single-threaded filters -> 3 without the post effect -> 4 plain safe transition
      GL (gl-transitions) recipes: 1 GL shader -> 2 without the post effect -> 3 plain safe transition."""
    if offset is None:
        offset = max(0.0, running_duration - transition_duration)
    is_gl = recipe is not None and recipe.get("kind") == "gl"
    base_spec, post_id = transition_name, None
    if recipe is not None:
        post_id = recipe["post"]
        if not is_gl:
            base_spec = f"custom:expr='{recipe['expr']}'" if recipe["kind"] != "xfade" else recipe["name"]

    def run_xfade(spec, post_on, single_thread):
        post = build_post_filter(post_id, offset, transition_duration) if (post_on and post_id) else None
        vout = "[xv]" if post else "[v]"
        chain = f"[0:v][1:v]xfade=transition={spec}:duration={transition_duration:.3f}:offset={offset:.3f}{vout}"
        if post:
            chain += f";[xv]{post}[v]"
        fc = (f"{chain};[0:a][1:a]acrossfade=d={transition_duration:.3f}:c1={audio_curve}:c2={audio_curve}[a]")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
        if single_thread:
            cmd += ["-filter_threads", "1", "-filter_complex_threads", "1"]
        cmd += ["-i", running_path, "-i", next_path, "-filter_complex", fc,
                "-map", "[v]", "-map", "[a]", "-pix_fmt", "yuv420p",
                "-c:v", VIDEO_CODEC, "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET, "-c:a", "aac", out_path]
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def run_gl(post_on):
        try:
            return _merge_gl(running_path, next_path, out_path, offset, transition_duration, audio_curve,
                              recipe["gl"], post_id if post_on else None)
        except Exception as e:
            return subprocess.CompletedProcess([], 1, "", f"GL transition '{recipe['gl']}' failed: {e}")

    ladder = []   # (description, runner, is_base_step)
    if is_gl:
        ladder.append((f"GL '{recipe['gl']}'", lambda: run_gl(True), True))
        if post_id:
            ladder.append((f"GL '{recipe['gl']}' without post effect", lambda: run_gl(False), True))
    else:
        ladder.append((f"'{transition_name}'", lambda: run_xfade(base_spec, True, False), True))
        ladder.append((f"'{transition_name}' single-threaded", lambda: run_xfade(base_spec, True, True), True))
        if post_id:
            ladder.append((f"'{transition_name}' without post effect", lambda: run_xfade(base_spec, False, True), True))
    for f in SAFE_FALLBACK_TRANSITIONS:
        ladder.append((f"safe '{f}'", (lambda f=f: run_xfade(f, False, True)), False))

    last = None
    for step, (desc, runner, is_base) in enumerate(ladder):
        last = runner()
        if last.returncode != 0:
            continue
        if step > 0:
            log.warning(f"  Merge recovered on fallback step {step + 1}/{len(ladder)} ({desc}).")
        if not is_base and recipe is not None:
            _record_crash(recipe["base"])        # every base variant failed: never pick this transition again
        return
    _record_crash(recipe["base"] if recipe else transition_name)
    raise RuntimeError(
        f"ffmpeg exit {last.returncode} merging with transition '{transition_name}' "
        f"(dur {transition_duration:.2f}s, offset {offset:.2f}s, post={post_id}) - every fallback failed: "
        f"{last.stderr[-1500:].strip() or 'no error text - ffmpeg was killed or crashed'}")


def normalize_cut_timeline(cuts, total: float):
    """Guarantees the cuts tile [0, total] exactly: contiguous, no gaps/overlaps, last cut ends at the
    end of the music (DNA snapping, arc windows or rounding used to leave the tail uncovered)."""
    cuts = sorted([c for c in cuts if c is not None], key=lambda c: c["start"])
    out = []
    for c in cuts:
        if c["start"] >= total - 0.05:
            continue
        if out and c["start"] - out[-1]["start"] < 0.15:
            continue
        out.append(c)
    if not out:
        out = [{"start": 0.0, "end": total, "energy": "low"}]
    out[0]["start"] = 0.0
    for a, b in zip(out, out[1:]):
        a["end"] = b["start"]
    out[-1]["end"] = total
    if len(out) > 1 and out[-1]["end"] - out[-1]["start"] < 0.3:
        out.pop()
        out[-1]["end"] = total
    return _split_long_cuts(out)


def plan_transitions(cut_infos, transition_set, dna_d):
    """Decides every transition BEFORE cutting, so each segment can be cut with the exact handle
    (half the transition on each side, centred on the beat). Then the crossfades eat no timeline."""
    plan, recent = [None], []
    for i in range(1, len(cut_infos)):
        energy = cut_infos[i]["energy"]
        recipe = None
        if transition_set is not None:
            recipe = transition_set.pick(energy, i)
            for _ in range(8):
                if (recipe["name"] or recipe["base"]) not in crashed_bases() and recipe["base"] not in crashed_bases():
                    break
                recipe = transition_set.pick(energy, i)
            name, dur = recipe["name"] or recipe["base"], recipe["dur"]
        else:
            name, dur = choose_transition(energy, recent, dna_d)
            recent.append(name)
            if len(recent) > AVOID_REPEAT_WINDOW_TRANSITION:
                recent.pop(0)
        c_prev = cut_infos[i - 1]["end"] - cut_infos[i - 1]["start"]
        c_cur = cut_infos[i]["end"] - cut_infos[i]["start"]
        dur = max(0.05, min(float(dur), 0.45 * min(c_prev, c_cur)))
        plan.append({"name": name, "dur": dur, "recipe": recipe, "curve": random.choice(AUDIO_CROSSFADE_CURVES)})
    return plan


def _pad_segment(path: str, target: float):
    """Last resort for one segment whose source was too short: hold its final frame."""
    cur = get_video_duration_sec(path)
    if target - cur <= 0.03:
        return
    tmp = path + ".pad.mp4"
    cmd = ["ffmpeg", "-y", "-i", path, "-vf", f"tpad=stop_mode=clone:stop_duration={target - cur:.3f}",
           "-af", f"apad=pad_dur={target - cur:.3f}", "-t", f"{target:.3f}",
           "-c:v", VIDEO_CODEC, "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET, "-c:a", "aac", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0:
        os.replace(tmp, path)


def extend_edit_to_duration(video_path: str, current: float, target: float, shots, work_dir: str):
    """Instead of freezing the last frame, cover any remaining shortfall with real extra footage."""
    overlap = 0.10
    for n in range(6):
        shortfall = target - current
        if shortfall <= 0.05:
            break
        seg_len = min(shortfall + overlap + 0.1, 3.0)
        idx, src, cs = choose_shot_for_cut(shots, seg_len, [], "medium")
        seg = os.path.join(work_dir, f"fill_{n}.mp4")
        merged = os.path.join(work_dir, f"fill_merged_{n}.mp4")
        try:
            cut_segment(src, cs, seg_len, seg, None)
            seg_len = get_video_duration_sec(seg)
            merge_with_transition(video_path, seg, merged, current, "fade", overlap,
                                   "tri", None, offset=max(0.0, current - overlap))
        except Exception as e:
            log.warning(f"  Gap-fill step failed ({e}).")
            break
        video_path, current = merged, max(0.0, current - overlap) + seg_len
        log.info(f"  Gap-fill: covered the tail with {seg_len:.2f}s of extra footage (now {current:.2f}s).")
    return video_path, current


def build_transitioned_edit(segment_paths, cut_infos, work_dir: str,
                             checkpoint_state=None, save_state_fn=None, dna=None, transition_set=None,
                             plan=None):
    """Joins the segments with their planned transitions using a BALANCED PAIRWISE TREE instead of one
    ever-growing file: neighbours are merged in pairs, then pairs of pairs, and so on. Every merge only
    re-encodes the two pieces it joins, so the total work is ~n*log2(n) segment-lengths instead of n^2/2,
    files stay small, and each clip is re-encoded ~log2(n) times instead of up to n times (better quality).
    Beat lock is kept with absolute timeline times: each transition is centred on its cut's beat."""
    n = len(segment_paths)
    if plan is None:
        plan = plan_transitions(cut_infos, transition_set, dna)
    tstate = (checkpoint_state or {}).get("transitions") or {}
    tree = tstate.get("tree")
    choices = {int(k): v for k, v in (tstate.get("choices") or {}).items()}
    if tree and any(not os.path.isfile(it["path"]) for it in tree["items"] + tree["out"]):
        log.warning("  A checkpointed merge file is missing - restarting the transitions stage.")
        tree, choices = None, {}
    if not tree:
        items = []
        for i, pth in enumerate(segment_paths):
            head = plan[i]["dur"] / 2 if (i > 0 and plan[i]) else 0.0
            items.append({"path": pth, "abs_start": cut_infos[i]["start"] - head,
                          "dur": get_video_duration_sec(pth), "first": i})
        tree = {"level": 0, "items": items, "out": [], "pos": 0}

    def persist():
        if save_state_fn is not None:
            save_state_fn({"tree": tree, "choices": {str(k): v for k, v in choices.items()}})

    pieces = len(tree["out"]) + len(tree["items"]) - tree["pos"]
    with tqdm(total=max(n - 1, 1), initial=n - pieces, desc="Building transitions", unit="merge", ncols=80) as pbar:
        while len(tree["items"]) > 1:
            items, out, pos, level = tree["items"], tree["out"], tree["pos"], tree["level"]
            while pos < len(items):
                if pos + 1 >= len(items):
                    out.append(items[pos])
                    pos += 1
                    tree["pos"] = pos
                    continue
                R, N = items[pos], items[pos + 1]
                idx = N["first"]
                pl = plan[idx]
                d = max(0.05, min(pl["dur"], R["dur"] * 0.4, N["dur"] * 0.4))
                target = cut_infos[idx]["start"] - d / 2 - R["abs_start"]      # beat-locked, local to R
                offset = max(0.0, min(target, R["dur"] - d))
                merged_path = os.path.join(work_dir, f"merged_L{level}_{pos // 2:04}.mp4")
                merge_with_transition(R["path"], N["path"], merged_path, R["dur"], pl["name"], d,
                                       pl["curve"], pl["recipe"], offset=offset)
                out.append({"path": merged_path, "abs_start": R["abs_start"],
                            "dur": get_video_duration_sec(merged_path), "first": R["first"]})
                choices[idx] = pl["recipe"]["id"] if pl["recipe"] else pl["name"]
                pos += 2
                tree["pos"] = pos
                persist()                                   # save BEFORE deleting what was consumed
                for it in (R, N):
                    if os.path.basename(it["path"]).startswith("merged_L") and it["path"] != merged_path:
                        try:
                            os.remove(it["path"])
                        except OSError:
                            pass
                pbar.update(1)
            tree = {"level": level + 1, "items": out, "out": [], "pos": 0}
            persist()

    final = tree["items"][0]
    transition_choices = [None] + [choices.get(i) for i in range(1, n)]
    segment_windows = [(cut_infos[i]["start"], cut_infos[i]["end"]) for i in range(n)]
    return final["path"], final["dur"], segment_windows, transition_choices


# --------------------------------------------------------------------------
# Final audio: plain music, trimmed/padded — NO ducking, NO VO (removed)
# --------------------------------------------------------------------------

def build_plain_music_audio(music_path: str, total_duration: float, out_path: str, start_offset: float = 0.0):
    cmd = ["ffmpeg", "-y"]
    if start_offset > 0:
        cmd += ["-ss", f"{start_offset:.3f}"]
    cmd += ["-i", music_path, "-t", f"{total_duration:.3f}"]
    if is_trimmed() and TRIM_FADE_OUT_SEC > 0 and total_duration > TRIM_FADE_OUT_SEC * 2:
        cmd += ["-af", f"afade=t=out:st={total_duration - TRIM_FADE_OUT_SEC:.3f}:d={TRIM_FADE_OUT_SEC:.3f}"]
    cmd += ["-c:a", "aac", "-ac", "2", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Music audio extraction failed: {result.stderr[-800:]}")


def mux_video_with_audio(video_path: str, audio_path: str, output_path: str, duration: float = None):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Final mux failed: {result.stderr[-500:]}")


def pad_video_to_duration(video_path: str, target_duration: float, out_path: str) -> tuple:
    current_duration = get_video_duration_sec(video_path)
    shortfall = target_duration - current_duration
    if shortfall <= 0.05:
        return video_path, current_duration

    if shortfall > 2.0:
        log.warning(f"  Edited video is {shortfall:.1f}s shorter than the music track — padding with a "
                     f"frozen last frame so the full song still plays. If this keeps happening, your "
                     f"source footage may be too short/limited for MAX_CUT_DURATION_SEC — consider adding "
                     f"more footage or lowering MAX_CUT_DURATION_SEC.")
    else:
        log.info(f"  Padding edited video by {shortfall:.1f}s (frozen last frame) to match the music's full length.")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"tpad=stop_mode=clone:stop_duration={shortfall:.3f}",
        "-af", f"apad=pad_dur={shortfall:.3f}",
        "-c:v", VIDEO_CODEC, "-crf", VIDEO_CRF, "-preset", VIDEO_PRESET,
        "-c:a", "aac",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(f"  Padding failed ({result.stderr[-500:]}) — the final video will stay short of the full song.")
        return video_path, current_duration
    return out_path, target_duration


# --------------------------------------------------------------------------
# Per-track deterministic-but-distinct seeding
# --------------------------------------------------------------------------

def seed_for_track(music_path: str) -> int:
    if not PER_TRACK_SEEDING:
        return RANDOM_SEED
    name = os.path.basename(music_path)
    digest = 0
    for ch in name:
        digest = (digest * 131 + ord(ch)) % 1_000_003
    return RANDOM_SEED + digest


def _safe_name(music_path: str, dna_names=None) -> str:
    track_name = os.path.splitext(os.path.basename(music_path))[0]
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", track_name)
    # Different lengths/start points must NOT share checkpoints or segment folders.
    if TARGET_DURATION_SEC:
        name += f"_{TARGET_DURATION_SEC:g}s"
    if START_OFFSET_SEC > 0:
        name += f"_from{START_OFFSET_SEC:g}s"
    if dna_names is None:
        dna_names = dna_names_for_track(music_path)
    if dna_names:
        name += "_dna-" + "+".join(re.sub(r"[^A-Za-z0-9_.\-]+", "_", n) for n in dna_names)
    return name


def _checkpoint_dir_for(music_path: str, dna_names=None) -> str:
    d = os.path.join(CHECKPOINT_FOLDER, _safe_name(music_path, dna_names))
    os.makedirs(d, exist_ok=True)
    return d


def _load_state(work_dir: str) -> dict:
    state_path = os.path.join(work_dir, "state.json")
    if RESUME_FROM_CHECKPOINT and os.path.isfile(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("  Checkpoint state.json is corrupt — starting this track fresh.")
    return {}


def _save_state(work_dir: str, state: dict):
    state_path = os.path.join(work_dir, "state.json")
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


# --------------------------------------------------------------------------
# Per-track pipeline
# --------------------------------------------------------------------------

def process_track(music_path, shots, vfx_library, transcripts, dna_names=None):
    if dna_names is None:
        dna_names = dna_names_for_track(music_path)
    safe_name = _safe_name(music_path, dna_names)
    output_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}_edit.mp4") if PROCESS_ALL_TRACKS else OUTPUT_PATH_SINGLE
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    log.info("")
    log.info("=" * 60)
    log.info(f"Processing track: {os.path.basename(music_path)}")
    log.info("=" * 60)

    track_seed = seed_for_track(music_path)
    random.seed(track_seed)
    log.info(f"  RNG seed for this track: {track_seed}")

    t0 = time.time()
    result = {"track": os.path.basename(music_path), "dna_run": "+".join(dna_names) or "none", "output": None, "status": "failed", "error": None}

    work_dir = _checkpoint_dir_for(music_path, dna_names)
    state = _load_state(work_dir)
    if state and state.get("plan_version") != 2:
        log.info("  Old-format checkpoint found (cuts were planned without transition handles) - starting this track fresh.")
        state = {}
    if state:
        log.info(f"  Found existing checkpoint (stage: {state.get('stage', 'unknown')}) — resuming.")

    seg_dir = os.path.join("video_cutaways", safe_name)
    os.makedirs(seg_dir, exist_ok=True)

    try:
        # ---------------- Step 1: analyze the track ----------------
        beat_times, beat_labels, music_duration, tempo, rms, rms_times, onset_times = analyze_music(music_path)
        beat_times, beat_labels, music_duration, rms, rms_times, onset_times, window_start = apply_duration_window(
            beat_times, beat_labels, music_duration, rms, rms_times, onset_times)

        # ---- Editing DNA for this track ----
        dna_profile = load_dna_profile(dna_names)
        dna_d = dna_profile["dna"] if dna_profile else None
        ACTIVE_BEATS_PER_CUT.update(derive_beats_per_cut(dna_d, tempo))
        low_motion_target = dna_low_motion_target(dna_d)
        grade_state = {}
        result["dna_modules"] = dna_profile["names"] if dna_profile else []
        if dna_profile:
            log.info(f"  Editing DNA: {'+'.join(dna_profile['names'])} | beats/cut {ACTIVE_BEATS_PER_CUT} "
                      f"| influence {DNA_INFLUENCE}")
        else:
            log.info("  Editing DNA: none (editor defaults).")
        if len(beat_times) < 2:
            log.warning(f"  Not enough beats detected in {music_path}, skipping.")
            result["status"] = "skipped_no_beats"
            return result

        result["tempo_bpm"] = round(tempo, 1)
        result["duration_sec"] = round(music_duration, 1)

        # ---------------- Step 2: load story-arc script, build the timeline ----------------
        arc_windows = None
        arc_script_path = find_arc_script(music_path)
        if arc_script_path:
            sections = parse_story_arc_script(arc_script_path)
            if sections:
                arc_windows = build_story_arc_windows(sections, music_duration, rms, rms_times)
                n_cycles = max(s["cycle"] for s in sections)
                log.info(f"  Story-arc script loaded: {os.path.basename(arc_script_path)} "
                          f"({n_cycles} cycle(s), {len(arc_windows)} stage window(s)).")
            else:
                log.warning(f"  Story-arc script found ({os.path.basename(arc_script_path)}) but had no "
                             f"recognizable stage headers (Hook/Rising Action/Conflict/Comeback/Payoff) — "
                             f"falling back to plain beat-driven cutting for this track.")
        else:
            log.info(f"  No story-arc script found in {REMIX_FOLDER}/ for this track "
                      f"(expected {os.path.splitext(os.path.basename(music_path))[0]}.txt) — falling back to plain beat-driven cutting; "
                      f"the built-in ducking planner will still run.")

        if arc_windows:
            cut_infos = build_cut_timeline_from_arc(arc_windows, beat_times, beat_labels)
        else:
            cut_infos = build_cut_timeline(beat_times, beat_labels, music_duration)
        if dna_d:
            cut_infos = apply_dna_sync(cut_infos, onset_times, dna_d)
        cut_infos = normalize_cut_timeline(cut_infos, music_duration)
        result["cut_planning_mode"] = "story_arc" if arc_windows else "beat_only"
        log.info(f"  Planned {len(cut_infos)} cut(s) for a {music_duration:.1f}s edit ({result['cut_planning_mode']}).")
        result["planned_cuts"] = len(cut_infos)

        # ---------------- Step 2b: plan transitions up front (so cuts get exact handles) ----------------
        transition_set, plan = None, None
        use_trans = TRANSITIONS_ENABLED and len(cut_infos) > 1
        if use_trans:
            old_plan = state.get("transition_plan")
            if old_plan and len(old_plan) != len(cut_infos):
                old_plan = None
            if TRANSITION_SETS_ENABLED:
                try:
                    transition_set = allocate_transition_set(safe_name, dna_profile, get_transition_registry(),
                                                              fresh=old_plan is None)
                except Exception as e:
                    log.warning(f"  Transition engine unavailable ({e}) — using the legacy transition lists "
                                 f"(fade-to-black still excluded).")
                    transition_set = None
            plan = old_plan or plan_transitions(cut_infos, transition_set, dna_d)
            state["transition_plan"] = plan
        state["plan_version"] = 2
        _save_state(work_dir, state)

        # ---------------- Step 3: cut video segments (resumable per-segment) ----------------
        segment_paths, cut_meta = [], []
        cached_segments = {s["index"]: s for s in state.get("segments", [])}
        recent_shots, recent_vfx = [], []

        with tqdm(total=len(cut_infos), desc="Cutting shots to the beat", unit="cut", ncols=80,
                  initial=len(cached_segments)) as pbar:
            for i, cut in enumerate(cut_infos):
                if i in cached_segments:
                    meta = cached_segments[i]
                    if os.path.isfile(meta["path"]):
                        segment_paths.append(meta["path"])
                        cut_meta.append(meta)
                        continue

                base_len = max(0.15, cut["end"] - cut["start"])
                last = i + 1 >= len(cut_infos)
                head = plan[i]["dur"] / 2 if (plan and i > 0) else 0.0
                tail = plan[i + 1]["dur"] / 2 if (plan and not last) else 0.0
                margin = (1.5 / TARGET_FPS) if (plan and not last) else 0.0
                duration = base_len + head + tail + margin     # exact handles -> crossfades cost no time

                seg_path = os.path.join(seg_dir, f"seg_{i:04}.mp4")
                tried, ok_cut = [], False
                for attempt in range(4):
                    shot_idx, shot_source, clip_start = choose_shot_for_cut(
                        shots, duration, recent_shots + tried, cut["energy"],
                        stage_text=cut.get("stage_text"), transcripts=transcripts,
                        low_motion_target=low_motion_target,
                    )
                    tried.append(shot_idx)
                    vfx_filter = None
                    if VFX_ENABLED and attempt < 2:
                        if dna_d:
                            vfx_filter, vfx_name = choose_vfx_with_dna(
                                vfx_library, cut["energy"], recent_vfx, dna_d, grade_state, cut["start"])
                        else:
                            vfx = choose_vfx(vfx_library, cut["energy"], recent_vfx)
                            vfx_filter, vfx_name = vfx["filter"], vfx["name"]
                    try:
                        cut_segment(shot_source, clip_start, duration, seg_path, vfx_filter)
                        got = get_video_duration_sec(seg_path)
                    except Exception as e:
                        tqdm.write(f"  Cut {i:04} attempt {attempt + 1} failed: {str(e)[-200:]}")
                        continue
                    if got >= duration - 0.08 or attempt == 3:
                        if got < duration - 0.08:
                            _pad_segment(seg_path, duration)      # only if every source was too short
                        ok_cut = True
                        break
                if not ok_cut:
                    raise RuntimeError(f"Cut {i:04} could not be produced from any shot.")
                recent_shots.append(shot_idx)
                if len(recent_shots) > AVOID_REPEAT_WINDOW:
                    recent_shots.pop(0)
                if VFX_ENABLED and vfx_filter:
                    recent_vfx.append(vfx_name)
                    if len(recent_vfx) > AVOID_REPEAT_WINDOW_VFX:
                        recent_vfx.pop(0)
                try:
                    meta = {
                        "index": i, "path": seg_path, "source": shot_source, "clip_start": clip_start,
                        "duration": duration, "energy": cut["energy"], "stage": cut.get("stage"),
                        "has_face": shots[shot_idx].get("has_face", False),
                        "motion_norm": shots[shot_idx].get("motion_score_norm", 0.0),
                    }
                    segment_paths.append(seg_path)
                    cut_meta.append(meta)
                    state["stage"] = "cutting"
                    state["segments"] = cut_meta
                    state["cut_infos"] = cut_infos
                    _save_state(work_dir, state)
                except Exception as e:
                    tqdm.write(f"  Cut {i:04} failed, skipping: {e}")
                    cut_infos[i] = None

                pbar.update(1)

        kept_indices = {m["index"] for m in cut_meta}
        cut_infos = [c for idx, c in enumerate(cut_infos) if c is not None and idx in kept_indices]
        cut_meta.sort(key=lambda m: m["index"])
        segment_paths = [m["path"] for m in cut_meta]

        if not segment_paths:
            log.warning("  No segments were successfully cut for this track. Skipping.")
            result["status"] = "skipped_no_segments"
            return result

        # ---------------- Step 4: transitions (resumable per-merge) ----------------
        if TRANSITIONS_ENABLED and len(segment_paths) > 1 and plan is not None and len(plan) == len(segment_paths):
            log.info("  Building seamless transitions between shots...")

            def save_transition_state(tstate):
                state["stage"] = "transitions"
                state["transitions"] = tstate
                _save_state(work_dir, state)

            resume_state = state if state.get("stage") == "transitions" and (state.get("transitions") or {}).get(
                "tree") else None

            edited_path, total_duration, segment_windows, transition_choices = build_transitioned_edit(
                segment_paths, cut_infos, work_dir,
                checkpoint_state=resume_state, save_state_fn=save_transition_state, dna=dna_d,
                transition_set=transition_set, plan=plan)
            if transition_set is not None:
                transition_set.export_used(os.path.join(OUTPUT_FOLDER, f"{safe_name}_transitions.json"))
                log.info(f"  Transitions used: {transition_set.summary()}")
        else:
            list_path = os.path.join(work_dir, "concat_list.txt")
            with open(list_path, "w") as f:
                for p in segment_paths:
                    f.write(f"file '{os.path.abspath(p)}'\n")
            edited_path = os.path.join(work_dir, "concat_video.mp4")
            subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", edited_path],
                            capture_output=True, text=True)
            total_duration = sum(get_video_duration_sec(p) for p in segment_paths)
            cum = 0.0
            segment_windows, transition_choices = [], [None] * len(segment_paths)
            for p in segment_paths:
                d = get_video_duration_sec(p)
                segment_windows.append((cum, cum + d))
                cum += d

        state["stage"] = "mixing"
        _save_state(work_dir, state)

        log.info(f"  Assembled {total_duration:.2f}s of video for {music_duration:.2f}s of music.")
        if music_duration - total_duration > 0.05:
            edited_path, total_duration = extend_edit_to_duration(edited_path, total_duration, music_duration,
                                                                   shots, work_dir)
        padded_path = os.path.join(work_dir, "padded_edit.mp4")
        edited_path, total_duration = pad_video_to_duration(edited_path, music_duration, padded_path)  # tiny remainder only
        total_duration = min(total_duration, music_duration)
        result["video_vs_music_sec"] = round(total_duration - music_duration, 3)

        # ---------------- Step 5: ducking (script optional) + final mix ----------------
        duck = None
        try:
            duck = run_ducking_pipeline(music_path, safe_name, work_dir, window_start, music_duration,
                                        arc_windows, beat_times, tempo, edited_video_path=edited_path)
        except Exception as e:
            log.warning(f"  Ducking failed ({e}) — continuing with the plain music track.")
            log.debug(traceback.format_exc())

        music_audio_path = os.path.join(work_dir, "music_audio.aac")
        audio_mode = "plain"
        if duck and FINAL_AUDIO_MODE in ("mix", "ducked", "blend"):
            if FINAL_AUDIO_MODE == "mix" and duck.get("mix_path"):
                src, audio_mode = duck["mix_path"], "mix"
            elif FINAL_AUDIO_MODE == "blend" and duck.get("blend_path"):
                src, audio_mode = duck["blend_path"], "blend"
            else:
                src, audio_mode = duck["ducked_path"], "ducked"
            build_audio_from_file(src, total_duration, music_audio_path)
        else:
            build_plain_music_audio(music_path, total_duration, music_audio_path, start_offset=window_start)
        log.info(f"  Muxing final video with {audio_mode} audio...")
        mux_video_with_audio(edited_path, music_audio_path, output_path, duration=music_duration)
        ducking_placed = bool(duck)

        state["stage"] = "done"
        _save_state(work_dir, state)

        result.update({
            "status": "ok",
            "output": os.path.abspath(output_path),
            "final_cuts": len(segment_paths),
            "transition_set": transition_set.summary() if transition_set else None,
            "ducking_track_exported": ducking_placed,
            "ducking": duck,
            "final_audio": audio_mode,
            "seconds_elapsed": round(time.time() - t0, 1),
        })

        log.info(f"  Done: {output_path}")
        log.info(f"    Ducking track exported: {ducking_placed}, {result.get('seconds_elapsed', 0):.1f}s elapsed.")

        if CLEAN_CHECKPOINT_ON_SUCCESS:
            try:
                shutil.rmtree(work_dir)
            except OSError as e:
                log.warning(f"  Could not clean up checkpoint folder {work_dir}: {e}")

        return result

    except Exception as e:
        log.error(f"  Track failed: {e}")
        log.debug(traceback.format_exc())
        log.error(f"  Progress so far is saved in: {work_dir} — rerun the script to resume this track.")
        result["status"] = "failed"
        result["error"] = str(e)
        return result


# --------------------------------------------------------------------------
# Main: batch driver
# --------------------------------------------------------------------------

def parse_cli():
    global TARGET_DURATION_SEC, START_OFFSET_SEC, DNA_MODULES, DNA_JOIN_MODE
    import argparse
    ap = argparse.ArgumentParser(description="Beat-synced auto editor")
    ap.add_argument("--duration", type=float, default=None,
                    help="edit length in seconds (overrides TARGET_DURATION_SEC; 0 = full track)")
    ap.add_argument("--start", type=float, default=None,
                    help="start point in the track, seconds (overrides START_OFFSET_SEC)")
    ap.add_argument("--dna", type=str, default=None,
                    help='DNA module name(s), comma-separated (overrides DNA_MODULES; "" = none)')
    ap.add_argument("--dna-join", type=str, default=None,
                    help='join variant(s): full_outer | inner | left | right | all, or comma list '
                         '(one edit per variant; overrides DNA_JOIN_MODE)')
    ap.add_argument("--build-transition-sets", type=int, default=0, metavar="N",
                    help="pre-build N unique transition sets for the selected DNA, then exit")
    ap.add_argument("--transition-report", action="store_true", help="print the transition ledger, then exit")
    ap.add_argument("--online-refresh", action="store_true",
                    help="re-download the online transition/easing library now, then exit")
    args = ap.parse_args()
    if args.dna is not None:
        DNA_MODULES = [n.strip() for n in args.dna.split(",") if n.strip()]
    if args.dna_join:
        parts = [x.strip() for x in args.dna_join.split(",") if x.strip()]
        bad = [x for x in parts if x != "all" and x not in JOIN_VARIANTS]
        if bad:
            ap.error(f"--dna-join: unknown {bad}; use {JOIN_VARIANTS + ['all']}")
        DNA_JOIN_MODE = "all" if "all" in parts else (parts[0] if len(parts) == 1 else parts)
    if args.duration is not None:
        TARGET_DURATION_SEC = args.duration if args.duration > 0 else None
    if args.start is not None:
        START_OFFSET_SEC = max(0.0, args.start)
    return args


def main():
    args = parse_cli()
    setup_logging()
    if is_trimmed():
        log.info(f"Edit length: {TARGET_DURATION_SEC or 'rest of track'}s, starting at {START_OFFSET_SEC}s.")
    run_preflight_checks()
    if handle_transition_cli(args):
        return
    random.seed(RANDOM_SEED)

    if not os.path.isdir(SOURCE_VIDEO_FOLDER):
        log.error(f"ERROR: Source footage folder not found: {SOURCE_VIDEO_FOLDER}")
        sys.exit(1)
    if not os.path.isdir(MUSIC_FOLDER):
        log.error(f"ERROR: Music folder not found: {MUSIC_FOLDER}")
        sys.exit(1)
    if not os.path.isdir(REMIX_FOLDER):
        log.warning(f"  Remix folder not found: {REMIX_FOLDER} — every track will use plain beat-driven "
                     f"cutting with no story-arc guidance (ducking still runs on its own). Create it and add "
                     f"<track_name>.txt files to enable the story-arc system.")

    source_videos = find_source_videos(SOURCE_VIDEO_FOLDER)
    if not source_videos:
        log.error(f"ERROR: No video files found in: {SOURCE_VIDEO_FOLDER}")
        sys.exit(1)
    log.info(f"Found {len(source_videos)} source video(s) in {SOURCE_VIDEO_FOLDER}:")
    for v in source_videos:
        log.info(f"  - {os.path.basename(v)}")

    music_tracks = find_music_tracks(MUSIC_FOLDER)
    if not music_tracks:
        log.error(f"ERROR: No audio files found in: {MUSIC_FOLDER}")
        sys.exit(1)

    tracks_to_process = music_tracks if PROCESS_ALL_TRACKS else music_tracks[:1]
    log.info(f"\nFound {len(music_tracks)} audio file(s) in {MUSIC_FOLDER}; processing {len(tracks_to_process)}:")
    for m in tracks_to_process:
        log.info(f"  - {os.path.basename(m)}")

    log.info(f"\nTranscript-guided shot selection: {'ON' if TRANSCRIPT_GUIDED_ENABLED else 'OFF'}")
    transcripts = load_transcripts(source_videos)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    shots = detect_shots(source_videos)
    if not shots:
        log.error("No usable shots found across the source videos.")
        return

    vfx_library = generate_vfx_library(VFX_LIBRARY_SIZE, VFX_LIBRARY_SEED) if VFX_ENABLED else []
    if VFX_ENABLED:
        log.info(f"VFX library ready: {len(vfx_library)} presets across {len({p['family'] for p in vfx_library})} families.")

    run_started = time.time()
    results = []
    jobs = []
    for music_path in tracks_to_process:
        try:
            runs = dna_runs_for_track(music_path)
        except Exception as e:
            log.error(f"DNA setup failed for {music_path}: {e}")
            results.append({"track": os.path.basename(music_path), "status": "failed", "error": str(e), "output": None})
            continue
        jobs += [(music_path, names) for names in runs]
    log.info(f"\n{len(jobs)} edit(s) planned (tracks x DNA variants).")
    stop = False
    for music_path, names in jobs:
        try:
            r = process_track(music_path, shots, vfx_library, transcripts, dna_names=names)
        except Exception as e:
            log.error(f"Unexpected error processing {music_path} [{'+'.join(names) or 'no DNA'}]: {e}")
            log.debug(traceback.format_exc())
            r = {"track": os.path.basename(music_path), "dna_run": "+".join(names) or "none",
                 "status": "failed", "error": str(e), "output": None}
            if not CONTINUE_ON_TRACK_FAILURE:
                stop = True
        results.append(r)
        if r and r.get("status") == "failed" and not CONTINUE_ON_TRACK_FAILURE:
            stop = True
        if stop:
            log.error("Stopping run because CONTINUE_ON_TRACK_FAILURE is False.")
            break

    ok = [r for r in results if r and r.get("status") == "ok"]
    failed = [r for r in results if r and r.get("status") == "failed"]
    skipped = [r for r in results if r and r.get("status", "").startswith("skipped")]

    log.info("")
    log.info("=" * 60)
    log.info(f"Run complete in {time.time() - run_started:.1f}s. "
              f"{len(ok)}/{len(results)} edit(s) produced, {len(skipped)} skipped, {len(failed)} failed.")
    for r in ok:
        log.info(f"  OK      {r['track']:30s} [{r.get('dna_run', '')}] -> {r['output']} "
                  f"(ducking exported: {r.get('ducking_track_exported', False)}, {r.get('cut_planning_mode', '?')})")
    for r in skipped:
        log.info(f"  SKIPPED {r['track']:40s} ({r.get('status')})")
    for r in failed:
        log.info(f"  FAILED  {r['track']:40s} ({r.get('error')})")

    if WRITE_MANIFEST:
        manifest_path = os.path.join(OUTPUT_FOLDER, "manifest.json")
        manifest = {
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_videos": [os.path.basename(v) for v in source_videos],
            "shot_pool_size": len(shots),
            "tracks": results,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"Manifest written to: {os.path.abspath(manifest_path)}")


if __name__ == "__main__":
    main()