"""
transcribe.py

Scans the current folder for audio files and transcribes each one into a
timestamped .txt transcript that:
  - shows a live progress bar (%) while transcribing
  - labels WHO is speaking (speaker diarization) — fully offline, no
    Hugging Face account/token needed
  - flags stretches of music / non-speech as [MUSIC / NON-SPEECH]
  - skips files that already have a transcript

Usage:
    python transcribe.py

That's it. No arguments needed. Re-running later skips anything already
transcribed.

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
pip install faster-whisper tqdm resemblyzer scikit-learn numpy

ffmpeg must be installed and on PATH.

Speaker diarization here uses Resemblyzer (a self-contained, offline voice
embedding model — nothing to download from a gated repo, no account or
token required) plus simple clustering to group segments by voice. It's
lighter-weight than pyannote and works fully offline after the one-time
pip install.
--------------------------------------------------------------------------

To change settings (model size, language, etc.), edit the CONFIG section
below.
"""

import os
import sys
from datetime import timedelta

import numpy as np
from faster_whisper import WhisperModel
from tqdm import tqdm

# ============================== CONFIG ===================================
MODEL_SIZE = "small"        # tiny, base, small, medium, large-v2, large-v3
LANGUAGE = None            # None = auto-detect, or force e.g. "en", "sw", "fr"
DEVICE = "auto"            # "auto", "cpu", or "cuda"
SCAN_FOLDER = "."          # folder to scan for audio files (current folder)
TRANSCRIPT_SUFFIX = "_transcript.txt"  # how output files are named/detected

# Speaker diarization (who's speaking). Fully offline — no account/token
# needed. Set NUM_SPEAKERS to a fixed number if you know it (e.g. 2 for an
# interview), or leave as "auto" to let the script estimate it.
ENABLE_SPEAKER_DIARIZATION = True
NUM_SPEAKERS = "auto"  # "auto", or an integer like 2, 3, 4...

# Music / non-speech detection. Flags segments Whisper thinks are music,
# applause, noise, etc. rather than speech, and labels them [MUSIC] instead
# of transcribing gibberish.
ENABLE_MUSIC_DETECTION = True
# ===========================================================================

# Extensions we'll attempt to transcribe. ffmpeg (used internally) supports
# far more than this, but this covers essentially everything you'd run into.
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
    ".aac", ".wma", ".aiff", ".aif", ".alac", ".amr", ".mkv", ".webm", ".mov", ".avi",  # video files (audio track only)
}

# Whisper's built-in tags for stretches that aren't spoken words.
NON_SPEECH_MARKERS = {
    "[music]", "(music)", "♪", "[applause]", "(applause)",
    "[laughter]", "(laughter)", "[noise]", "(noise)", "[silence]", "(silence)",
}


def format_timestamp(seconds: float) -> str:
    """Convert seconds (float) into HH:MM:SS,mmm format."""
    td = timedelta(seconds=max(0, seconds))
    total_ms = int(td.total_seconds() * 1000)
    hours, remainder = divmod(total_ms, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def find_audio_files(folder: str):
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def transcript_path_for(audio_path: str) -> str:
    base, _ = os.path.splitext(audio_path)
    return base + TRANSCRIPT_SUFFIX


def already_transcribed(audio_path: str) -> bool:
    out_path = transcript_path_for(audio_path)
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def is_non_speech(text: str) -> bool:
    cleaned = text.strip().lower()
    if not cleaned:
        return True
    return cleaned in NON_SPEECH_MARKERS


def resolve_device_and_compute_type():
    device = DEVICE
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


# --------------------------------------------------------------------------
# Offline speaker diarization using Resemblyzer + clustering.
# For each Whisper segment, we extract a voice embedding for that audio
# slice, then cluster all embeddings in the file to assign speaker labels.
# --------------------------------------------------------------------------

def get_speaker_labels(audio_path: str, segments):
    """
    Returns a dict {segment_index: "Speaker N"} or None if diarization
    isn't available/failed. `segments` is the list of Whisper segments
    (already materialized).
    """
    if not ENABLE_SPEAKER_DIARIZATION:
        return None

    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        print("  [Speaker ID disabled] resemblyzer isn't installed. "
              "Run: pip install resemblyzer scikit-learn")
        return None

    try:
        import librosa
    except ImportError:
        print("  [Speaker ID disabled] librosa isn't installed. "
              "Run: pip install librosa")
        return None

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print("  [Speaker ID disabled] scikit-learn isn't installed. "
              "Run: pip install scikit-learn")
        return None

    speech_segments = [
        (i, s) for i, s in enumerate(segments)
        if not (ENABLE_MUSIC_DETECTION and is_non_speech(s.text))
    ]
    if len(speech_segments) < 2:
        return None  # not enough speech to bother clustering

    try:
        encoder = VoiceEncoder()
        # Load and resample to 16kHz ourselves (what the encoder expects),
        # skipping resemblyzer's own preprocess_wav() so we don't depend on
        # a working webrtcvad silence-trimmer.
        sample_rate = 16000
        wav, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

        embeddings = []
        valid_indices = []
        for i, seg in speech_segments:
            start_sample = int(seg.start * sample_rate)
            end_sample = int(seg.end * sample_rate)
            clip = wav[start_sample:end_sample]
            if len(clip) < sample_rate * 0.3:  # skip very short clips (<0.3s)
                continue
            embeddings.append(encoder.embed_utterance(clip))
            valid_indices.append(i)

        if len(embeddings) < 2:
            return None

        embeddings = np.array(embeddings)

        if NUM_SPEAKERS == "auto":
            # Try a range of cluster counts, pick the best by silhouette score
            from sklearn.metrics import silhouette_score
            best_k, best_score, best_labels = 1, -1, np.zeros(len(embeddings), dtype=int)
            max_k = min(6, len(embeddings))
            for k in range(2, max_k + 1):
                clustering = AgglomerativeClustering(n_clusters=k)
                labels = clustering.fit_predict(embeddings)
                if len(set(labels)) < 2:
                    continue
                try:
                    score = silhouette_score(embeddings, labels)
                except ValueError:
                    continue
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
            labels = best_labels
        else:
            k = max(1, int(NUM_SPEAKERS))
            if k == 1:
                labels = np.zeros(len(embeddings), dtype=int)
            else:
                clustering = AgglomerativeClustering(n_clusters=k)
                labels = clustering.fit_predict(embeddings)

        result = {}
        for idx, label in zip(valid_indices, labels):
            result[idx] = f"Speaker {label + 1}"
        return result

    except Exception as e:
        print(f"  WARNING: speaker diarization failed for this file ({e}). "
              f"Continuing without speaker labels.")
        return None


def transcribe_one(whisper_model, audio_path: str) -> str:
    """Transcribe + diarize a single audio file, write its .txt transcript."""
    output_path = transcript_path_for(audio_path)

    print(f"\nTranscribing: {audio_path}")
    segments_gen, info = whisper_model.transcribe(
        audio_path,
        language=LANGUAGE,      # None = auto-detect
        beam_size=5,
        vad_filter=True,        # skips silence, improves accuracy
    )

    total_duration = info.duration  # seconds, used to drive the progress bar
    segments = []

    with tqdm(total=100, unit="%", desc="  Progress", ncols=80) as pbar:
        last_pct = 0
        for segment in segments_gen:
            segments.append(segment)
            if total_duration > 0:
                pct = min(100, int((segment.end / total_duration) * 100))
                if pct > last_pct:
                    pbar.update(pct - last_pct)
                    last_pct = pct
        if last_pct < 100:
            pbar.update(100 - last_pct)

    print(f"  Detected language: {info.language} (probability {info.language_probability:.2f})")

    speaker_labels = get_speaker_labels(audio_path, segments) if segments else None
    if speaker_labels:
        speaker_count = len(set(speaker_labels.values()))
        print(f"  Detected {speaker_count} speaker(s)")

    lines = []
    lines.append(f"Transcript of: {os.path.basename(audio_path)}")
    lines.append(f"Detected language: {info.language}")
    if speaker_labels:
        lines.append(f"Speakers detected: {len(set(speaker_labels.values()))}")
    lines.append("-" * 60)
    lines.append("")

    segment_count = 0
    for i, segment in enumerate(segments):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()

        if ENABLE_MUSIC_DETECTION and is_non_speech(text):
            line = f"[{start} --> {end}] [MUSIC / NON-SPEECH]"
            lines.append(line)
            segment_count += 1
            continue

        speaker_label = speaker_labels.get(i) if speaker_labels else None
        if speaker_label:
            line = f"[{start} --> {end}] {speaker_label}: {text}"
        else:
            line = f"[{start} --> {end}] {text}"

        lines.append(line)
        segment_count += 1

    if segment_count == 0:
        lines.append("(No speech detected.)")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved -> {output_path}")
    return output_path


def main():
    folder = os.path.abspath(SCAN_FOLDER)
    print(f"Scanning for audio files in: {folder}")

    all_files = find_audio_files(folder)
    if not all_files:
        print("No audio files found in this folder.")
        return

    to_process = [f for f in all_files if not already_transcribed(f)]
    skipped = [f for f in all_files if already_transcribed(f)]

    print(f"\nFound {len(all_files)} audio file(s).")
    if skipped:
        print(f"Skipping {len(skipped)} already-transcribed file(s):")
        for f in skipped:
            print(f"  - {os.path.basename(f)} (found {os.path.basename(transcript_path_for(f))})")

    if not to_process:
        print("\nNothing new to transcribe. All done.")
        return

    print(f"\n{len(to_process)} file(s) to transcribe:")
    for f in to_process:
        print(f"  - {os.path.basename(f)}")

    device, compute_type = resolve_device_and_compute_type()
    print(f"\nLoading Whisper model '{MODEL_SIZE}' on device '{device}' ({compute_type})...")
    whisper_model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)

    succeeded, failed = [], []
    for audio_path in to_process:
        try:
            transcribe_one(whisper_model, audio_path)
            succeeded.append(audio_path)
        except Exception as e:
            print(f"  ERROR transcribing {audio_path}: {e}", file=sys.stderr)
            failed.append(audio_path)

    print("\n" + "=" * 60)
    print(f"Done. {len(succeeded)} transcribed, {len(failed)} failed, {len(skipped)} skipped.")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()