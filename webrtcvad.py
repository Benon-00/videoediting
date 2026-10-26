"""
Minimal stub for webrtcvad.

Resemblyzer imports this module at the top of resemblyzer/audio.py but
only actually calls it inside trim_long_silences(), which our transcribe.py
does not use (we do our own lightweight preprocessing instead). This stub
exists purely so the `import webrtcvad` line succeeds without needing to
compile the real C extension (which has no prebuilt wheel for newer Python
versions on Windows yet).
"""


class Vad:
    def __init__(self, mode=0):
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def is_speech(self, frame, sample_rate):
        # Always report "speech" — real VAD filtering isn't used by our
        # pipeline, so this just needs to not crash.
        return True
