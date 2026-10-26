# Publishing checklist (not for the repo — delete after use)
1. Unzip your project, **delete**: `__pycache__/`, `.kilo/`, `beat_sync_editor.log`, `beat_sync_editor_new.py.bak`, empty output folders.
2. Copy these docs into the project root (keep `.github/`).
3. Replace every `<your-username>`, `<repo>`, `<Your Name>`.
4. `git init && git add . && git commit -m "Initial release" && git branch -M main`
5. Create the repo on GitHub (empty, no README), then:
   `git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main`
6. `yolov8n.pt` is 6 MB — fine to commit, but AGPL applies; or delete it and let Ultralytics auto-download.
7. Repo Settings → add topics (video-editing, beat-sync, ffmpeg, librosa, whisper, anime, amv, voice-over), enable Discussions + Issues.
8. Create issues from ROADMAP.md; pin the voice-over one. Add a release (zip) and a demo GIF.
