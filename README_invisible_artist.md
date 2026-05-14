# InvisibleArtist

Hides hands and arms from overhead drawing videos so the artwork appears to build itself.

**Unlike** `remove_hand_frames.py` (which deletes frames), this tool keeps every frame and fills hand/arm regions using a rolling clean canvas built from nearby hand-free moments.

---

## How It Works (Simple Version)

1. Reads your video and builds a **background plate** — the blank paper, free of hands.
2. Detects hands using Google MediaPipe.
3. Detects moving arms using motion analysis (comparing each frame to the running background).
4. For every frame, paints hand/arm regions using the **most recent clean view** of the paper at that spot.
5. Keeps track of new ink, pencil marks, and color fills as the artist's hand moves away.
6. Writes a smooth, continuous output MP4 at the original resolution and FPS.

### Important Limitation

If a hand fully covers a brand-new mark before the camera ever sees it clearly, the tool cannot know what is underneath. It will fill with the last-known clean state at that spot. The moment the hand moves away, the new mark snaps in. This is expected behavior for v1 — not a bug.

---

## Quick Start

```bash
# Basic run
python3 invisible_artist.py input.mkv output_invisible_artist.mp4

# Quick test on first 5 seconds (at 30fps ≈ 150 frames)
python3 invisible_artist.py input.mkv test_out.mp4 --max-frames 150 --debug
```

---

## All Options

| Flag | Default | What It Does |
|------|---------|--------------|
| `--mask-dilate N` | `25` | Pixels to expand the hand/arm mask outward. Higher = catches more of the arm but may erase nearby paper. |
| `--motion-threshold N` | `25` | How sensitive motion detection is (0–255). Lower catches more movement. Raise if blank paper is being masked. |
| `--feather N` | `9` | Softness of the mask edge in pixels. Higher = smoother blend but less precise. |
| `--debug` | off | Saves `debug_masks.mp4` (red overlay showing what gets masked) and `debug_summary.txt`. |
| `--max-frames N` | none | Stop after N frames. Use for quick tests before running the full video. |

---

## Recommended Workflow

### Step 1 — Quick Test
Run on just a few seconds with debug enabled:
```bash
python3 invisible_artist.py 1.mkv test_out.mp4 --max-frames 150 --debug
```

### Step 2 — Check the Debug Mask Video
Open `debug_masks.mp4`. Red areas = what gets replaced by the canvas.
- If arms are not fully red → increase `--mask-dilate` (try 35 or 45)
- If too much paper turns red → lower `--motion-threshold` (try 35) or reduce `--mask-dilate`

### Step 3 — Tune and Re-run
```bash
python3 invisible_artist.py 1.mkv test_out.mp4 --max-frames 150 --mask-dilate 35 --motion-threshold 30
```

### Step 4 — Full Run
```bash
python3 invisible_artist.py 1.mkv output_invisible_artist.mp4
```

---

## Comparison: Old vs New

| | `remove_hand_frames.py` | `invisible_artist.py` |
|---|---|---|
| Method | Deletes hand frames | Fills hand regions from canvas |
| Output length | Shorter (frames removed) | Same length as input |
| Smoothness | Can be choppy | Smooth and continuous |
| Speed | Fast | Slower (processes every frame) |
| Best for | Quick cleanup | Polished "self-drawing" effect |

---

## Recording Tips for Best Results

- **Use a fixed overhead camera** — any camera shake confuses the motion detector
- **Tape your paper down** — paper movement looks like motion
- **Use even, consistent lighting** — shadows confuse the detector
- **Pull your hand fully away** after each section — gives the tool clean frames to build the canvas
- **Pause briefly** after adding a new section so the canvas can capture the fresh marks
- **Avoid skin-colored desks** — similar color to hands reduces detection accuracy
- **Avoid strong moving shadows** — they can trigger the motion mask

---

## Troubleshooting

**Hands not being removed** → Increase `--mask-dilate` (try 40) or lower `--motion-threshold` (try 15)

**Too much paper being erased** → Raise `--motion-threshold` (try 40) or lower `--mask-dilate` (try 15)

**Ghosting / smearing around hands** → This is the canvas fill. Try increasing `--feather` for a softer blend, or ensure you have frequent clean frames (pull hand away often).

**New ink disappears then reappears** → Expected behavior when hand covers fresh marks. Pull hand away briefly after each new mark.

**Script is slow** → Normal. A 2-minute video at 1080p may take 10–20 minutes on a MacBook. Use `--max-frames` to test settings before the full run.

**"Model not found" error** → Delete `hand_landmarker.task` if it exists and re-run. It will re-download.
