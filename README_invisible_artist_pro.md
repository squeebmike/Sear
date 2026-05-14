# InvisibleArtist Pro

Advanced two-pass self-drawing video engine. Hides hands, arms, sleeves, and markers while preserving the paper and artwork as it builds over time.

---

## How It Differs From the Other Scripts

| Script | Method | Output |
|---|---|---|
| `remove_hand_frames.py` | Deletes hand frames | Shorter, choppy |
| `invisible_artist.py` | Rolling forward canvas fill | Smooth, but fills from the *past* |
| `invisible_artist_pro.py` | Two-pass + final canvas fill | Smooth, fills from the *future* (look-ahead) |

The key improvement in Pro: **Pass 1** scans the entire video and builds a complete clean canvas (the drawing in its most finished state per pixel). **Pass 2** uses that canvas to fill hand regions — so when the hand covers a spot, the fill already shows what that spot will eventually look like when the hand moves away.

---

## Quick Start

### Short test (first 10 seconds, ~300 frames at 30fps)
```bash
python3 invisible_artist_pro.py 1.mkv test_pro.mp4 --max-frames 300 --debug
```

### Full recommended test run
```bash
python3 invisible_artist_pro.py 1.mkv test_pro.mp4 \
  --max-frames 300 --debug \
  --mask-dilate 55 --arm-length 500 --arm-thickness 220 \
  --enable-sleeve-mask --enable-marker-mask
```

### Full video
```bash
python3 invisible_artist_pro.py 1.mkv output_self_drawing.mp4 \
  --mask-dilate 55 --arm-length 500 --arm-thickness 220 \
  --enable-sleeve-mask --enable-marker-mask
```

---

## All Flags

| Flag | Default | Purpose |
|---|---|---|
| `--debug` | off | Save `debug_mask.mp4`, `debug_canvas.mp4`, `debug_summary.txt` |
| `--max-frames N` | none | Stop after N frames — great for testing |
| `--confidence X` | `0.45` | MediaPipe hand detection sensitivity |
| `--mask-dilate N` | `45` | Pixels to expand hand hull outward |
| `--arm-length N` | `450` | How far the arm ray extends from the wrist (pixels) |
| `--arm-thickness N` | `190` | How wide the arm ray is (pixels) |
| `--feather N` | `15` | Mask edge softness |
| `--enable-sleeve-mask` | off | Also mask dark pixels connected to the arm region (for black/dark sleeves) |
| `--enable-marker-mask` | off | Also mask marker/pen body near the index finger |
| `--stable-frames N` | `2` | Clean frames required before committing new artwork to canvas |
| `--art-threshold N` | `18` | Pixel change needed to count as new artwork |
| `--strict-canvas` | off | Never update canvas from any masked pixel |

---

## Tuning Workflow

### Step 1 — Run with debug on a short clip
```bash
python3 invisible_artist_pro.py 1.mkv test.mp4 --max-frames 300 --debug
```

### Step 2 — Open `debug_mask.mp4`
Red = what gets masked (replaced by canvas).
- **Hand/fingers not fully red?** → increase `--mask-dilate` (try 60)
- **Arm/sleeve not covered?** → increase `--arm-length` / `--arm-thickness`, add `--enable-sleeve-mask`
- **Too much paper being covered?** → reduce `--mask-dilate` or `--arm-thickness`

### Step 3 — Open `debug_canvas.mp4`
This shows the final canvas used for fills.
- Should look like the drawing in its most complete state
- If blank paper areas show through the drawing, the canvas needs more clean frames

### Step 4 — Check `test.mp4`
- Hand visible? → increase mask settings
- Ghosting/artifacts around hand edges? → increase `--feather` (try 25)
- Drawing disappears under hand and doesn't come back? → expected — the hand blocked those pixels the whole time

### Step 5 — Full run
Once settings look good, remove `--max-frames` and run the full video.

---

## How the Two-Pass Look-Ahead Works

```
PASS 1 (full video, forward):
  For every frame:
    Detect hands → build mask
    For each pixel NOT in mask:
      If pixel has been clean for >= stable_frames: add to final_canvas
  
  Result: final_canvas = the most complete clean view of
          every pixel across the entire video

PASS 2 (full video, forward):
  For every frame:
    Detect hands → build mask
    For unmasked pixels: output the live frame
    For masked pixels: fill from final_canvas
    Feather mask edges for smooth blending
```

Why this helps: in Pass 2, when the hand covers a region, we fill it with `final_canvas`, which already knows what that region will look like *after* the hand moves away (because Pass 1 already saw the whole video). This avoids the "blank paper under the hand" problem common in forward-only approaches.

---

## Known Limitations

- If a hand covers brand-new artwork and **never fully moves away**, that artwork is never captured and the canvas fills with the last clean view before the hand arrived.
- Very fast hand movements may leave 1–2 frames of ghosting at the mask edge. Increasing `--feather` reduces this.
- The script runs MediaPipe twice (both passes), so processing time is roughly 2× the single-pass version.
- Dark sleeves are detected by brightness (`< 80` value in grayscale). Very light-colored sleeves may not be detected by `--enable-sleeve-mask`.

---

## Recording Tips

- **Pull your hand fully away** after each major stroke or color pass — gives the canvas clean frames to lock in the artwork
- **Use even, flat lighting** — moving shadows can confuse the arm mask
- **Tape paper down** — paper movement looks like motion to the detector
- **Avoid skin-toned backgrounds** — reduces detection accuracy
- **Keep the camera fixed** — any camera shake causes registration problems between the canvas and live frames
