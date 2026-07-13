# NEET OMR Checker â€” Robust Version

## What changed vs. the fragile version

| Old vulnerability | Fix in this version |
|---|---|
| Crashed unless exactly 720 bubble contours were found | Detected bubbles are only used to **estimate** where the 4 columns and 45 rows sit (via percentile-based extent + gap clustering). The full 180Ã—4 grid is then generated mathematically and **every expected position is sampled directly** â€” a missed bubble or a stray shadow contour can no longer break the array math. |
| No alignment anchors | The grid-interpolation approach acts as a soft anchor system: it only needs the *overall spread* of detected ink, not a perfect anchor mark or a perfect count. |
| Broke on rotation/skew | Two-tier correction: (1) full 4-point perspective warp when a clean sheet boundary is found, (2) a rotation-only deskew fallback (via `cv2.minAreaRect` on ink pixels) when it isn't â€” safe even on pre-cropped photos. |
| Single global threshold, bad in shadows | Adaptive Gaussian thresholding (per local region) + morphological open/close cleans up speckle noise and uneven lighting. |
| Silent misreads | Every question shows its raw fill ratios in the debug expander, and both the answer key and student sheet go through an **editable table** before scoring, so any misdetection is a two-second manual fix, not a re-shoot. |

## How the grid-interpolation trick works

1. Loosely detect bubble-like blobs (broad size/aspect-ratio filter â€” not strict).
2. Split them into 4 columns using **gap-based 1D clustering**: sort all x-centroids, find the 3 largest horizontal gaps (the natural whitespace gutters between columns), and cut there. This is deterministic â€” no k-means random-seed flakiness.
3. Within each column, take the 3rdâ€“97th percentile of detected y-positions as the column's vertical extent (percentiles ignore stray outliers), then `np.linspace` 45 evenly spaced row centers across that range.
4. Do the same horizontally for the 4 option centers within each column.
5. For **every** one of the resulting 4 Ã— 45 Ã— 4 = 720 expected coordinates, sample a small circular region on the adaptive-threshold ink mask and measure fill ratio â€” regardless of whether a contour was actually found there.

If detection finds almost nothing (e.g., very poor photo), the code falls back to sensible even-spacing defaults across the sheet rather than crashing.

## Tuning knobs (top of `app.py`)

- `FILL_THRESHOLD` (default `0.28`) â€” raise if faint smudges are being read as marks; lower if legitimately filled bubbles are being missed.
- `AMBIGUOUS_MARGIN` (default `0.06`) â€” how close two options' fill ratios must be before a question is flagged "multi-mark" instead of picking a winner.
- `WARP_W`, `WARP_H` â€” canvas size after perspective correction; only affects processing resolution, not scoring logic.

## Deployment (Streamlit Community Cloud)

1. Push `app.py`, `requirements.txt` to your `Omr-scanner-for-neet` GitHub repo (replacing the old files).
2. On [share.streamlit.io](https://share.streamlit.io), redeploy pointing at `app.py`.
3. On Android, open the deployed URL in Chrome â†’ menu â†’ **Add to Home Screen** for the PWA-style experience.

## Known remaining limitations

- Assumes the answer key photo is itself an OMR-style bubble sheet in the same layout (the app scores by comparing detected bubbles, not a typed key). If you'd rather type the key once and reuse it for many students, that's a small follow-up: skip the key camera step and just prefill the editable table.
- Extremely blurry or motion-blurred photos will still need the manual-correction table â€” no OCR/CV pipeline can fully substitute for a legible source image.
- Multi-marked questions are currently scored as incorrect (-1), matching typical NEET rules; change the `"multi-mark"` branch in `score_sheet()` if your practice test treats them as 0.
