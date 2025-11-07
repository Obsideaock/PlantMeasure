# edge_first_leaves.py
from __future__ import annotations
import cv2, numpy as np

def _largest_bright_sheet(Ls: np.ndarray) -> np.ndarray:
    """Return mask (255) of the largest bright component = paper sheet."""
    # 1) Try Otsu (bright paper)
    _, bright = cv2.threshold(Ls, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frac = float(np.mean(bright > 0))

    # 2) Fallbacks if Otsu is degenerate (too small or almost everything)
    if frac < 0.05 or frac > 0.98:
        t = int(np.percentile(Ls, 92))  # robust high-tail cut for paper
        _, bright = cv2.threshold(Ls, t, 255, cv2.THRESH_BINARY)
        frac = float(np.mean(bright > 0))
    if frac < 0.02:  # last resort: adaptive
        bright = cv2.adaptiveThreshold(Ls, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 101, -5)

    # 3) Clean & keep largest CC
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, se)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  se)

    num, labs, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    sheet = np.zeros_like(bright)
    if num > 1:
        i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        sheet[labs == i] = 255
    return sheet

def _fill_holes(bin_img: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask via flood fill from borders."""
    h, w = bin_img.shape[:2]
    pad = cv2.copyMakeBorder(bin_img, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
    ff = pad.copy()
    cv2.floodFill(ff, None, (0,0), 255)
    ff = ff[1:-1,1:-1]
    filled = cv2.bitwise_or(bin_img, cv2.bitwise_not(ff))
    return filled

def _thin_peaks_to_centers(peaks: np.ndarray) -> np.ndarray:
    """Reduce plateau peaks to a single pixel at each component centroid."""
    num, labs, _, cents = cv2.connectedComponentsWithStats(peaks)
    out = np.zeros_like(peaks)
    for i in range(1, num):
        cx, cy = int(round(cents[i][0])), int(round(cents[i][1]))
        if 0 <= cy < out.shape[0] and 0 <= cx < out.shape[1]:
            out[cy, cx] = 255
    return out

def _local_maxima(dt: np.ndarray, min_dist_px: int = 9, rel_thresh: float = 0.35) -> np.ndarray:
    """Find DT peaks with non-max suppression."""
    dtf = dt.astype(np.float32)

    print(f"[peaks.in] dtf_max={float(dtf.max()):.3f} rel_thresh={rel_thresh}  "
          f"min_dist={min_dist_px}")

    if dtf.max() <= 0:
        return np.zeros_like(dt, np.uint8)
    thr = dtf.max() * float(rel_thresh)
    m = (dtf >= thr).astype(np.uint8) * 255
    # non-max suppression by dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, min_dist_px)|1,)*2)
    dil = cv2.dilate(dtf, kernel)
    peaks = ((dtf == dil) & (dtf >= thr)).astype(np.uint8) * 255
    peaks = cv2.bitwise_and(peaks, m)

    print(f"[peaks.out] peaks_px={int(np.count_nonzero(peaks))}")

    return peaks

def segment_leaves_edgefirst(
    bgr: np.ndarray,
    *,
    bilateral_d: int = 9,
    bilateral_sigma_color: int = 55,
    bilateral_sigma_space: int = 7,
    canny_low_rel: float = 0.35,
    canny_high_rel: float = 0.7,
    edge_close: int = 3,
    min_leaf_area: int = 500,
    dt_rel_thresh: float = 0.38,
    dt_min_dist_px: int = 11,
    post_split: bool = True
):
    """
    Edge-first, marker-controlled watershed.
    Returns (leaf_count, leaf_mask, leaf_labels, overlay_bgr).
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image")

    print(f"[edge] input shape={bgr.shape} dtype={bgr.dtype} "
          f"min={int(bgr.min())} max={int(bgr.max())}")

    h, w = bgr.shape[:2]

    # CHANGED: make min-distance scale with image size (helps huge sheets)
    dt_min_dist_px = max(dt_min_dist_px, int(min(h, w) * 0.007))

    # 1) Paper/plant mask on luminance
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_s = cv2.bilateralFilter(L, d=bilateral_d,
                              sigmaColor=bilateral_sigma_color,
                              sigmaSpace=bilateral_sigma_space)

    # (i) detect the paper sheet (local helper)
    def _largest_bright_sheet(Ls: np.ndarray) -> np.ndarray:
        _, bright = cv2.threshold(Ls, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if not (0.05 <= np.mean(bright > 0) <= 0.98):
            t = int(np.percentile(Ls, 92))
            _, bright = cv2.threshold(Ls, t, 255, cv2.THRESH_BINARY)
        se7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, se7)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, se7)
        num, labs, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        sheet = np.zeros_like(bright)
        if num > 1:
            i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            sheet[labs == i] = 255
        sheet = cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
        return sheet

    sheet = _largest_bright_sheet(L_s)
    print(f"[sheet] frac={np.mean(sheet > 0):.3f}")

    # (ii) adaptive foreground on the sheet (dark-on-paper measured on the sheet only)
    p98 = float(np.percentile(L_s[sheet > 0], 98))
    target = 0.25
    low, high = 0.10, 0.45
    margins = [50, 44, 38, 34, 30, 26, 22, 18, 14, 10]

    best = None;
    best_err = 1e9
    plant0_lum = None;
    fg_frac = 0.0
    for m in margins:
        thr = int(np.clip(p98 - m, 0, 255))
        _, p0 = cv2.threshold(L_s, thr, 255, cv2.THRESH_BINARY_INV)  # dark on paper
        p0 = cv2.bitwise_and(p0, sheet)
        frac = float(np.mean(p0 > 0))
        err = abs(frac - target)
        if low <= frac <= high:
            plant0_lum, fg_frac = p0, frac
            break
        if err < best_err:
            best, best_err, fg_frac = p0, err, frac
    if plant0_lum is None:
        plant0_lum = best
    print(f"[mask.adapt] p98={p98:.1f} fg_frac0={fg_frac:.3f} (luminance)")

    # --- soft color gate to drop gray/black artifacts but keep olive frond ---
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    chroma = np.hypot(a, b)  # robust chroma
    color_gate = (chroma > 6).astype(np.uint8) * 255  # softer than 12

    S = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    sat_gate = (S > 8).astype(np.uint8) * 255  # softer than 15

    # plant0 = luminance mask gated by color/sat (still pre fill)
    plant0 = cv2.bitwise_and(plant0_lum, color_gate)
    plant0 = cv2.bitwise_and(plant0, sat_gate)

    # Fallback if color gate is too aggressive
    if np.mean(plant0 > 0) < 0.06:  # ~6% pixels
        plant0 = plant0_lum
        print("[mask.gate] fallback to luminance-only (color gate too strict)")

    # Clean once, then fill holes -> final plant
    plant0 = cv2.morphologyEx(plant0, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    plant0 = cv2.morphologyEx(plant0, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    plant = _fill_holes(plant0)

    # Optionally keep only largest 1–2 blobs inside sheet
    num, labs, stats, _ = cv2.connectedComponentsWithStats(plant, 8)
    if num > 2:
        keep_ids = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2:]
        kept = np.zeros_like(plant)
        for i in keep_ids:
            kept[labs == i] = 255
        plant = kept

    print(f"[mask.B] plant(after color-gate+clean): frac={np.mean(plant > 0):.3f}")

    # 2) Edge magnitude (still useful for stats/visuals)
    gx = cv2.Scharr(L_s, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(L_s, cv2.CV_32F, 0, 1)
    gm = cv2.magnitude(gx, gy)
    gm8 = cv2.normalize(gm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    med = np.median(gm8[plant>0]) if np.count_nonzero(plant) else np.median(gm8)
    lo = int(max(0, (canny_low_rel * med)))
    hi = int(min(255, (canny_high_rel * med + 32)))
    edges = cv2.Canny(gm8, lo, hi)
    print(f"[edge] med={med:.1f}  canny_lo={lo}  canny_hi={hi}  edge_frac={np.mean(edges > 0):.3f}")
    if edge_close > 0:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_close*2+1,)*2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, se, iterations=1)

    # 3) Seeds from DT peaks
    print(f"[mask.C] BEFORE DT: plant frac={np.mean(plant > 0):.3f}  "
          f"unique={np.unique(plant).tolist()[:6]}")
    dt = cv2.distanceTransform(plant, cv2.DIST_L2, 5)
    print(f"[dt] min={float(np.min(dt)):.3f}  max={float(np.max(dt)):.3f}  "
          f"all_finite={np.isfinite(dt).all()}")

    fg_peaks = _local_maxima(dt, min_dist_px=dt_min_dist_px, rel_thresh=dt_rel_thresh)
    fg_peaks = _thin_peaks_to_centers(fg_peaks)

    # CHANGED: auto-raise peak threshold to cap the number of seeds on huge images
    plant_px = int(np.count_nonzero(plant))
    max_peaks = max(8, plant_px // (80 * 80))  # ~1 seed per 80×80 block
    rel = float(dt_rel_thresh)
    peaks_px = int(np.count_nonzero(fg_peaks))
    while peaks_px > max_peaks and rel < 0.85:
        rel += 0.05
        fg_peaks = _local_maxima(dt, min_dist_px=dt_min_dist_px, rel_thresh=rel)
        fg_peaks = _thin_peaks_to_centers(fg_peaks)
        peaks_px = int(np.count_nonzero(fg_peaks))

    # give each seed a small footprint so background can’t cut between pixels
    seed_mask = cv2.dilate(
        fg_peaks,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        1
    )
    _, fg_labels = cv2.connectedComponents(seed_mask)

    u, c = np.unique(fg_labels, return_counts=True)
    print(f"[seeds.cc] labels={len(u) - 1}  top_counts={c[1:6].tolist()}")
    num_seed_px = int(np.count_nonzero(fg_peaks))
    num_seed_cc = cv2.connectedComponents(fg_peaks)[0] - 1
    print(f"[seeds] dt_max={dt.max():.2f}  peaks_px={num_seed_px}  seeds={num_seed_cc}")

    # Background marker (label 1)
    bg = cv2.bitwise_not(plant)
    bg = cv2.erode(bg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 2)

    # Markers: 0 unknown, 1 paper, >1 seeds
    markers = np.zeros_like(fg_labels, dtype=np.int32)
    markers[bg > 0] = 1
    markers[fg_labels > 0] = fg_labels[fg_labels > 0] + 1

    print(f"[markers] bg_count={(markers == 1).sum()}  zero_count={(markers == 0).sum()}  "
          f"fg_count={(markers > 1).sum()}")

    # 4) CHANGED: watershed on –DT (not edge magnitude)
    neg = cv2.normalize(-dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    markers_ws = markers.copy()
    cv2.watershed(cv2.cvtColor(neg, cv2.COLOR_GRAY2BGR), markers_ws)

    u = np.unique(markers_ws)
    print(f"[ws.raw] uniq={u.tolist()[:10]}  neg_count={(markers_ws == -1).sum()}")

    labels = markers_ws.copy()
    labels[(plant==0) | (labels==1) | (labels==-1)] = 0
    # relabel to 1..K
    ids = np.unique(labels)
    ids = ids[(ids!=0) & (ids!=1)]
    relabeled = np.zeros_like(labels, np.int32)
    for new_id, old_id in enumerate(ids, 1):
        relabeled[labels==old_id] = new_id
    labels = relabeled
    print(f"[ws.clean] labels>0 frac={np.mean(labels > 0):.3f}  max_id={int(labels.max())}")

    # 5) Area filter (pre split)
    leaf_mask = (labels>0).astype(np.uint8)*255
    if min_leaf_area > 0:
        num, labs, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, 8)
        keep = np.zeros(num, bool); keep[0] = False
        for i in range(1, num):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= int(min_leaf_area)
        leaf_mask[:] = 0
        for i in range(1, num):
            if keep[i]: leaf_mask[labs==i] = 255
        labels = labels * (leaf_mask>0)

    if post_split:
        out_labels = np.zeros_like(labels, np.int32)
        next_id = 1
        num, labs, stats, _ = cv2.connectedComponentsWithStats((labels>0).astype(np.uint8), 8)
        for i in range(1, num):
            roi = (labs == i).astype(np.uint8)*255
            if stats[i, cv2.CC_STAT_AREA] < max(4*min_leaf_area, 400):
                out_labels[roi>0] = next_id; next_id += 1
                continue
            dt_i = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
            peaks = _local_maxima(dt_i, min_dist_px=dt_min_dist_px, rel_thresh=dt_rel_thresh)
            peaks = _thin_peaks_to_centers(peaks)

            # label the seed pixels
            num_labels, seed_cc = cv2.connectedComponents(peaks)  # seed_cc is 0 outside seeds, 1..K on seeds
            num_seeds = num_labels - 1

            if num_seeds >= 2:
                # --- CORRECT MARKERS ---
                # 0 = unknown (inside ROI), 1 = background (outside ROI), >1 = seeds
                mk = np.zeros_like(seed_cc, dtype=np.int32)
                mk[roi == 0] = 1  # background OUTSIDE the ROI
                mk[seed_cc > 0] = seed_cc[seed_cc > 0] + 1  # seeds become 2..K+1

                neg_i = cv2.normalize(-dt_i, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                mk_ws = mk.copy()
                cv2.watershed(cv2.cvtColor(neg_i, cv2.COLOR_GRAY2BGR), mk_ws)

                # keep only non-background labels inside the ROI
                mk_ws[(roi == 0) | (mk_ws == 1) | (mk_ws == -1)] = 0

                for sid in np.unique(mk_ws):
                    if sid <= 1:
                        continue
                    out_labels[mk_ws == sid] = next_id
                    next_id += 1
            else:
                out_labels[roi > 0] = next_id
                next_id += 1
        labels = out_labels
        leaf_mask = (labels>0).astype(np.uint8)*255
        # Area filter AFTER split
        if min_leaf_area > 0:
            num, labs, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, 8)
            keep = np.zeros(num, dtype=bool); keep[0] = False
            for i_cc in range(1, num):
                keep[i_cc] = stats[i_cc, cv2.CC_STAT_AREA] >= int(min_leaf_area)
            leaf_mask[:] = 0
            for i_cc in range(1, num):
                if keep[i_cc]:
                    leaf_mask[labs == i_cc] = 255
            labels = labels * (leaf_mask > 0)

    # CHANGED: Final relabel + robust count from kept mask
    num_cc, cc_labels = cv2.connectedComponents((leaf_mask > 0).astype(np.uint8), connectivity=8)
    labels = cc_labels.astype(np.int32)
    leaf_count = num_cc - 1

    print(f"[area] comps_final={leaf_count}  leaf_mask_frac={np.mean(leaf_mask > 0):.3f}")

    # 6) Overlay
    overlay = bgr.copy().astype(np.float32)
    alpha = 0.35
    mask = (leaf_mask > 0)
    green = np.array([0, 255, 0], dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + green * alpha
    overlay = overlay.clip(0, 255).astype(np.uint8)

    print(f"[final] leaf_count={leaf_count}  leaf_mask_frac={np.mean(leaf_mask > 0):.3f}")

    return leaf_count, leaf_mask, labels, overlay

def colorize_labels(leaf_labels: np.ndarray, seed: int = 42) -> np.ndarray:
    k = int(leaf_labels.max())
    if k <= 0:
        return np.zeros((*leaf_labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(k+1, 3), dtype=np.uint8)
    out = np.zeros((*leaf_labels.shape, 3), dtype=np.uint8)
    nz = leaf_labels > 0
    out[nz] = palette[leaf_labels[nz]]

    return out
