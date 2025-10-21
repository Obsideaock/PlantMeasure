#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import numpy as np
import argparse
import pandas as pd
from typing import Dict, Tuple, Optional, List

DEFAULT_IMAGE = "data/ASU0115004_lg.jpg"

# =============================
# RULER: dark band + text inside
# =============================

def _deskew_crop(img_bgr: np.ndarray, rect):
    H, W = img_bgr.shape[:2]
    (cx, cy), (w, h), ang = rect
    if w < h:
        w, h = h, w
        ang += 90.0
    M = cv.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv.warpAffine(img_bgr, M, (W, H), flags=cv.INTER_CUBIC,
                        borderMode=cv.BORDER_REPLICATE)
    x0 = max(0, int(cx - w/2)); y0 = max(0, int(cy - h/2))
    x1 = min(W, int(cx + w/2)); y1 = min(H, int(cy + h/2))
    roi = rot[y0:y1, x0:x1].copy()
    return roi, rot, (x0, y0, x1, y1)


def _find_texty_dark_candidates(img_bgr: np.ndarray):
    """
    Find long, dark, horizontal bands near the borders that have *text-like*
    content inside (edge-dense bright bits). This matches the ASU ruler band.
    """
    H, W = img_bgr.shape[:2]
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    L = lab[:, :, 0]

    # dark mask
    dark = (L < 90).astype(np.uint8) * 255
    dark = cv.morphologyEx(dark, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
    dark = cv.morphologyEx(dark, cv.MORPH_CLOSE, np.ones((21,21), np.uint8))

    cnts, _ = cv.findContours(dark, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cand_rects = []
    for c in cnts:
        rect = cv.minAreaRect(c)
        (cx, cy), (w, h), ang = rect
        short, long_ = sorted([w, h])
        if short < 0.035*H or short > 0.15*H:
            continue
        if long_ / max(short, 1e-3) < 4.0:
            continue

        x, y, ww, hh = cv.boundingRect(c)
        # rulers live near edges
        near_border = (y < 0.30*H) or (y + hh > 0.70*H) or \
                      (x < 0.30*W) or (x + ww > 0.70*W)
        if not near_border:
            continue

        # deskew & check for "textiness" (lots of edges on a dark band)
        roi, _, _ = _deskew_crop(img_bgr, rect)
        if roi.size == 0:
            continue
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (edges.size + 1e-6)
        mean_L = float(np.mean(cv.cvtColor(roi, cv.COLOR_BGR2LAB)[:, :, 0]))
        if edge_density < 0.02:    # not enough fine detail → not text
            continue
        if mean_L > 150:           # too bright for the black band
            continue

        cand_rects.append((rect, edge_density))

    cand_rects.sort(key=lambda t: t[1], reverse=True)
    return [r for r, _ in cand_rects]


def _pick_period_with_harmonic(ac: np.ndarray, lo: int, hi: int) -> Tuple[Optional[int], float]:
    """
    From an autocorrelation vector, pick a period in [lo,hi) by favoring
    local maxima whose 2× harmonic is also strong. Returns (period, score).
    """
    seg = ac[lo:hi]
    seg_s = cv.GaussianBlur(seg.reshape(1, -1), (1, 11), 0).ravel()
    peaks = [i for i in range(1, len(seg_s)-1)
             if seg_s[i] > seg_s[i-1] and seg_s[i] > seg_s[i+1]]
    if not peaks:
        return None, 0.0
    best_i, best_score = None, -1.0
    for i in peaks:
        period = lo + i
        h2 = seg_s[2*i - lo] if (2*period) < hi else 0.0
        score = float(seg_s[i]) + 0.5*float(h2)
        if score > best_score:
            best_i, best_score = period, score
    return best_i, best_score


def _tick_spacing_px_inside_band(roi_bgr: np.ndarray):
    """
    Estimate 1-cm spacing inside the dark ruler band by:
      1) finding the bright baseline and cropping a thin strip around it,
      2) enhancing vertical bright lines (ticks),
      3) adaptively selecting major ticks by height among slender components,
      4) computing a robust median spacing, with a safety that halves spacing
         if we accidentally selected every other tick.

    Returns (tick_px, normalized_profile_for_debug, peak_positions, score).
    """
    if roi_bgr.size == 0:
        return None, None, None, 0.0

    # Ignore logo / color card on the right and dense mm cluster at the very left
    W_all = roi_bgr.shape[1]
    left_cut  = int(0.04 * W_all)       # drop the tiny-mm cluster leftmost
    right_cut = int(0.80 * W_all)       # drop ASU logo & color card
    roi = roi_bgr[:, left_cut:right_cut].copy()
    if roi.size == 0:
        return None, None, None, 0.0

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3,3), 0)

    # --- 1) locate bright horizontal baseline (white scale bar)
    row_mean = gray.mean(axis=1).astype(np.float32)
    H = gray.shape[0]
    top, bot = int(0.10*H), int(0.90*H)
    if bot <= top:
        return None, None, None, 0.0
    y0 = top + int(np.argmax(row_mean[top:bot]))

    # --- 2) crop a thin strip around the baseline (keeps ticks, drops text)
    strip_half = max(10, int(0.12 * H))  # ~24% of band height total
    yA = max(0, y0 - strip_half); yB = min(H, y0 + strip_half)
    strip = gray[yA:yB, :]

    # --- 3) enhance vertical bright lines (ticks)
    vker = cv.getStructuringElement(cv.MORPH_RECT, (1, 23))
    toph = cv.morphologyEx(strip, cv.MORPH_TOPHAT, vker)  # bright-on-dark
    _, bw = cv.threshold(toph, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # bridge hairline breaks
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3,3)))

    # --- 4) connected components -> adaptively choose "major" ticks
    num, labels, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)
    strip_h = strip.shape[0]

    xs, hs, ws = [], [], []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        # slender vertical candidates first
        if w > 9:                  # too wide -> likely text
            continue
        if h / max(w,1) < 4.0:     # not vertical enough
            continue
        if h < 0.25*strip_h:       # very short
            continue
        xs.append(x + w/2.0)
        hs.append(float(h))
        ws.append(float(w))

    if not xs:
        # nothing usable; bail to periodic fallback
        proj = bw.sum(axis=0).astype(np.float32)
        proj_s = cv.GaussianBlur(proj.reshape(1,-1),(1,11),0).ravel().astype(np.float32)
        rng = float(np.ptp(proj_s)); mn = float(np.min(proj_s))
        if rng < 1e-6:
            return None, None, None, 0.0
        p = (proj_s - mn) / (rng + 1e-6)
        ac = np.correlate(p, p, mode="full")[len(p)-1:]
        period, rawscore = _pick_period_with_harmonic(ac, lo=30, hi=180)
        if period is None:
            return None, p, None, 0.0
        tick_px = float(period)
        # preview peaks on the best phase
        sums = [np.sum(p[o::int(max(1,round(tick_px)))]) for o in range(int(max(1,round(tick_px))))]
        best_off = int(np.argmax(sums))
        peaks = list(np.arange(best_off, len(p), int(max(1,round(tick_px)))))
        score = float(sums[best_off] / (np.sum(p) + 1e-6))
        return tick_px, p, peaks, score

    xs = np.array(xs, dtype=np.float32)
    hs = np.array(hs, dtype=np.float32)

    # Adaptive height threshold: keep the top ~35% tallest candidates,
    # but never below 0.35*strip height.
    thr_h = max(0.35*strip_h, float(np.percentile(hs, 65)))
    majors = xs[hs >= thr_h]
    if majors.size < 3:
        # relax if too strict
        thr_h = max(0.30*strip_h, float(np.percentile(hs, 55)))
        majors = xs[hs >= thr_h]

    if majors.size < 3:
        # fallback to using all slender candidates
        majors = np.sort(xs)
    else:
        majors = np.sort(majors)

    # Robust median spacing
    diffs = np.diff(majors)
    if diffs.size < 2:
        return None, None, None, 0.0

    med = float(np.median(diffs))
    p20 = float(np.percentile(diffs, 20))
    good = diffs[(diffs > 0.5*med) & (diffs < 1.5*med)]
    if good.size >= 2:
        d_est = float(np.median(good))
    else:
        d_est = med

    # Safety: if we accidentally picked every other tick, diffs cluster near 2*true.
    # Heuristic: if med ~ 2 * p20 (within 20%), halve it.
    if p20 > 0 and abs(med/(2.0*p20) - 1.0) < 0.20:
        d_est *= 0.5

    tick_px = float(d_est)

    # --- Build a simple debug profile and lattice-aligned peaks
    proj = bw.sum(axis=0).astype(np.float32)
    proj_s = cv.GaussianBlur(proj.reshape(1,-1),(1,11),0).ravel().astype(np.float32)
    rng = float(np.ptp(proj_s)); mn = float(np.min(proj_s))
    p = (proj_s - mn) / (rng + 1e-6)

    # choose phase as the median remainder of majors modulo the period
    T = max(1, int(round(tick_px)))
    rema = (majors % T)
    # wrap rema around [0,T)
    off = int(round(float(np.median(rema))))
    peaks = list(np.arange(off, len(p), T))

    # score: how consistent majors are with the lattice
    if len(peaks) > 0:
        sampled = np.take(p, np.clip(peaks, 0, len(p)-1))
        score = float(np.sum(sampled) / (np.sum(p) + 1e-6))
    else:
        score = 0.0

    # sanity window for 1-cm spacing in large scans
    if not (12.0 <= tick_px <= 450.0):
        return None, p, np.array(peaks, dtype=int), 0.0

    return tick_px, p, np.array(peaks, dtype=int), score

def pixels_per_cm(img_bgr: np.ndarray):
    """
    1) Find the dark, texty band (ruler strip).
    2) Deskew, then estimate tick spacing with a harmonic-aware autocorr.
    Always emit proof images in `info`.
    """
    info: Dict = {}
    rects = _find_texty_dark_candidates(img_bgr)
    if not rects:
        info["reason"] = "No dark texty band found near the borders."
        return None, info

    best = None
    best_tick = None
    best_info = None
    best_score = -1.0

    for rect in rects:
        roi, rot, _ = _deskew_crop(img_bgr, rect)
        tick_px, prof, peaks, score = _tick_spacing_px_inside_band(roi)
        if tick_px is None:
            continue
        # super-wide plausibility range; this ruler image is large
        if not (12.0 <= tick_px <= 450.0):
            continue
        if score > best_score:
            best_score = score
            best = rect
            best_tick = tick_px
            best_info = (roi, prof, peaks)

    if best is None:
        info["reason"] = "No candidate had usable tick periodicity."
        return None, info

    roi, prof, peaks = best_info

    # Proof images
    roi_proof = roi.copy()
    if roi_proof.ndim == 2:
        roi_proof = cv.cvtColor(roi_proof, cv.COLOR_GRAY2BGR)
    for x in (peaks if peaks is not None else []):
        cv.line(roi_proof, (int(x), 0),
                (int(x), roi_proof.shape[0]-1), (0, 255, 255), 1)

    full_proof = img_bgr.copy()
    box = cv.boxPoints(best).astype(int)
    cv.polylines(full_proof, [box], True, (0, 255, 0), 2)

    info["roi_preview"] = roi_proof
    info["full_preview"] = full_proof
    info["tick_px"] = float(best_tick)
    info["periodicity_score"] = float(best_score)
    info["roi_shape"] = tuple(roi.shape)
    info["n_peaks"] = int(len(peaks) if peaks is not None else 0)

    # 1-cm ticks → px/cm
    return float(best_tick), info

# =============================
# SPECIMEN SEGMENTATION
# =============================

def _remove_border_touching(mask: np.ndarray) -> np.ndarray:
    # Keep specimen even if it touches the page edge; just trim a thin frame.
    out = mask.copy()
    b = 8
    out[:b, :] = 0; out[-b:, :] = 0; out[:, :b] = 0; out[:, -b:] = 0
    return out


def _is_light_rect_with_text(img_bgr: np.ndarray, cnt) -> bool:
    x, y, w, h = cv.boundingRect(cnt)
    if w * h < 2000:
        return False
    roi = img_bgr[y:y+h, x:x+w]
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    edges = cv.Canny(gray, 50, 150)
    density = float(np.count_nonzero(edges)) / (w * h + 1e-6)
    rect = cv.minAreaRect(cnt)
    (cx, cy), (rw, rh), _ = rect
    long_side, short_side = (max(rw, rh), min(rw, rh))
    rectangularity = (cv.contourArea(cnt) / (w * h + 1e-6))
    aspect = long_side / max(short_side, 1e-6)
    return (mean > 180 and density > 0.02 and rectangularity > 0.75 and aspect > 2.0)


def segment_specimen(img_bgr: np.ndarray) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    m = int(round(0.015 * min(H, W)))
    crop = img_bgr[m:H-m, m:W-m].copy()
    lab = cv.cvtColor(crop, cv.COLOR_BGR2LAB)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    L_inv = cv.normalize(255 - L, None, 0, 255, cv.NORM_MINMAX)
    chroma = cv.absdiff(A, 128) + cv.absdiff(B, 128)
    mask = ((L_inv > 25) | (chroma > 12)).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7,7), np.uint8))
    full = np.zeros((H, W), np.uint8); full[m:H-m, m:W-m] = mask
    full = _remove_border_touching(full)

    cnts, _ = cv.findContours(full, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rm = np.zeros_like(full)
    for c in cnts:
        if _is_light_rect_with_text(img_bgr, c):
            cv.drawContours(rm, [c], -1, 255, -1)
    cleaned = cv.bitwise_and(full, cv.bitwise_not(rm))
    return cleaned

# =============================
# LEAF MEASUREMENT (simple baseline)
# =============================

def _is_leaf_like(cnt) -> bool:
    area = cv.contourArea(cnt)
    if area < 1200:
        return False
    per = cv.arcLength(cnt, True)
    if per < 100:
        return False

    rect = cv.minAreaRect(cnt)
    (_, _), (w, h), _ = rect
    long_side, short_side = (max(w, h), min(w, h))
    aspect = long_side / max(short_side, 1e-6)
    if aspect > 12.0:
        return False

    hull = cv.convexHull(cnt)
    solidity = cv.contourArea(cnt) / (cv.contourArea(hull) + 1e-6)
    if solidity < 0.70:
        return False

    x,y,w_b,h_b = cv.boundingRect(cnt)
    rectangularity = cv.contourArea(cnt) / (float(w_b*h_b) + 1e-6)
    if rectangularity > 0.92 and aspect < 3.0:
        return False
    return True


def measure_leaves(img_bgr: np.ndarray, px_per_cm: float, mask: Optional[np.ndarray]=None):
    if mask is None:
        mask = segment_specimen(img_bgr)

    # crude separation of connected parts
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, se)

    num, labels, stats, _ = cv.connectedComponentsWithStats(opened, connectivity=8)
    vis = img_bgr.copy()
    recs: List[dict] = []
    leaf_id = 1

    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area < 1200:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv.contourArea)
        if not _is_leaf_like(c):
            continue

        rect = cv.minAreaRect(c)
        (cx,cy),(w2,h2),_ = rect
        long_side, short_side = (max(w2,h2), min(w2,h2))
        box = cv.boxPoints(rect).astype(int)
        cv.polylines(vis, [box], True, (36,255,12), 2)
        cv.putText(vis, str(leaf_id), (int(cx), int(cy)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        area_px = cv.contourArea(c)
        recs.append(dict(
            id=leaf_id,
            area_cm2=float(area_px)/(px_per_cm**2),
            length_cm=float(long_side)/px_per_cm,
            width_cm=float(short_side)/px_per_cm,
            centroid_x_px=float(cx),
            centroid_y_px=float(cy)
        ))
        leaf_id += 1

    return recs, vis, opened

# =============================
# I/O HELPERS
# =============================

def _save_proofs(info: Dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if "roi_preview" in info and isinstance(info["roi_preview"], np.ndarray):
        cv.imwrite(os.path.join(out_dir, "ruler_roi.png"), info["roi_preview"])
    if "full_preview" in info and isinstance(info["full_preview"], np.ndarray):
        cv.imwrite(os.path.join(out_dir, "ruler_on_image.png"), info["full_preview"])
    with open(os.path.join(out_dir, "calibration.txt"), "w", encoding="utf-8") as f:
        for k,v in info.items():
            if isinstance(v, np.ndarray):
                continue
            f.write(f"{k}: {v}\n")


def _write_mask(mask: np.ndarray, out_dir: str):
    cv.imwrite(os.path.join(out_dir, "mask.png"), mask)


def _write_detections(vis: np.ndarray, out_dir: str):
    cv.imwrite(os.path.join(out_dir, "leaf_detections.png"), vis)


def _write_csv(records: List[dict], out_dir: str):
    if not records:
        return
    df = pd.DataFrame.from_records(records).sort_values("id")
    df.to_csv(os.path.join(out_dir, "measurements.csv"), index=False)

# =============================
# CLI
# =============================

def main():
    ap = argparse.ArgumentParser(description="Herbarium leaf measurement with ruler proof images.")
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                    help=f"Path to input image (default: {DEFAULT_IMAGE})")
    ap.add_argument("--out", default="./out", help="Output directory")
    ap.add_argument("--px-per-cm", type=float, default=None,
                    help="Bypass auto-calibration with known px/cm")
    ap.add_argument("--two-points", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"),
                    help="Fallback: two pixel coords on ruler")
    ap.add_argument("--cm", type=float, help="Distance in cm for --two-points")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    img = cv.imread(args.image, cv.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    # Calibration (always write proofs)
    pxcm = args.px_per_cm
    info = {}
    if pxcm is None:
        pxcm, info = pixels_per_cm(img)
        if (pxcm is None) and (args.two_points is not None) and (args.cm is not None):
            x1,y1,x2,y2 = map(int, args.two_points)
            pxcm = float(np.hypot(x2-x1, y2-y1)) / float(args.cm)
            info["fallback"] = "two_points"
            info["tick_px"] = pxcm

    _save_proofs(info, args.out)

    if pxcm is None:
        print("Calibration FAILED.")
        if "reason" in info: print("Reason:", info["reason"])
        print("Tip: supply --px-per-cm or --two-points ... --cm ...")
        raise SystemExit(1)

    if not (12.0 <= pxcm <= 450.0):
        print(f"Suspicious px/cm = {pxcm:.2f}. Aborting; check ruler detection.")
        raise SystemExit(1)

    print(f"[OK] Pixels per centimeter: {pxcm:.6f} (periodicity={info.get('periodicity_score','?')}, peaks={info.get('n_peaks','?')})")

    # Segmentation + measurement
    mask = segment_specimen(img)
    _write_mask(mask, args.out)

    records, vis, _ = measure_leaves(img, pxcm, mask=mask)
    _write_detections(vis, args.out)
    _write_csv(records, args.out)

    print(f"[OK] Measured {len(records)} leaves.")
    if records:
        areas = [r['area_cm2'] for r in records]
        lengths = [r['length_cm'] for r in records]
        widths  = [r['width_cm']  for r in records]
        print(f"  area_cm2: mean={np.mean(areas):.2f}, median={np.median(areas):.2f}")
        print(f"  length_cm: mean={np.mean(lengths):.2f}, median={np.median(lengths):.2f}")
        print(f"  width_cm:  mean={np.mean(widths):.2f}, median={np.median(widths):.2f}")
    print(f"Outputs: {os.path.abspath(args.out)}")
    print("  - ruler_roi.png (tick peaks) & ruler_on_image.png (band outline)")
    print("  - mask.png, leaf_detections.png, measurements.csv")

if __name__ == "__main__":
    main()
