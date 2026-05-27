#!/usr/bin/env python3
"""
VISUAL METRICS EXTRACTION SCRIPT
Adapted from Styliani Katsoulis

DESCRIPTION
-----------
This script processes images and computes visual features spanning intensity, 
contrast, color, texture, frequency, geometry, and complexity.

Usage:
  python src/extract_visual_features.py --stimuli-dir rendered_images/
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.segmentation import slic
from skimage.color import rgb2lab
from scipy.fft import fft2, fftshift
from math import log2

# Try to import Pillow for color counting
HAS_PIL = True
try:
    from PIL import Image
except Exception:
    HAS_PIL = False

# Try to import openpyxl (for Excel coloring); script still works if missing
HAS_OPENPYXL = True
try:
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter
except Exception:
    HAS_OPENPYXL = False

# ================================
# CONFIG
# ================================
# GLCM configuration
GLCM_DISTANCES = [1]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS    = 256

# Explicit, stable column order
EXPECTED_COLS = [
    "filename",
    "brightness","contrast","sharpness",
    "colorfulness","mean_R","mean_G","mean_B","saturation","num_colours",
    "image_entropy","edge_density",
    "glcm_energy","glcm_entropy","glcm_contrast","glcm_homogeneity","glcm_correlation",
    "self_similarity",
    "power_db","spectral_power_db","anisotropy",
    "feature_congestion",
    "subband_entropy",
    "num_photo_objects",
    "MIG_h","MIG_s","MIG_v","MIG_mean",
    "mean_gradient_strength",
    "center_offset","object_fraction","fraction_plate_covered"
]

# ================================
# UTILS
# ================================
def safe_entropy_from_hist(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def image_entropy_gray(gray):
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    return safe_entropy_from_hist(hist)

def gaussian_local_std(img, ksize=7, sigma=1.5):
    blur  = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    blur2 = cv2.GaussianBlur(img*img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(0.0, blur2 - blur*blur)
    return np.sqrt(var)

def quantize_to_bins(arr01, bins=11):
    q = np.clip((arr01 * (bins - 1)).round().astype(np.int32), 0, bins-1)
    return q

def shannon_entropy_from_values(vals, bins):
    hist = np.bincount(vals.ravel(), minlength=bins).astype(np.float64)
    hist /= hist.sum() if hist.sum() > 0 else 1.0
    return safe_entropy_from_hist(hist)

# ================================
# METRIC FUNCTIONS
# ================================
def compute_brightness(gray):
    return float(np.mean(gray))

def compute_contrast(gray):
    return float(np.std(gray))

def compute_sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_colorfulness(image_bgr):
    B, G, R = cv2.split(image_bgr.astype("float32"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3*np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

def compute_mean_rgb(image_bgr):
    B, G, R = cv2.split(image_bgr)
    return float(np.mean(R)), float(np.mean(G)), float(np.mean(B))

def compute_saturation(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))

def compute_edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).mean())

def compute_glcm_features(gray):
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    glcm = graycomatrix(gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    energy       = float(np.mean(graycoprops(glcm, 'energy')))
    contrast     = float(np.mean(graycoprops(glcm, 'contrast')))
    homogeneity  = float(np.mean(graycoprops(glcm, 'homogeneity')))
    correlation  = float(np.mean(graycoprops(glcm, 'correlation')))
    nz = glcm[glcm > 0]
    glcm_entropy = float(-np.sum(nz * np.log2(nz)))
    return energy, glcm_entropy, contrast, homogeneity, correlation

def compute_spectral_power(gray):
    f = fft2(gray)
    fshift = fftshift(f)
    power = np.abs(fshift)**2
    return float(10 * np.log10(np.mean(power) + 1e-8))

def compute_anisotropy(gray, bins=18):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
    mag = np.sqrt(gx**2 + gy**2)
    ori = (np.rad2deg(np.arctan2(gy, gx)) % 180)
    hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)
    s = hist.sum()
    return float(np.std(hist / s)) if s > 0 else np.nan

def compute_center_offset(gray, shape, thr=240):
    _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    m = cv2.moments(th)
    if m["m00"] == 0:
        return np.nan
    cx, cy = m["m10"]/m["m00"], m["m01"]/m["m00"]
    return float(np.hypot(cx - shape[1]/2, cy - shape[0]/2))

def compute_object_fraction(gray, thr=240):
    _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    return float((th > 0).mean())

def compute_num_colours(image_bgr):
    if not HAS_PIL:
        return np.nan
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    q = pil.convert("P", palette=Image.ADAPTIVE, colors=256)
    arr = np.array(q)
    return int(np.unique(arr).size)

def compute_mean_gradient_strength(image_bgr):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mags = []
    for c in range(3):
        gx = cv2.Sobel(lab[:,:,c], cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(lab[:,:,c], cv2.CV_32F, 0, 1, 3)
        mags.append(np.sqrt(gx*gx + gy*gy))
    mag_max = np.maximum(np.maximum(mags[0], mags[1]), mags[2])
    return float(np.mean(mag_max))

def compute_self_similarity(gray, levels=3, orientations=8):
    h, w = gray.shape
    ss_vals = []
    for l in range(levels):
        cells = 2**l
        cell_h = h // cells
        cell_w = w // cells
        hists = []
        for i in range(cells):
            for j in range(cells):
                y0, y1 = i*cell_h, (i+1)*cell_h
                x0, x1 = j*cell_w, (j+1)*cell_w
                tile = gray[y0:y1, x0:x1]
                if tile.size < 64*64:
                    tile = cv2.resize(tile, (max(64, tile.shape[1]), max(64, tile.shape[0])), interpolation=cv2.INTER_LINEAR)
                feat = hog(tile, orientations=orientations, pixels_per_cell=(16,16), cells_per_block=(1,1), feature_vector=True)
                s = feat.sum()
                if s > 0:
                    feat = feat / s
                hists.append(feat)
        m = len(hists)
        if m < 2: continue
        inter_sum = 0.0
        cnt = 0
        for a in range(m):
            ha = hists[a]
            for b in range(a+1, m):
                hb = hists[b]
                k = np.minimum(ha, hb).sum()
                inter_sum += float(k)
                cnt += 1
        ss_vals.append(inter_sum / max(cnt,1))
    return float(np.mean(ss_vals)) if ss_vals else np.nan

def compute_feature_congestion(image_bgr):
    lab = rgb2lab(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).astype(np.float32)
    L  = (lab[:,:,0] / 100.0).clip(0,1)
    a  = ((lab[:,:,1] + 128.0) / 255.0).clip(0,1)
    b  = ((lab[:,:,2] + 128.0) / 255.0).clip(0,1)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    gmag = np.sqrt(gx*gx + gy*gy)
    scales = [(7,1.0),(11,2.0),(15,4.0)]
    fc_maps = []
    for ksize, sigma in scales:
        stdL = gaussian_local_std(L, ksize=ksize, sigma=sigma)
        stdC = np.sqrt(gaussian_local_std(a, ksize, sigma)**2 + gaussian_local_std(b, ksize, sigma)**2)
        stdO = gaussian_local_std(gmag, ksize, sigma)
        fc_maps.append(stdL + stdC + stdO)
    fc = np.mean(fc_maps)
    return float(np.mean(fc))

def compute_subband_entropy(image_bgr):
    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    entropies = []
    for ch in range(3):
        im = ycc[:,:,ch]
        im0 = im.copy()
        for lvl in range(3):
            down = cv2.pyrDown(im0)
            up   = cv2.pyrUp(down, dstsize=(im0.shape[1], im0.shape[0]))
            band = cv2.absdiff(im0, up)
            b = np.clip((band - band.min()) / (band.max() - band.min() + 1e-8), 0, 1)
            hist, _ = np.histogram((b*255).astype(np.uint8), bins=256, range=(0,256), density=True)
            entropies.append(safe_entropy_from_hist(hist))
            im0 = down
    return float(np.mean(entropies)) if entropies else np.nan

def compute_num_photo_objects(image_bgr, n_segments=200, compactness=10, sigma=1):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    labels = slic(rgb, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)
    return int(np.unique(labels).size)

def compute_MIG_channels(image_bgr, bins=11):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = (hsv[:,:,0] / 179.0).clip(0,1)
    S = (hsv[:,:,1] / 255.0).clip(0,1)
    V = (hsv[:,:,2] / 255.0).clip(0,1)
    def mig_of(ch01):
        q = quantize_to_bins(ch01, bins=bins)
        H0 = shannon_entropy_from_values(q, bins)
        q_r = np.roll(q, -1, axis=1)
        joint = q * bins + q_r
        Hxy = shannon_entropy_from_values(joint, bins*bins)
        mig = (Hxy - H0) / max(log2(bins), 1e-8)
        return float(np.clip(mig, 0.0, 1.0))
    mig_h, mig_s, mig_v = mig_of(H), mig_of(S), mig_of(V)
    return mig_h, mig_s, mig_v, float(np.mean([mig_h, mig_s, mig_v]))

# ================================
# EXCEL STYLING
# ================================
def style_excel_columns(xlsx_path):
    if not HAS_OPENPYXL: return
    wb = load_workbook(xlsx_path)
    ws = wb.active
    groups = {
        "filename": (["filename"], "D9D9D9"),
        "intensity": (["brightness","contrast","sharpness","mean_gradient_strength"], "C6E0B4"),
        "color": (["colorfulness","mean_R","mean_G","mean_B","saturation","num_colours"], "F8CBAD"),
        "texture": (["image_entropy","edge_density","glcm_energy","glcm_entropy","glcm_contrast","glcm_homogeneity","glcm_correlation","self_similarity"], "BDD7EE"),
        "frequency": (["power_db","spectral_power_db","anisotropy"], "B7DEE8"),
        "complexity": (["feature_congestion","subband_entropy","num_photo_objects","MIG_h","MIG_s","MIG_v","MIG_mean"], "FFE699"),
        "geometry": (["center_offset","object_fraction","fraction_plate_covered"], "D9C2E9"),
    }
    header_row = 1
    headers = {cell.value: idx+1 for idx, cell in enumerate(ws[header_row]) if cell.value is not None}
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"
    for cols, color in groups.values():
        fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        for colname in cols:
            if colname not in headers: continue
            col_letter = get_column_letter(headers[colname])
            cell = ws[f"{col_letter}{header_row}"]
            cell.fill, cell.font = fill, Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
    for name, idx in headers.items():
        ws.column_dimensions[get_column_letter(idx)].width = max(12, len(str(name)) + 4)
    wb.save(xlsx_path)

# ================================
# MAIN
# ================================
def main():
    parser = argparse.ArgumentParser(description="Extract low-level visual features from images.")
    parser.add_argument("--stimuli-dir", type=str, required=True, help="Folder containing images.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to save CSV output. Defaults to <stimuli-dir>/visual_metrics.csv")
    parser.add_argument("--merge-canonical", action="store_true", help="Merge results into the canonical CSV in ../data/")
    args = parser.parse_args()

    stimuli_dir = Path(args.stimuli_dir).resolve()
    if not args.output_csv:
        output_csv = stimuli_dir / "visual_metrics.csv"
    else:
        output_csv = Path(args.output_csv)

    print(f"[INFO] Processing images in: {stimuli_dir}")
    files = sorted([f for f in os.listdir(stimuli_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])
    print(f"[INFO] Found {len(files)} images")

    results = []
    for fname in tqdm(files):
        path = stimuli_dir / fname
        img = cv2.imread(str(path))
        if img is None: continue
        row = {c: np.nan for c in EXPECTED_COLS}; row["filename"] = fname
        try: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except: results.append(row); continue
        try: row["brightness"] = compute_brightness(gray)
        except: pass
        try: row["contrast"] = compute_contrast(gray)
        except: pass
        try: row["sharpness"] = compute_sharpness(gray)
        except: pass
        try: row["colorfulness"] = compute_colorfulness(img)
        except: pass
        try:
            r, g, b = compute_mean_rgb(img)
            row["mean_R"], row["mean_G"], row["mean_B"] = r, g, b
        except: pass
        try: row["saturation"] = compute_saturation(img)
        except: pass
        try: row["num_colours"] = compute_num_colours(img)
        except: pass
        try: row["image_entropy"] = image_entropy_gray(gray)
        except: pass
        try: row["edge_density"] = compute_edge_density(gray)
        except: pass
        try:
            ge, gent, gc, gh, gcor = compute_glcm_features(gray)
            row["glcm_energy"], row["glcm_entropy"], row["glcm_contrast"], row["glcm_homogeneity"], row["glcm_correlation"] = ge, gent, gc, gh, gcor
        except: pass
        try: row["self_similarity"] = compute_self_similarity(gray)
        except: pass
        try:
            p_db = compute_spectral_power(gray)
            row["spectral_power_db"] = row["power_db"] = p_db
        except: pass
        try: row["anisotropy"] = compute_anisotropy(gray)
        except: pass
        try: row["feature_congestion"] = compute_feature_congestion(img)
        except: pass
        try: row["subband_entropy"] = compute_subband_entropy(img)
        except: pass
        try: row["num_photo_objects"] = compute_num_photo_objects(img)
        except: pass
        try:
            mh, ms, mv, mmean = compute_MIG_channels(img)
            row["MIG_h"], row["MIG_s"], row["MIG_v"], row["MIG_mean"] = mh, ms, mv, mmean
        except: pass
        try: row["mean_gradient_strength"] = compute_mean_gradient_strength(img)
        except: pass
        try: row["center_offset"] = compute_center_offset(gray, img.shape)
        except: pass
        try:
            frac = compute_object_fraction(gray)
            row["object_fraction"] = row["fraction_plate_covered"] = frac
        except: pass
        results.append(row)

    df_metrics = pd.DataFrame(results, columns=EXPECTED_COLS)
    df_metrics.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved CSV: {output_csv}")

    if HAS_OPENPYXL:
        output_xlsx = output_csv.with_suffix(".xlsx")
        df_metrics.to_excel(output_xlsx, index=False, engine="openpyxl")
        style_excel_columns(output_xlsx)
        print(f"[OK] Saved styled Excel: {output_xlsx}")

    if args.merge_canonical:
        src_dir = Path(__file__).resolve().parent
        canonical_path = src_dir.parent / "data" / "Foodpictures_information_dynamic.csv"
        if canonical_path.exists():
            print(f"[INFO] Merging results into: {canonical_path}")
            try:
                df_can = pd.read_csv(canonical_path)
                if "filename" not in df_can.columns:
                    print(f"[ERROR] 'filename' column missing in {canonical_path}. Merge failed.")
                else:
                    # Left join on filename
                    df_merged = df_can.merge(df_metrics, on="filename", how="left", suffixes=('', '_new'))
                    # Save back
                    df_merged.to_csv(canonical_path, index=False)
                    print("[OK] Merged results into canonical CSV.")
            except Exception as e:
                print(f"[ERROR] Could not merge into canonical: {e}")
        else:
            print(f"[WARN] Canonical CSV not found at {canonical_path}. Merge skipped.")

if __name__ == "__main__":
    main()
