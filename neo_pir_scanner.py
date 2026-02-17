from __future__ import annotations

import io
import csv
import math
import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, List, Generator, Optional

import numpy as np
import cv2
from PIL import Image
import streamlit as st

import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import csv
import io
import os

def load_scoring_key_from_csv(path: str = "scoring_key.csv") -> dict[int, list[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier '{path}' introuvable. Ajoute scoring_key.csv à la racine du projet."
        )

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key: dict[int, list[int]] = {}
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]

    # vérifs
    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet, items manquants: {missing[:20]}")

    bad = [i for i,v in key.items() if len(v) != 5 or any(x < 0 or x > 4 for x in v)]
    if bad:
        raise ValueError(f"scoring_key.csv invalide, items problématiques: {bad[:20]}")

    return key


# ============================================================
# 0) CONSTANTES / LABELS
# ============================================================

OPTIONS = ["FD", "D", "N", "A", "FA"]  # indices 0..4
N_OPTION_INDEX = 2

domain_labels = {
    "N": "Névrosisme",
    "E": "Extraversion",
    "O": "Ouverture",
    "A": "Agréabilité",
    "C": "Conscience",
}

facet_labels = {
    "N1": "N1 - Anxiété", "N2": "N2 - Hostilité colérique", "N3": "N3 - Dépression",
    "N4": "N4 - Timidité", "N5": "N5 - Impulsivité", "N6": "N6 - Vulnérabilité",
    "E1": "E1 - Chaleur", "E2": "E2 - Grégarité", "E3": "E3 - Affirmation de soi",
    "E4": "E4 - Activité", "E5": "E5 - Recherche d'excitation", "E6": "E6 - Émotions positives",
    "O1": "O1 - Imagination", "O2": "O2 - Esthétique", "O3": "O3 - Sentiments",
    "O4": "O4 - Actions", "O5": "O5 - Idées", "O6": "O6 - Valeurs",
    "A1": "A1 - Confiance", "A2": "A2 - Franchise", "A3": "A3 - Altruisme",
    "A4": "A4 - Conformité", "A5": "A5 - Modestie", "A6": "A6 - Tendresse",
    "C1": "C1 - Compétence", "C2": "C2 - Ordre", "C3": "C3 - Sens du devoir",
    "C4": "C4 - Effort pour réussir", "C5": "C5 - Autodiscipline", "C6": "C6 - Délibération",
}
scoring_key = load_scoring_key_from_csv("scoring_key.csv")

}

# ============================================================
# 2) ITEMS -> FACETS (généré)
# ============================================================
facet_bases = {
    "N1": [1], "N2": [6], "N3": [11], "N4": [16], "N5": [21], "N6": [26],
    "E1": [2], "E2": [7], "E3": [12], "E4": [17], "E5": [22], "E6": [27],
    "O1": [3], "O2": [8], "O3": [13], "O4": [18], "O5": [23], "O6": [28],
    "A1": [4], "A2": [9], "A3": [14], "A4": [19], "A5": [24], "A6": [29],
    "C1": [5], "C2": [10], "C3": [15], "C4": [20], "C5": [25], "C6": [30],
}

item_to_facet: Dict[int, str] = {}
for fac, bases in facet_bases.items():
    for b in bases:
        for k in range(0, 240, 30):
            item_to_facet[b + k] = fac

facet_to_domain = {
    **{f"N{i}": "N" for i in range(1, 7)},
    **{f"E{i}": "E" for i in range(1, 7)},
    **{f"O{i}": "O" for i in range(1, 7)},
    **{f"A{i}": "A" for i in range(1, 7)},
    **{f"C{i}": "C" for i in range(1, 7)},
}


# ============================================================
# 3) PROTOCOLE
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = N_OPTION_INDEX


# ============================================================
# 4) NORMES (T-scores, percentiles) - sans SciPy
# ============================================================
@dataclass
class NormRow:
    scale_type: str   # "facet" | "domain"
    scale: str        # "N".."C" or "N1".."C6"
    sex: str          # "M" | "F"
    age_min: int
    age_max: int
    mean: float
    sd: float


def normal_cdf(z: float) -> float:
    # Phi(z) via erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def percentile_from_t(t: float) -> float:
    # T-score -> z = (t-50)/10
    z = (t - 50.0) / 10.0
    return 100.0 * normal_cdf(z)


def load_norms_csv(file_bytes: bytes) -> List[NormRow]:
    text = file_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    out: List[NormRow] = []
    required = {"scale_type", "scale", "sex", "age_min", "age_max", "mean", "sd"}
    if set(reader.fieldnames or []) & required != required:
        raise ValueError("norms.csv: colonnes requises: scale_type,scale,sex,age_min,age_max,mean,sd")

    for row in reader:
        out.append(NormRow(
            scale_type=row["scale_type"].strip(),
            scale=row["scale"].strip(),
            sex=row["sex"].strip().upper(),
            age_min=int(row["age_min"]),
            age_max=int(row["age_max"]),
            mean=float(row["mean"]),
            sd=float(row["sd"]),
        ))
    return out


def find_norm(norms: List[NormRow], scale_type: str, scale: str, sex: str, age: int) -> Optional[NormRow]:
    sex = sex.upper()
    candidates = [
        n for n in norms
        if n.scale_type == scale_type and n.scale == scale and n.sex == sex and (n.age_min <= age <= n.age_max)
    ]
    if not candidates:
        return None
    # If multiple, pick the narrowest age band
    candidates.sort(key=lambda r: (r.age_max - r.age_min))
    return candidates[0]


def raw_to_t(raw: float, mean: float, sd: float) -> float:
    if sd <= 0:
        return 50.0
    return 50.0 + 10.0 * ((raw - mean) / sd)


# ============================================================
# 5) VISION (rotation + warp + bbox + flexible grid + micro-adjust)
# ============================================================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_keep_ratio(bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= 0:
        return bgr
    new_h = int(h * (target_width / float(w)))
    return cv2.resize(bgr, (target_width, max(1, new_h)))


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform_cv(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(1, int(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(1, int(max(heightA, heightB)))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def find_document_warp(bgr: np.ndarray, target_width: int = 1800) -> np.ndarray:
    resized = resize_keep_ratio(bgr, target_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return resized

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    doc = None
    for c in cnts[:12]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break

    if doc is None:
        # fallback: bounding rect of the largest contour
        x, y, w, h = cv2.boundingRect(cnts[0])
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")
        return four_point_transform_cv(resized, pts)

    return four_point_transform_cv(resized, doc.reshape(4, 2))


def apply_clahe(gray: np.ndarray, clip: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize_for_marks(bgr: np.ndarray, mode: str = "photo") -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray, clip=2.0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if mode == "scan":
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thr

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    return thr


def rotate_bgr(bgr: np.ndarray, rot: int) -> np.ndarray:
    if rot == 0:
        return bgr
    if rot == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rot must be 0/90/180/270")


def find_table_bbox_robust(
    thr_inv: np.ndarray,
    min_w_ratio: float,
    min_h_ratio: float,
    kernel_div: int,
    dilate_size: int,
) -> Tuple[int, int, int, int]:
    H, W = thr_inv.shape[:2]

    def ok(w: int, h: int) -> bool:
        return (w >= W * min_w_ratio) and (h >= H * min_h_ratio)

    hk = max(10, W // max(10, kernel_div))
    vk = max(10, H // max(10, kernel_div))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    hor = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    ver = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size)), iterations=1)

    cnts, _ = cv2.findContours(grid.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(cnts[0])
        if ok(w, h):
            return x, y, w, h

    cnts2, _ = cv2.findContours(thr_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts2:
        return 0, 0, W, H

    cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)
    for c in cnts2[:20]:
        x, y, w, h = cv2.boundingRect(c)
        if ok(w, h):
            return x, y, w, h

    x, y, w, h = cv2.boundingRect(cnts2[0])
    return x, y, w, h


def score_table_candidate(thr_inv: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    H, W = thr_inv.shape[:2]
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return -1e9

    area_ratio = (w * h) / float(W * H)
    crop = thr_inv[y:y+h, x:x+w]
    if crop.size == 0:
        return -1e9

    hk = max(10, crop.shape[1] // 18)
    vk = max(10, crop.shape[0] // 18)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    hor = cv2.morphologyEx(crop, cv2.MORPH_OPEN, h_kernel, iterations=1)
    ver = cv2.morphologyEx(crop, cv2.MORPH_OPEN, v_kernel, iterations=1)
    grid = cv2.bitwise_or(hor, ver)
    grid_energy = np.count_nonzero(grid) / float(grid.size)

    cx = x + w / 2.0
    cy = y + h / 2.0
    center_dist = ((cx - W/2.0)**2 + (cy - H/2.0)**2) ** 0.5 / ((W**2 + H**2) ** 0.5)

    return (2.5 * area_ratio) + (3.0 * grid_energy) - (1.0 * center_dist)


def auto_rotate_and_warp(
    bgr: np.ndarray,
    target_width: int,
    mode: str,
    min_w_ratio: float,
    min_h_ratio: float,
    kernel_div: int,
    dilate_size: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], int]:
    best = None
    for rot in (0, 90, 180, 270):
        bgr_r = rotate_bgr(bgr, rot)
        warped = find_document_warp(bgr_r, target_width=target_width)
        thr = binarize_for_marks(warped, mode=mode)
        bbox = find_table_bbox_robust(thr, min_w_ratio, min_h_ratio, kernel_div, dilate_size)
        s = score_table_candidate(thr, bbox)
        cand = (s, warped, thr, bbox, rot)
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None
    _, warped_best, thr_best, bbox_best, rot_best = best
    return warped_best, thr_best, bbox_best, rot_best


def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, int(k) | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def pick_peaks(proj: np.ndarray, n_peaks: int, min_dist: int) -> List[int]:
    proj = proj.copy()
    peaks: List[int] = []
    for _ in range(n_peaks):
        i = int(np.argmax(proj))
        if proj[i] <= 0:
            break
        peaks.append(i)
        lo = max(0, i - min_dist)
        hi = min(len(proj), i + min_dist + 1)
        proj[lo:hi] = 0
    return sorted(peaks)


def grid_lines_from_bbox(
    thr_inv: np.ndarray,
    bbox: Tuple[int, int, int, int],
    rows: int = 30,
    cols: int = 8,
    kernel_div: int = 18,
) -> Tuple[List[int], List[int], np.ndarray]:
    x, y, w, h = bbox
    crop = thr_inv[y:y+h, x:x+w]
    if crop.size == 0:
        raise ValueError("Empty table crop")

    hk = max(10, crop.shape[1] // max(10, kernel_div))
    vk = max(10, crop.shape[0] // max(10, kernel_div))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    hor = cv2.morphologyEx(crop, cv2.MORPH_OPEN, h_kernel, iterations=1)
    ver = cv2.morphologyEx(crop, cv2.MORPH_OPEN, v_kernel, iterations=1)
    grid = cv2.bitwise_or(hor, ver)

    proj_x = np.sum(ver > 0, axis=0)
    proj_y = np.sum(hor > 0, axis=1)

    proj_x = smooth_1d(proj_x, k=max(9, w // 120))
    proj_y = smooth_1d(proj_y, k=max(9, h // 120))

    min_dx = max(8, w // (cols * 4))
    min_dy = max(6, h // (rows * 4))

    xs = pick_peaks(proj_x, n_peaks=cols+1, min_dist=min_dx)
    ys = pick_peaks(proj_y, n_peaks=rows+1, min_dist=min_dy)

    if len(xs) < cols+1 or len(ys) < rows+1:
        raise ValueError("Not enough grid lines detected")

    xs = sorted([int(v) for v in xs[:cols+1]])
    ys = sorted([int(v) for v in ys[:rows+1]])

    if (xs[-1] - xs[0]) < w * 0.65 or (ys[-1] - ys[0]) < h * 0.65:
        raise ValueError("Grid span too small")

    xs_abs = [x + v for v in xs]
    ys_abs = [y + v for v in ys]

    grid_vis = cv2.cvtColor(crop.copy(), cv2.COLOR_GRAY2BGR)
    for xv in xs:
        cv2.line(grid_vis, (xv, 0), (xv, h-1), (255, 0, 0), 1)
    for yv in ys:
        cv2.line(grid_vis, (0, yv), (w-1, yv), (0, 255, 0), 1)

    return xs_abs, ys_abs, grid_vis


def split_grid_uniform(bbox: Tuple[int, int, int, int], rows: int = 30, cols: int = 8) -> Generator[Tuple[int, int, Tuple[int, int, int, int]], None, None]:
    x, y, w, h = bbox
    cell_w = w / cols
    cell_h = h / rows
    for r in range(rows):
        for c in range(cols):
            cx1 = int(x + c * cell_w)
            cy1 = int(y + r * cell_h)
            cx2 = int(x + (c + 1) * cell_w)
            cy2 = int(y + (r + 1) * cell_h)
            yield r, c, (cx1, cy1, cx2, cy2)


def split_grid_flexible(xs: List[int], ys: List[int], rows: int = 30, cols: int = 8) -> Generator[Tuple[int, int, Tuple[int, int, int, int]], None, None]:
    xs = xs[:cols+1]
    ys = ys[:rows+1]
    for r in range(rows):
        for c in range(cols):
            cx1, cx2 = xs[c], xs[c+1]
            cy1, cy2 = ys[r], ys[r+1]
            yield r, c, (int(cx1), int(cy1), int(cx2), int(cy2))


def build_grid_mask(thr_inv_crop: np.ndarray, kernel_div: int = 18) -> np.ndarray:
    h, w = thr_inv_crop.shape[:2]
    hk = max(10, w // max(10, kernel_div))
    vk = max(10, h // max(10, kernel_div))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    hor = cv2.morphologyEx(thr_inv_crop, cv2.MORPH_OPEN, h_kernel, iterations=1)
    ver = cv2.morphologyEx(thr_inv_crop, cv2.MORPH_OPEN, v_kernel, iterations=1)
    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return grid


def border_energy(grid_mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, band: int = 2) -> float:
    h, w = grid_mask.shape[:2]
    x1 = max(0, min(w - 2, x1))
    y1 = max(0, min(h - 2, y1))
    x2 = max(x1 + 2, min(w, x2))
    y2 = max(y1 + 2, min(h, y2))
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return -1e9

    top = grid_mask[y1:y1 + band, x1:x2]
    bot = grid_mask[max(y2 - band, y1):y2, x1:x2]
    left = grid_mask[y1:y2, x1:x1 + band]
    right = grid_mask[y1:y2, max(x2 - band, x1):x2]

    s = np.count_nonzero(top) + np.count_nonzero(bot) + np.count_nonzero(left) + np.count_nonzero(right)
    denom = float(top.size + bot.size + left.size + right.size)
    return s / max(1.0, denom)


def refine_cell_bbox(grid_mask: np.ndarray, cell: Tuple[int, int, int, int], search: int, band: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = cell
    best_score = border_energy(grid_mask, x1, y1, x2, y2, band=band)
    best_cell = cell

    for dx1 in range(-search, search + 1):
        for dx2 in range(-search, search + 1):
            for dy1 in range(-search, search + 1):
                for dy2 in range(-search, search + 1):
                    nx1 = x1 + dx1
                    nx2 = x2 + dx2
                    ny1 = y1 + dy1
                    ny2 = y2 + dy2
                    if (nx2 - nx1) < 18 or (ny2 - ny1) < 18:
                        continue
                    sc = border_energy(grid_mask, nx1, ny1, nx2, ny2, band=band)
                    if sc > best_score:
                        best_score = sc
                        best_cell = (nx1, ny1, nx2, ny2)
    return best_cell


def item_id_from_rc(r: int, c: int) -> int:
    return (r + 1) + 30 * c


def option_rois_in_cell(cell: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = cell
    w = x2 - x1
    h = y2 - y1

    left = x1 + int(0.18 * w)
    right = x2 - int(0.05 * w)
    top = y1 + int(0.18 * h)
    bottom = y2 - int(0.18 * h)

    inner_w = max(1, right - left)
    band_w = inner_w / 5.0

    rois = []
    for k in range(5):
        rx1 = int(left + k * band_w + 0.10 * band_w)
        rx2 = int(left + (k + 1) * band_w - 0.10 * band_w)
        rois.append((rx1, top, rx2, bottom))
    return rois


def ink_score(thr_inv: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = roi
    patch = thr_inv[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size) * 100.0


def read_grid_responses(
    thr_inv: np.ndarray,
    table_bbox: Tuple[int, int, int, int],
    mark_threshold: float,
    ambiguity_gap: float,
    multi_mark_extra_gap: float,
    use_micro_adjust: bool,
    micro_search: int,
    micro_band: int,
    try_flexible_grid: bool,
    grid_kernel_div: int,
) -> Tuple[Dict[int, int], Dict[int, dict], np.ndarray, str, Optional[np.ndarray]]:
    overlay = cv2.cvtColor(thr_inv.copy(), cv2.COLOR_GRAY2BGR)
    responses: Dict[int, int] = {}
    meta: Dict[int, dict] = {}
    grid_mode = "uniform"
    grid_vis = None

    # grid iterator
    try:
        if try_flexible_grid:
            xs, ys, grid_vis = grid_lines_from_bbox(thr_inv, table_bbox, rows=30, cols=8, kernel_div=grid_kernel_div)
            grid_iter = split_grid_flexible(xs, ys, rows=30, cols=8)
            grid_mode = "flexible"
        else:
            grid_iter = split_grid_uniform(table_bbox, rows=30, cols=8)
    except Exception:
        grid_iter = split_grid_uniform(table_bbox, rows=30, cols=8)
        grid_mode = "uniform"

    # micro-adjust grid mask (local crop)
    tx, ty, tw, th = table_bbox
    crop_thr = thr_inv[ty:ty+th, tx:tx+tw]
    grid_mask_local = build_grid_mask(crop_thr, kernel_div=grid_kernel_div) if use_micro_adjust else None

    for r, c, cell in grid_iter:
        item_id = item_id_from_rc(r, c)

        # micro adjust cell using grid lines
        if use_micro_adjust and grid_mask_local is not None:
            cx1, cy1, cx2, cy2 = cell
            lc1, ly1, lc2, ly2 = cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty
            ref_local = refine_cell_bbox(grid_mask_local, (lc1, ly1, lc2, ly2), search=micro_search, band=micro_band)
            cell = (ref_local[0] + tx, ref_local[1] + ty, ref_local[2] + tx, ref_local[3] + ty)

        rois = option_rois_in_cell(cell)
        fills = [ink_score(thr_inv, roi) for roi in rois]

        best_idx = int(np.argmax(fills))
        best_fill = float(fills[best_idx])
        sorted_f = sorted(fills, reverse=True)
        second_fill = float(sorted_f[1]) if len(sorted_f) > 1 else 0.0

        blank = best_fill < mark_threshold
        ambiguous = (not blank) and ((best_fill - second_fill) < ambiguity_gap)
        above = [i for i, v in enumerate(fills) if v >= (mark_threshold + multi_mark_extra_gap)]
        multi_mark = (not blank) and (len(above) >= 2)

        # Confidence (simple but useful): margin best-second
        confidence = max(0.0, best_fill - second_fill)

        x1, y1, x2, y2 = map(int, cell)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 70, 70), 1)

        if blank:
            responses[item_id] = -1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            responses[item_id] = best_idx
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if ambiguous or multi_mark:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)

        if not blank:
            rx1, ry1, rx2, ry2 = rois[best_idx]
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        meta[item_id] = {
            "fills": fills,
            "chosen_idx": best_idx,
            "chosen_fill": best_fill,
            "second_fill": second_fill,
            "blank": blank,
            "ambiguous": ambiguous,
            "multi_mark": multi_mark,
            "above_idx": above,
            "confidence": confidence,
        }

    return responses, meta, overlay, grid_mode, grid_vis


# ============================================================
# 6) SCORING + PROTOCOLE + IMPUTATION
# ============================================================
def compute_scores(responses: Dict[int, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    facet_scores = {fac: 0 for fac in facet_labels.keys()}

    for item_id, idx in responses.items():
        if idx == -1:
            continue
        if item_id in scoring_key and item_id in item_to_facet:
            fac = item_to_facet[item_id]
            facet_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facet_scores.items():
        domain_scores[facet_to_domain[fac]] += sc

    return facet_scores, domain_scores


def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_blank = len(blanks)
    n_count = sum(1 for v in responses.values() if v == N_OPTION_INDEX)

    status = {
        "valid": True,
        "reasons": [],
        "blank_items": blanks,
        "n_blank": n_blank,
        "n_count": n_count,
        "imputed": 0,
    }

    if n_blank >= rules.max_blank_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop d'items vides : {n_blank} (>= {rules.max_blank_invalid})")

    if n_count >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(f"Trop de réponses 'N' : {n_count} (>= {rules.max_N_invalid})")

    new_resp = dict(responses)
    if status["valid"] and 0 < n_blank <= rules.impute_blank_if_leq:
        for item_id in blanks:
            new_resp[item_id] = rules.impute_option_index
            status["imputed"] += 1

    return new_resp, status


# ============================================================
# 7) CHARTS (bars + radar)
# ============================================================
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_bar_domains(domain_scores: Dict[str, float], title: str) -> bytes:
    domains = ["N", "E", "O", "A", "C"]
    labels = [domain_labels[d] for d in domains]
    values = [domain_scores[d] for d in domains]

    fig = plt.figure(figsize=(8, 3.2))
    plt.bar(labels, values)
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    return fig_to_png_bytes(fig)


def plot_bar_facets(facet_scores: Dict[str, float], title: str) -> bytes:
    facets = sorted(facet_scores.keys(), key=lambda x: (x[0], int(x[1])))
    values = [facet_scores[f] for f in facets]
    fig = plt.figure(figsize=(10, 4.2))
    plt.bar(facets, values)
    plt.title(title)
    plt.tight_layout()
    return fig_to_png_bytes(fig)


def plot_radar_domains(domain_scores: Dict[str, float], title: str) -> bytes:
    domains = ["N", "E", "O", "A", "C"]
    values = [domain_scores[d] for d in domains]
    labels = [domain_labels[d] for d in domains]

    angles = np.linspace(0, 2*np.pi, len(domains), endpoint=False).tolist()
    values_loop = values + [values[0]]
    angles_loop = angles + [angles[0]]

    fig = plt.figure(figsize=(6.2, 6.2))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_loop, values_loop, linewidth=2)
    ax.fill(angles_loop, values_loop, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_title(title, pad=18)
    return fig_to_png_bytes(fig)


# ============================================================
# 8) EXPORTS (CSV/TXT/PDF)
# ============================================================
def build_csv(facet_raw: Dict[str, int], domain_raw: Dict[str, int],
              facet_t: Optional[Dict[str, float]] = None, domain_t: Optional[Dict[str, float]] = None,
              facet_p: Optional[Dict[str, float]] = None, domain_p: Optional[Dict[str, float]] = None) -> str:
    out = io.StringIO()
    w = csv.writer(out)

    w.writerow(["Type", "Scale", "Label", "Raw", "T", "Percentile"])
    for f in sorted(facet_raw.keys(), key=lambda x: (x[0], int(x[1]))):
        w.writerow(["facet", f, facet_labels[f], facet_raw[f],
                    "" if facet_t is None else round(facet_t.get(f, float("nan")), 2),
                    "" if facet_p is None else round(facet_p.get(f, float("nan")), 2)])

    for d in ["N", "E", "O", "A", "C"]:
        w.writerow(["domain", d, domain_labels[d], domain_raw[d],
                    "" if domain_t is None else round(domain_t.get(d, float("nan")), 2),
                    "" if domain_p is None else round(domain_p.get(d, float("nan")), 2)])
    return out.getvalue()


def build_report_txt(status: dict, grid_mode: str, rot: int,
                     facet_raw: Dict[str, int], domain_raw: Dict[str, int],
                     facet_t: Optional[Dict[str, float]], domain_t: Optional[Dict[str, float]],
                     facet_p: Optional[Dict[str, float]], domain_p: Optional[Dict[str, float]],
                     review_list: List[int]) -> str:
    lines = []
    lines.append("RAPPORT NEO PI-R")
    lines.append(f"Date: {datetime.date.today().isoformat()}")
    lines.append(f"Rotation détectée: {rot}°")
    lines.append(f"Mode grille: {grid_mode}")
    lines.append("")
    lines.append(f"STATUT PROTOCOLE : {'VALIDE' if status['valid'] else 'INVALIDE'}")
    if status["reasons"]:
        lines.append("RAISONS:")
        for r in status["reasons"]:
            lines.append(f"- {r}")
    lines.append("")
    lines.append(f"Items vides : {status['n_blank']}")
    lines.append(f"Réponses 'N' (idx=2) : {status['n_count']}")
    lines.append(f"Imputations : {status['imputed']}")
    lines.append("")
    if review_list:
        lines.append("ITEMS A VERIFIER (ambigus/multi/faible confiance):")
        lines.append(", ".join(map(str, review_list[:80])) + ("..." if len(review_list) > 80 else ""))
        lines.append("")

    def line_scale(scale_type: str, key: str, label: str, raw: int, t: Optional[float], p: Optional[float]) -> str:
        parts = [f"{label}: raw={raw}"]
        if t is not None and not math.isnan(t):
            parts.append(f"T={t:.2f}")
        if p is not None and not math.isnan(p):
            parts.append(f"pct={p:.2f}")
        return " | ".join(parts)

    lines.append("TOTAUX DOMAINES")
    for d in ["N", "E", "O", "A", "C"]:
        t = None if domain_t is None else domain_t.get(d)
        p = None if domain_p is None else domain_p.get(d)
        lines.append(line_scale("domain", d, domain_labels[d], domain_raw[d], t, p))

    lines.append("")
    lines.append("SCORES FACETTES")
    for f in sorted(facet_raw.keys(), key=lambda x: (x[0], int(x[1]))):
        t = None if facet_t is None else facet_t.get(f)
        p = None if facet_p is None else facet_p.get(f)
        lines.append(line_scale("facet", f, facet_labels[f], facet_raw[f], t, p))

    return "\n".join(lines)


def build_pdf_report(
    status: dict, grid_mode: str, rot: int,
    domain_chart_png: bytes, facet_chart_png: bytes, radar_png: bytes,
    report_txt: str
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # Page 1 summary + radar + domain chart
    y = H - 2.0 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2.0 * cm, y, "Rapport NEO PI-R")
    y -= 1.0 * cm

    c.setFont("Helvetica", 11)
    c.drawString(2.0 * cm, y, f"Date: {datetime.date.today().isoformat()}")
    y -= 0.6 * cm
    c.drawString(2.0 * cm, y, f"Rotation: {rot}°  |  Grille: {grid_mode}")
    y -= 0.8 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.0 * cm, y, f"Statut protocole: {'VALIDE' if status['valid'] else 'INVALIDE'}")
    y -= 0.6 * cm
    c.setFont("Helvetica", 10)
    c.drawString(2.0 * cm, y, f"Items vides: {status['n_blank']}  |  N (idx=2): {status['n_count']}  |  Imputations: {status['imputed']}")
    y -= 0.8 * cm

    if status["reasons"]:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2.0 * cm, y, "Raisons:")
        y -= 0.5 * cm
        c.setFont("Helvetica", 9)
        for r in status["reasons"][:6]:
            c.drawString(2.2 * cm, y, f"- {r[:120]}")
            y -= 0.42 * cm

    # Radar
    radar = ImageReader(io.BytesIO(radar_png))
    c.drawImage(radar, 2.0 * cm, 11.0 * cm, width=8.5 * cm, height=8.5 * cm, preserveAspectRatio=True, anchor="sw")

    # Domain chart
    dom = ImageReader(io.BytesIO(domain_chart_png))
    c.drawImage(dom, 11.0 * cm, 12.0 * cm, width=8.0 * cm, height=6.0 * cm, preserveAspectRatio=True, anchor="sw")

    c.showPage()

    # Page 2 facets bar chart
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2.0 * cm, H - 2.0 * cm, "Facettes")
    facet = ImageReader(io.BytesIO(facet_chart_png))
    c.drawImage(facet, 1.5 * cm, 6.0 * cm, width=18.5 * cm, height=11.0 * cm, preserveAspectRatio=True, anchor="sw")
    c.showPage()

    # Page 3+ text report
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2.0 * cm, H - 2.0 * cm, "Détails (TXT)")
    y = H - 2.8 * cm
    c.setFont("Helvetica", 8.5)
    for line in report_txt.splitlines():
        c.drawString(2.0 * cm, y, line[:160])
        y -= 0.35 * cm
        if y < 2.0 * cm:
            c.showPage()
            y = H - 2.0 * cm
            c.setFont("Helvetica", 8.5)

    c.save()
    return buf.getvalue()


# ============================================================
# 9) STREAMLIT APP
# ============================================================
st.set_page_config(page_title="NEO PI-R – OMR Pro", page_icon="🧠", layout="wide")
st.title("NEO PI-R – Scanner OMR (Version Pro / Scientifique)")
st.caption("Rotation auto • Grille flexible • Micro-ajustement cellule • Validité protocole • Raw + T + percentiles • PDF pro")

with st.sidebar:
    st.subheader("Image")
    mode = st.selectbox("Type", ["photo", "scan"], index=0)
    target_width = st.slider("Largeur warp", 1200, 2600, 1800, 100)

    st.subheader("Détection réponses")
    mark_threshold = st.slider("Seuil détection (%)", 0.1, 25.0, 1.2, 0.1)
    ambiguity_gap = st.slider("Ambiguïté (écart %)", 0.1, 20.0, 2.0, 0.1)
    multi_mark_extra_gap = st.slider("Multi-cochage (marge %)", 0.0, 5.0, 0.3, 0.1)

    st.subheader("Table bbox (tolérance photo)")
    min_w_ratio = st.slider("BBox largeur min", 0.20, 0.95, 0.40, 0.05)
    min_h_ratio = st.slider("BBox hauteur min", 0.15, 0.95, 0.28, 0.05)
    kernel_div = st.slider("kernel_div", 10, 40, 18, 1)
    dilate_size = st.slider("dilate_size", 1, 9, 5, 1)

    st.subheader("Grille")
    try_flexible_grid = st.toggle("Grille flexible (lignes réelles)", value=True)
    grid_kernel_div = st.slider("Grille kernel_div", 10, 40, 18, 1)

    st.subheader("Micro-ajustement cellule")
    use_micro_adjust = st.toggle("Activer micro-ajustement", value=True)
    micro_search = st.slider("Recherche ± px", 0, 12, 6, 1)
    micro_band = st.slider("Bande bordure px", 1, 5, 2, 1)

    st.subheader("Protocole")
    max_blank_invalid = st.number_input("Invalide si blancs ≥", 0, 240, 15, 1)
    max_N_invalid = st.number_input("Invalide si N ≥", 0, 240, 42, 1)
    impute_blank_if_leq = st.number_input("Imputation si blancs ≤", 0, 240, 10, 1)

    st.subheader("Normes (optionnel)")
    sex = st.selectbox("Sexe", ["M", "F"], index=0)
    age = st.number_input("Âge", 10, 99, 25, 1)
    norms_file = st.file_uploader("Importer norms.csv", type=["csv"])

    debug = st.toggle("Debug scientifique", value=False)

RULES = ProtocolRules(
    max_blank_invalid=int(max_blank_invalid),
    max_N_invalid=int(max_N_invalid),
    impute_blank_if_leq=int(impute_blank_if_leq),
    impute_option_index=N_OPTION_INDEX,
)

uploaded = st.file_uploader("Importer une photo/scanner (JPG/PNG)", type=["jpg", "jpeg", "png"])
run_btn = st.button("Scanner & Calculer", type="primary", disabled=(uploaded is None))

if run_btn and uploaded is not None:
    try:
        if len(scoring_key) < 240:
            st.warning("⚠️ Ta scoring_key n'est pas complète (1..240). Colle ta clé complète dans le code.")
        pil_img = Image.open(uploaded)
        bgr = pil_to_bgr(pil_img)

        warped, thr, table_bbox, rot = auto_rotate_and_warp(
            bgr=bgr,
            target_width=int(target_width),
            mode=str(mode),
            min_w_ratio=float(min_w_ratio),
            min_h_ratio=float(min_h_ratio),
            kernel_div=int(kernel_div),
            dilate_size=int(dilate_size),
        )

        raw_resp, meta, overlay, grid_mode, grid_vis = read_grid_responses(
            thr_inv=thr,
            table_bbox=table_bbox,
            mark_threshold=float(mark_threshold),
            ambiguity_gap=float(ambiguity_gap),
            multi_mark_extra_gap=float(multi_mark_extra_gap),
            use_micro_adjust=bool(use_micro_adjust),
            micro_search=int(micro_search),
            micro_band=int(micro_band),
            try_flexible_grid=bool(try_flexible_grid),
            grid_kernel_div=int(grid_kernel_div),
        )

        final_resp, status = apply_protocol_rules(raw_resp, RULES)
        facet_raw, domain_raw = compute_scores(final_resp)

        # Items à vérifier: ambigus, multi, ou faible confiance
        review = []
        for item_id, m in meta.items():
            if m["blank"]:
                continue
            if m["ambiguous"] or m["multi_mark"] or (m["confidence"] < max(0.4, ambiguity_gap * 0.35)):
                review.append(item_id)
        review.sort()

        # Normes: compute T + percentile (si norms.csv)
        norms: Optional[List[NormRow]] = None
        if norms_file is not None:
            norms = load_norms_csv(norms_file.getvalue())

        facet_t = None
        facet_p = None
        domain_t = None
        domain_p = None

        if norms is not None:
            facet_t = {}
            facet_p = {}
            for f, raw in facet_raw.items():
                n = find_norm(norms, "facet", f, sex, int(age))
                if n is None:
                    facet_t[f] = float("nan")
                    facet_p[f] = float("nan")
                else:
                    t = raw_to_t(raw, n.mean, n.sd)
                    facet_t[f] = t
                    facet_p[f] = percentile_from_t(t)

            domain_t = {}
            domain_p = {}
            for d, raw in domain_raw.items():
                n = find_norm(norms, "domain", d, sex, int(age))
                if n is None:
                    domain_t[d] = float("nan")
                    domain_p[d] = float("nan")
                else:
                    t = raw_to_t(raw, n.mean, n.sd)
                    domain_t[d] = t
                    domain_p[d] = percentile_from_t(t)

        # KPI
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Items vides", status["n_blank"])
        c2.metric("N (idx=2)", status["n_count"])
        c3.metric("Imputations", status["imputed"])
        c4.metric("Statut", "VALIDE" if status["valid"] else "INVALIDE")
        c5.metric("À vérifier", len(review))

        if debug:
            st.write("Rotation:", rot, " | Grille:", grid_mode, " | Table bbox:", table_bbox)

        if not status["valid"]:
            st.error("Protocole INVALIDE")
            for r in status["reasons"]:
                st.write("•", r)
        else:
            st.success("Protocole VALIDE")

        tab1, tab2, tab3, tab4 = st.tabs(["Contrôle OMR", "Facettes", "Domaines", "Exports"])

        with tab1:
            colA, colB = st.columns(2)
            with colA:
                st.subheader("Image redressée")
                st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)
            with colB:
                st.subheader("Overlay (vert=renseigné, rouge=vide, orange=ambigu/multi, bleu=choix)")
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            if grid_vis is not None and debug:
                st.subheader("Debug grille (bleu=vertical, vert=horizontal)")
                st.image(cv2.cvtColor(grid_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

            if review:
                st.subheader("Items à vérifier")
                st.write(", ".join(map(str, review[:120])) + (" ..." if len(review) > 120 else ""))

        with tab2:
            st.subheader("Scores Facettes")
            rows = []
            for f in sorted(facet_raw.keys(), key=lambda x: (x[0], int(x[1]))):
                rows.append({
                    "Facette": facet_labels[f],
                    "Raw": facet_raw[f],
                    "T": None if facet_t is None else facet_t.get(f),
                    "Percentile": None if facet_p is None else facet_p.get(f),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            title = "Facettes (Raw)" if facet_t is None else "Facettes (T-scores)"
            plot_source = facet_raw if facet_t is None else {k: (0.0 if math.isnan(v) else v) for k, v in facet_t.items()}
            st.image(plot_bar_facets(plot_source, title), use_container_width=True)

        with tab3:
            st.subheader("Scores Domaines")
            drows = []
            for d in ["N", "E", "O", "A", "C"]:
                drows.append({
                    "Domaine": domain_labels[d],
                    "Raw": domain_raw[d],
                    "T": None if domain_t is None else domain_t.get(d),
                    "Percentile": None if domain_p is None else domain_p.get(d),
                })
            st.dataframe(drows, use_container_width=True, hide_index=True)

            title_dom = "Domaines (Raw)" if domain_t is None else "Domaines (T-scores)"
            dom_source = domain_raw if domain_t is None else {k: (0.0 if math.isnan(v) else v) for k, v in domain_t.items()}

            dom_png = plot_bar_domains(dom_source, title_dom)
            radar_png = plot_radar_domains(dom_source, "Profil (Radar)")

            st.image(dom_png, use_container_width=True)
            st.image(radar_png, use_container_width=True)

        with tab4:
            st.subheader("Exports")
            # Charts for PDF
            dom_source = domain_raw if domain_t is None else {k: (0.0 if math.isnan(v) else v) for k, v in domain_t.items()}
            fac_source = facet_raw if facet_t is None else {k: (0.0 if math.isnan(v) else v) for k, v in facet_t.items()}

            domain_chart = plot_bar_domains(dom_source, "Domaines (T)" if domain_t is not None else "Domaines (Raw)")
            facet_chart = plot_bar_facets(fac_source, "Facettes (T)" if facet_t is not None else "Facettes (Raw)")
            radar_chart = plot_radar_domains(dom_source, "Radar (T)" if domain_t is not None else "Radar (Raw)")

            report_txt = build_report_txt(
                status=status, grid_mode=grid_mode, rot=rot,
                facet_raw=facet_raw, domain_raw=domain_raw,
                facet_t=facet_t, domain_t=domain_t,
                facet_p=facet_p, domain_p=domain_p,
                review_list=review
            )

            csv_text = build_csv(facet_raw, domain_raw, facet_t, domain_t, facet_p, domain_p)
            pdf_bytes = build_pdf_report(status, grid_mode, rot, domain_chart, facet_chart, radar_chart, report_txt)

            st.download_button("📥 Télécharger CSV", csv_text, "neo_pir_scores.csv", "text/csv")
            st.download_button("📥 Télécharger Rapport TXT", report_txt, "neo_pir_report.txt", "text/plain")
            st.download_button("📥 Télécharger Rapport PDF", pdf_bytes, "neo_pir_report.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Erreur : {e}")
        if debug:
            st.exception(e)
