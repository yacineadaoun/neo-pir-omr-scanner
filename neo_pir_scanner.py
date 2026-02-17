# app.py
# NEO PI-R Scanner & Calculator - Version 1.0
# Author: Yacine Adaoun
# Streamlit + OpenCV + Psychometric scoring (Facets + Domains) + Protocol validity + Exports

from __future__ import annotations

import io
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Generator, Optional

import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from PIL import Image
import streamlit as st

# PDF export
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


# ============================================================
# 1) SCORING KEY (1..240)
# ============================================================
# IMPORTANT: Remplace/colle ici ta version complète si tu l'as déjà.
# Je garde exactement ton style: item_id: [score_for_FD, score_for_D, score_for_N, score_for_A, score_for_FA]
scoring_key: Dict[int, List[int]] = {
    1 : [4,3,2,1,0], 31 : [0,1,2,3,4], 61 : [4,3,2,1,0], 91 : [0,1,2,3,4], 121 : [4,3,2,1,0], 151 : [0,1,2,3,4], 181 : [4,3,2,1,0], 211 : [0,1,2,3,4],
    2 : [0,1,2,3,4], 32 : [4,3,2,1,0], 62 : [0,1,2,3,4], 92 : [4,3,2,1,0], 122 : [0,1,2,3,4], 152 : [4,3,2,1,0], 182 : [0,1,2,3,4], 212 : [4,3,2,1,0],
    3 : [0,1,2,3,4], 33 : [4,3,2,1,0], 63 : [0,1,2,3,4], 93 : [4,3,2,1,0], 123 : [0,1,2,3,4], 153 : [4,3,2,1,0], 183 : [0,1,2,3,4], 213 : [4,3,2,1,0],
    4 : [4,3,2,1,0], 34 : [0,1,2,3,4], 64 : [4,3,2,1,0], 94 : [0,1,2,3,4], 124 : [4,3,2,1,0], 154 : [0,1,2,3,4], 184 : [4,3,2,1,0], 214 : [0,1,2,3,4],
    5 : [0,1,2,3,4], 35 : [4,3,2,1,0], 65 : [0,1,2,3,4], 95 : [4,3,2,1,0], 125 : [0,1,2,3,4], 155 : [4,3,2,1,0], 185 : [0,1,2,3,4], 215 : [4,3,2,1,0],
    6 : [0,1,2,3,4], 36 : [4,3,2,1,0], 66 : [0,1,2,3,4], 96 : [4,3,2,1,0], 126 : [0,1,2,3,4], 156 : [4,3,2,1,0], 186 : [0,1,2,3,4], 216 : [4,3,2,1,0],
    7 : [4,3,2,1,0], 37 : [0,1,2,3,4], 67 : [4,3,2,1,0], 97 : [0,1,2,3,4], 127 : [4,3,2,1,0], 157 : [0,1,2,3,4], 187 : [4,3,2,1,0], 217 : [0,1,2,3,4],
    8 : [4,3,2,1,0], 38 : [0,1,2,3,4], 68 : [4,3,2,1,0], 98 : [0,1,2,3,4], 128 : [4,3,2,1,0], 158 : [0,1,2,3,4], 188 : [4,3,2,1,0], 218 : [0,1,2,3,4],
    9 : [0,1,2,3,4], 39 : [4,3,2,1,0], 69 : [0,1,2,3,4], 99 : [4,3,2,1,0], 129 : [0,1,2,3,4], 159 : [4,3,2,1,0], 189 : [0,1,2,3,4], 219 : [4,3,2,1,0],
    10 : [4,3,2,1,0], 40 : [0,1,2,3,4], 70 : [4,3,2,1,0], 100 : [0,1,2,3,4], 130 : [4,3,2,1,0], 160 : [0,1,2,3,4], 190 : [4,3,2,1,0], 220 : [0,1,2,3,4],
    11 : [4,3,2,1,0], 41 : [0,1,2,3,4], 71 : [4,3,2,1,0], 101 : [0,1,2,3,4], 131 : [4,3,2,1,0], 161 : [0,1,2,3,4], 191 : [4,3,2,1,0], 221 : [0,1,2,3,4],
    12 : [0,1,2,3,4], 42 : [4,3,2,1,0], 72 : [0,1,2,3,4], 102 : [4,3,2,1,0], 132 : [0,1,2,3,4], 162 : [4,3,2,1,0], 192 : [0,1,2,3,4], 222 : [4,3,2,1,0],
    13 : [0,1,2,3,4], 43 : [4,3,2,1,0], 73 : [0,1,2,3,4], 103 : [4,3,2,1,0], 133 : [0,1,2,3,4], 163 : [4,3,2,1,0], 193 : [0,1,2,3,4], 223 : [4,3,2,1,0],
    14 : [4,3,2,1,0], 44 : [0,1,2,3,4], 74 : [4,3,2,1,0], 104 : [0,1,2,3,4], 134 : [4,3,2,1,0], 164 : [0,1,2,3,4], 194 : [4,3,2,1,0], 224 : [0,1,2,3,4],
    15 : [0,1,2,3,4], 45 : [4,3,2,1,0], 75 : [0,1,2,3,4], 105 : [4,3,2,1,0], 135 : [0,1,2,3,4], 165 : [4,3,2,1,0], 195 : [0,1,2,3,4], 225 : [4,3,2,1,0],
    16 : [0,1,2,3,4], 46 : [4,3,2,1,0], 76 : [0,1,2,3,4], 106 : [4,3,2,1,0], 136 : [0,1,2,3,4], 166 : [4,3,2,1,0], 196 : [0,1,2,3,4], 226 : [4,3,2,1,0],
    17 : [4,3,2,1,0], 47 : [0,1,2,3,4], 77 : [4,3,2,1,0], 107 : [0,1,2,3,4], 137 : [4,3,2,1,0], 167 : [0,1,2,3,4], 197 : [4,3,2,1,0], 227 : [0,1,2,3,4],
    18 : [4,3,2,1,0], 48 : [0,1,2,3,4], 78 : [4,3,2,1,0], 108 : [0,1,2,3,4], 138 : [4,3,2,1,0], 168 : [0,1,2,3,4], 198 : [4,3,2,1,0], 228 : [0,1,2,3,4],
    19 : [0,1,2,3,4], 49 : [4,3,2,1,0], 79 : [0,1,2,3,4], 109 : [4,3,2,1,0], 139 : [0,1,2,3,4], 169 : [4,3,2,1,0], 199 : [0,1,2,3,4], 229 : [4,3,2,1,0],
    20 : [4,3,2,1,0], 50 : [0,1,2,3,4], 80 : [4,3,2,1,0], 110 : [0,1,2,3,4], 140 : [4,3,2,1,0], 170 : [0,1,2,3,4], 200 : [4,3,2,1,0], 230 : [0,1,2,3,4],
    21 : [4,3,2,1,0], 51 : [0,1,2,3,4], 81 : [4,3,2,1,0], 111 : [0,1,2,3,4], 141 : [4,3,2,1,0], 171 : [0,1,2,3,4], 201 : [4,3,2,1,0], 231 : [0,1,2,3,4],
    22 : [0,1,2,3,4], 52 : [4,3,2,1,0], 82 : [0,1,2,3,4], 112 : [4,3,2,1,0], 142 : [0,1,2,3,4], 172 : [4,3,2,1,0], 202 : [0,1,2,3,4], 232 : [4,3,2,1,0],
    23 : [0,1,2,3,4], 53 : [4,3,2,1,0], 83 : [0,1,2,3,4], 113 : [4,3,2,1,0], 143 : [0,1,2,3,4], 173 : [4,3,2,1,0], 203 : [0,1,2,3,4], 233 : [4,3,2,1,0],
    24 : [4,3,2,1,0], 54 : [0,1,2,3,4], 84 : [4,3,2,1,0], 114 : [0,1,2,3,4], 144 : [4,3,2,1,0], 174 : [0,1,2,3,4], 204 : [4,3,2,1,0], 234 : [0,1,2,3,4],
    25 : [0,1,2,3,4], 55 : [4,3,2,1,0], 85 : [0,1,2,3,4], 115 : [4,3,2,1,0], 145 : [0,1,2,3,4], 175 : [4,3,2,1,0], 205 : [0,1,2,3,4], 235 : [4,3,2,1,0],
    26 : [0,1,2,3,4], 56 : [4,3,2,1,0], 86 : [0,1,2,3,4], 116 : [4,3,2,1,0], 146 : [0,1,2,3,4], 176 : [4,3,2,1,0], 206 : [0,1,2,3,4], 236 : [4,3,2,1,0],
    27 : [4,3,2,1,0], 57 : [0,1,2,3,4], 87 : [4,3,2,1,0], 117 : [0,1,2,3,4], 147 : [4,3,2,1,0], 177 : [0,1,2,3,4], 207 : [4,3,2,1,0], 237 : [0,1,2,3,4],
    28 : [4,3,2,1,0], 58 : [0,1,2,3,4], 88 : [4,3,2,1,0], 118 : [0,1,2,3,4], 148 : [4,3,2,1,0], 178 : [0,1,2,3,4], 208 : [4,3,2,1,0], 238 : [0,1,2,3,4],
    29 : [0,1,2,3,4], 59 : [4,3,2,1,0], 89 : [0,1,2,3,4], 119 : [4,3,2,1,0], 149 : [0,1,2,3,4], 179 : [4,3,2,1,0], 209 : [0,1,2,3,4], 239 : [4,3,2,1,0],
    30 : [4,3,2,1,0], 60 : [0,1,2,3,4], 90 : [4,3,2,1,0], 120 : [0,1,2,3,4], 150 : [4,3,2,1,0], 180 : [0,1,2,3,4], 210 : [4,3,2,1,0], 240 : [0,1,2,3,4]
}


# ============================================================
# 2) ITEMS -> FACETS (generated, no 240 lines)
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

domain_labels = {
    "N": "Névrosisme", "E": "Extraversion", "O": "Ouverture", "A": "Agréabilité", "C": "Conscience"
}


# ============================================================
# 3) PROTOCOL RULES
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15     # >= 15 blanks => invalid
    max_N_invalid: int = 42         # >= 42 neutral => invalid
    impute_blank_if_leq: int = 10   # <= 10 blanks: impute
    impute_option_index: int = 2    # neutral index
    impute_score_value: int = 2     # (informative)


# ============================================================
# 4) VISION UTILITIES
# ============================================================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def find_document_warp(bgr: np.ndarray, target_width: int = 1800) -> np.ndarray:
    # resize for stability
    ratio = bgr.shape[1] / float(target_width)
    resized = imutils.resize(bgr, width=target_width)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        raise ValueError("Impossible de détecter la feuille (aucun contour).")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    doc = None
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break

    if doc is None:
        raise ValueError("Impossible de détecter un contour à 4 coins (feuille).")

    warped = four_point_transform(resized, doc.reshape(4, 2))
    # upscale back (optional): not needed because we already use target_width
    return warped


def binarize_for_marks(bgr: np.ndarray, mode: str = "photo") -> np.ndarray:
    """
    Returns a binary inverted image (white marks on black background) as uint8.
    mode:
      - "photo": adaptive threshold, more robust to shadows
      - "scan": Otsu threshold, usually cleaner
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if mode == "scan":
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thr

    # photo
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    return thr


def find_table_bbox(thr_inv: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detect the main grid area by extracting horizontal & vertical lines.
    Returns (x, y, w, h).
    """
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, thr_inv.shape[1] // 18), 1))
    hor = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, thr_inv.shape[0] // 18)))
    ver = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    cnts = cv2.findContours(grid.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        raise ValueError("Éléments du tableau introuvables (lignes non détectées).")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnts[0])

    # sanity check
    if w < thr_inv.shape[1] * 0.60 or h < thr_inv.shape[0] * 0.40:
        raise ValueError("BBox du tableau incohérente (angle/zoom trop fort).")

    return x, y, w, h


def split_grid(table_bbox: Tuple[int, int, int, int], rows: int = 30, cols: int = 8) -> Generator[Tuple[int, int, Tuple[int, int, int, int]], None, None]:
    x, y, w, h = table_bbox
    cell_w = w / cols
    cell_h = h / rows
    for r in range(rows):
        for c in range(cols):
            cx1 = int(x + c * cell_w)
            cy1 = int(y + r * cell_h)
            cx2 = int(x + (c + 1) * cell_w)
            cy2 = int(y + (r + 1) * cell_h)
            yield r, c, (cx1, cy1, cx2, cy2)


def item_id_from_rc(r: int, c: int) -> int:
    # column 0 => items 1..30, col 1 => 31..60, ... col 7 => 211..240
    return (r + 1) + 30 * c


def option_rois_in_cell(cell: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    """
    In each cell: 5 option zones (FD, D, N, A, FA).
    We cut the interior area in 5 vertical stripes, excluding borders and the item number area.
    """
    x1, y1, x2, y2 = cell
    w = x2 - x1
    h = y2 - y1

    # margins to avoid borders + left item number
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
        ry1 = top
        ry2 = bottom
        rois.append((rx1, ry1, rx2, ry2))
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
    multi_mark_extra_gap: float = 0.3
) -> Tuple[Dict[int, int], Dict[int, dict], np.ndarray]:
    """
    Returns:
      responses[item_id] = option_index (0..4) or -1 if blank
      meta[item_id] with fills + flags
      overlay BGR for visual QA
    """
    overlay = cv2.cvtColor(thr_inv.copy(), cv2.COLOR_GRAY2BGR)

    responses: Dict[int, int] = {}
    meta: Dict[int, dict] = {}

    for r, c, cell in split_grid(table_bbox, rows=30, cols=8):
        item_id = item_id_from_rc(r, c)
        rois = option_rois_in_cell(cell)
        fills = [ink_score(thr_inv, roi) for roi in rois]

        best_idx = int(np.argmax(fills))
        best_fill = float(fills[best_idx])

        sorted_f = sorted(fills, reverse=True)
        second_fill = float(sorted_f[1]) if len(sorted_f) > 1 else 0.0

        blank = best_fill < mark_threshold
        ambiguous = (not blank) and ((best_fill - second_fill) < ambiguity_gap)

        # multi-mark: if at least 2 zones clearly above threshold
        above = [i for i, v in enumerate(fills) if v >= (mark_threshold + multi_mark_extra_gap)]
        multi_mark = (not blank) and (len(above) >= 2)

        x1, y1, x2, y2 = cell

        # base border
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 70, 70), 1)

        if blank:
            responses[item_id] = -1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
        else:
            responses[item_id] = best_idx
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green

        if ambiguous or multi_mark:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange

        # chosen option
        rx1, ry1, rx2, ry2 = rois[best_idx]
        if not blank:
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)  # blue

        meta[item_id] = {
            "fills": fills,
            "chosen_idx": best_idx,
            "chosen_fill": best_fill,
            "blank": blank,
            "ambiguous": ambiguous,
            "multi_mark": multi_mark,
            "above_idx": above,
        }

    return responses, meta, overlay


# ============================================================
# 5) SCORING + PROTOCOL
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

    # "N" response = index 2
    n_count = sum(1 for v in responses.values() if v == 2)

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
        status["reasons"].append(
            f"Trop d'items vides : {n_blank} (>= {rules.max_blank_invalid})"
        )

    if n_count >= rules.max_N_invalid:
        status["valid"] = False
        status["reasons"].append(
            f"Trop de réponses 'N' : {n_count} (>= {rules.max_N_invalid})"
        )

    new_resp = dict(responses)

    if status["valid"] and 0 < n_blank <= rules.impute_blank_if_leq:
        for item_id in blanks:
            new_resp[item_id] = rules.impute_option_index
            status["imputed"] += 1

    return new_resp, status


# ============================================================
# 6) EXPORTS
# ============================================================
def build_csv(facet_scores: Dict[str, int], domain_scores: Dict[str, int]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Facette", "Score brut"])
    for fac in sorted(facet_scores.keys()):
        writer.writerow([facet_labels[fac], facet_scores[fac]])

    writer.writerow([])
    writer.writerow(["Domaine", "Score brut"])
    for d in sorted(domain_scores.keys()):
        writer.writerow([domain_labels[d], domain_scores[d]])

    return output.getvalue()


def build_report_txt(status: dict, facet_scores: Dict[str, int], domain_scores: Dict[str, int]) -> str:
    lines = []
    lines.append("RAPPORT NEO PI-R")
    lines.append("")
    lines.append(f"STATUT PROTOCOLE : {'VALIDE' if status['valid'] else 'INVALIDE'}")

    if status["reasons"]:
        lines.append("RAISONS:")
        for r in status["reasons"]:
            lines.append(f"- {r}")

    lines.append("")
    lines.append(f"Items vides : {status['n_blank']}")
    lines.append(f"Réponses 'N' observées (index=2) : {status['n_count']}")
    lines.append(f"Imputations : {status['imputed']}")
    lines.append("")
    lines.append("SCORES PAR FACETTE")
    for fac in sorted(facet_scores.keys()):
        lines.append(f"{facet_labels[fac]}: {facet_scores[fac]}")
    lines.append("")
    lines.append("TOTAUX PAR DOMAINE")
    for d in sorted(domain_scores.keys()):
        lines.append(f"{domain_labels[d]}: {domain_scores[d]}")
    lines.append("")
    return "\n".join(lines)


def build_report_pdf(
    status: dict,
    facet_scores: Dict[str, int],
    domain_scores: Dict[str, int],
    title: str = "Rapport NEO PI-R"
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    def write_line(text: str, x: float, y: float, size: int = 11, bold: bool = False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x, y, text)

    y = h - 2.0 * cm
    write_line(title, 2.0 * cm, y, size=16, bold=True)
    y -= 1.0 * cm

    write_line(f"Statut protocole : {'VALIDE' if status['valid'] else 'INVALIDE'}", 2.0 * cm, y, bold=True)
    y -= 0.7 * cm

    write_line(f"Items vides : {status['n_blank']}", 2.0 * cm, y)
    y -= 0.5 * cm
    write_line(f"Réponses 'N' (index=2) : {status['n_count']}", 2.0 * cm, y)
    y -= 0.5 * cm
    write_line(f"Imputations : {status['imputed']}", 2.0 * cm, y)
    y -= 0.8 * cm

    if status["reasons"]:
        write_line("Raisons:", 2.0 * cm, y, bold=True)
        y -= 0.6 * cm
        for r in status["reasons"]:
            write_line(f"- {r}", 2.2 * cm, y)
            y -= 0.5 * cm
        y -= 0.3 * cm

    # Domain scores
    write_line("Totaux par domaine", 2.0 * cm, y, bold=True)
    y -= 0.7 * cm
    for d in sorted(domain_scores.keys()):
        write_line(f"{domain_labels[d]} : {domain_scores[d]}", 2.2 * cm, y)
        y -= 0.5 * cm

    y -= 0.8 * cm
    if y < 6 * cm:
        c.showPage()
        y = h - 2.0 * cm

    # Facet scores (may span pages)
    write_line("Scores par facette", 2.0 * cm, y, bold=True)
    y -= 0.7 * cm

    for fac in sorted(facet_scores.keys()):
        write_line(f"{facet_labels[fac]} : {facet_scores[fac]}", 2.2 * cm, y)
        y -= 0.45 * cm
        if y < 2.0 * cm:
            c.showPage()
            y = h - 2.0 * cm

    c.save()
    return buf.getvalue()


# ============================================================
# 7) STREAMLIT APP
# ============================================================
st.set_page_config(page_title="Scanner NEO PI-R", page_icon="🧠", layout="wide")

st.title("NEO PI-R – Scanner & Calculateur (Feuille de réponses)")
st.caption("Lecture par grille 30×8 – support stylo bleu/noir – validation protocolaire – exports CSV/TXT/PDF")

with st.sidebar:
    st.subheader("Mode & Paramètres vision")
    mode = st.selectbox("Mode d'image", ["photo", "scan"], index=0, help="Photo = robuste aux ombres. Scan = plus strict/clean.")
    target_width = st.slider("Largeur cible (warp)", 1200, 2600, 1800, 100)

    mark_threshold = st.slider("Seuil 'réponse détectée' (%)", 0.1, 20.0, 1.2, 0.1)
    ambiguity_gap = st.slider("Seuil ambiguïté (écart % entre 1er et 2e)", 0.1, 15.0, 2.0, 0.1)
    multi_mark_extra_gap = st.slider("Multi-cochage: marge extra au-dessus du seuil (%)", 0.0, 3.0, 0.3, 0.1)

    st.markdown("---")
    st.subheader("Règles protocole")
    max_blank_invalid = st.number_input("Items vides ⇒ invalide si ≥", min_value=0, max_value=240, value=15, step=1)
    max_N_invalid = st.number_input("Réponses 'N' ⇒ invalide si ≥", min_value=0, max_value=240, value=42, step=1)
    impute_blank_if_leq = st.number_input("Imputation si blancs ≤", min_value=0, max_value=240, value=10, step=1)

    st.caption("Imputation = choisir 'N' (index 2, score 2) sur chaque item vide.")
    debug = st.toggle("Debug", value=False)

RULES = ProtocolRules(
    max_blank_invalid=int(max_blank_invalid),
    max_N_invalid=int(max_N_invalid),
    impute_blank_if_leq=int(impute_blank_if_leq),
    impute_option_index=2,
    impute_score_value=2
)

uploaded = st.file_uploader("Importer une photo/scanner de la feuille (JPG/PNG)", type=["jpg", "jpeg", "png"])
run_btn = st.button("Scanner & Calculer", type="primary", disabled=(uploaded is None))

if run_btn and uploaded is not None:
    try:
        pil_img = Image.open(uploaded)
        bgr = pil_to_bgr(pil_img)

        warped = find_document_warp(bgr, target_width=target_width)
        thr = binarize_for_marks(warped, mode=mode)

        table_bbox = find_table_bbox(thr)

        raw_resp, meta, overlay = read_grid_responses(
            thr, table_bbox,
            mark_threshold=mark_threshold,
            ambiguity_gap=ambiguity_gap,
            multi_mark_extra_gap=multi_mark_extra_gap
        )

        final_resp, status = apply_protocol_rules(raw_resp, RULES)
        facet_scores, domain_scores = compute_scores(final_resp)

        n_items = 240
        n_blank = status["n_blank"]
        n_count = status["n_count"]
        imputed = status["imputed"]
        valid = status["valid"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Items vides", n_blank)
        c2.metric("N observés (idx=2)", n_count)
        c3.metric("Imputations", imputed)
        c4.metric("Statut protocole", "VALIDE" if valid else "INVALIDE")

        if not valid:
            st.error("Protocole INVALIDE")
            for r in status["reasons"]:
                st.write("•", r)
        else:
            st.success("Protocole VALIDE")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overlay contrôle", "Scores facettes", "Scores domaines", "Exports"])

        with tab1:
            colA, colB = st.columns(2)
            with colA:
                st.subheader("Image redressée")
                st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)
            with colB:
                st.subheader("Overlay (vert=renseigné, rouge=vide, orange=ambigu/multi, bleu=choix)")
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            if debug:
                st.write("Table bbox:", table_bbox)
                amb = [i for i, m in meta.items() if (not m["blank"]) and (m["ambiguous"] or m["multi_mark"])]
                st.write("Éléments ambigus ou multi-cochés :", len(amb))
                st.write("Exemples:", amb[:40])

        with tab2:
            st.subheader("Scores par facette (bruts)")
            rows = [{"Facette": facet_labels[f], "Score brut": facet_scores[f]} for f in sorted(facet_scores.keys())]
            st.dataframe(rows, use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("Totaux par domaine (bruts)")
            drows = [{"Domaine": domain_labels[d], "Score brut": domain_scores[d]} for d in sorted(domain_scores.keys())]
            st.dataframe(drows, use_container_width=True, hide_index=True)

        with tab4:
            st.subheader("Exports")

            csv_text = build_csv(facet_scores, domain_scores)
            report_txt = build_report_txt(status, facet_scores, domain_scores)
            report_pdf = build_report_pdf(status, facet_scores, domain_scores)

            st.download_button("📥 Télécharger CSV", csv_text, "neo_pir_scores.csv", "text/csv")
            st.download_button("📥 Télécharger Rapport TXT", report_txt, "neo_pir_report.txt", "text/plain")
            st.download_button("📥 Télécharger Rapport PDF", report_pdf, "neo_pir_report.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Erreur : {e}")
