from __future__ import annotations

import io
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from PIL import Image


# ============================================================
# NEO PI-R — OMR Clinique (Feuille sans bulles)
# - Lettres FD / D / N / A / FA entourées
# - Validité :
#   * Items vides >= 15 => INVALIDE
#   * N cochés >= 42     => INVALIDE (N réellement cochés, pas imputés)
# - Imputation scientifique :
#   * item vide => N (2 points via scoring_key)
# ============================================================

# ====================== SCORING KEY (COMPLET 1..240) ======================
# scoring_key[item] = [score_FD, score_D, score_N, score_A, score_FA]
scoring_key: Dict[int, List[int]] = {
    1: [4,3,2,1,0], 31: [0,1,2,3,4], 61: [4,3,2,1,0], 91: [0,1,2,3,4], 121: [4,3,2,1,0], 151: [0,1,2,3,4], 181: [4,3,2,1,0], 211: [0,1,2,3,4],
    2: [0,1,2,3,4], 32: [4,3,2,1,0], 62: [0,1,2,3,4], 92: [4,3,2,1,0], 122: [0,1,2,3,4], 152: [4,3,2,1,0], 182: [0,1,2,3,4], 212: [4,3,2,1,0],
    3: [0,1,2,3,4], 33: [4,3,2,1,0], 63: [0,1,2,3,4], 93: [4,3,2,1,0], 123: [0,1,2,3,4], 153: [4,3,2,1,0], 183: [0,1,2,3,4], 213: [4,3,2,1,0],
    4: [4,3,2,1,0], 34: [0,1,2,3,4], 64: [4,3,2,1,0], 94: [0,1,2,3,4], 124: [4,3,2,1,0], 154: [0,1,2,3,4], 184: [4,3,2,1,0], 214: [0,1,2,3,4],
    5: [0,1,2,3,4], 35: [4,3,2,1,0], 65: [0,1,2,3,4], 95: [4,3,2,1,0], 125: [0,1,2,3,4], 155: [4,3,2,1,0], 185: [0,1,2,3,4], 215: [4,3,2,1,0],
    6: [0,1,2,3,4], 36: [4,3,2,1,0], 66: [0,1,2,3,4], 96: [4,3,2,1,0], 126: [0,1,2,3,4], 156: [4,3,2,1,0], 186: [0,1,2,3,4], 216: [4,3,2,1,0],
    7: [4,3,2,1,0], 37: [0,1,2,3,4], 67: [4,3,2,1,0], 97: [0,1,2,3,4], 127: [4,3,2,1,0], 157: [0,1,2,3,4], 187: [4,3,2,1,0], 217: [0,1,2,3,4],
    8: [4,3,2,1,0], 38: [0,1,2,3,4], 68: [4,3,2,1,0], 98: [0,1,2,3,4], 128: [4,3,2,1,0], 158: [0,1,2,3,4], 188: [4,3,2,1,0], 218: [0,1,2,3,4],
    9: [0,1,2,3,4], 39: [4,3,2,1,0], 69: [0,1,2,3,4], 99: [4,3,2,1,0], 129: [0,1,2,3,4], 159: [4,3,2,1,0], 189: [0,1,2,3,4], 219: [4,3,2,1,0],
    10: [4,3,2,1,0], 40: [0,1,2,3,4], 70: [4,3,2,1,0], 100: [0,1,2,3,4], 130: [4,3,2,1,0], 160: [0,1,2,3,4], 190: [4,3,2,1,0], 220: [0,1,2,3,4],
    11: [4,3,2,1,0], 41: [0,1,2,3,4], 71: [4,3,2,1,0], 101: [0,1,2,3,4], 131: [4,3,2,1,0], 161: [0,1,2,3,4], 191: [4,3,2,1,0], 221: [0,1,2,3,4],
    12: [0,1,2,3,4], 42: [4,3,2,1,0], 72: [0,1,2,3,4], 102: [4,3,2,1,0], 132: [0,1,2,3,4], 162: [4,3,2,1,0], 192: [0,1,2,3,4], 222: [4,3,2,1,0],
    13: [0,1,2,3,4], 43: [4,3,2,1,0], 73: [0,1,2,3,4], 103: [4,3,2,1,0], 133: [0,1,2,3,4], 163: [4,3,2,1,0], 193: [0,1,2,3,4], 223: [4,3,2,1,0],
    14: [4,3,2,1,0], 44: [0,1,2,3,4], 74: [4,3,2,1,0], 104: [0,1,2,3,4], 134: [4,3,2,1,0], 164: [0,1,2,3,4], 194: [4,3,2,1,0], 224: [0,1,2,3,4],
    15: [0,1,2,3,4], 45: [4,3,2,1,0], 75: [0,1,2,3,4], 105: [4,3,2,1,0], 135: [0,1,2,3,4], 165: [4,3,2,1,0], 195: [0,1,2,3,4], 225: [4,3,2,1,0],
    16: [0,1,2,3,4], 46: [4,3,2,1,0], 76: [0,1,2,3,4], 106: [4,3,2,1,0], 136: [0,1,2,3,4], 166: [4,3,2,1,0], 196: [0,1,2,3,4], 226: [4,3,2,1,0],
    17: [4,3,2,1,0], 47: [0,1,2,3,4], 77: [4,3,2,1,0], 107: [0,1,2,3,4], 137: [4,3,2,1,0], 167: [0,1,2,3,4], 197: [4,3,2,1,0], 227: [0,1,2,3,4],
    18: [4,3,2,1,0], 48: [0,1,2,3,4], 78: [4,3,2,1,0], 108: [0,1,2,3,4], 138: [4,3,2,1,0], 168: [0,1,2,3,4], 198: [4,3,2,1,0], 228: [0,1,2,3,4],
    19: [0,1,2,3,4], 49: [4,3,2,1,0], 79: [0,1,2,3,4], 109: [4,3,2,1,0], 139: [0,1,2,3,4], 169: [4,3,2,1,0], 199: [0,1,2,3,4], 229: [4,3,2,1,0],
    20: [4,3,2,1,0], 50: [0,1,2,3,4], 80: [4,3,2,1,0], 110: [0,1,2,3,4], 140: [4,3,2,1,0], 170: [0,1,2,3,4], 200: [4,3,2,1,0], 230: [0,1,2,3,4],
    21: [4,3,2,1,0], 51: [0,1,2,3,4], 81: [4,3,2,1,0], 111: [0,1,2,3,4], 141: [4,3,2,1,0], 171: [0,1,2,3,4], 201: [4,3,2,1,0], 231: [0,1,2,3,4],
    22: [0,1,2,3,4], 52: [4,3,2,1,0], 82: [0,1,2,3,4], 112: [4,3,2,1,0], 142: [0,1,2,3,4], 172: [4,3,2,1,0], 202: [0,1,2,3,4], 232: [4,3,2,1,0],
    23: [0,1,2,3,4], 53: [4,3,2,1,0], 83: [0,1,2,3,4], 113: [4,3,2,1,0], 143: [0,1,2,3,4], 173: [4,3,2,1,0], 203: [0,1,2,3,4], 233: [4,3,2,1,0],
    24: [4,3,2,1,0], 54: [0,1,2,3,4], 84: [4,3,2,1,0], 114: [0,1,2,3,4], 144: [4,3,2,1,0], 174: [0,1,2,3,4], 204: [4,3,2,1,0], 234: [0,1,2,3,4],
    25: [0,1,2,3,4], 55: [4,3,2,1,0], 85: [0,1,2,3,4], 115: [4,3,2,1,0], 145: [0,1,2,3,4], 175: [4,3,2,1,0], 205: [0,1,2,3,4], 235: [4,3,2,1,0],
    26: [0,1,2,3,4], 56: [4,3,2,1,0], 86: [0,1,2,3,4], 116: [4,3,2,1,0], 146: [0,1,2,3,4], 176: [4,3,2,1,0], 206: [0,1,2,3,4], 236: [4,3,2,1,0],
    27: [4,3,2,1,0], 57: [0,1,2,3,4], 87: [4,3,2,1,0], 117: [0,1,2,3,4], 147: [4,3,2,1,0], 177: [0,1,2,3,4], 207: [4,3,2,1,0], 237: [0,1,2,3,4],
    28: [4,3,2,1,0], 58: [0,1,2,3,4], 88: [4,3,2,1,0], 118: [0,1,2,3,4], 148: [4,3,2,1,0], 178: [0,1,2,3,4], 208: [4,3,2,1,0], 238: [0,1,2,3,4],
    29: [0,1,2,3,4], 59: [4,3,2,1,0], 89: [0,1,2,3,4], 119: [4,3,2,1,0], 149: [0,1,2,3,4], 179: [4,3,2,1,0], 209: [0,1,2,3,4], 239: [4,3,2,1,0],
    30: [4,3,2,1,0], 60: [0,1,2,3,4], 90: [4,3,2,1,0], 120: [0,1,2,3,4], 150: [4,3,2,1,0], 180: [0,1,2,3,4], 210: [4,3,2,1,0], 240: [0,1,2,3,4]
}

# ====================== ITEM -> FACETTE (COMPLET 1..240) ======================
item_to_facette: Dict[int, str] = {
    1: 'N1', 31: 'N1', 61: 'N1', 91: 'N1', 121: 'N1', 151: 'N1', 181: 'N1', 211: 'N1',
    6: 'N2', 36: 'N2', 66: 'N2', 96: 'N2', 126: 'N2', 156: 'N2', 186: 'N2', 216: 'N2',
    11: 'N3', 41: 'N3', 71: 'N3', 101: 'N3', 131: 'N3', 161: 'N3', 191: 'N3', 221: 'N3',
    16: 'N4', 46: 'N4', 76: 'N4', 106: 'N4', 136: 'N4', 166: 'N4', 196: 'N4', 226: 'N4',
    21: 'N5', 51: 'N5', 81: 'N5', 111: 'N5', 141: 'N5', 171: 'N5', 201: 'N5', 231: 'N5',
    26: 'N6', 56: 'N6', 86: 'N6', 116: 'N6', 146: 'N6', 176: 'N6', 206: 'N6', 236: 'N6',

    2: 'E1', 32: 'E1', 62: 'E1', 92: 'E1', 122: 'E1', 152: 'E1', 182: 'E1', 212: 'E1',
    7: 'E2', 37: 'E2', 67: 'E2', 97: 'E2', 127: 'E2', 157: 'E2', 187: 'E2', 217: 'E2',
    12: 'E3', 42: 'E3', 72: 'E3', 102: 'E3', 132: 'E3', 162: 'E3', 192: 'E3', 222: 'E3',
    17: 'E4', 47: 'E4', 77: 'E4', 107: 'E4', 137: 'E4', 167: 'E4', 197: 'E4', 227: 'E4',
    22: 'E5', 52: 'E5', 82: 'E5', 112: 'E5', 142: 'E5', 172: 'E5', 202: 'E5', 232: 'E5',
    27: 'E6', 57: 'E6', 87: 'E6', 117: 'E6', 147: 'E6', 177: 'E6', 207: 'E6', 237: 'E6',

    3: 'O1', 33: 'O1', 63: 'O1', 93: 'O1', 123: 'O1', 153: 'O1', 183: 'O1', 213: 'O1',
    8: 'O2', 38: 'O2', 68: 'O2', 98: 'O2', 128: 'O2', 158: 'O2', 188: 'O2', 218: 'O2',
    13: 'O3', 43: 'O3', 73: 'O3', 103: 'O3', 133: 'O3', 163: 'O3', 193: 'O3', 223: 'O3',
    18: 'O4', 48: 'O4', 78: 'O4', 108: 'O4', 138: 'O4', 168: 'O4', 198: 'O4', 228: 'O4',
    23: 'O5', 53: 'O5', 83: 'O5', 113: 'O5', 143: 'O5', 173: 'O5', 203: 'O5', 233: 'O5',
    28: 'O6', 58: 'O6', 88: 'O6', 118: 'O6', 148: 'O6', 178: 'O6', 208: 'O6', 238: 'O6',

    4: 'A1', 34: 'A1', 64: 'A1', 94: 'A1', 124: 'A1', 154: 'A1', 184: 'A1', 214: 'A1',
    9: 'A2', 39: 'A2', 69: 'A2', 99: 'A2', 129: 'A2', 159: 'A2', 189: 'A2', 219: 'A2',
    14: 'A3', 44: 'A3', 74: 'A3', 104: 'A3', 134: 'A3', 164: 'A3', 194: 'A3', 224: 'A3',
    19: 'A4', 49: 'A4', 79: 'A4', 109: 'A4', 139: 'A4', 169: 'A4', 199: 'A4', 229: 'A4',
    24: 'A5', 54: 'A5', 84: 'A5', 114: 'A5', 144: 'A5', 174: 'A5', 204: 'A5', 234: 'A5',
    29: 'A6', 59: 'A6', 89: 'A6', 119: 'A6', 149: 'A6', 179: 'A6', 209: 'A6', 239: 'A6',

    5: 'C1', 35: 'C1', 65: 'C1', 95: 'C1', 125: 'C1', 155: 'C1', 185: 'C1', 215: 'C1',
    10: 'C2', 40: 'C2', 70: 'C2', 100: 'C2', 130: 'C2', 160: 'C2', 190: 'C2', 220: 'C2',
    15: 'C3', 45: 'C3', 75: 'C3', 105: 'C3', 135: 'C3', 165: 'C3', 195: 'C3', 225: 'C3',
    20: 'C4', 50: 'C4', 80: 'C4', 110: 'C4', 140: 'C4', 170: 'C4', 200: 'C4', 230: 'C4',
    25: 'C5', 55: 'C5', 85: 'C5', 115: 'C5', 145: 'C5', 175: 'C5', 205: 'C5', 235: 'C5',
    30: 'C6', 60: 'C6', 90: 'C6', 120: 'C6', 150: 'C6', 180: 'C6', 210: 'C6', 240: 'C6'
}

facettes_to_domain = {
    'N1': 'N', 'N2': 'N', 'N3': 'N', 'N4': 'N', 'N5': 'N', 'N6': 'N',
    'E1': 'E', 'E2': 'E', 'E3': 'E', 'E4': 'E', 'E5': 'E', 'E6': 'E',
    'O1': 'O', 'O2': 'O', 'O3': 'O', 'O4': 'O', 'O5': 'O', 'O6': 'O',
    'A1': 'A', 'A2': 'A', 'A3': 'A', 'A4': 'A', 'A5': 'A', 'A6': 'A',
    'C1': 'C', 'C2': 'C', 'C3': 'C', 'C4': 'C', 'C5': 'C', 'C6': 'C'
}

facette_labels = {
    'N1': 'N1 - Anxiété',
    'N2': 'N2 - Hostilité colérique',
    'N3': 'N3 - Dépression',
    'N4': 'N4 - Timidité',
    'N5': 'N5 - Impulsivité',
    'N6': 'N6 - Vulnérabilité',
    'E1': 'E1 - Chaleur',
    'E2': 'E2 - Grégarité',
    'E3': 'E3 - Affirmation de soi',
    'E4': 'E4 - Activité',
    'E5': "E5 - Recherche d'excitation",
    'E6': 'E6 - Émotions positives',
    'O1': 'O1 - Imagination',
    'O2': 'O2 - Esthétique',
    'O3': 'O3 - Sentiments',
    'O4': 'O4 - Actions',
    'O5': 'O5 - Idées',
    'O6': 'O6 - Valeurs',
    'A1': 'A1 - Confiance',
    'A2': 'A2 - Franchise',
    'A3': 'A3 - Altruisme',
    'A4': 'A4 - Compliance',
    'A5': 'A5 - Modestie',
    'A6': 'A6 - Tendresse',
    'C1': 'C1 - Compétence',
    'C2': 'C2 - Ordre',
    'C3': 'C3 - Sens du devoir',
    'C4': 'C4 - Effort pour réussir',
    'C5': 'C5 - Autodiscipline',
    'C6': 'C6 - Délibération'
}

domain_labels = {
    'N': 'Névrosisme',
    'E': 'Extraversion',
    'O': 'Ouverture',
    'A': 'Agréabilité',
    'C': 'Conscience'
}

CHOICES = ["FD", "D", "N", "A", "FA"]  # indices 0..4


# ==============================
# CONFIGS
# ==============================
@dataclass
class OMRConfig:
    # zone globale de la grille (ratios sur la feuille redressée)
    grid_left: float = 0.05
    grid_right: float = 0.95
    grid_top: float = 0.205
    grid_bottom: float = 0.86

    rows: int = 30
    cols: int = 8

    # centres relatifs des 5 lettres dans une case (FD D N A FA)
    option_centers: Tuple[float, float, float, float, float] = (0.14, 0.34, 0.54, 0.74, 0.92)

    # taille ROI dans une case (en ratio de la cellule)
    box_w_ratio: float = 0.18
    box_h_ratio: float = 0.62

    # seuil auto (encre)
    auto_threshold_factor: float = 1.8
    auto_threshold_floor: int = 280

    # détecter vide + ambigu + faible
    blank_detect_margin: float = 0.92
    ambiguity_rel_gap: float = 0.12
    weak_rel_margin: float = 1.35

    # imputation
    impute_blank_to_N: bool = True


@dataclass
class ValidityConfig:
    blank_invalid_threshold: int = 15
    neutral_invalid_threshold: int = 42


@dataclass
class OMRStats:
    total_items: int
    blank_count: int
    ambiguous_count: int
    weak_mark_count: int
    neutral_marked_count: int
    neutral_imputed_count: int
    threshold_ink: int
    ink_median: int
    ink_p10: int
    ink_p90: int


# ==============================
# IMAGE PIPELINE (smartphone)
# ==============================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    img_rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

def remove_shadows(gray: np.ndarray) -> np.ndarray:
    dil = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dil, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

def find_document_and_warp(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    doc = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc = approx
                break

    if doc is None:
        return img_bgr.copy(), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), False

    paper = four_point_transform(img_bgr, doc.reshape(4, 2))
    warped_gray = four_point_transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), doc.reshape(4, 2))
    return paper, warped_gray, True

def robust_threshold(gray: np.ndarray) -> np.ndarray:
    g = remove_shadows(gray)
    thr = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    thr = cv2.dilate(thr, kernel, iterations=1)
    return thr

def normalize_width(gray: np.ndarray, thr: np.ndarray, target_w: int = 1700) -> Tuple[np.ndarray, np.ndarray]:
    H, W = gray.shape[:2]
    scale = target_w / float(W)
    new_size = (target_w, max(1, int(H * scale)))
    return cv2.resize(gray, new_size), cv2.resize(thr, new_size)


# ==============================
# OMR ROIs (lettres)
# ==============================
def extract_inks(thr: np.ndarray, cfg: OMRConfig):
    H, W = thr.shape[:2]
    x0 = int(cfg.grid_left * W); x1 = int(cfg.grid_right * W)
    y0 = int(cfg.grid_top * H);  y1 = int(cfg.grid_bottom * H)

    x0 = max(0, min(W - 2, x0)); x1 = max(x0 + 1, min(W - 1, x1))
    y0 = max(0, min(H - 2, y0)); y1 = max(y0 + 1, min(H - 1, y1))

    grid_w = x1 - x0
    grid_h = y1 - y0
    cell_w = grid_w / cfg.cols
    cell_h = grid_h / cfg.rows

    all_inks: List[int] = []
    raw_items: List[Tuple[int, int, List[int], List[Tuple[int,int,int,int]]]] = []

    for r in range(cfg.rows):
        for c in range(cfg.cols):
            cx0 = int(x0 + c * cell_w)
            cy0 = int(y0 + r * cell_h)
            cw = int(cell_w)
            ch = int(cell_h)

            bw = max(8, int(cw * cfg.box_w_ratio))
            bh = max(8, int(ch * cfg.box_h_ratio))
            by = int(cy0 + (ch - bh) * 0.50)

            inks: List[int] = []
            rois: List[Tuple[int,int,int,int]] = []

            for oc in cfg.option_centers:
                bx = int(cx0 + oc * cw - bw // 2)
                bx = max(0, min(W - bw - 1, bx))
                by2 = max(0, min(H - bh - 1, by))

                roi = thr[by2:by2 + bh, bx:bx + bw]
                ink = int(cv2.countNonZero(roi))

                inks.append(ink)
                all_inks.append(ink)
                rois.append((bx, by2, bw, bh))

            raw_items.append((r, c, inks, rois))

    return all_inks, raw_items

def auto_threshold_from_inks(all_inks: List[int], cfg: OMRConfig) -> Tuple[int, int, int, int]:
    v = np.array(all_inks, dtype=np.float32)
    med = int(np.median(v))
    p10 = int(np.percentile(v, 10))
    p90 = int(np.percentile(v, 90))
    thr = max(cfg.auto_threshold_floor, int(med * cfg.auto_threshold_factor))
    return thr, med, p10, p90

def decide_responses(raw_items, thr_img: np.ndarray, cfg: OMRConfig, thr_ink: int, overlay: bool):
    responses: Dict[int, int] = {}
    mark_state: Dict[int, str] = {}  # marked / blank_imputed / blank_unimputed
    warnings: List[str] = []

    blank = 0
    ambiguous = 0
    weak = 0
    neutral_marked = 0
    neutral_imputed = 0

    ov = cv2.cvtColor(thr_img.copy(), cv2.COLOR_GRAY2BGR) if overlay else None
    blank_thr = int(thr_ink * cfg.blank_detect_margin)

    for (r, c, inks, rois) in raw_items:
        item_id = (r + 1) + 30 * c

        best_idx = int(np.argmax(inks))
        sorted_inks = sorted(inks, reverse=True)
        best_ink = int(sorted_inks[0])

        rel_gap = 1.0
        if sorted_inks[0] > 0:
            rel_gap = (sorted_inks[0] - sorted_inks[1]) / float(sorted_inks[0])

        # BLANK
        if best_ink < blank_thr:
            blank += 1
            if cfg.impute_blank_to_N:
                responses[item_id] = 2  # N
                neutral_imputed += 1
                mark_state[item_id] = "blank_imputed"
                warnings.append(f"Item {item_id}: non répondu → imputé à N.")
            else:
                responses[item_id] = best_idx
                mark_state[item_id] = "blank_unimputed"
                warnings.append(f"Item {item_id}: non répondu.")
        else:
            responses[item_id] = best_idx
            mark_state[item_id] = "marked"

            if best_idx == 2:
                neutral_marked += 1

            if best_ink < int(thr_ink * cfg.weak_rel_margin):
                weak += 1
                warnings.append(f"Item {item_id}: marquage faible (ink={best_ink}, seuil={thr_ink}).")

            if rel_gap < cfg.ambiguity_rel_gap:
                ambiguous += 1
                warnings.append(f"Item {item_id}: ambiguïté (gap relatif={rel_gap:.2f}).")

        if ov is not None:
            chosen = responses[item_id]
            for j, (bx, by, bw, bh) in enumerate(rois):
                col = (0, 200, 0) if j == chosen else (170, 170, 170)
                cv2.rectangle(ov, (bx, by), (bx + bw, by + bh), col, 1)

    stats = OMRStats(
        total_items=cfg.rows * cfg.cols,
        blank_count=blank,
        ambiguous_count=ambiguous,
        weak_mark_count=weak,
        neutral_marked_count=neutral_marked,
        neutral_imputed_count=neutral_imputed,
        threshold_ink=thr_ink,
        ink_median=0,
        ink_p10=0,
        ink_p90=0
    )
    return responses, mark_state, warnings, stats, ov


# ==============================
# VALIDITÉ + SCORING
# ==============================
def protocol_validity(stats: OMRStats, vcfg: ValidityConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if stats.blank_count >= vcfg.blank_invalid_threshold:
        reasons.append(f"Items non répondus: {stats.blank_count} (seuil ≥ {vcfg.blank_invalid_threshold}).")
    if stats.neutral_marked_count >= vcfg.neutral_invalid_threshold:
        reasons.append(f"Réponses N cochées: {stats.neutral_marked_count} (seuil ≥ {vcfg.neutral_invalid_threshold}).")
    return (len(reasons) == 0), reasons

def calculate_scores(responses: Dict[int, int]) -> Tuple[Dict[str,int], Dict[str,int]]:
    fac_scores: Dict[str, int] = {fac: 0 for fac in facette_labels.keys()}
    for item_id, opt in responses.items():
        fac = item_to_facette[item_id]
        fac_scores[fac] += scoring_key[item_id][opt]

    dom_scores: Dict[str, int] = {d: 0 for d in domain_labels.keys()}
    for fac, sc in fac_scores.items():
        dom_scores[facettes_to_domain[fac]] += sc
    return fac_scores, dom_scores


# ==============================
# EXPORTS
# ==============================
def export_csv(valid: bool, reasons: List[str], stats: OMRStats, fac_scores: Dict[str,int], dom_scores: Dict[str,int]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["SECTION", "LIBELLÉ", "VALEUR"])
    w.writerow(["VALIDITÉ", "Protocole", "VALIDE" if valid else "INVALIDE"])
    for r in reasons:
        w.writerow(["VALIDITÉ", "Raison", r])

    w.writerow([])
    w.writerow(["QUALITÉ", "Seuil encrage (auto)", stats.threshold_ink])
    w.writerow(["QUALITÉ", "Items vides", stats.blank_count])
    w.writerow(["QUALITÉ", "Ambiguïtés", stats.ambiguous_count])
    w.writerow(["QUALITÉ", "Marquages faibles", stats.weak_mark_count])
    w.writerow(["QUALITÉ", "N cochés", stats.neutral_marked_count])
    w.writerow(["QUALITÉ", "N imputés", stats.neutral_imputed_count])

    w.writerow([])
    w.writerow(["DOMAINES", "Domaine", "Score brut"])
    for d in sorted(dom_scores.keys()):
        w.writerow(["DOMAINES", domain_labels[d], dom_scores[d]])

    w.writerow([])
    w.writerow(["FACETTES", "Facette", "Score brut"])
    for fac in sorted(fac_scores.keys()):
        w.writerow(["FACETTES", facette_labels.get(fac, fac), fac_scores[fac]])

    return out.getvalue()

def export_txt(valid: bool, reasons: List[str], stats: OMRStats, fac_scores: Dict[str,int], dom_scores: Dict[str,int]) -> str:
    lines = []
    lines.append("RAPPORT NEO PI-R — OMR CLINIQUE (FEUILLE SANS BULLES)")
    lines.append("")
    lines.append("VALIDITÉ")
    lines.append("VALIDE" if valid else "INVALIDE")
    for r in reasons:
        lines.append(f"- {r}")

    lines.append("")
    lines.append("QUALITÉ DE LECTURE")
    lines.append(f"Seuil encrage auto: {stats.threshold_ink}")
    lines.append(f"Items vides: {stats.blank_count}/{stats.total_items}")
    lines.append(f"Ambiguïtés: {stats.ambiguous_count}")
    lines.append(f"Marquages faibles: {stats.weak_mark_count}")
    lines.append(f"N cochés: {stats.neutral_marked_count}")
    lines.append(f"N imputés: {stats.neutral_imputed_count}")

    lines.append("")
    lines.append("DOMAINES (scores bruts)")
    for d in sorted(dom_scores.keys()):
        lines.append(f"{domain_labels[d]}: {dom_scores[d]}")

    lines.append("")
    lines.append("FACETTES (scores bruts)")
    for fac in sorted(fac_scores.keys()):
        lines.append(f"{facette_labels.get(fac, fac)}: {fac_scores[fac]}")

    return "\n".join(lines)


# ==============================
# UI
# ==============================
st.set_page_config(page_title="NEO PI-R — OMR Clinique", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
      div.stButton > button { width: 100%; border-radius: 10px; padding: 0.75rem 1rem; font-weight: 650; }
      .card { border: 1px solid rgba(49, 51, 63, 0.14); border-radius: 14px; padding: 14px 16px; background: rgba(255,255,255,0.65); }
      .label { font-size: 12px; color: rgba(49, 51, 63, 0.65); margin-bottom: 4px; }
      .value { font-size: 22px; font-weight: 800; }
      .sub { font-size: 12px; color: rgba(49, 51, 63, 0.65); margin-top: 4px; }
      .footer { text-align:center; color: rgba(49,51,63,0.55); font-size: 12px; padding-top: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("NEO PI-R — OMR Clinique (Feuille sans bulles)")
st.caption("Lecture photo smartphone (lettres FD/D/N/A/FA entourées) + Validité automatique + Scoring brut.")

with st.sidebar:
    st.markdown("## Calibration (si nécessaire)")
    cfg = OMRConfig(
        grid_left=st.slider("Grille — gauche", 0.00, 0.20, 0.05, 0.005),
        grid_right=st.slider("Grille — droite", 0.80, 1.00, 0.95, 0.005),
        grid_top=st.slider("Grille — haut", 0.10, 0.35, 0.205, 0.005),
        grid_bottom=st.slider("Grille — bas", 0.70, 0.95, 0.86, 0.005),
        impute_blank_to_N=st.toggle("Imputer item vide → N (2 points)", value=True),
    )
    cfg.auto_threshold_factor = st.slider("Auto-seuil (facteur)", 1.2, 3.0, 1.8, 0.05)
    cfg.blank_detect_margin = st.slider("Détection vide (marge)", 0.70, 1.10, 0.92, 0.01)
    cfg.ambiguity_rel_gap = st.slider("Ambiguïté (gap relatif)", 0.02, 0.40, 0.12, 0.01)
    cfg.weak_rel_margin = st.slider("Marquage faible (marge)", 1.05, 2.00, 1.35, 0.05)

    st.markdown("---")
    st.markdown("## Règles validité")
    vcfg = ValidityConfig(
        blank_invalid_threshold=st.number_input("Invalide si items vides ≥", 0, 240, 15, 1),
        neutral_invalid_threshold=st.number_input("Invalide si N cochés ≥", 0, 240, 42, 1),
    )

    st.markdown("---")
    show_overlay = st.toggle("Afficher overlay ROIs", value=False)

uploaded = st.file_uploader("Importer la feuille (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    c1, c2 = st.columns([0.7, 0.3], vertical_alignment="top")
    with c1:
        raw_img = Image.open(uploaded)
        st.image(raw_img, caption="Image importée", use_container_width=True)
    with c2:
        run = st.button("Lancer l'analyse", type="primary")

    if run:
        try:
            pil_img = Image.open(uploaded)
            img_bgr = pil_to_bgr(pil_img)

            paper_bgr, warped_gray, doc_found = find_document_and_warp(img_bgr)
            thr = robust_threshold(warped_gray)
            warped_gray, thr = normalize_width(warped_gray, thr, target_w=1700)

            all_inks, raw_items = extract_inks(thr, cfg)
            thr_ink, med, p10, p90 = auto_threshold_from_inks(all_inks, cfg)

            responses, mark_state, warnings, stats, overlay = decide_responses(
                raw_items, thr, cfg, thr_ink, overlay=show_overlay
            )
            stats.ink_median = med
            stats.ink_p10 = p10
            stats.ink_p90 = p90

            # validité
            valid, reasons = protocol_validity(stats, vcfg)

            # scoring
            fac_scores, dom_scores = calculate_scores(responses)

            # KPIs
            response_rate = 100.0 * (1.0 - stats.blank_count / max(1, stats.total_items))
            quality_proxy = max(
                0.0,
                100.0 * (1.0 - (stats.ambiguous_count + 0.5 * stats.weak_mark_count) / max(1, stats.total_items))
            )

            st.markdown("### Résumé")
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(
                f"<div class='card'><div class='label'>Validité</div>"
                f"<div class='value'>{'VALIDE' if valid else 'INVALIDE'}</div>"
                f"<div class='sub'>Règles: vides ≥ {vcfg.blank_invalid_threshold} · N cochés ≥ {vcfg.neutral_invalid_threshold}</div></div>",
                unsafe_allow_html=True
            )
            k2.markdown(
                f"<div class='card'><div class='label'>Taux de réponse</div>"
                f"<div class='value'>{response_rate:.1f}%</div>"
                f"<div class='sub'>Vides: {stats.blank_count}/{stats.total_items}</div></div>",
                unsafe_allow_html=True
            )
            k3.markdown(
                f"<div class='card'><div class='label'>N cochés</div>"
                f"<div class='value'>{stats.neutral_marked_count}</div>"
                f"<div class='sub'>N imputés: {stats.neutral_imputed_count}</div></div>",
                unsafe_allow_html=True
            )
            k4.markdown(
                f"<div class='card'><div class='label'>Qualité lecture</div>"
                f"<div class='value'>{quality_proxy:.1f}%</div>"
                f"<div class='sub'>Ambiguïtés: {stats.ambiguous_count} · Faibles: {stats.weak_mark_count} · Seuil: {stats.threshold_ink}</div></div>",
                unsafe_allow_html=True
            )

            if valid:
                st.success("✅ Protocole valide.")
            else:
                st.error("❌ Protocole invalide.")
                for r in reasons:
                    st.warning(r)

            tab1, tab2, tab3, tab4 = st.tabs(["Scores", "Qualité & images", "Avertissements", "Exports"])

            with tab1:
                st.subheader("Domaines (scores bruts)")
                dom_table = [{"Domaine": domain_labels[d], "Score brut": dom_scores[d]} for d in sorted(dom_scores.keys())]
                st.dataframe(dom_table, use_container_width=True, hide_index=True)

                st.subheader("Facettes (scores bruts)")
                fac_table = [{"Facette": facette_labels.get(f, f), "Score brut": fac_scores[f]} for f in sorted(fac_scores.keys())]
                st.dataframe(fac_table, use_container_width=True, hide_index=True)

            with tab2:
                st.subheader("Contrôle technique")
                st.write({
                    "document_detecté": doc_found,
                    "seuil_encrage_auto": stats.threshold_ink,
                    "ink_médian": stats.ink_median,
                    "ink_p10": stats.ink_p10,
                    "ink_p90": stats.ink_p90,
                })

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.image(paper_bgr, channels="BGR", caption="Feuille redressée (perspective)", use_container_width=True)
                with cc2:
                    st.image(thr, clamp=True, caption="Binarisation (encre = blanc)", use_container_width=True)

                if show_overlay and overlay is not None:
                    st.image(overlay, channels="BGR", caption="Overlay ROIs (contrôle)", use_container_width=True)

            with tab3:
                st.subheader("Avertissements")
                if warnings:
                    with st.expander("Afficher", expanded=True):
                        for w in warnings[:600]:
                            st.warning(w)
                    if len(warnings) > 600:
                        st.info(f"{len(warnings)} avertissements. Affichage limité.")
                else:
                    st.success("Aucun avertissement.")

            with tab4:
                st.subheader("Exports")
                csv_text = export_csv(valid, reasons, stats, fac_scores, dom_scores)
                st.download_button("Télécharger CSV", csv_text, file_name="neo_pir_export.csv", mime="text/csv")

                txt_text = export_txt(valid, reasons, stats, fac_scores, dom_scores)
                st.download_button("Télécharger rapport TXT", txt_text, file_name="neo_pir_report.txt", mime="text/plain")

        except Exception as e:
            st.error(f"Erreur : {e}")

st.markdown("<div class='footer'>NEO PI-R —ADAOUN· © 2026</div>", unsafe_allow_html=True)
