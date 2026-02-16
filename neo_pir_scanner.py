# neo_pir_scanner_app.py
# ------------------------------------------------------------
# NEO PI-R — feuille unique (lettres FD/D/N/A/FA entourées)
# Détection par ENCRE (caméra) + grille 30x8 + 5 choix
# Règles protocole : N>=42 invalide ; vides>=15 invalide ; vides<10 => +2 points/imputation
# ------------------------------------------------------------

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import csv

# ============================================================
# 1) SCORING + MAPPINGS (COLLER ICI TES TABLES COMPLETES)
# ============================================================

# ---------------------- scoring_key (1..240) ----------------------
# option_idx: 0=FD, 1=D, 2=N, 3=A, 4=FA
scoring_key = {
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

# ---------------------- item_to_facette (1..240) ----------------------
item_to_facette = {
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

facettes_to_domain = {**{f"N{i}": "N" for i in range(1,7)},
                      **{f"E{i}": "E" for i in range(1,7)},
                      **{f"O{i}": "O" for i in range(1,7)},
                      **{f"A{i}": "A" for i in range(1,7)},
                      **{f"C{i}": "C" for i in range(1,7)}}

facette_labels = {
    'N1': 'N1 - Anxiété', 'N2': 'N2 - Hostilité colérique', 'N3': 'N3 - Dépression',
    'N4': 'N4 - Timidité', 'N5': 'N5 - Impulsivité', 'N6': 'N6 - Vulnérabilité',
    'E1': 'E1 - Chaleur', 'E2': 'E2 - Grégarité', 'E3': 'E3 - Affirmation de soi',
    'E4': 'E4 - Activité', 'E5': "E5 - Recherche d'excitation", 'E6': 'E6 - Émotions positives',
    'O1': 'O1 - Imagination', 'O2': 'O2 - Esthétique', 'O3': 'O3 - Sentiments',
    'O4': 'O4 - Actions', 'O5': 'O5 - Idées', 'O6': 'O6 - Valeurs',
    'A1': 'A1 - Confiance', 'A2': 'A2 - Franchise', 'A3': 'A3 - Altruisme',
    'A4': 'A4 - Compliance', 'A5': 'A5 - Modestie', 'A6': 'A6 - Tendresse',
    'C1': 'C1 - Compétence', 'C2': 'C2 - Ordre', 'C3': 'C3 - Sens du devoir',
    'C4': 'C4 - Effort pour réussir', 'C5': 'C5 - Autodiscipline', 'C6': 'C6 - Délibération'
}

domain_labels = {'N': 'Névrosisme', 'E': 'Extraversion', 'O': 'Ouverture', 'A': 'Agréabilité', 'C': 'Conscience'}

# ============================================================
# 2) VISION — REDRESSEMENT + MASQUE ENCRE (PAS DE "BULLES")
# ============================================================

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def warp_document(image_bgr: np.ndarray) -> np.ndarray:
    """Détecte le contour feuille, perspective transform, retourne image redressée."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("Aucun contour détecté (photo trop sombre/floue).")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    doc = None
    for c in cnts[:15]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break
    if doc is None:
        raise ValueError("Contour feuille non détecté (essaye photo plus cadrée).")

    pts = _order_points(doc)
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxW, maxH))

    # Normalisation taille (stabilité ROI)
    target_w = 1600
    scale = target_w / warped.shape[1]
    warped = cv2.resize(warped, (target_w, int(warped.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return warped

def crop_table(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Trouve la grande zone grille (tableau) via lignes/contours,
    sinon fallback : coupe zone centrale (robuste si photo correcte).
    """
    g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    bin_inv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)

    # renforcer les bordures (morph)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thick = cv2.dilate(bin_inv, kernel, iterations=1)

    cnts, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return warped_bgr

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    h, w = warped_bgr.shape[:2]

    # Cherche un rectangle "table" (pas toute la page)
    for c in cnts[:20]:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 0.20 * (w * h):
            continue
        if area > 0.98 * (w * h):
            continue
        # table plutôt large et haute
        if ww > 0.75 * w and hh > 0.45 * h:
            pad = 6
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(w, x + ww + pad); y1 = min(h, y + hh + pad)
            return warped_bgr[y0:y1, x0:x1]

    # fallback : zone centrale où se trouve la grille
    y0 = int(0.18 * h); y1 = int(0.78 * h)
    x0 = int(0.03 * w); x1 = int(0.97 * w)
    return warped_bgr[y0:y1, x0:x1]

def ink_mask(table_bgr: np.ndarray, sat_min: int, val_max: int) -> np.ndarray:
    """
    Masque encre (stylo) basé sur HSV (saturation) + un peu de bleu,
    pour ignorer le texte imprimé noir/gris autant que possible.
    """
    hsv = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # encre colorée (bleu/noir bleuté) : saturation suffisante, valeur pas trop haute
    m1 = ((S >= sat_min) & (V <= val_max)).astype(np.uint8) * 255

    # bonus : bleu typique
    lower_blue = np.array([90, 35, 40])
    upper_blue = np.array([145, 255, 255])
    m2 = cv2.inRange(hsv, lower_blue, upper_blue)

    m = cv2.bitwise_or(m1, m2)

    # nettoyage
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.dilate(m, k, iterations=1)
    return m

def kmeans_1d(data: np.ndarray, k: int = 5, iters: int = 30) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if len(data) < k:
        # fallback centres réguliers
        return np.linspace(0.15, 0.90, k).astype(np.float32)

    centers = np.quantile(data, np.linspace(0.05, 0.95, k))
    for _ in range(iters):
        dists = np.abs(data[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([data[labels == j].mean() if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers
    return np.sort(centers).astype(np.float32)

def read_responses(table_bgr: np.ndarray, mask_ink: np.ndarray,
                   rows: int = 30, cols: int = 8, choices: int = 5,
                   xwin: int = 14, y_margin: int = 2,
                   min_pixels: int = 18,
                   ambiguity_ratio: float = 1.15):
    """
    Lecture par grille uniforme + centres X estimés par colonne (kmeans sur l’encre).
    Retourne:
      - responses: dict item_id -> option_idx (0..4) ou None si vide
      - empty_items: list
      - ambiguous_items: list
      - overlay: image BGR avec contrôle (vert sélection, rouge vide)
    """
    H, W = mask_ink.shape[:2]
    col_edges = [int(i * W / cols) for i in range(cols + 1)]
    row_edges = [int(i * H / rows) for i in range(rows + 1)]

    # centres options (X) par colonne via clustering des pixels d’encre
    centers_by_col = []
    for c in range(cols):
        x0, x1 = col_edges[c], col_edges[c + 1]
        sub = mask_ink[:, x0:x1]
        ys, xs = np.where(sub > 0)
        if len(xs) < 200:
            # fallback centres "à la main" (relatifs)
            centers_by_col.append(np.array([0.22, 0.40, 0.58, 0.74, 0.90]) * (x1 - x0))
        else:
            centers_by_col.append(kmeans_1d(xs, k=choices))

    responses = {}
    empty_items = []
    ambiguous_items = []

    overlay = table_bgr.copy()

    for r in range(rows):
        y0 = row_edges[r] + y_margin
        y1 = row_edges[r + 1] - y_margin
        y0 = max(0, y0); y1 = min(H, y1)

        for c in range(cols):
            x0, x1 = col_edges[c], col_edges[c + 1]
            centers = centers_by_col[c]

            counts = []
            for ctr in centers:
                xx0 = int(max(0, ctr - xwin))
                xx1 = int(min((x1 - x0) - 1, ctr + xwin))
                roi = mask_ink[y0:y1, x0 + xx0:x0 + xx1]
                counts.append(int(roi.sum() / 255))

            best_idx = int(np.argmax(counts))
            best_val = counts[best_idx]
            sorted_counts = sorted(counts, reverse=True)

            item_id = (r + 1) + 30 * c  # colonne 0: 1..30 ; col1: 31..60 ; ... ; col7: 211..240

            if best_val < min_pixels:
                responses[item_id] = None
                empty_items.append(item_id)
                # dessine rouge (zone item)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
                continue

            # ambiguïté si le top1 trop proche du top2
            if len(sorted_counts) > 1 and sorted_counts[0] < ambiguity_ratio * sorted_counts[1]:
                ambiguous_items.append(item_id)

            responses[item_id] = best_idx

            # dessine vert autour de l'option gagnante
            ctr = centers[best_idx]
            xx0 = int(max(0, ctr - xwin))
            xx1 = int(min((x1 - x0) - 1, ctr + xwin))
            cv2.rectangle(overlay, (x0 + xx0, y0), (x0 + xx1, y1), (0, 255, 0), 2)

    return responses, empty_items, ambiguous_items, overlay

# ============================================================
# 3) SCORING + PROTOCOLE (tes règles)
# ============================================================

def compute_protocol(responses: dict, empty_items: list, n_limit: int, empty_limit: int, impute_if_lt: int):
    """
    - N_count = nombre de réponses N (idx=2)
    - invalid si N_count >= n_limit ou empty >= empty_limit
    - imputation : si empty < impute_if_lt => +2 points par item vide (sans inventer une réponse)
    """
    n_count = sum(1 for _, v in responses.items() if v == 2)
    empty_count = len(empty_items)

    invalid_reasons = []
    if n_count >= n_limit:
        invalid_reasons.append(f"Trop de réponses 'N' : {n_count} (>= {n_limit})")
    if empty_count >= empty_limit:
        invalid_reasons.append(f"Trop d'items vides : {empty_count} (>= {empty_limit})")

    is_valid = (len(invalid_reasons) == 0)

    imputations = 0
    imputation_points = 0
    if empty_count < impute_if_lt:
        imputations = empty_count
        imputation_points = 2 * imputations

    return {
        "N_count": n_count,
        "empty_count": empty_count,
        "is_valid": is_valid,
        "reasons": invalid_reasons,
        "imputations": imputations,
        "imputation_points": imputation_points
    }

def calculate_scores(responses: dict, empty_items: list, imputation_points: int):
    """
    Score brut facettes/domaines selon scoring_key.
    Items vides: pas de score via key, mais on ajoute imputation_points au total domaine/facette?
    Tu as dit: "chaque item vide donne 2 points" => c'est une correction globale.
    Ici: on ajoute cette correction au TOTAL GLOBAL uniquement + on l'affiche.
    (Si tu veux répartir par domaine, dis-moi la règle exacte.)
    """
    facette_scores = {fac: 0 for fac in facette_labels.keys()}

    for item_id, option_idx in responses.items():
        if option_idx is None:
            continue
        if item_id in scoring_key and item_id in item_to_facette:
            score = scoring_key[item_id][option_idx]
            fac = item_to_facette[item_id]
            facette_scores[fac] += score

    domain_scores = {dom: 0 for dom in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        dom = facettes_to_domain[fac]
        domain_scores[dom] += sc

    total_raw = sum(domain_scores.values())
    total_with_imputation = total_raw + imputation_points

    return facette_scores, domain_scores, total_raw, total_with_imputation

# ============================================================
# 4) UI STREAMLIT (PRO)
# ============================================================

st.set_page_config(page_title="NEO PI-R Scanner", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;}
.kpi{border:1px solid rgba(49,51,63,.15);border-radius:14px;padding:14px 16px;background:rgba(255,255,255,.6);}
.kpi-label{font-size:12px;color:rgba(49,51,63,.6);margin-bottom:4px;}
.kpi-value{font-size:22px;font-weight:800;}
.small{font-size:12px;color:rgba(49,51,63,.6);}
</style>
""", unsafe_allow_html=True)

st.title("NEO PI-R — Scanner & Calculateur (Feuille unique)")
st.caption("Lecture caméra (lettres entourées) • Contrôle protocole • Scores facettes/domaines")

with st.sidebar:
    st.markdown("## Paramètres lecture")
    sat_min = st.slider("Saturation min (détecter encre)", 10, 120, 45)
    val_max = st.slider("Valeur max (rejeter papier clair)", 120, 255, 225)
    min_pixels = st.slider("Seuil item répondu (pixels encre)", 5, 80, 18)
    xwin = st.slider("Largeur fenêtre option (px)", 6, 30, 14)
    ambiguity_ratio = st.slider("Seuil ambiguïté (top1/top2)", 1.01, 2.0, 1.15, 0.01)

    st.markdown("---")
    st.markdown("## Règles protocole")
    n_limit = st.number_input("Invalide si N ≥", min_value=1, max_value=240, value=42, step=1)
    empty_limit = st.number_input("Invalide si items vides ≥", min_value=1, max_value=240, value=15, step=1)
    impute_if_lt = st.number_input("Imputation (+2/vides) si vides <", min_value=0, max_value=240, value=10, step=1)

    st.markdown("---")
    debug = st.toggle("Mode debug", value=False)

uploaded = st.file_uploader("Importer la photo/scan de la feuille (JPG/PNG)", type=["jpg", "jpeg", "png"])

run = st.button("Scanner & Calculer", type="primary", disabled=(uploaded is None))

if uploaded and run:
    try:
        pil = Image.open(uploaded).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        warped = warp_document(img_bgr)
        table = crop_table(warped)
        m = ink_mask(table, sat_min=sat_min, val_max=val_max)

        responses, empty_items, ambiguous_items, overlay = read_responses(
            table_bgr=table,
            mask_ink=m,
            min_pixels=min_pixels,
            xwin=xwin,
            ambiguity_ratio=ambiguity_ratio
        )

        proto = compute_protocol(
            responses=responses,
            empty_items=empty_items,
            n_limit=int(n_limit),
            empty_limit=int(empty_limit),
            impute_if_lt=int(impute_if_lt)
        )

        fac_scores, dom_scores, total_raw, total_with_imp = calculate_scores(
            responses=responses,
            empty_items=empty_items,
            imputation_points=proto["imputation_points"]
        )

        # ---------------- KPIs ----------------
        c1, c2, c3, c4 = st.columns(4)
        status = "VALIDE" if proto["is_valid"] else "INVALIDE"
        c1.markdown(f"<div class='kpi'><div class='kpi-label'>Statut protocole</div><div class='kpi-value'>{status}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi'><div class='kpi-label'>Items vides</div><div class='kpi-value'>{proto['empty_count']}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='kpi'><div class='kpi-label'>N observés (idx=2)</div><div class='kpi-value'>{proto['N_count']}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='kpi'><div class='kpi-label'>Imputations</div><div class='kpi-value'>{proto['imputations']}</div><div class='small'>+{proto['imputation_points']} pts</div></div>", unsafe_allow_html=True)

        if not proto["is_valid"]:
            st.error("Protocole INVALIDE")
            for r in proto["reasons"]:
                st.write("•", r)
        else:
            st.success("Protocole VALIDE")

        if ambiguous_items:
            st.warning(f"Ambiguïtés possibles sur {len(ambiguous_items)} items (double marquage / encre faible).")

        # ---------------- Tabs ----------------
        tab1, tab2, tab3 = st.tabs(["Contrôle visuel", "Scores", "Exports"])

        with tab1:
            a, b = st.columns(2)
            with a:
                st.subheader("Image redressée (table)")
                st.image(cv2.cvtColor(table, cv2.COLOR_BGR2RGB), use_container_width=True)
            with b:
                st.subheader("Overlay contrôle (Vert=réponse, Rouge=vide)")
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            if debug:
                st.subheader("Masque encre (debug)")
                st.image(m, clamp=True, use_container_width=True)

        with tab2:
            st.subheader("Scores par facette")
            fac_rows = []
            for fac in sorted(facette_labels.keys()):
                items = [str(k) for k, v in item_to_facette.items() if v == fac]
                fac_rows.append({"Facette": facette_labels[fac], "Items": ", ".join(items), "Score brut": fac_scores[fac]})
            st.dataframe(fac_rows, use_container_width=True, hide_index=True)

            st.subheader("Scores domaines")
            dom_rows = [{"Domaine": domain_labels[d], "Score": dom_scores[d]} for d in sorted(dom_scores.keys())]
            st.dataframe(dom_rows, use_container_width=True, hide_index=True)

            st.markdown(f"**Total brut : {total_raw}**")
            st.markdown(f"**Total avec imputation : {total_with_imp}**  (imputation: +{proto['imputation_points']} pts)")

        with tab3:
            # Export CSV
            out = io.StringIO()
            w = csv.writer(out)
            w.writerow(["STATUT_PROTOCOLE", status])
            w.writerow(["ITEMS_VIDES", proto["empty_count"]])
            w.writerow(["N_OBSERVES", proto["N_count"]])
            w.writerow(["IMPUTATIONS", proto["imputations"]])
            w.writerow(["IMPUTATION_POINTS", proto["imputation_points"]])
            w.writerow([])
            w.writerow(["FACETTE", "SCORE_BRUT"])
            for fac in sorted(facette_labels.keys()):
                w.writerow([facette_labels[fac], fac_scores[fac]])
            w.writerow([])
            w.writerow(["DOMAINE", "SCORE"])
            for d in sorted(dom_scores.keys()):
                w.writerow([domain_labels[d], dom_scores[d]])
            w.writerow([])
            w.writerow(["TOTAL_BRUT", total_raw])
            w.writerow(["TOTAL_AVEC_IMPUTATION", total_with_imp])

            st.download_button("Télécharger CSV", out.getvalue(), "neo_pir_results.csv", "text/csv")

            # Export TXT
            txt = []
            txt.append("RAPPORT NEO PI-R")
            txt.append(f"Statut protocole: {status}")
            if proto["reasons"]:
                txt.append("Raisons invalidité:")
                for r in proto["reasons"]:
                    txt.append(f"- {r}")
            txt.append("")
            txt.append(f"Items vides: {proto['empty_count']}")
            txt.append(f"N observés (idx=2): {proto['N_count']}")
            txt.append(f"Imputations: {proto['imputations']} (+{proto['imputation_points']} pts)")
            txt.append("")
            txt.append("SCORES FACETTES")
            for fac in sorted(facette_labels.keys()):
                txt.append(f"{facette_labels[fac]}: {fac_scores[fac]}")
            txt.append("")
            txt.append("SCORES DOMAINES")
            for d in sorted(dom_scores.keys()):
                txt.append(f"{domain_labels[d]}: {dom_scores[d]}")
            txt.append("")
            txt.append(f"TOTAL BRUT: {total_raw}")
            txt.append(f"TOTAL AVEC IMPUTATION: {total_with_imp}")
            if ambiguous_items:
                txt.append("")
                txt.append(f"Ambiguïtés (à vérifier): {', '.join(map(str, ambiguous_items[:80]))}" + (" ..." if len(ambiguous_items) > 80 else ""))

            st.download_button("Télécharger rapport TXT", "\n".join(txt), "neo_pir_report.txt", "text/plain")

    except Exception as e:
        st.error(f"Erreur : {e}")

st.caption("NEO PI-R — Scanner & Calculateur • Offline • Feuille unique")
