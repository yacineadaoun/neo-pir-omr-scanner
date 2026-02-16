import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils.perspective import four_point_transform
import io
import csv

# ============================================================
# 1) SCORING KEY (complet 1..240) : FD/D/N/A/FA -> 0..4
# Ici option_idx = 0..4 correspond à (FD, D, N, A, FA)
# Ta feuille "0..4" / "4..0" est déjà encodée par scoring_key
# ============================================================
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

# ============================================================
# 2) item_to_facette complet (1..240) : tu l’avais déjà complet
# ============================================================
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

facettes_to_domain = {f: f[0] for f in ['N1','N2','N3','N4','N5','N6','E1','E2','E3','E4','E5','E6','O1','O2','O3','O4','O5','O6','A1','A2','A3','A4','A5','A6','C1','C2','C3','C4','C5','C6']}

facette_labels = {
    'N1':'N1 - Anxiété','N2':'N2 - Hostilité','N3':'N3 - Dépression','N4':'N4 - Timidité','N5':'N5 - Impulsivité','N6':'N6 - Vulnérabilité',
    'E1':'E1 - Chaleur','E2':'E2 - Grégarité','E3':'E3 - Affirmation','E4':'E4 - Activité','E5':"E5 - Excitation",'E6':'E6 - Émotions +',
    'O1':'O1 - Imagination','O2':'O2 - Esthétique','O3':'O3 - Sentiments','O4':'O4 - Actions','O5':'O5 - Idées','O6':'O6 - Valeurs',
    'A1':'A1 - Confiance','A2':'A2 - Franchise','A3':'A3 - Altruisme','A4':'A4 - Compliance','A5':'A5 - Modestie','A6':'A6 - Tendresse',
    'C1':'C1 - Compétence','C2':'C2 - Ordre','C3':'C3 - Devoir','C4':'C4 - Réussite','C5':'C5 - Autodiscipline','C6':'C6 - Délibération'
}
domain_labels = {'N':'Névrosisme','E':'Extraversion','O':'Ouverture','A':'Agréabilité','C':'Conscience'}

# ============================================================
# 3) PREPROCESS robuste (caméra)
# ============================================================
def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 60, 160)

    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    docCnt = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        # fallback = image entière
        warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        paper = img.copy()
    else:
        paper = four_point_transform(img, docCnt.reshape(4,2))
        warped = four_point_transform(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), docCnt.reshape(4,2))

    # Adaptive threshold + légère morpho pour stylo
    thr = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    return paper, thr

# ============================================================
# 4) Découpage grille (30 x 8) + choix (5)
#    IMPORTANT : réglable par ratios (selon cadrage)
# ============================================================
def get_grid_roi(paper_bgr, thr,
                 top=0.205, bottom=0.83,
                 left=0.045, right=0.965):
    """
    Ces ratios marchent bien sur tes photos.
    Si ta photo est trop zoomée/décalée, ajuste un peu.
    """
    h, w = thr.shape[:2]
    y1 = int(h * top)
    y2 = int(h * bottom)
    x1 = int(w * left)
    x2 = int(w * right)
    return (x1,y1,x2,y2)

def read_item_choices(thr, cell, choices=5,
                      ink_min=250, ambiguity_ratio=0.10):
    """
    Retourne:
    - idx choisi (0..4) ou None si vide
    - scores d'encre par choix
    - warning si ambigu
    """
    x1,y1,x2,y2 = cell
    roi = thr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, [0]*choices, "roi vide"

    # découper en 5 sous-zones égales
    h, w = roi.shape
    step = w / choices

    inks = []
    for i in range(choices):
        sx1 = int(i * step)
        sx2 = int((i+1) * step)

        sub = roi[:, sx1:sx2]

        # on mesure l'encre dans une zone centrale (évite bords)
        hh, ww = sub.shape
        cx1 = int(ww*0.15); cx2 = int(ww*0.85)
        cy1 = int(hh*0.10); cy2 = int(hh*0.90)
        core = sub[cy1:cy2, cx1:cx2]

        inks.append(int(cv2.countNonZero(core)))

    best = int(np.argmax(inks))
    best_val = inks[best]
    sorted_inks = sorted(inks, reverse=True)

    # item vide
    if best_val < ink_min:
        return None, inks, None

    # ambiguïté
    warn = None
    if len(sorted_inks) > 1:
        if (sorted_inks[0] - sorted_inks[1]) < (sorted_inks[0] * ambiguity_ratio):
            warn = "ambiguïté"

    return best, inks, warn

def scan_sheet(thr, paper_bgr,
               grid_top=0.205, grid_bottom=0.83,
               grid_left=0.045, grid_right=0.965,
               ink_min=250, ambiguity_ratio=0.10):
    """
    Lit 240 items via 30 lignes x 8 colonnes.
    item_id = (row+1) + 30*col  (col=0 => 1..30, col=1 => 31..60, etc.)
    """
    x1,y1,x2,y2 = get_grid_roi(paper_bgr, thr, grid_top, grid_bottom, grid_left, grid_right)
    grid = thr[y1:y2, x1:x2]
    H, W = grid.shape

    rows, cols, choices = 30, 8, 5
    cell_h = H / rows
    cell_w = W / cols

    responses = {}
    warnings = []
    overlay = paper_bgr.copy()

    for r in range(rows):
        for c in range(cols):
            item_id = (r+1) + 30*c

            cx1 = x1 + int(c*cell_w)
            cx2 = x1 + int((c+1)*cell_w)
            cy1 = y1 + int(r*cell_h)
            cy2 = y1 + int((r+1)*cell_h)

            # un peu de padding interne (évite les traits)
            pad_x = int((cx2-cx1)*0.06)
            pad_y = int((cy2-cy1)*0.12)
            cell = (cx1+pad_x, cy1+pad_y, cx2-pad_x, cy2-pad_y)

            idx, inks, warn = read_item_choices(
                thr, cell, choices=choices,
                ink_min=ink_min,
                ambiguity_ratio=ambiguity_ratio
            )

            responses[item_id] = idx  # None si vide

            # overlay: cellule
            color = (0,255,0) if idx is not None else (0,0,255)
            cv2.rectangle(overlay, (cell[0], cell[1]), (cell[2], cell[3]), color, 1)

            if warn:
                warnings.append(f"Item {item_id}: {warn} (inks={inks})")

            if idx is None:
                warnings.append(f"Item {item_id}: non répondu")

    return responses, warnings, overlay

# ============================================================
# 5) Validité + imputation + scores
# ============================================================
def apply_validity_and_impute(responses, max_missing_valid=14, max_neutral=41, impute_if_missing_lt=10):
    """
    Règles demandées:
    - missing >= 15 -> invalide
    - N (score=2) >= 42 -> invalide
    - missing < 10 -> chaque item vide = 2 points (imputation score=2)
    """
    missing = [k for k,v in responses.items() if v is None]
    missing_count = len(missing)

    # compter N sur réponses EXISTANTES (sans imputation)
    neutral_count_observed = sum(1 for k,v in responses.items() if v == 2)

    valid = True
    reasons = []

    if missing_count >= (max_missing_valid+1):
        valid = False
        reasons.append(f"Trop d'items vides: {missing_count} (>= 15)")

    if neutral_count_observed >= (max_neutral+1):
        valid = False
        reasons.append(f"Trop de réponses 'N' (score=2): {neutral_count_observed} (>= 42)")

    imputed = []
    if missing_count < impute_if_missing_lt:
        # impute en "N" (idx=2)
        for k in missing:
            responses[k] = 2
            imputed.append(k)

    return valid, reasons, missing_count, neutral_count_observed, imputed

def calculate_scores(responses):
    facette_scores = {fac: 0 for fac in facette_labels.keys()}

    for item_id, idx in responses.items():
        if idx is None:
            continue
        if item_id in scoring_key and item_id in item_to_facette:
            score = scoring_key[item_id][idx]
            fac = item_to_facette[item_id]
            facette_scores[fac] += score

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc

    return facette_scores, domain_scores

# ============================================================
# 6) STREAMLIT UI
# ============================================================
st.set_page_config(page_title="NEO PI-R Scanner", page_icon="🧠", layout="wide")
st.title("NEO PI-R Scanner — Feuille 240 items (scan photo, sans bulles)")

with st.sidebar:
    st.header("Paramètres de lecture")
    grid_top = st.slider("Grid top (ratio)", 0.10, 0.35, 0.205, 0.005)
    grid_bottom = st.slider("Grid bottom (ratio)", 0.60, 0.95, 0.83, 0.005)
    grid_left = st.slider("Grid left (ratio)", 0.00, 0.20, 0.045, 0.005)
    grid_right = st.slider("Grid right (ratio)", 0.80, 1.00, 0.965, 0.005)

    st.header("Qualité & ambiguïtés")
    ink_min = st.slider("Seuil encre (item vide)", 50, 1200, 250, 10)
    ambiguity_ratio = st.slider("Ambiguïté (ratio)", 0.02, 0.35, 0.10, 0.01)

    st.header("Validité protocole")
    st.caption("Règles: vides>=15 invalide, N>=42 invalide, vides<10 -> imputation=2")
    debug = st.toggle("Afficher debug (warnings détaillés)", value=False)

uploaded = st.file_uploader("📷 Charger une photo (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded:
    pil = Image.open(uploaded)
    paper, thr = preprocess_image(pil)

    st.subheader("Aperçu")
    c1, c2 = st.columns(2)
    with c1:
        st.image(pil, caption="Original", use_container_width=True)
    with c2:
        st.image(thr, caption="Binarisation (thr)", use_container_width=True)

    if st.button("🚀 Scanner & Calculer", type="primary"):
        responses, warnings, overlay = scan_sheet(
            thr, paper,
            grid_top=grid_top, grid_bottom=grid_bottom,
            grid_left=grid_left, grid_right=grid_right,
            ink_min=ink_min,
            ambiguity_ratio=ambiguity_ratio
        )

        valid, reasons, missing_count, neutral_count, imputed = apply_validity_and_impute(responses)

        fac_scores, dom_scores = calculate_scores(responses)

        st.subheader("Résultats")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Items vides", missing_count)
        k2.metric("N observés (idx=2)", neutral_count)
        k3.metric("Imputations (score=2)", len(imputed))
        k4.metric("Statut protocole", "VALIDE" if valid else "INVALIDE")

        if not valid:
            st.error("Protocole INVALIDE")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.success("Protocole VALIDE (selon règles)")

        st.subheader("Overlay contrôle")
        st.image(overlay, channels="BGR", caption="Vert=réponse détectée | Rouge=vide", use_container_width=True)

        # Tables scores
        st.subheader("Scores par facette")
        fac_table = []
        for fac in sorted(facette_labels.keys()):
            fac_table.append({"Facette": facette_labels[fac], "Score brut": fac_scores[fac]})
        st.dataframe(fac_table, use_container_width=True, hide_index=True)

        st.subheader("Scores domaines")
        dom_table = [{"Domaine": domain_labels[d], "Score": dom_scores[d]} for d in sorted(domain_labels.keys())]
        st.dataframe(dom_table, use_container_width=True, hide_index=True)

        # Warnings
        st.subheader("Avertissements")
        if warnings:
            if debug:
                for w in warnings:
                    st.warning(w)
            else:
                st.warning(f"{len(warnings)} avertissements (active Debug pour le détail).")
        else:
            st.success("Aucun avertissement.")

        # Export CSV + TXT
        csv_buf = io.StringIO()
        w = csv.writer(csv_buf)
        w.writerow(["Item", "OptionIdx(0..4)", "Score", "Facette"])
        for item in range(1, 241):
            idx = responses.get(item, None)
            if idx is None:
                score = ""
                fac = item_to_facette.get(item, "")
            else:
                score = scoring_key[item][idx]
                fac = item_to_facette.get(item, "")
            w.writerow([item, idx if idx is not None else "", score, fac])

        st.download_button("📥 Télécharger réponses CSV", csv_buf.getvalue(), "neo_pir_responses.csv", "text/csv")

        report_lines = []
        report_lines.append("RAPPORT NEO PI-R")
        report_lines.append("")
        report_lines.append(f"Statut protocole: {'VALIDE' if valid else 'INVALIDE'}")
        report_lines.append(f"Items vides: {missing_count}")
        report_lines.append(f"N observés (idx=2): {neutral_count}")
        report_lines.append(f"Imputations (score=2): {len(imputed)}")
        if reasons:
            report_lines.append("Raisons invalidité:")
            report_lines.extend([f"- {r}" for r in reasons])

        report_lines.append("")
        report_lines.append("SCORES DOMAINES")
        for d in sorted(dom_scores.keys()):
            report_lines.append(f"{domain_labels[d]}: {dom_scores[d]}")

        report_lines.append("")
        report_lines.append("SCORES FACETTES")
        for fac in sorted(fac_scores.keys()):
            report_lines.append(f"{facette_labels[fac]}: {fac_scores[fac]}")

        report = "\n".join(report_lines)
        st.download_button("📥 Télécharger rapport TXT", report, "neo_pir_report.txt", "text/plain")
