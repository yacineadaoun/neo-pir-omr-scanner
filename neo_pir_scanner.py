import streamlit as st
import cv2
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
import imutils
from PIL import Image
import io
import csv

# ====================== CLÉ DE CORRECTION ======================
# ✅ INCHANGÉE (ta scoring_key complète)
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

# ====================== MAPPINGS ======================
# ✅ INCHANGÉS (ton mapping complet)
item_to_facette = {  # ... ton mapping complet ...
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

# ====================== FONCTIONS ======================
def preprocess_image(image):
    """PIL -> OpenCV, détection doc, redressement, binarisation."""
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image is None or not hasattr(image, "shape"):
        raise ValueError("Image invalide.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        raise ValueError("Contour du document non détecté.")

    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    thresh = cv2.adaptiveThreshold(
        warped, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return paper, thresh


def detect_bubbles(thresh, min_size=20, ar_min=0.85, ar_max=1.15):
    """Détecte toutes les bulles de la feuille unique (240 items → 1200 bulles)."""
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    bubbleCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if h == 0:
            continue
        ar = w / float(h)
        if w >= min_size and h >= min_size and (ar_min <= ar <= ar_max):
            bubbleCnts.append(c)

    expected = 240 * 5  # ✅ 1200
    if len(bubbleCnts) != expected:
        raise ValueError(f"Nombre de bulles incorrect (détecté {len(bubbleCnts)}, attendu {expected}).")

    # Tri top-to-bottom, puis on découpera par lignes (40 bulles / ligne)
    bubbleCnts = contours.sort_contours(bubbleCnts, method="top-to-bottom")[0]
    return bubbleCnts


def evaluate_responses_single_sheet(bubbleCnts, thresh, min_fill=5, weak_fill=30, ambiguity_diff=15):
    """
    Feuille unique : 30 lignes.
    Chaque ligne contient 8 items ; chaque item a 5 bulles => 40 bulles / ligne.
    """
    responses = {}
    warnings = []

    rows = 30
    items_per_row = 8
    choices_per_item = 5
    bubbles_per_row = items_per_row * choices_per_item  # 40

    for r in range(rows):
        row_cnts = bubbleCnts[r * bubbles_per_row:(r + 1) * bubbles_per_row]
        row_cnts = contours.sort_contours(row_cnts, method="left-to-right")[0]

        for c in range(items_per_row):
            item_cnts = row_cnts[c * choices_per_item:(c + 1) * choices_per_item]
            item_cnts = contours.sort_contours(item_cnts, method="left-to-right")[0]

            bubbled = None
            fills = []

            for (j, cnt) in enumerate(item_cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mask)

                total = cv2.countNonZero(masked)
                area = cv2.contourArea(cnt)
                fill_percent = (total / area * 100) if area > 0 else 0

                fills.append(fill_percent)
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            # ✅ item_id : colonne 0 = items 1-30, colonne 1 = 31-60, ...
            item_id = (r + 1) + 30 * c
            responses[item_id] = bubbled[1]

            max_fill = fills[bubbled[1]]
            if max_fill < weak_fill:
                warnings.append(f"Item {item_id}: Détection faible ({max_fill:.1f}%)")

            sorted_fills = sorted(fills, reverse=True)
            if len(sorted_fills) > 1 and (sorted_fills[0] - sorted_fills[1]) < ambiguity_diff:
                warnings.append(f"Item {item_id}: Ambiguïté détectée")

            if max_fill < min_fill:
                warnings.append(f"Item {item_id}: Item non répondu")

    return responses, warnings


def calculate_scores(all_responses):
    facette_scores = {fac: 0 for fac in facette_labels}

    for item_id, option_idx in all_responses.items():
        if item_id in scoring_key and item_id in item_to_facette:
            score = scoring_key[item_id][option_idx]
            fac = item_to_facette[item_id]
            facette_scores[fac] += score

    domain_scores = {dom: 0 for dom in domain_labels}
    for fac, score in facette_scores.items():
        dom = facettes_to_domain[fac]
        domain_scores[dom] += score

    return facette_scores, domain_scores


# ====================== APPLICATION STREAMLIT ======================
    import streamlit as st
import io
import csv
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="NEO PI-R Scanner",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS PRO ----------
st.markdown(
    """
    <style>
      /* Global spacing */
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* Titles */
      h1, h2, h3 { letter-spacing: -0.2px; }

      /* Buttons */
      div.stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
      }

      /* KPI cards */
      .kpi {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.6);
      }
      .kpi-label { font-size: 12px; color: rgba(49, 51, 63, 0.6); margin-bottom: 4px; }
      .kpi-value { font-size: 22px; font-weight: 700; }
      .kpi-sub { font-size: 12px; color: rgba(49, 51, 63, 0.6); margin-top: 4px; }

      /* Section dividers */
      .section {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 16px;
        padding: 16px;
        background: rgba(255,255,255,0.4);
      }

      /* Footer */
      .footer {
        text-align: center;
        color: rgba(49, 51, 63, 0.55);
        font-size: 12px;
        padding-top: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HEADER ----------
left, right = st.columns([0.75, 0.25], vertical_alignment="bottom")
with left:
    st.title("NEO PI-R Scanner")
    st.caption("Analyse automatisée d’une feuille unique (240 items) — Usage professionnel")
with right:
    st.write("")  # spacing

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## Paramètres")
    debug = st.toggle("Mode debug", value=False)

    st.markdown("---")
    st.markdown("### Détection des bulles")
    min_bubble_size = st.slider("Taille minimale (px)", 10, 80, 20)
    ar_min = st.slider("Ratio aspect min", 0.50, 1.00, 0.85, 0.05)
    ar_max = st.slider("Ratio aspect max", 1.00, 1.80, 1.15, 0.05)

    st.markdown("---")
    st.markdown("### Seuils qualité")
    min_fill = st.slider("Seuil 'non répondu' (%)", 0, 20, 5)
    weak_fill = st.slider("Seuil 'détection faible' (%)", 5, 60, 30)
    ambiguity_diff = st.slider("Seuil 'ambiguïté' (diff %)", 5, 40, 15)

    st.markdown("---")
    st.markdown("### À propos")
    st.caption("NEO PI-R Scanner — v1.0 (Essai)")
    st.caption("© 2026 Yacine Adaoun — Tous droits réservés")

# ---------- INPUT ----------
st.markdown("### Import")
colA, colB = st.columns([0.65, 0.35], vertical_alignment="top")

with colA:
    uploaded_file = st.file_uploader(
        "Déposer la feuille de réponses (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    st.caption("Conseil : photo nette, feuille à plat, bonne lumière, pas d’ombre.")

with colB:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("**Contrôles**")
    run = st.button("Lancer l’analyse", type="primary", disabled=(uploaded_file is None))
    st.markdown("Cette action détecte les bulles, calcule les scores et génère les exports.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RUN ----------
if uploaded_file and run:
    all_warnings = []

    try:
        # 1) load image
        original = Image.open(uploaded_file)

        # 2) preprocess + detect
        paper, thresh = preprocess_image(original)

        bubbleCnts = detect_bubbles(
            thresh,
            min_size=min_bubble_size,
            ar_min=ar_min,
            ar_max=ar_max
        )

        responses, warnings = evaluate_responses_single_sheet(
            bubbleCnts,
            thresh,
            min_fill=min_fill,
            weak_fill=weak_fill,
            ambiguity_diff=ambiguity_diff
        )
        all_warnings.extend(warnings)

        # 3) draw selected bubbles in green
        rows = 30
        items_per_row = 8
        choices_per_item = 5
        bubbles_per_row = items_per_row * choices_per_item  # 40

        for r in range(rows):
            row_cnts = bubbleCnts[r * bubbles_per_row:(r + 1) * bubbles_per_row]
            row_cnts = contours.sort_contours(row_cnts, method="left-to-right")[0]

            for c in range(items_per_row):
                item_id = (r + 1) + 30 * c
                idx = responses[item_id]
                item_cnts = row_cnts[c * choices_per_item:(c + 1) * choices_per_item]
                item_cnts = contours.sort_contours(item_cnts, method="left-to-right")[0]
                cv2.drawContours(paper, [item_cnts[idx]], -1, (0, 255, 0), 3)

        # 4) scores
        facette_scores, domain_scores = calculate_scores(responses)

        # 5) KPIs
        total_items = 240
        warning_count = len(all_warnings)
        conf = 100 * (1 - warning_count / total_items)

        st.markdown("### Résultats")
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Confiance globale</div>
              <div class="kpi-value">{conf:.1f}%</div>
              <div class="kpi-sub">Basée sur {total_items} items</div>
            </div>
        """, unsafe_allow_html=True)

        k2.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Avertissements</div>
              <div class="kpi-value">{warning_count}</div>
              <div class="kpi-sub">Non-réponses / faible / ambiguïtés</div>
            </div>
        """, unsafe_allow_html=True)

        k3.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">Items analysés</div>
              <div class="kpi-value">{len(responses)}</div>
              <div class="kpi-sub">Attendu: {total_items}</div>
            </div>
        """, unsafe_allow_html=True)

        if conf < 95:
            st.warning(f"Qualité à vérifier : confiance {conf:.1f}% (relecture recommandée).")
        else:
            st.success(f"Analyse OK : confiance {conf:.1f}%.")

        # 6) tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Scores", "Images", "Avertissements", "Exports"])

        with tab1:
            # facettes
            data = []
            for fac in sorted(facette_labels):
                items = [str(k) for k, v in item_to_facette.items() if v == fac]
                data.append({
                    "Facette": facette_labels[fac],
                    "Items": ", ".join(items),
                    "Score brut": facette_scores[fac]
                })
            st.subheader("Scores par facette")
            st.dataframe(data, use_container_width=True, hide_index=True)

            dom_data = [{"Domaine": domain_labels[d], "Score": domain_scores[d]} for d in sorted(domain_labels)]
            st.subheader("Totaux par domaine")
            st.dataframe(dom_data, use_container_width=True, hide_index=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original")
                st.image(original, use_container_width=True)
            with c2:
                st.subheader("Détection")
                st.image(paper, channels="BGR", use_container_width=True)

        with tab3:
            st.subheader("Journal de contrôle")
            if all_warnings:
                with st.expander("Afficher les avertissements", expanded=True):
                    for w in all_warnings:
                        st.warning(w)
            else:
                st.success("Aucun avertissement détecté.")

            if debug:
                st.code(
                    f"debug: warnings={warning_count}, conf={conf:.2f}, bubbles={len(bubbleCnts)}",
                    language="text"
                )

        with tab4:
            st.subheader("Exports")
            # CSV
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["Facette", "Items", "Score brut"])
            writer.writeheader()
            writer.writerows(data)

            output.write("\n--- TOTAUX PAR DOMAINE ---\n")
            dom_writer = csv.DictWriter(output, fieldnames=["Domaine", "Score"])
            dom_writer.writeheader()
            dom_writer.writerows(dom_data)

            st.download_button(
                "Télécharger le CSV",
                output.getvalue(),
                file_name="neo_pir_scores.csv",
                mime="text/csv"
            )

            # TXT report
            report_lines = ["RAPPORT NEO PI-R", ""]
            report_lines.append("SCORES PAR FACETTE")
            for row in data:
                report_lines.append(f"{row['Facette']}: {row['Score brut']}")

            report_lines.append("")
            report_lines.append("TOTAUX DOMAINES")
            for row in dom_data:
                report_lines.append(f"{row['Domaine']}: {row['Score']}")

            report_lines.append("")
            report_lines.append("Avertissements:")
            if all_warnings:
                report_lines.extend(all_warnings)
            else:
                report_lines.append("Aucun avertissement.")

            report = "\n".join(report_lines)
            st.download_button(
                "Télécharger le rapport TXT",
                report,
                file_name="neo_pir_report.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Erreur : {e}")

st.markdown("<div class='footer'>NEO PI-R Scanner — v1.0 (Essai) · © 2026 Yacine Adaoun</div>", unsafe_allow_html=True)
