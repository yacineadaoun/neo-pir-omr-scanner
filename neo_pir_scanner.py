import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from PIL import Image
import io
import csv

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="NEO PI-R OMR Clinique",
    layout="wide"
)

# =========================================================
# >>>>>>> COLLER ICI TES TABLES COMPLETES <<<<<<<<
# =========================================================

scoring_key = {
    # COLLE ICI TES 240 ITEMS
}

item_to_facette = {
    # COLLE ICI TES 240 ITEMS
}

facettes_to_domain = {
    'N1': 'N','N2': 'N','N3': 'N','N4': 'N','N5': 'N','N6': 'N',
    'E1': 'E','E2': 'E','E3': 'E','E4': 'E','E5': 'E','E6': 'E',
    'O1': 'O','O2': 'O','O3': 'O','O4': 'O','O5': 'O','O6': 'O',
    'A1': 'A','A2': 'A','A3': 'A','A4': 'A','A5': 'A','A6': 'A',
    'C1': 'C','C2': 'C','C3': 'C','C4': 'C','C5': 'C','C6': 'C'
}

# =========================================================
# IMAGE PROCESSING ROBUSTE SMARTPHONE
# =========================================================

def remove_shadows(gray):
    dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm

def preprocess(pil_img):
    img = pil_img.convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détection contour feuille
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
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

    if doc is not None:
        paper = four_point_transform(img, doc.reshape(4,2))
        gray = four_point_transform(gray, doc.reshape(4,2))
    else:
        paper = img

    # Correction ombres
    gray = remove_shadows(gray)

    # Threshold robuste
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    # Morphologie renforcement stylo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel)

    return paper, thresh

# =========================================================
# LECTURE OMR LETTRES
# =========================================================

def read_sheet(thresh,
               grid_left=0.05, grid_right=0.95,
               grid_top=0.205, grid_bottom=0.86,
               rows=30, cols=8,
               option_centers=(0.12,0.32,0.52,0.72,0.90),
               impute_blank=True):

    H,W = thresh.shape
    x0 = int(grid_left*W)
    x1 = int(grid_right*W)
    y0 = int(grid_top*H)
    y1 = int(grid_bottom*H)

    cell_w = (x1-x0)/cols
    cell_h = (y1-y0)/rows

    all_inks = []
    raw_data = []

    for r in range(rows):
        for c in range(cols):
            cx0 = int(x0 + c*cell_w)
            cy0 = int(y0 + r*cell_h)

            cw = int(cell_w)
            ch = int(cell_h)

            inks = []
            for oc in option_centers:
                bw = int(cw*0.16)
                bh = int(ch*0.6)
                bx = int(cx0 + oc*cw - bw/2)
                by = int(cy0 + (ch-bh)/2)

                roi = thresh[by:by+bh, bx:bx+bw]
                ink = cv2.countNonZero(roi)
                inks.append(ink)
                all_inks.append(ink)

            raw_data.append((r,c,inks))

    # Seuil automatique
    median_ink = np.median(all_inks)
    threshold = max(300, int(median_ink*1.8))

    responses = {}
    blank = 0
    ambiguous = 0

    for (r,c,inks) in raw_data:
        item_id = (r+1) + 30*c

        best = int(np.argmax(inks))
        sorted_inks = sorted(inks, reverse=True)

        if sorted_inks[0] < threshold:
            blank += 1
            responses[item_id] = 2 if impute_blank else best
        else:
            responses[item_id] = best

            if (sorted_inks[0]-sorted_inks[1]) < threshold*0.15:
                ambiguous += 1

    stats = {
        "blank": blank,
        "ambiguous": ambiguous,
        "threshold": threshold
    }

    return responses, stats

# =========================================================
# SCORING
# =========================================================

def compute_scores(responses):
    fac_scores = {}
    for item, choice in responses.items():
        if item in scoring_key and item in item_to_facette:
            fac = item_to_facette[item]
            score = scoring_key[item][choice]
            fac_scores[fac] = fac_scores.get(fac,0) + score

    dom_scores = {}
    for fac, val in fac_scores.items():
        dom = facettes_to_domain[fac]
        dom_scores[dom] = dom_scores.get(dom,0) + val

    return fac_scores, dom_scores

# =========================================================
# UI
# =========================================================

st.title("NEO PI-R — OMR Clinique Smartphone")

uploaded = st.file_uploader("Importer la feuille (photo caméra)", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    paper, thresh = preprocess(img)

    responses, stats = read_sheet(thresh)

    fac_scores, dom_scores = compute_scores(responses)

    blank = stats["blank"]
    ambiguous = stats["ambiguous"]

    # VALIDITÉ
    valid = True
    reasons = []

    if blank >= 15:
        valid = False
        reasons.append("≥15 items non répondus")

    n_total = sum(1 for v in responses.values() if v==2)
    if n_total >= 42:
        valid = False
        reasons.append("≥42 réponses N")

    st.subheader("Synthèse protocolaire")

    col1,col2,col3 = st.columns(3)
    col1.metric("Items vides", blank)
    col2.metric("Réponses N", n_total)
    col3.metric("Ambiguïtés", ambiguous)

    if valid:
        st.success("PROTOCOLE VALIDE")
    else:
        st.error("PROTOCOLE INVALIDE")
        for r in reasons:
            st.warning(r)

    st.subheader("Scores domaines")
    st.write(dom_scores)

    st.subheader("Qualité lecture")
    st.write(stats)

    st.image(paper, caption="Feuille redressée")
    st.image(thresh, caption="Binarisation")

st.caption("NEO PI-R OMR Clinique v2.0 — Optimisé Smartphone")
