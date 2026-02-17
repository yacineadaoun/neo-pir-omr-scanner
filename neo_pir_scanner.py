import io
import os
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# 0) OUTILS: perspective + rotation (sans imutils/scipy)
# ============================================================
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0],
         [maxW - 1, 0],
         [maxW - 1, maxH - 1],
         [0, maxH - 1]], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def rotate_k90(bgr: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return bgr
    if k == 1:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


def resize_keep_aspect(bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= target_width:
        return bgr
    scale = target_width / float(w)
    nh = int(h * scale)
    return cv2.resize(bgr, (target_width, nh), interpolation=cv2.INTER_AREA)


# ============================================================
# 1) SCORING KEY depuis scoring_key.csv
# ============================================================
@st.cache_resource
def load_scoring_key_from_csv(path: str = "scoring_key.csv") -> Dict[int, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' introuvable. Mets scoring_key.csv à la racine du projet."
        )

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        key: Dict[int, List[int]] = {}
        for row in reader:
            item = int(row["item"])
            key[item] = [int(row["FD"]), int(row["D"]), int(row["N"]), int(row["A"]), int(row["FA"])]

    missing = [i for i in range(1, 241) if i not in key]
    if missing:
        raise ValueError(f"scoring_key.csv incomplet. Items manquants: {missing[:20]}")

    bad = [i for i, v in key.items() if len(v) != 5 or any((x < 0 or x > 4) for x in v)]
    if bad:
        raise ValueError(f"scoring_key.csv invalide. Items problématiques: {bad[:20]}")

    return key


scoring_key = load_scoring_key_from_csv("scoring_key.csv")


# ============================================================
# 2) ITEM -> FACETTE (généré)
# ============================================================
facet_bases = {
    "N1": [1],  "N2": [6],  "N3": [11], "N4": [16], "N5": [21], "N6": [26],
    "E1": [2],  "E2": [7],  "E3": [12], "E4": [17], "E5": [22], "E6": [27],
    "O1": [3],  "O2": [8],  "O3": [13], "O4": [18], "O5": [23], "O6": [28],
    "A1": [4],  "A2": [9],  "A3": [14], "A4": [19], "A5": [24], "A6": [29],
    "C1": [5],  "C2": [10], "C3": [15], "C4": [20], "C5": [25], "C6": [30],
}

item_to_facette: Dict[int, str] = {}
for fac, bases in facet_bases.items():
    for b in bases:
        for k in range(0, 240, 30):
            item_to_facette[b + k] = fac

facettes_to_domain = {**{f"N{i}": "N" for i in range(1, 7)},
                      **{f"E{i}": "E" for i in range(1, 7)},
                      **{f"O{i}": "O" for i in range(1, 7)},
                      **{f"A{i}": "A" for i in range(1, 7)},
                      **{f"C{i}": "C" for i in range(1, 7)}}

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
# 3) PROTOCOLE
# ============================================================
@dataclass
class ProtocolRules:
    max_blank_invalid: int = 15
    max_N_invalid: int = 42
    impute_blank_if_leq: int = 10
    impute_option_index: int = 2


# ============================================================
# 4) VISION: redressement + tableau + micro-ajustement
# ============================================================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def find_document_warp_auto_rotate(bgr: np.ndarray, target_width: int = 1800) -> Tuple[np.ndarray, int]:
    """
    Essaie rotations 0/90/180/270 et garde celle qui maximise l'aire du quadrilatère détecté.
    """
    best_area = -1.0
    best_warp = None
    best_k = 0

    for k in [0, 1, 2, 3]:
        img = rotate_k90(bgr, k)
        resized = resize_keep_aspect(img, target_width)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)

        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:12]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > best_area:
                    best_area = area
                    best_warp = four_point_transform(resized, approx.reshape(4, 2))
                    best_k = k
                break

    if best_warp is None:
        raise ValueError("Feuille non détectée (4 coins). Photo trop loin / floue / fond chargé.")

    return best_warp, best_k


def binarize_inv(bgr: np.ndarray) -> np.ndarray:
    """
    THRESH_BINARY_INV: imprimé noir + stylo -> blanc.
    Sert à détecter la grille/print, pas à lire les réponses.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    return thr


def build_grid_mask(thr_inv: np.ndarray) -> np.ndarray:
    h, w = thr_inv.shape[:2]
    hk = max(20, w // 18)
    vk = max(20, h // 18)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    hor = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    ver = cv2.morphologyEx(thr_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return grid


def find_table_bbox_soft(thr_inv: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Détection souple du tableau. Si échec -> fallback zone centrale.
    """
    grid = build_grid_mask(thr_inv)
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = thr_inv.shape[:2]
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:8]:
            x, y, w, h = cv2.boundingRect(c)
            area_ok = (w * h) > 0.18 * (W * H)
            cx, cy = x + w / 2, y + h / 2
            center_ok = (W * 0.15) < cx < (W * 0.85) and (H * 0.15) < cy < (H * 0.88)
            if area_ok and center_ok:
                return x, y, w, h

    fx = int(W * 0.05)
    fy = int(H * 0.12)
    fw = int(W * 0.90)
    fh = int(H * 0.78)
    return fx, fy, fw, fh


def item_id_from_rc(r: int, c: int) -> int:
    return (r + 1) + 30 * c


def split_grid_uniform(table_bbox: Tuple[int, int, int, int], rows: int = 30, cols: int = 8):
    x, y, w, h = table_bbox
    cw = w / cols
    ch = h / rows
    for r in range(rows):
        for c in range(cols):
            x1 = int(x + c * cw)
            y1 = int(y + r * ch)
            x2 = int(x + (c + 1) * cw)
            y2 = int(y + (r + 1) * ch)
            yield r, c, (x1, y1, x2, y2)


def split_grid_micro_adjust(thr_inv: np.ndarray, table_bbox: Tuple[int, int, int, int], rows: int = 30, cols: int = 8):
    """
    Micro-ajustement: tente d'estimer les frontières via projection du masque de grille.
    Si insuffisant -> fallback uniforme.
    """
    x, y, w, h = table_bbox
    roi = thr_inv[y:y + h, x:x + w]
    grid = build_grid_mask(roi)

    proj_x = grid.sum(axis=0).astype(np.float32)
    proj_y = grid.sum(axis=1).astype(np.float32)

    if proj_x.max() > 0:
        proj_x /= proj_x.max()
    if proj_y.max() > 0:
        proj_y /= proj_y.max()

    def pick_peaks(proj: np.ndarray, n_needed: int, min_dist: int, thr: float) -> List[int]:
        idxs = np.where(proj >= thr)[0].tolist()
        if not idxs:
            return []
        peaks = []
        last = -10**9
        for i in idxs:
            if i - last >= min_dist:
                peaks.append(i)
                last = i
        if len(peaks) > n_needed:
            keep = np.linspace(0, len(peaks) - 1, n_needed).round().astype(int)
            peaks = [peaks[i] for i in keep]
        return peaks

    vx = pick_peaks(proj_x, n_needed=9, min_dist=max(6, w // 60), thr=0.35)
    hy = pick_peaks(proj_y, n_needed=31, min_dist=max(6, h // 90), thr=0.35)

    if len(vx) != 9 or len(hy) != 31:
        yield from split_grid_uniform(table_bbox, rows=rows, cols=cols)
        return

    vx = sorted(vx)
    hy = sorted(hy)
    vxg = [x + v for v in vx]
    hyg = [y + u for u in hy]

    for r in range(rows):
        for c in range(cols):
            x1, x2 = vxg[c], vxg[c + 1]
            y1, y2 = hyg[r], hyg[r + 1]
            yield r, c, (int(x1), int(y1), int(x2), int(y2))


# ============================================================
# 5) MASQUES STYLO (BLEU + NOIR) + SUPPRESSION IMPRIMÉ
# ============================================================
def build_blue_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # normalisation V (ombre)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])

    lower = np.array([85, 40, 40], dtype=np.uint8)
    upper = np.array([145, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.medianBlur(mask, 3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)
    return mask


def build_black_pen_mask(bgr: np.ndarray, print_mask: np.ndarray) -> np.ndarray:
    """
    Détecte le stylo noir en limitant l'imprimé:
      - candidat noir: pixels sombres + black-hat
      - puis suppression de l'imprimé via print_mask (thr_inv de la feuille)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # contraste local (black-hat)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)

    # candidat = sombre OU blackhat fort
    dark = cv2.inRange(gray, 0, 80)
    bh = cv2.inRange(blackhat, 25, 255)

    cand = cv2.bitwise_or(dark, bh)

    # nettoyage
    cand = cv2.medianBlur(cand, 3)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k2, iterations=1)

    # suppression de l'imprimé: print_mask est 255 sur texte+grille+stylo
    # on veut retirer au max texte+grille -> on dilate print_mask pour englober l'imprimé
    pm = cv2.dilate(print_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    pm_inv = cv2.bitwise_not(pm)

    black_pen = cv2.bitwise_and(cand, pm_inv)

    # petit boost final
    black_pen = cv2.dilate(black_pen, k2, iterations=1)
    return black_pen


def build_pen_mask(bgr: np.ndarray, thr_inv_print: np.ndarray, detect_blue: bool, detect_black: bool) -> np.ndarray:
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

    if detect_blue:
        mask = cv2.bitwise_or(mask, build_blue_mask(bgr))

    if detect_black:
        mask = cv2.bitwise_or(mask, build_black_pen_mask(bgr, thr_inv_print))

    return mask


def ink_score_mask(mask: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = roi
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return (float(np.count_nonzero(patch)) / float(patch.size)) * 100.0


# ============================================================
# 6) ROIs INTELLIGENTES (entourage)
# ============================================================
def option_rois_in_cell(cell: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    """
    ROIs "chirurgicales" centrées sur la lettre (zone d'entourage).
    Réduit le bruit de l'imprimé.
    """
    x1, y1, x2, y2 = cell
    w = x2 - x1
    h = y2 - y1

    left = x1 + int(0.22 * w)
    right = x2 - int(0.06 * w)
    top = y1 + int(0.20 * h)
    bottom = y2 - int(0.20 * h)

    inner_w = max(1, right - left)
    band_w = inner_w / 5.0

    rois = []
    for k in range(5):
        bx1 = left + k * band_w
        bx2 = left + (k + 1) * band_w

        rx1 = int(bx1 + 0.25 * band_w)
        rx2 = int(bx2 - 0.25 * band_w)

        ry1 = int(top + 0.12 * (bottom - top))
        ry2 = int(bottom - 0.12 * (bottom - top))

        rois.append((rx1, ry1, rx2, ry2))
    return rois


# ============================================================
# 7) LECTURE SMART (masque stylo + seuil adaptatif + ambiguïté)
# ============================================================
def read_responses_from_grid_smart(
    warped_bgr: np.ndarray,
    thr_inv_print: np.ndarray,
    table_bbox: Tuple[int, int, int, int],
    use_micro_adjust: bool,
    mark_threshold: float,
    ambiguity_gap: float,
    detect_blue: bool,
    detect_black: bool
):
    """
    Lecture robuste: on lit sur masque stylo (bleu/noir),
    on évite l'imprimé via thr_inv_print, et on utilise un seuil adaptatif.
    """
    pen_mask = build_pen_mask(warped_bgr, thr_inv_print, detect_blue=detect_blue, detect_black=detect_black)
    overlay = warped_bgr.copy()

    responses: Dict[int, int] = {}
    meta: Dict[int, dict] = {}

    if use_micro_adjust:
        iterator = split_grid_micro_adjust(thr_inv_print, table_bbox, rows=30, cols=8)
    else:
        iterator = split_grid_uniform(table_bbox, rows=30, cols=8)

    for r, c, cell in iterator:
        item_id = item_id_from_rc(r, c)
        rois = option_rois_in_cell(cell)

        fills = [ink_score_mask(pen_mask, roi) for roi in rois]
        best_idx = int(np.argmax(fills))
        best = float(fills[best_idx])

        sorted_f = sorted(fills, reverse=True)
        second = float(sorted_f[1]) if len(sorted_f) > 1 else 0.0
        gap = best - second

        mean_f = float(np.mean(fills))
        rel = best - mean_f

        # blank: soit best trop petit, soit pas au-dessus du bruit local
        blank = (best < mark_threshold) or (rel < 0.12)

        ambiguous = (not blank) and (gap < ambiguity_gap)

        x1, y1, x2, y2 = cell
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), 1)

        if blank:
            responses[item_id] = -1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            responses[item_id] = best_idx
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if ambiguous:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)

        rx1, ry1, rx2, ry2 = rois[best_idx]
        if not blank:
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        meta[item_id] = {
            "fills": fills,
            "chosen_idx": best_idx,
            "best": best,
            "second": second,
            "gap": gap,
            "mean": mean_f,
            "rel": rel,
            "blank": blank,
            "ambiguous": ambiguous
        }

    return responses, meta, overlay, pen_mask


# ============================================================
# 8) SCORING + PROTOCOLE
# ============================================================
def compute_scores(responses: Dict[int, int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    facette_scores = {fac: 0 for fac in facette_labels.keys()}
    for item_id, idx in responses.items():
        if idx == -1:
            continue
        fac = item_to_facette.get(item_id)
        if fac is None:
            continue
        facette_scores[fac] += scoring_key[item_id][idx]

    domain_scores = {d: 0 for d in domain_labels.keys()}
    for fac, sc in facette_scores.items():
        domain_scores[facettes_to_domain[fac]] += sc
    return facette_scores, domain_scores


def apply_protocol_rules(responses: Dict[int, int], rules: ProtocolRules) -> Tuple[Dict[int, int], dict]:
    blanks = [i for i, v in responses.items() if v == -1]
    n_blank = len(blanks)
    n_count = sum(1 for v in responses.values() if v == 2)

    status = {
        "valid": True,
        "reasons": [],
        "blank_items": blanks,
        "n_blank": n_blank,
        "n_count": n_count,
        "imputed": 0
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
# 9) GRAPHIQUE PROFIL + export
# ============================================================
def plot_profile(facette_scores: Dict[str, int], domain_scores: Dict[str, int]):
    x_labels = ["N", "E", "O", "A", "C"] + [
        "N1","N2","N3","N4","N5","N6",
        "E1","E2","E3","E4","E5","E6",
        "O1","O2","O3","O4","O5","O6",
        "A1","A2","A3","A4","A5","A6",
        "C1","C2","C3","C4","C5","C6",
    ]

    y = [domain_scores[d] for d in ["N","E","O","A","C"]] + [facette_scores[k] for k in x_labels[5:]]

    fig = plt.figure(figsize=(16, 5))
    ax = plt.gca()
    ax.plot(range(len(x_labels)), y, marker="o", linewidth=2)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=60, ha="right")
    ax.set_title("Profil NEO PI-R — Scores bruts (Domaines + 30 facettes)")
    ax.set_ylabel("Score brut")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_to_bytes(fig, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# 10) STREAMLIT UI
# ============================================================
st.set_page_config(page_title="NEO PI-R Scanner", page_icon="🧾", layout="wide")

st.title("NEO PI-R — Scanner & Calculateur (Feuille de réponses)")
st.caption("Rotation auto • Redressement • Grille 30×8 • Micro-ajustement • Lecture stylo (bleu + noir) • Scoring • Exports • Profil")

RULES_DEFAULT = ProtocolRules()

with st.sidebar:
    st.subheader("Lecture photo")
    mark_threshold = st.slider("Seuil réponse (encre %) — trait fin", 0.05, 5.0, 0.35, 0.05)
    ambiguity_gap = st.slider("Seuil ambiguïté (best - second) %", 0.05, 5.0, 0.40, 0.05)
    use_micro = st.toggle("Micro-ajustement cellules (recommandé)", value=True)

    st.markdown("---")
    st.subheader("Stylo")
    detect_blue = st.toggle("Détecter stylo BLEU", value=True)
    detect_black = st.toggle("Détecter stylo NOIR (anti-imprimé)", value=True)

    st.markdown("---")
    st.subheader("Protocole")
    max_blank_invalid = st.number_input("Items vides ⇒ invalide si ≥", 0, 240, RULES_DEFAULT.max_blank_invalid)
    max_N_invalid = st.number_input("Réponses 'N' ⇒ invalide si ≥", 0, 240, RULES_DEFAULT.max_N_invalid)
    impute_blank_if_leq = st.number_input("Imputation si blancs ≤", 0, 240, RULES_DEFAULT.impute_blank_if_leq)

    debug = st.toggle("Debug", value=False)

RULES = ProtocolRules(
    max_blank_invalid=int(max_blank_invalid),
    max_N_invalid=int(max_N_invalid),
    impute_blank_if_leq=int(impute_blank_if_leq),
    impute_option_index=2
)

uploaded = st.file_uploader("Importer une photo/scanner (JPG/PNG)", type=["jpg", "jpeg", "png"])
run = st.button("Scanner & Calculer", type="primary", disabled=(uploaded is None))

if run and uploaded:
    try:
        pil_img = Image.open(uploaded)
        bgr = pil_to_bgr(pil_img)

        warped, rot_k = find_document_warp_auto_rotate(bgr, target_width=1800)
        thr_inv_print = binarize_inv(warped)  # print + grille + stylo -> blanc

        table_bbox = find_table_bbox_soft(thr_inv_print)

        raw_responses, meta, overlay, pen_mask = read_responses_from_grid_smart(
            warped_bgr=warped,
            thr_inv_print=thr_inv_print,
            table_bbox=table_bbox,
            use_micro_adjust=use_micro,
            mark_threshold=float(mark_threshold),
            ambiguity_gap=float(ambiguity_gap),
            detect_blue=detect_blue,
            detect_black=detect_black
        )

        final_responses, status = apply_protocol_rules(raw_responses, RULES)
        facette_scores, domain_scores = compute_scores(final_responses)

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

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overlay", "Scores facettes", "Scores domaines", "Profil graphique", "Exports"]
        )

        with tab1:
            colA, colB = st.columns(2)
            with colA:
                st.subheader(f"Image redressée (rotation={rot_k*90}°)")
                st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)
                if debug:
                    st.subheader("thr_inv_print (imprimé + grille)")
                    st.image(thr_inv_print, use_container_width=True)

            with colB:
                st.subheader("Overlay (vert=renseigné, rouge=vide, orange=ambigu)")
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

                if debug:
                    st.subheader("Masque stylo (bleu+noir) — debug")
                    st.image(pen_mask, use_container_width=True)

                    amb = [i for i, m in meta.items() if (not m["blank"]) and m["ambiguous"]]
                    st.write("Items ambigus:", len(amb))
                    st.write(amb[:40])
                    st.write("table_bbox:", table_bbox)

        with tab2:
            data = []
            for fac in sorted(facette_labels.keys()):
                items = [str(k) for k, v in sorted(item_to_facette.items()) if v == fac]
                data.append({"Facette": facette_labels[fac], "Items": ", ".join(items), "Score brut": facette_scores[fac]})
            st.dataframe(data, use_container_width=True, hide_index=True)

        with tab3:
            dom_data = [{"Domaine": domain_labels[d], "Score": domain_scores[d]} for d in ["N", "E", "O", "A", "C"]]
            st.dataframe(dom_data, use_container_width=True, hide_index=True)

        with tab4:
            st.subheader("Profil (scores bruts)")
            fig = plot_profile(facette_scores, domain_scores)
            st.pyplot(fig)

            st.download_button("📥 Télécharger profil PNG", fig_to_bytes(fig, "png"), "neo_pir_profile.png", "image/png")
            st.download_button("📥 Télécharger profil PDF", fig_to_bytes(fig, "pdf"), "neo_pir_profile.pdf", "application/pdf")

        with tab5:
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=["Facette", "Items", "Score brut"])
            writer.writeheader()
            writer.writerows(data)

            out.write("\n--- TOTAUX PAR DOMAINE ---\n")
            dom_writer = csv.DictWriter(out, fieldnames=["Domaine", "Score"])
            dom_writer.writeheader()
            dom_writer.writerows(dom_data)

            st.download_button("📥 Télécharger CSV", out.getvalue(), "neo_pir_scores.csv", "text/csv")

            report_lines = ["RAPPORT NEO PI-R", ""]
            report_lines.append(f"STATUT PROTOCOLE: {'VALIDE' if valid else 'INVALIDE'}")
            if status["reasons"]:
                report_lines.append("RAISONS:")
                report_lines.extend([f"- {r}" for r in status["reasons"]])
            report_lines.append("")
            report_lines.append(f"Items vides: {n_blank}")
            report_lines.append(f"N observés: {n_count}")
            report_lines.append(f"Imputations: {imputed}")
            report_lines.append("")
            report_lines.append("SCORES PAR FACETTE")
            for row in data:
                report_lines.append(f"{row['Facette']}: {row['Score brut']}")
            report_lines.append("")
            report_lines.append("TOTAUX DOMAINES")
            for row in dom_data:
                report_lines.append(f"{row['Domaine']}: {row['Score']}")

            st.download_button("📥 Télécharger rapport TXT", "\n".join(report_lines), "neo_pir_report.txt", "text/plain")

    except Exception as e:
        st.error(f"Erreur : {e}")
