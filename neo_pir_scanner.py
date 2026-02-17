import io
import os
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, List

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# 0) PERSPECTIVE + ROTATION (sans imutils/scipy)
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
# 1) SCORING KEY depuis scoring_key.csv (pro)
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
    Essaie 0/90/180/270. Garde la rotation qui maximise l'aire du quadrilatère détecté.
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
        doc = None
        for c in cnts[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc = approx.reshape(4, 2)
                area = cv2.contourArea(approx)
                if area > best_area:
                    best_area = area
                    best_warp = four_point_transform(resized, doc)
                    best_k = k
                break

    if best_warp is None:
        raise ValueError("Feuille non détectée (4 coins). Photo trop loin / trop floue / fond chargé.")

    return best_warp, best_k


def binarize(bgr: np.ndarray) -> np.ndarray:
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
        for c in cnts[:6]:
            x, y, w, h = cv2.boundingRect(c)
            area_ok = (w * h) > 0.20 * (W * H)
            cx, cy = x + w / 2, y + h / 2
            center_ok = (W * 0.20) < cx < (W * 0.80) and (H * 0.20) < cy < (H * 0.80)
            if area_ok and center_ok:
                return x, y, w, h

    # fallback central
    fx = int(W * 0.05)
    fy = int(H * 0.12)
    fw = int(W * 0.90)
    fh = int(H * 0.78)
    return fx, fy, fw, fh


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
    return (float(np.count_nonzero(patch)) / float(patch.size)) * 100.0


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
    Micro-ajustement: essaie d'estimer les frontières via la grille.
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

    vx = pick_peaks(proj_x, n_needed=9, min_dist=max(5, w // 50), thr=0.35)
    hy = pick_peaks(proj_y, n_needed=31, min_dist=max(5, h // 80), thr=0.35)

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


def read_responses_from_grid(
    thr_inv: np.ndarray,
    table_bbox: Tuple[int, int, int, int],
    mark_threshold: float,
    ambiguity_gap: float,
    use_micro_adjust: bool
) -> Tuple[Dict[int, int], Dict[int, dict], np.ndarray]:

    overlay = cv2.cvtColor(thr_inv.copy(), cv2.COLOR_GRAY2BGR)
    responses: Dict[int, int] = {}
    meta: Dict[int, dict] = {}

    splitter = split_grid_micro_adjust if use_micro_adjust else (lambda a, b, rows=30, cols=8: split_grid_uniform(b, rows, cols))

    for r, c, cell in splitter(thr_inv, table_bbox, rows=30, cols=8):
        item_id = item_id_from_rc(r, c)
        rois = option_rois_in_cell(cell)
        fills = [ink_score(thr_inv, roi) for roi in rois]

        best_idx = int(np.argmax(fills))
        best = fills[best_idx]
        sorted_f = sorted(fills, reverse=True)
        second = sorted_f[1] if len(sorted_f) > 1 else 0.0

        blank = best < mark_threshold
        ambiguous = (best - second) < ambiguity_gap and not blank

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
            "chosen_fill": best,
            "blank": blank,
            "ambiguous": ambiguous
        }

    return responses, meta, overlay


# ============================================================
# 5) SCORING + PROTOCOLE
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
# 6) GRAPHIQUE PROFIL (Domaines + 30 facettes) + export
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
# 7) STREAMLIT UI
# ============================================================
st.set_page_config(page_title="NEO PI-R Scanner", page_icon="🧾", layout="wide")

st.title("NEO PI-R — Scanner & Calculateur (Feuille de réponses)")
st.caption("Rotation auto • Redressement • Grille 30×8 • Micro-ajustement • Scoring • Exports • Profil graphique")

RULES_DEFAULT = ProtocolRules()

with st.sidebar:
    st.subheader("Lecture photo")
    mark_threshold = st.slider("Seuil 'réponse détectée' (%)", 0.1, 10.0, 1.2, 0.1)
    ambiguity_gap = st.slider("Seuil ambiguïté (écart %)", 0.5, 10.0, 2.0, 0.5)
    use_micro = st.toggle("Micro-ajustement cellules (recommandé)", value=True)

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
        thr = binarize(warped)

        table_bbox = find_table_bbox_soft(thr)

        raw_responses, meta, overlay = read_responses_from_grid(
            thr, table_bbox,
            mark_threshold=mark_threshold,
            ambiguity_gap=ambiguity_gap,
            use_micro_adjust=use_micro
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
            with colB:
                st.subheader("Overlay (vert=renseigné, rouge=vide, orange=ambigu)")
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            if debug:
                st.write("table_bbox:", table_bbox)
                amb = [i for i, m in meta.items() if (not m["blank"]) and m["ambiguous"]]
                st.write("items ambigus:", len(amb))
                st.write(amb[:40])

        with tab2:
            data = []
            for fac in sorted(facette_labels.keys()):
                items = [str(k) for k, v in sorted(item_to_facette.items()) if v == fac]
                data.append({"Facette": facette_labels[fac], "Items": ", ".join(items), "Score brut": facette_scores[fac]})
            st.dataframe(data, use_container_width=True, hide_index=True)

        with tab3:
            dom_data = [{"Domaine": domain_labels[d], "Score": domain_scores[d]} for d in ["N","E","O","A","C"]]
            st.dataframe(dom_data, use_container_width=True, hide_index=True)

        with tab4:
            st.subheader("Profil (style pro, scores bruts)")
            fig = plot_profile(facette_scores, domain_scores)
            st.pyplot(fig)

            png_bytes = fig_to_bytes(fig, "png")
            pdf_bytes = fig_to_bytes(fig, "pdf")

            st.download_button("📥 Télécharger profil PNG", png_bytes, "neo_pir_profile.png", "image/png")
            st.download_button("📥 Télécharger profil PDF", pdf_bytes, "neo_pir_profile.pdf", "application/pdf")

        with tab5:
            # CSV
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=["Facette", "Items", "Score brut"])
            writer.writeheader()
            writer.writerows(data)
            out.write("\n--- TOTAUX PAR DOMAINE ---\n")
            dom_writer = csv.DictWriter(out, fieldnames=["Domaine", "Score"])
            dom_writer.writeheader()
            dom_writer.writerows(dom_data)

            st.download_button("📥 Télécharger CSV", out.getvalue(), "neo_pir_scores.csv", "text/csv")

            # TXT
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
