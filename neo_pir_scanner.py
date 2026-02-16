# ============================================================
# NEO PI-R — OMR Clinique (Photo smartphone) — VERSION FINALE
# ------------------------------------------------------------
# ✅ Feuille SANS bulles (lettres FD / D / N / A / FA entourées)
# ✅ Lecture robuste par ROIs (zones) + auto-seuil
# ✅ Validité protocolaire :
#     - Items vides ≥ 15  -> INVALIDE
#     - N cochés ≥ 42     -> INVALIDE  (⚠️ uniquement N réellement cochés, pas les N imputés)
# ✅ Imputation : item vide -> N (index 2) pour le scoring (donc 2 points via scoring_key)
# ✅ Scoring OPTIONNEL : si scoring_key + item_to_facette fournis → scores facettes/domaines
# ============================================================

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

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(
    page_title="NEO PI-R — OMR Clinique",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
      h1, h2, h3 { letter-spacing: -0.2px; }
      div.stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-weight: 650;
      }
      .card {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.65);
      }
      .label { font-size: 12px; color: rgba(49, 51, 63, 0.65); margin-bottom: 4px; }
      .value { font-size: 22px; font-weight: 800; }
      .sub { font-size: 12px; color: rgba(49, 51, 63, 0.65); margin-top: 4px; }
      .footer {
        text-align: center;
        color: rgba(49, 51, 63, 0.55);
        font-size: 12px;
        padding-top: 16px;
      }
      .note {
        border-left: 4px solid rgba(49, 51, 63, 0.25);
        padding: 10px 12px;
        background: rgba(49, 51, 63, 0.03);
        border-radius: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 1) TABLES (OPTIONNELLES)
# ------------------------------------------------------------
# ➜ Tu peux laisser vides : l’app fera quand même OMR + VALIDITÉ
# ➜ Si tu colles les tables complètes : l’app calculera les SCORES
# ============================================================

# scoring_key[item] = [score_FD, score_D, score_N, score_A, score_FA]
# Doit être complet sur 1..240 pour activer le scoring.
scoring_key: Dict[int, List[int]] = {
    # 👉 COLLE ICI TON scoring_key COMPLET si tu veux les scores
}

# item_to_facette[item] = "N1".."C6"
item_to_facette: Dict[int, str] = {
    # 👉 COLLE ICI TON item_to_facette COMPLET si tu veux les scores
}

facettes_to_domain = {
    'N1': 'N', 'N2': 'N', 'N3': 'N', 'N4': 'N', 'N5': 'N', 'N6': 'N',
    'E1': 'E', 'E2': 'E', 'E3': 'E', 'E4': 'E', 'E5': 'E', 'E6': 'E',
    'O1': 'O', 'O2': 'O', 'O3': 'O', 'O4': 'O', 'O5': 'O', 'O6': 'O',
    'A1': 'A', 'A2': 'A', 'A3': 'A', 'A4': 'A', 'A5': 'A', 'A6': 'A',
    'C1': 'C', 'C2': 'C', 'C3': 'C', 'C4': 'C', 'C5': 'C', 'C6': 'C'
}
domain_labels = {
    'N': 'Névrosisme',
    'E': 'Extraversion',
    'O': 'Ouverture',
    'A': 'Agréabilité',
    'C': 'Conscience'
}
# Facette labels optionnels (affichage)
facette_labels = {k: k for k in facettes_to_domain.keys()}

ChoiceLabels = ["FD", "D", "N", "A", "FA"]  # index 0..4

# ============================================================
# 2) CONFIGS
# ============================================================

@dataclass
class OMRConfig:
    # zone globale de la grille (en ratios de la feuille redressée)
    grid_left: float = 0.05
    grid_right: float = 0.95
    grid_top: float = 0.205
    grid_bottom: float = 0.86

    # structure de la feuille (240 items = 30 lignes x 8 colonnes)
    rows: int = 30
    cols: int = 8

    # positions relatives des 5 options dans une case
    # (à ajuster si besoin, mais ces valeurs conviennent bien à la feuille standard)
    option_centers: Tuple[float, float, float, float, float] = (0.14, 0.34, 0.54, 0.74, 0.92)

    # taille des ROIs (boîtes) autour des lettres
    box_w_ratio: float = 0.18
    box_h_ratio: float = 0.62

    # auto-seuil d’encre
    auto_threshold_factor: float = 1.8
    auto_threshold_floor: int = 280

    # qualité
    ambiguity_rel_gap: float = 0.12
    weak_rel_margin: float = 1.35

    # comportement vide
    impute_blank_to_N: bool = True  # vide -> N pour scoring (2 points via scoring_key)
    blank_detect_margin: float = 0.92  # marge tolérée (évite de compter des micro-bruits)

@dataclass
class ValidityConfig:
    blank_invalid_threshold: int = 15
    neutral_invalid_threshold: int = 42  # ⚠️ N réellement cochés (pas imputés)

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

# ============================================================
# 3) AUDITS (scoring)
# ============================================================

def scoring_ready(sk: Dict[int, List[int]], m: Dict[int, str]) -> bool:
    return len(sk) == 240 and len(m) == 240

def audit_scoring_key(sk: Dict[int, List[int]]) -> List[str]:
    errs: List[str] = []
    if not sk:
        return ["scoring_key vide (scoring désactivé, mais validité OK)."]
    if len(sk) != 240:
        errs.append(f"scoring_key: {len(sk)} items (attendu 240).")
    for item_id in range(1, 241):
        if item_id not in sk:
            errs.append(f"scoring_key: item manquant {item_id}.")
            continue
        v = sk[item_id]
        if not isinstance(v, list) or len(v) != 5:
            errs.append(f"scoring_key: item {item_id} doit avoir 5 valeurs, reçu {v}.")
            continue
        if v not in ([4, 3, 2, 1, 0], [0, 1, 2, 3, 4]):
            errs.append(f"scoring_key: item {item_id} pattern invalide {v}.")
        if v[2] != 2:
            errs.append(f"scoring_key: item {item_id} valeur N != 2 (reçu {v[2]}).")
    return errs

def audit_item_to_facette(m: Dict[int, str]) -> List[str]:
    errs: List[str] = []
    if not m:
        return ["item_to_facette vide (scoring désactivé, mais validité OK)."]
    if len(m) != 240:
        errs.append(f"item_to_facette: {len(m)} items (attendu 240).")
    for item_id in range(1, 241):
        if item_id not in m:
            errs.append(f"item_to_facette: item manquant {item_id}.")
            continue
        fac = m[item_id]
        if fac not in facettes_to_domain:
            errs.append(f"item_to_facette: item {item_id} facette invalide '{fac}'.")
    return errs

# ============================================================
# 4) IMAGE PIPELINE ROBUSTE SMARTPHONE
# ============================================================

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    img_rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

def remove_shadows(gray: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm

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
        # fallback: pas de perspective, on garde l'image entière
        return img_bgr.copy(), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), False

    paper = four_point_transform(img_bgr, doc.reshape(4, 2))
    warped_gray = four_point_transform(gray, doc.reshape(4, 2))
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
    if W <= 0:
        raise ValueError("Image invalide (largeur nulle).")
    scale = target_w / float(W)
    new_size = (target_w, max(1, int(H * scale)))
    return cv2.resize(gray, new_size), cv2.resize(thr, new_size)

# ============================================================
# 5) OMR — EXTRACTION ENCRE PAR ROI
# ============================================================

def extract_inks(thr: np.ndarray, cfg: OMRConfig):
    H, W = thr.shape[:2]

    x0 = int(cfg.grid_left * W)
    x1 = int(cfg.grid_right * W)
    y0 = int(cfg.grid_top * H)
    y1 = int(cfg.grid_bottom * H)

    # clamp
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
    if v.size < 100:
        thr = 900
        return thr, int(np.median(v)) if v.size else 0, 0, 0
    med = int(np.median(v))
    p10 = int(np.percentile(v, 10))
    p90 = int(np.percentile(v, 90))
    thr = max(cfg.auto_threshold_floor, int(med * cfg.auto_threshold_factor))
    return thr, med, p10, p90

def decide_responses(raw_items, thr_img: np.ndarray, cfg: OMRConfig, thr_ink: int, overlay: bool):
    responses: Dict[int, int] = {}            # item -> idx
    mark_state: Dict[int, str] = {}           # "marked" / "blank_imputed" / "blank_unimputed"
    warnings: List[str] = []

    blank = 0
    ambiguous = 0
    weak = 0
    neutral_marked = 0
    neutral_imputed = 0

    ov = cv2.cvtColor(thr_img.copy(), cv2.COLOR_GRAY2BGR) if overlay else None

    # seuil "blank" un peu plus strict pour éviter le bruit
    blank_thr = int(thr_ink * cfg.blank_detect_margin)

    for (r, c, inks, rois) in raw_items:
        item_id = (r + 1) + 30 * c

        best_idx = int(np.argmax(inks))
        sorted_inks = sorted(inks, reverse=True)
        best_ink = int(sorted_inks[0])

        # gap relatif entre 1er et 2e
        rel_gap = 1.0
        if sorted_inks[0] > 0:
            rel_gap = (sorted_inks[0] - sorted_inks[1]) / float(sorted_inks[0])

        # BLANK (non répondu)
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

        # overlay
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

# ============================================================
# 6) VALIDITÉ + STYLE
# ============================================================

def protocol_validity(stats: OMRStats, vcfg: ValidityConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if stats.blank_count >= vcfg.blank_invalid_threshold:
        reasons.append(f"Items non répondus: {stats.blank_count} (seuil ≥ {vcfg.blank_invalid_threshold}).")
    if stats.neutral_marked_count >= vcfg.neutral_invalid_threshold:
        reasons.append(f"Réponses N cochées: {stats.neutral_marked_count} (seuil ≥ {vcfg.neutral_invalid_threshold}).")
    return (len(reasons) == 0), reasons

def response_style(responses: Dict[int,int], mark_state: Dict[int,str]) -> Dict[str, float]:
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    blanks = 0
    imputed = 0

    for item_id, idx in responses.items():
        counts[idx] += 1
        stt = mark_state.get(item_id, "marked")
        if stt.startswith("blank"):
            blanks += 1
        if stt == "blank_imputed":
            imputed += 1

    total = max(1, sum(counts.values()))
    return {
        "FD": counts[0],
        "D": counts[1],
        "N_total": counts[2],
        "A": counts[3],
        "FA": counts[4],
        "Vides": blanks,
        "N_imputés": imputed,
        "FD%": 100*counts[0]/total,
        "D%": 100*counts[1]/total,
        "N%": 100*counts[2]/total,
        "A%": 100*counts[3]/total,
        "FA%": 100*counts[4]/total,
        "Extrêmes (FD+FA)": counts[0] + counts[4],
        "Acquiescence (A+FA)": counts[3] + counts[4],
    }

# ============================================================
# 7) SCORING (OPTIONNEL)
# ============================================================

def calculate_scores(responses: Dict[int, int]) -> Tuple[Dict[str,int], Dict[str,int]]:
    fac_scores: Dict[str, int] = {fac: 0 for fac in facettes_to_domain.keys()}
    for item_id, opt in responses.items():
        fac = item_to_facette[item_id]
        fac_scores[fac] += scoring_key[item_id][opt]

    dom_scores: Dict[str, int] = {d: 0 for d in domain_labels.keys()}
    for fac, sc in fac_scores.items():
        dom_scores[facettes_to_domain[fac]] += sc
    return fac_scores, dom_scores

# ============================================================
# 8) EXPORTS
# ============================================================

def export_csv(valid: bool, reasons: List[str], stats: OMRStats, style: Dict[str,float],
               fac_scores: Optional[Dict[str,int]], dom_scores: Optional[Dict[str,int]]) -> str:
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
    w.writerow(["STYLE", "Indice", "Valeur"])
    for k, v in style.items():
        w.writerow(["STYLE", k, v])

    if dom_scores is not None and fac_scores is not None:
        w.writerow([])
        w.writerow(["DOMAINES", "Domaine", "Score brut"])
        for d in sorted(dom_scores.keys()):
            w.writerow(["DOMAINES", domain_labels[d], dom_scores[d]])

        w.writerow([])
        w.writerow(["FACETTES", "Facette", "Score brut"])
        for fac in sorted(fac_scores.keys()):
            w.writerow(["FACETTES", facette_labels.get(fac, fac), fac_scores[fac]])
    else:
        w.writerow([])
        w.writerow(["SCORING", "Statut", "Désactivé (tables non fournies)"])

    return out.getvalue()

def export_txt(valid: bool, reasons: List[str], stats: OMRStats, style: Dict[str,float],
               fac_scores: Optional[Dict[str,int]], dom_scores: Optional[Dict[str,int]]) -> str:
    lines: List[str] = []
    lines.append("RAPPORT NEO PI-R — OMR CLINIQUE (PHOTO SMARTPHONE)")
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
    lines.append("STYLE DE RÉPONSE (DESCRIPTIF)")
    for k, v in style.items():
        lines.append(f"{k}: {v}")

    if dom_scores is not None and fac_scores is not None:
        lines.append("")
        lines.append("TOTAUX DOMAINES (BRUT)")
        for d in sorted(dom_scores.keys()):
            lines.append(f"{domain_labels[d]}: {dom_scores[d]}")

        lines.append("")
        lines.append("SCORES FACETTES (BRUT)")
        for fac in sorted(fac_scores.keys()):
            lines.append(f"{facette_labels.get(fac, fac)}: {fac_scores[fac]}")
    else:
        lines.append("")
        lines.append("SCORING")
        lines.append("Désactivé : tables scoring_key et/ou item_to_facette non fournies.")

    return "\n".join(lines)

# ============================================================
# 9) UI
# ============================================================

st.title("NEO PI-R — OMR Clinique (Photo smartphone)")
st.caption("Lecture des lettres entourées FD/D/N/A/FA — validité protocolaire automatique — scoring optionnel si tables collées.")

with st.sidebar:
    st.markdown("## Paramètres (calibration)")
    cfg = OMRConfig(
        grid_left=st.slider("Grille — gauche", 0.00, 0.20, 0.05, 0.005),
        grid_right=st.slider("Grille — droite", 0.80, 1.00, 0.95, 0.005),
        grid_top=st.slider("Grille — haut", 0.10, 0.35, 0.205, 0.005),
        grid_bottom=st.slider("Grille — bas", 0.70, 0.95, 0.86, 0.005),
        impute_blank_to_N=st.toggle("Imputer item vide à N (pour scoring)", value=True),
    )
    cfg.auto_threshold_factor = st.slider("Auto-seuil (facteur)", 1.2, 3.0, 1.8, 0.05)
    cfg.ambiguity_rel_gap = st.slider("Ambiguïté (gap relatif)", 0.02, 0.40, 0.12, 0.01)
    cfg.weak_rel_margin = st.slider("Marquage faible (marge)", 1.05, 2.00, 1.35, 0.05)
    cfg.blank_detect_margin = st.slider("Détection vide (marge)", 0.70, 1.10, 0.92, 0.01)

    st.markdown("---")
    st.markdown("## Validité protocolaire")
    vcfg = ValidityConfig(
        blank_invalid_threshold=st.number_input("Invalide si items vides ≥", 0, 240, 15, 1),
        neutral_invalid_threshold=st.number_input("Invalide si N cochés ≥", 0, 240, 42, 1),
    )

    st.markdown("---")
    show_overlay = st.toggle("Afficher overlay ROIs", value=False)
    show_audit = st.toggle("Audit tables (scoring)", value=True)

if show_audit:
    with st.expander("Audit tables (scoring) — facultatif", expanded=False):
        e1 = audit_scoring_key(scoring_key)
        e2 = audit_item_to_facette(item_to_facette)
        # si les 2 tables sont complètes et correctes, l’audit peut être long -> on affiche résumé
        if scoring_ready(scoring_key, item_to_facette):
            st.success("Tables complètes (240/240). Scoring activé.")
        else:
            st.info("Scoring désactivé tant que les 2 tables ne sont pas complètes. Validité OMR fonctionne quand même.")
        st.code("\n".join((e1 + e2)[:200]), language="text")

st.markdown(
    "<div class='note'><b>Important :</b> Même si tu ne colles pas les tables, l’app analyse la feuille et "
    "retourne <b>VALIDE/INVALIDE</b>. Le scoring (facettes/domaines) devient actif seulement si les tables sont complètes.</div>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("Importer la feuille (photo JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    colA, colB = st.columns([0.7, 0.3], vertical_alignment="top")
    with colA:
        raw_img = Image.open(uploaded)
        st.image(raw_img, caption="Image importée (brute)", use_container_width=True)
    with colB:
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

            # Validité (✅ se calcule toujours)
            valid, reasons = protocol_validity(stats, vcfg)

            # Style descriptif
            style = response_style(responses, mark_state)

            # Scoring (✅ seulement si tables complètes)
            scoring_enabled = scoring_ready(scoring_key, item_to_facette)
            fac_scores = None
            dom_scores = None
            if scoring_enabled:
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

            if not scoring_enabled:
                st.info("Scoring désactivé : colle scoring_key + item_to_facette complets (1..240) pour activer les scores.")

            tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Qualité & images", "Avertissements", "Exports"])

            with tab1:
                st.subheader("Distribution des réponses (descriptif)")
                st.write(style)

                if scoring_enabled and dom_scores is not None and fac_scores is not None:
                    st.subheader("Domaines (scores bruts)")
                    dom_table = [{"Domaine": domain_labels[d], "Score brut": dom_scores[d]} for d in sorted(dom_scores.keys())]
                    st.dataframe(dom_table, use_container_width=True, hide_index=True)

                    st.subheader("Facettes (scores bruts)")
                    fac_table = [{"Facette": facette_labels.get(f, f), "Score brut": fac_scores[f]} for f in sorted(fac_scores.keys())]
                    st.dataframe(fac_table, use_container_width=True, hide_index=True)

            with tab2:
                st.subheader("Qualité")
                st.write({
                    "document_detecté": doc_found,
                    "seuil_encrage_auto": stats.threshold_ink,
                    "ink_médian": stats.ink_median,
                    "ink_p10": stats.ink_p10,
                    "ink_p90": stats.ink_p90,
                })

                c1, c2 = st.columns(2)
                with c1:
                    st.image(paper_bgr, channels="BGR", caption="Feuille redressée (perspective)", use_container_width=True)
                with c2:
                    st.image(thr, clamp=True, caption="Binarisation (encre = blanc)", use_container_width=True)

                if show_overlay and overlay is not None:
                    st.image(overlay, channels="BGR", caption="Overlay ROIs (contrôle)", use_container_width=True)

            with tab3:
                st.subheader("Avertissements")
                if warnings:
                    with st.expander("Afficher", expanded=True):
                        for w in warnings[:500]:
                            st.warning(w)
                    if len(warnings) > 500:
                        st.info(f"{len(warnings)} avertissements. Affichage limité.")
                else:
                    st.success("Aucun avertissement.")

            with tab4:
                st.subheader("Exports")
                csv_text = export_csv(valid, reasons, stats, style, fac_scores, dom_scores)
                st.download_button("Télécharger CSV", csv_text, file_name="neo_pir_export.csv", mime="text/csv")

                txt_text = export_txt(valid, reasons, stats, style, fac_scores, dom_scores)
                st.download_button("Télécharger rapport TXT", txt_text, file_name="neo_pir_report.txt", mime="text/plain")

        except Exception as e:
            st.error(f"Erreur : {e}")

st.markdown("<div class='footer'>NEO PI-R — OMR Clinique · © 2026</div>", unsafe_allow_html=True)
