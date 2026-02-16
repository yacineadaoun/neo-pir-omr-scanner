import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import csv

# ============================================================
# 1) ✅ TES TABLES (tu les as déjà)
#    -> garde-les ici OU charge-les depuis un JSON local
#    -> je ne recolle pas de clé complète issue d’un manuel.
# ============================================================

# 👉 COLLE ICI tes dicts (déjà présents dans ton projet) :
# scoring_key = {...}          # 1..240 -> [4,3,2,1,0] ou inversé selon item
# item_to_facette = {...}      # 1..240 -> 'N1'...'C6'
# facettes_to_domain = {...}
# facette_labels = {...}
# domain_labels = {...}

# ------------------------------------------------------------
# Helpers: sécurité si tu oublies de coller
# ------------------------------------------------------------
def _require_tables():
    missing = []
    for name in ["scoring_key", "item_to_facette", "facettes_to_domain", "facette_labels", "domain_labels"]:
        if name not in globals():
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Tables manquantes: " + ", ".join(missing) +
            ".\nColle tes dictionnaires (scoring_key, item_to_facette, etc.) en haut du fichier."
        )

# ============================================================
# 2) Vision: redressement + extraction zone tableau + grille
# ============================================================

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def find_biggest_quad(edged: np.ndarray):
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def order_points(pts):
    # standard ordering: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_warp(image_bgr, pts, out_w=2000, out_h=2600):
    rect = order_points(pts.astype("float32"))
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_w, out_h))
    return warped

def preprocess_for_detection(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 150)
    return gray, edged

def adaptive_bin(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 10
    )

def crop_grid_region(warped_bgr, top=0.20, bottom=0.83, left=0.05, right=0.95):
    """
    Zone qui contient le grand tableau des 240 items.
    Ajustable via sliders si besoin.
    """
    h, w = warped_bgr.shape[:2]
    y1 = int(h * top)
    y2 = int(h * bottom)
    x1 = int(w * left)
    x2 = int(w * right)
    return warped_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

def build_grid_boxes(grid_rect, rows=30, cols=8):
    """
    grid_rect = (x1,y1,x2,y2) dans l'image warped
    renvoie boxes[item_id] = (x1,y1,x2,y2)
    item_id suit: colonne 0 => 1..30, col 1 => 31..60, ..., col 7 => 211..240
    """
    x1, y1, x2, y2 = grid_rect
    W = x2 - x1
    H = y2 - y1
    cell_w = W / cols
    cell_h = H / rows

    boxes = {}
    for r in range(rows):
        for c in range(cols):
            ix1 = int(x1 + c * cell_w)
            ix2 = int(x1 + (c+1) * cell_w)
            iy1 = int(y1 + r * cell_h)
            iy2 = int(y1 + (r+1) * cell_h)

            item_id = (r + 1) + 30 * c
            boxes[item_id] = (ix1, iy1, ix2, iy2)
    return boxes

def choice_boxes_within_item(item_box, pad_x=0.10, pad_y=0.20):
    """
    Dans une case item, on crée 5 sous-zones (FD D N A FA).
    On évite le numéro item à gauche via pad_x.
    """
    x1, y1, x2, y2 = item_box
    w = x2 - x1
    h = y2 - y1

    # coupe les marges
    ax1 = int(x1 + w * pad_x)
    ax2 = int(x2 - w * 0.03)
    ay1 = int(y1 + h * pad_y)
    ay2 = int(y2 - h * pad_y)

    # 5 slots horizontaux
    slots = []
    span = ax2 - ax1
    slot_w = span / 5.0
    for k in range(5):
        sx1 = int(ax1 + k * slot_w)
        sx2 = int(ax1 + (k+1) * slot_w)
        slots.append((sx1, ay1, sx2, ay2))
    return slots

def ink_score(bin_inv, box):
    """
    Mesure simple: densité de pixels "encre" (blanc dans bin_inv)
    """
    x1,y1,x2,y2 = box
    roi = bin_inv[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)

def detect_responses(
    warped_bgr,
    grid_rect,
    ref_warped_bgr=None,
    pad_x=0.10,
    pad_y=0.20,
    min_mark=0.06,
    amb_diff=0.015
):
    """
    Retour:
      responses[item_id] = choice_index (0..4) ou None
      meta: dict (scores, ambigu, etc.)
      overlay_bgr
    """
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    warped_bin = adaptive_bin(warped_gray)

    # Si référence vierge fournie: absdiff pour isoler l’encre du stylo
    if ref_warped_bgr is not None:
        ref_gray = cv2.cvtColor(ref_warped_bgr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ref_gray, warped_gray)
        diff = cv2.GaussianBlur(diff, (3,3), 0)
        _, diff_bin = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        bin_for_marks = diff_bin
    else:
        # sinon on se base sur binarisation directe
        bin_for_marks = warped_bin

    boxes = build_grid_boxes(grid_rect, rows=30, cols=8)
    overlay = warped_bgr.copy()

    responses = {}
    meta = {
        "mark_scores": {},
        "ambiguous": set(),
        "empty": set()
    }

    for item_id in range(1, 241):
        item_box = boxes[item_id]
        slots = choice_boxes_within_item(item_box, pad_x=pad_x, pad_y=pad_y)

        scores = [ink_score(bin_for_marks, s) for s in slots]
        meta["mark_scores"][item_id] = scores

        best_idx = int(np.argmax(scores))
        best = scores[best_idx]
        sorted_scores = sorted(scores, reverse=True)

        # critères
        if best < min_mark:
            responses[item_id] = None
            meta["empty"].add(item_id)
            # rouge: item vide (encadre toute la case)
            x1,y1,x2,y2 = item_box
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,255), 2)
            continue

        # ambigu si top2 trop proches
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < amb_diff:
            meta["ambiguous"].add(item_id)
            responses[item_id] = best_idx
            # orange
            sx1,sy1,sx2,sy2 = slots[best_idx]
            cv2.rectangle(overlay, (sx1,sy1), (sx2,sy2), (0,165,255), 2)
        else:
            responses[item_id] = best_idx
            # vert
            sx1,sy1,sx2,sy2 = slots[best_idx]
            cv2.rectangle(overlay, (sx1,sy1), (sx2,sy2), (0,255,0), 2)

    return responses, meta, overlay

# ============================================================
# 3) Scoring + Validité (tes règles)
# ============================================================

CHOICE_LABELS = ["FD", "D", "N", "A", "FA"]
NEUTRAL_IDX = 2
NEUTRAL_SCORE = 2  # "item vide donne 2 points"

def apply_protocol_rules(responses: dict):
    """
    Règles fournies:
      - invalide si items_vides >= 15
      - invalide si N_observés >= 42
      - si valide: imputer chaque vide à N (score 2)
    """
    empty_items = [i for i,v in responses.items() if v is None]
    n_count = sum(1 for v in responses.values() if v == NEUTRAL_IDX)

    invalid_reasons = []
    if len(empty_items) >= 15:
        invalid_reasons.append(f"Trop d'items vides: {len(empty_items)} (>= 15)")
    if n_count >= 42:
        invalid_reasons.append(f"Trop de réponses 'N' (Neutre): {n_count} (>= 42)")

    is_valid = (len(invalid_reasons) == 0)

    imputed = 0
    final_responses = dict(responses)

    if is_valid and len(empty_items) > 0:
        # imputation: vide -> N (idx=2)
        for i in empty_items:
            final_responses[i] = NEUTRAL_IDX
            imputed += 1

    return {
        "is_valid": is_valid,
        "reasons": invalid_reasons,
        "empty_items": empty_items,
        "n_count": n_count,
        "imputed": imputed,
        "final_responses": final_responses
    }

def calculate_scores(final_responses):
    _require_tables()
    facette_scores = {fac: 0 for fac in facette_labels}
    # option_idx -> score via scoring_key[item_id][option_idx]
    for item_id, option_idx in final_responses.items():
        if item_id in scoring_key and item_id in item_to_facette and option_idx is not None:
            fac = item_to_facette[item_id]
            facette_scores[fac] += scoring_key[item_id][option_idx]

    domain_scores = {d: 0 for d in domain_labels}
    for fac, s in facette_scores.items():
        dom = facettes_to_domain[fac]
        domain_scores[dom] += s

    return facette_scores, domain_scores

# ============================================================
# 4) Streamlit UI (pro, mais simple et stable)
# ============================================================

st.set_page_config(page_title="NEO PI-R Scanner", page_icon="🧾", layout="wide")

st.title("NEO PI-R — Scanner & Calculateur (Feuille unique)")
st.caption("Détection par grille (lettres entourées), robuste photo caméra. Export CSV/TXT + overlay de contrôle.")

with st.sidebar:
    st.markdown("## Entrées")
    use_ref = st.toggle("Utiliser une photo de feuille vierge (recommandé)", value=False)

    st.markdown("## Réglages (si besoin)")
    out_w = st.slider("Largeur normalisée", 1400, 2600, 2000, 50)
    out_h = st.slider("Hauteur normalisée", 1800, 3400, 2600, 50)

    st.markdown("### Zone tableau (ratios)")
    top = st.slider("Top", 0.10, 0.35, 0.20, 0.01)
    bottom = st.slider("Bottom", 0.65, 0.95, 0.83, 0.01)
    left = st.slider("Left", 0.01, 0.15, 0.05, 0.01)
    right = st.slider("Right", 0.85, 0.99, 0.95, 0.01)

    st.markdown("### Sous-zones FD/D/N/A/FA")
    pad_x = st.slider("Pad X (éviter numéro)", 0.00, 0.25, 0.10, 0.01)
    pad_y = st.slider("Pad Y (marge verticale)", 0.00, 0.35, 0.20, 0.01)

    st.markdown("### Seuils encre")
    min_mark = st.slider("Seuil 'marqué'", 0.01, 0.20, 0.06, 0.005)
    amb_diff = st.slider("Seuil ambiguïté (diff)", 0.001, 0.05, 0.015, 0.001)

    st.markdown("### Protocole")
    st.write("Invalide si:")
    st.write("- items vides ≥ 15")
    st.write("- réponses 'N' ≥ 42")
    st.write("Imputation si valide: vide → score 2 (N)")

col1, col2 = st.columns([0.6, 0.4], vertical_alignment="top")

with col1:
    filled_file = st.file_uploader("📷 Feuille REMPLIE (jpg/png)", type=["jpg","jpeg","png"], accept_multiple_files=False)
    ref_file = None
    if use_ref:
        ref_file = st.file_uploader("🧾 Feuille VIERGE (même cadrage idéalement)", type=["jpg","jpeg","png"], accept_multiple_files=False)

    run = st.button("🚀 Scanner & Calculer", type="primary", disabled=(filled_file is None))

with col2:
    st.markdown("### Conseils caméra (important)")
    st.write("• Feuille à plat, bien éclairée, sans ombres.")
    st.write("• Prends la photo bien de face (pas d’angle).")
    st.write("• Évite le flou (tap-to-focus).")
    st.write("• La *référence vierge* améliore énormément la fiabilité.")

if run and filled_file:
    try:
        # Load PIL
        filled_pil = Image.open(filled_file)
        filled_bgr = pil_to_bgr(filled_pil)

        # Preprocess
        _, edged = preprocess_for_detection(filled_bgr)
        quad = find_biggest_quad(edged)
        if quad is None:
            raise RuntimeError("Impossible de détecter le contour principal de la feuille. Reprends la photo plus nette/centrée.")

        filled_warp = four_point_warp(filled_bgr, quad, out_w=out_w, out_h=out_h)

        ref_warp = None
        if use_ref and ref_file:
            ref_pil = Image.open(ref_file)
            ref_bgr = pil_to_bgr(ref_pil)
            _, edged_ref = preprocess_for_detection(ref_bgr)
            quad_ref = find_biggest_quad(edged_ref)
            if quad_ref is None:
                raise RuntimeError("Contour feuille vierge non détecté. Reprends la photo vierge.")
            ref_warp = four_point_warp(ref_bgr, quad_ref, out_w=out_w, out_h=out_h)

        # Grid region
        grid_crop, grid_rect_local = crop_grid_region(filled_warp, top=top, bottom=bottom, left=left, right=right)
        x1,y1,x2,y2 = grid_rect_local  # in warped coords

        # Detect responses
        responses, meta, overlay = detect_responses(
            filled_warp,
            grid_rect=(x1,y1,x2,y2),
            ref_warped_bgr=ref_warp,
            pad_x=pad_x,
            pad_y=pad_y,
            min_mark=min_mark,
            amb_diff=amb_diff
        )

        # Protocol
        proto = apply_protocol_rules(responses)

        # KPIs
        items_vides = len(proto["empty_items"])
        n_count = proto["n_count"]
        imputed = proto["imputed"]
        status = "VALIDE" if proto["is_valid"] else "INVALIDE"

        st.markdown("## Résultats")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Items vides", items_vides)
        k2.metric("N observés", n_count)
        k3.metric("Imputations (score=2)", imputed)
        k4.metric("Statut protocole", status)

        if not proto["is_valid"]:
            st.error("Protocole INVALIDE")
            for r in proto["reasons"]:
                st.write(f"• {r}")
            st.info("Même si invalide, l’overlay te montre où la détection a estimé vide / ambigu.")
        else:
            st.success("Protocole VALIDE")
            if imputed > 0:
                st.warning(f"{imputed} item(s) vide(s) imputé(s) à N (score=2).")

        # Scores (si tables présentes)
        facette_scores = None
        domain_scores = None
        score_error = None
        try:
            facette_scores, domain_scores = calculate_scores(proto["final_responses"])
        except Exception as e:
            score_error = str(e)

        tab1, tab2, tab3 = st.tabs(["🧾 Overlay contrôle", "📊 Scores", "📦 Exports"])

        with tab1:
            st.caption("Vert = réponse détectée · Rouge = item vide · Orange = ambigu")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        with tab2:
            if score_error:
                st.warning("Scores non calculés (tables manquantes).")
                st.code(score_error, language="text")
            else:
                # Tables
                data = []
                for fac in sorted(facette_labels):
                    items = [str(k) for k, v in item_to_facette.items() if v == fac]
                    data.append({"Facette": facette_labels[fac], "Items": ", ".join(items), "Score brut": facette_scores[fac]})
                st.subheader("Scores par facette")
                st.dataframe(data, use_container_width=True, hide_index=True)

                dom_data = [{"Domaine": domain_labels[d], "Score": domain_scores[d]} for d in sorted(domain_labels)]
                st.subheader("Scores par domaine")
                st.dataframe(dom_data, use_container_width=True, hide_index=True)

        with tab3:
            # Export réponses item->choix
            resp_rows = []
            for i in range(1,241):
                v = responses[i]
                v_final = proto["final_responses"][i]
                resp_rows.append({
                    "Item": i,
                    "Choix_detecte": "" if v is None else CHOICE_LABELS[v],
                    "Choix_final": "" if v_final is None else CHOICE_LABELS[v_final],
                    "Vide": (v is None),
                    "Ambigu": (i in meta["ambiguous"]),
                    "Scores_slots": meta["mark_scores"][i],
                })

            # CSV export (réponses + statut)
            out = io.StringIO()
            w = csv.writer(out)
            w.writerow(["Statut", status])
            w.writerow(["Items_vides", items_vides])
            w.writerow(["N_observes", n_count])
            w.writerow(["Imputations", imputed])
            if proto["reasons"]:
                w.writerow(["Raisons", " | ".join(proto["reasons"])])
            w.writerow([])
            w.writerow(["Item","Choix_detecte","Choix_final","Vide","Ambigu","Scores_slots"])
            for r in resp_rows:
                w.writerow([r["Item"], r["Choix_detecte"], r["Choix_final"], r["Vide"], r["Ambigu"], r["Scores_slots"]])

            st.download_button("⬇️ Télécharger CSV (réponses + protocole)", out.getvalue(), "neo_pir_scan.csv", "text/csv")

            # TXT report
            report = []
            report.append("RAPPORT NEO PI-R — Scanner")
            report.append("")
            report.append(f"Statut protocole: {status}")
            report.append(f"Items vides: {items_vides}")
            report.append(f"N observés: {n_count}")
            report.append(f"Imputations (score=2): {imputed}")
            if proto["reasons"]:
                report.append("Raisons invalidité:")
                report.extend([f"- {x}" for x in proto["reasons"]])
            report.append("")
            if facette_scores and domain_scores:
                report.append("SCORES PAR FACETTE")
                for fac in sorted(facette_labels):
                    report.append(f"{facette_labels[fac]}: {facette_scores[fac]}")
                report.append("")
                report.append("SCORES PAR DOMAINE")
                for d in sorted(domain_labels):
                    report.append(f"{domain_labels[d]}: {domain_scores[d]}")

            st.download_button("⬇️ Télécharger rapport TXT", "\n".join(report), "neo_pir_report.txt", "text/plain")

    except Exception as e:
        st.error(f"Erreur: {e}")
