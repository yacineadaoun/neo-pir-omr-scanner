import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from PIL import Image
import io
import csv

# =========================================================
# CONFIG PAGE (DOIT ÊTRE AVANT TOUT st.*)
# =========================================================
st.set_page_config(
    page_title="NEO PI-R — OMR Clinique",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS (sobre / professionnel)
# =========================================================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      h1, h2, h3 { letter-spacing: -0.2px; }

      div.stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.7rem 1rem;
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
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 1) COLLE ICI TES DICTIONNAIRES COMPLETS
# =========================================================

# ---- scoring_key COMPLET (240 items) ----
scoring_key = {
    # COLLE ICI TON scoring_key COMPLET
}

# ---- item_to_facette COMPLET ----
item_to_facette = {
    # COLLE ICI TON item_to_facette COMPLET
}

# ---- Mappings domaines (inchangés) ----
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

# =========================================================
# 2) IMAGE PIPELINE
# =========================================================
def preprocess_image(pil_img: Image.Image):
    """
    - Convertit PIL -> OpenCV BGR
    - Tente de détecter le contour du document, redresse (perspective)
    - Produit (paper_bgr, gray, thresh_inv)
    """
    img_rgb = pil_img.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        paper = img_bgr.copy()
        warped_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    else:
        paper = four_point_transform(img_bgr, docCnt.reshape(4, 2))
        warped_gray = four_point_transform(gray, docCnt.reshape(4, 2))

    # Binarisation: pixels "encre" -> 255
    thresh = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return paper, warped_gray, thresh


def normalize_width(gray, thresh, target_w=1700):
    H, W = gray.shape[:2]
    if W == 0:
        raise ValueError("Image invalide (largeur=0).")
    scale = target_w / float(W)
    new_size = (target_w, max(1, int(H * scale)))
    gray2 = cv2.resize(gray, new_size)
    thr2 = cv2.resize(thresh, new_size)
    return gray2, thr2


# =========================================================
# 3) OMR "LETTRES ENTOURÉES" (pas de bulles)
# =========================================================
def read_answers_letters(
    thresh,
    grid_left=0.05, grid_right=0.95,
    grid_top=0.205, grid_bottom=0.86,
    rows=30, cols=8,
    option_centers=(0.12, 0.32, 0.52, 0.72, 0.90),  # FD D N A FA
    box_w_ratio=0.16, box_h_ratio=0.60,
    min_mark_threshold=1200,
    ambiguity_gap=0.12,
    impute_blank_to_N=True,
    draw_debug=False
):
    """
    Mesure l'encrage dans 5 ROIs par item.
    - item_id = (ligne+1) + 30*colonne  (numérotation verticale par colonne)
    - si "vide" (encrage < seuil), imputation optionnelle à N (index 2)
    """
    H, W = thresh.shape[:2]

    # zone de grille
    x0 = int(grid_left * W);  x1 = int(grid_right * W)
    y0 = int(grid_top * H);   y1 = int(grid_bottom * H)

    x0 = max(0, min(W - 2, x0)); x1 = max(x0 + 1, min(W - 1, x1))
    y0 = max(0, min(H - 2, y0)); y1 = max(y0 + 1, min(H - 1, y1))

    grid_w = x1 - x0
    grid_h = y1 - y0
    cell_w = grid_w / cols
    cell_h = grid_h / rows

    responses = {}
    warnings = []

    blank_count = 0
    neutral_marked = 0
    neutral_imputed = 0
    low_mark = 0
    ambiguous = 0

    debug_canvas = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR) if draw_debug else None

    for r in range(rows):
        for c in range(cols):
            item_id = (r + 1) + 30 * c

            cx0 = int(x0 + c * cell_w)
            cy0 = int(y0 + r * cell_h)
            cw = int(cell_w)
            ch = int(cell_h)

            bw = max(8, int(cw * box_w_ratio))
            bh = max(8, int(ch * box_h_ratio))
            by = int(cy0 + (ch - bh) * 0.50)

            inks = []
            rois = []

            for oc in option_centers:
                bx = int(cx0 + oc * cw - bw // 2)
                bx = max(0, min(W - bw - 1, bx))
                by2 = max(0, min(H - bh - 1, by))

                roi = thresh[by2:by2 + bh, bx:bx + bw]
                ink = int(cv2.countNonZero(roi))
                inks.append(ink)
                rois.append((bx, by2, bw, bh))

            best_idx = int(np.argmax(inks))
            best_ink = int(inks[best_idx])

            sorted_inks = sorted(inks, reverse=True)
            if sorted_inks[0] > 0:
                rel_gap = (sorted_inks[0] - sorted_inks[1]) / float(sorted_inks[0])
            else:
                rel_gap = 1.0

            # item vide ?
            if best_ink < min_mark_threshold:
                blank_count += 1
                if impute_blank_to_N:
                    responses[item_id] = 2
                    neutral_imputed += 1
                    warnings.append(f"Item {item_id}: non répondu → imputé à N (2 points)")
                else:
                    responses[item_id] = best_idx
                    warnings.append(f"Item {item_id}: non répondu (non imputé)")
            else:
                responses[item_id] = best_idx
                if best_idx == 2:
                    neutral_marked += 1

                if best_ink < int(min_mark_threshold * 1.35):
                    low_mark += 1
                    warnings.append(f"Item {item_id}: marquage faible (ink={best_ink})")

                if rel_gap < ambiguity_gap:
                    ambiguous += 1
                    warnings.append(f"Item {item_id}: ambiguïté (top1={sorted_inks[0]}, top2={sorted_inks[1]})")

            # debug draw
            if draw_debug and debug_canvas is not None:
                for j, (bx, by2, bw, bh) in enumerate(rois):
                    color = (0, 255, 0) if j == responses[item_id] else (200, 200, 200)
                    cv2.rectangle(debug_canvas, (bx, by2), (bx + bw, by2 + bh), color, 1)

    stats = {
        "total_items": rows * cols,
        "blank_count": blank_count,
        "neutral_marked_count": neutral_marked,
        "neutral_imputed_count": neutral_imputed,
        "neutral_total_count": neutral_marked + neutral_imputed,
        "low_mark_count": low_mark,
        "ambiguity_count": ambiguous,
    }

    return responses, warnings, stats, debug_canvas


# =========================================================
# 4) SCORING + INDICES CLINIQUES
# =========================================================
def calculate_scores(responses):
    facette_scores = {fac: 0 for fac in facette_labels}
    for item_id, option_idx in responses.items():
        if item_id in scoring_key and item_id in item_to_facette:
            fac = item_to_facette[item_id]
            facette_scores[fac] += scoring_key[item_id][option_idx]

    domain_scores = {dom: 0 for dom in domain_labels}
    for fac, sc in facette_scores.items():
        dom = facettes_to_domain.get(fac)
        if dom:
            domain_scores[dom] += sc

    return facette_scores, domain_scores


def response_style_indices(responses):
    """
    Indices utiles (non diagnostiques à eux seuls) :
    - extrêmes: FD/FA
    - acquiescence: A/FA
    - neutralité: N
    - distribution brute des options
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for _, idx in responses.items():
        if idx in counts:
            counts[idx] += 1

    total = sum(counts.values()) if sum(counts.values()) else 1
    pct = {k: 100 * v / total for k, v in counts.items()}

    indices = {
        "FD": counts[0], "D": counts[1], "N": counts[2], "A": counts[3], "FA": counts[4],
        "FD%": pct[0], "D%": pct[1], "N%": pct[2], "A%": pct[3], "FA%": pct[4],
        "Extrêmes (FD+FA)": counts[0] + counts[4],
        "Acquiescence (A+FA)": counts[3] + counts[4],
    }
    return indices


def protocol_validity(stats, blank_invalid_threshold=15, neutral_invalid_threshold=42):
    reasons = []
    if stats["blank_count"] >= blank_invalid_threshold:
        reasons.append(f"Items non répondus: {stats['blank_count']} (seuil ≥ {blank_invalid_threshold}).")
    if stats["neutral_total_count"] >= neutral_invalid_threshold:
        reasons.append(f"Réponses N totales: {stats['neutral_total_count']} (seuil ≥ {neutral_invalid_threshold}).")
    return (len(reasons) == 0), reasons


# =========================================================
# UI
# =========================================================
left, right = st.columns([0.78, 0.22], vertical_alignment="bottom")
with left:
    st.title("NEO PI-R — OMR Clinique (feuille de réponses)")
    st.caption("Lecture des réponses par encrage sur les lettres FD / D / N / A / FA (240 items).")
with right:
    st.write("")

with st.sidebar:
    st.markdown("## Paramètres")
    debug_draw = st.toggle("Afficher overlay de lecture", value=False)
    impute_blank_to_N = st.toggle("Imputer item vide à N (2 points)", value=True)

    st.markdown("---")
    st.markdown("### Validité protocolaire (paramétrable)")
    blank_invalid_threshold = st.number_input("Items vides (invalide si ≥)", 0, 240, 15, 1)
    neutral_invalid_threshold = st.number_input("N total (invalide si ≥)", 0, 240, 42, 1)

    st.markdown("---")
    st.markdown("### Zone de la grille (calibration)")
    grid_left = st.slider("Gauche", 0.00, 0.20, 0.05, 0.005)
    grid_right = st.slider("Droite", 0.80, 1.00, 0.95, 0.005)
    grid_top = st.slider("Haut", 0.10, 0.35, 0.205, 0.005)
    grid_bottom = st.slider("Bas", 0.70, 0.95, 0.86, 0.005)

    st.markdown("---")
    st.markdown("### Seuils OMR")
    min_mark_threshold = st.slider("Seuil encrage (ink)", 200, 4000, 1200, 50)
    ambiguity_gap = st.slider("Ambiguïté (gap relatif)", 0.02, 0.40, 0.12, 0.01)

    st.markdown("---")
    st.caption("Référence documentaire : manuel fourni (scan).")
    st.caption("Usage professionnel sous responsabilité du psychologue.")

st.markdown("### Import")
c1, c2 = st.columns([0.65, 0.35], vertical_alignment="top")
with c1:
    uploaded_file = st.file_uploader("Charger la feuille (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.caption("Conseil: photo nette, feuille à plat, lumière homogène, sans ombres.")
with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Contrôles**")
    run = st.button("Analyser", type="primary", disabled=(uploaded_file is None))
    st.caption("Redressement → lecture OMR → scores → validité → exports.")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file and run:
    try:
        pil_img = Image.open(uploaded_file)

        paper, gray, thresh = preprocess_image(pil_img)
        gray_n, thr_n = normalize_width(gray, thresh, target_w=1700)

        responses, warnings, stats, overlay = read_answers_letters(
            thr_n,
            grid_left=grid_left, grid_right=grid_right,
            grid_top=grid_top, grid_bottom=grid_bottom,
            min_mark_threshold=min_mark_threshold,
            ambiguity_gap=ambiguity_gap,
            impute_blank_to_N=impute_blank_to_N,
            draw_debug=debug_draw
        )

        facette_scores, domain_scores = calculate_scores(responses)
        style = response_style_indices(responses)

        is_valid, reasons = protocol_validity(
            stats,
            blank_invalid_threshold=int(blank_invalid_threshold),
            neutral_invalid_threshold=int(neutral_invalid_threshold)
        )

        total = stats["total_items"]
        response_rate = 100 * (1 - stats["blank_count"] / max(1, total))

        # Qualité lecture = pénalise ambiguïtés + faibles (proxy)
        penalty = (stats["ambiguity_count"] * 1.0 + stats["low_mark_count"] * 0.5) / max(1, total)
        confidence = max(0.0, 100 * (1 - penalty))

        st.markdown("### Synthèse")
        k1, k2, k3, k4 = st.columns(4)

        k1.markdown(
            f"<div class='card'><div class='label'>Validité</div><div class='value'>{'Valide' if is_valid else 'Invalide'}</div>"
            f"<div class='sub'>Seuils: vides ≥ {blank_invalid_threshold} ; N ≥ {neutral_invalid_threshold}</div></div>",
            unsafe_allow_html=True
        )
        k2.markdown(
            f"<div class='card'><div class='label'>Taux de réponse</div><div class='value'>{response_rate:.1f}%</div>"
            f"<div class='sub'>Vides: {stats['blank_count']} / {total}</div></div>",
            unsafe_allow_html=True
        )
        k3.markdown(
            f"<div class='card'><div class='label'>N total</div><div class='value'>{stats['neutral_total_count']}</div>"
            f"<div class='sub'>N cochés: {stats['neutral_marked_count']} ; imputés: {stats['neutral_imputed_count']}</div></div>",
            unsafe_allow_html=True
        )
        k4.markdown(
            f"<div class='card'><div class='label'>Qualité de lecture</div><div class='value'>{confidence:.1f}%</div>"
            f"<div class='sub'>Ambiguïtés: {stats['ambiguity_count']} ; faibles: {stats['low_mark_count']}</div></div>",
            unsafe_allow_html=True
        )

        st.subheader("Décision protocolaire")
        if is_valid:
            st.success("Protocole valide.")
        else:
            st.error("Protocole invalide.")
            for r in reasons:
                st.warning(r)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Scores", "Style de réponse", "Images", "Avertissements", "Exports"]
        )

        with tab1:
            fac_data = []
            for fac in sorted(facette_labels):
                items = [str(k) for k, v in item_to_facette.items() if v == fac]
                fac_data.append({
                    "Facette": facette_labels[fac],
                    "Items": ", ".join(items),
                    "Score brut": facette_scores.get(fac, 0)
                })
            st.subheader("Scores par facette")
            st.dataframe(fac_data, use_container_width=True, hide_index=True)

            dom_data = [{"Domaine": domain_labels[d], "Score": domain_scores.get(d, 0)} for d in sorted(domain_labels)]
            st.subheader("Totaux par domaine")
            st.dataframe(dom_data, use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("Indices de style de réponse (descriptif)")
            st.caption("Ces indices aident à repérer des profils de réponse atypiques (sans conclure à eux seuls).")
            st.json(style)

        with tab3:
            cimg1, cimg2 = st.columns(2)
            with cimg1:
                st.subheader("Original")
                st.image(pil_img, use_container_width=True)
            with cimg2:
                st.subheader("Binarisé (encre=blanc)")
                st.image(thr_n, clamp=True, use_container_width=True)

            if debug_draw and overlay is not None:
                st.subheader("Overlay de lecture (ROIs)")
                st.image(overlay, channels="BGR", use_container_width=True)

        with tab4:
            st.subheader("Journal")
            if warnings:
                with st.expander("Afficher", expanded=True):
                    for w in warnings[:600]:
                        st.warning(w)
                if len(warnings) > 600:
                    st.info(f"{len(warnings)} avertissements au total. Affichage limité à 600.")
            else:
                st.success("Aucun avertissement.")

        with tab5:
            st.subheader("Exports")
            # CSV
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["Facette", "Items", "Score brut"])
            writer.writeheader()
            writer.writerows(fac_data)

            output.write("\n--- TOTAUX PAR DOMAINE ---\n")
            dom_writer = csv.DictWriter(output, fieldnames=["Domaine", "Score"])
            dom_writer.writeheader()
            dom_writer.writerows(dom_data)

            output.write("\n--- STYLE DE RÉPONSE ---\n")
            for k, v in style.items():
                output.write(f"{k},{v}\n")

            st.download_button("Télécharger CSV", output.getvalue(), "neo_pir_scores.csv", "text/csv")

            # TXT report
            lines = []
            lines.append("RAPPORT NEO PI-R — OMR CLINIQUE")
            lines.append("")
            lines.append("VALIDITÉ")
            lines.append("Valide" if is_valid else "Invalide")
            for r in reasons:
                lines.append(f"- {r}")

            lines.append("")
            lines.append("QUALITÉ LECTURE")
            lines.append(f"Taux de réponse: {response_rate:.1f}%")
            lines.append(f"Qualité proxy: {confidence:.1f}%")
            lines.append(f"Vides: {stats['blank_count']} / {total}")
            lines.append(f"N total: {stats['neutral_total_count']} (cochés={stats['neutral_marked_count']} ; imputés={stats['neutral_imputed_count']})")
            lines.append(f"Ambiguïtés: {stats['ambiguity_count']}")
            lines.append(f"Marquages faibles: {stats['low_mark_count']}")

            lines.append("")
            lines.append("STYLE DE RÉPONSE (DESCRIPTIF)")
            for k, v in style.items():
                lines.append(f"{k}: {v}")

            lines.append("")
            lines.append("SCORES PAR FACETTE")
            for row in fac_data:
                lines.append(f"{row['Facette']}: {row['Score brut']}")

            lines.append("")
            lines.append("TOTAUX DOMAINES")
            for row in dom_data:
                lines.append(f"{row['Domaine']}: {row['Score']}")

            lines.append("")
            lines.append("AVERTISSEMENTS")
            if warnings:
                lines.extend(warnings)
            else:
                lines.append("Aucun avertissement.")

            report = "\n".join(lines)
            st.download_button("Télécharger rapport TXT", report, "neo_pir_report.txt", "text/plain")

    except KeyError as e:
        st.error(f"Erreur de mapping/scoring : clé manquante {e}. Vérifie scoring_key/item_to_facette.")
    except Exception as e:
        st.error(f"Erreur : {e}")

st.markdown("<div class='footer'>NEO PI-R — OMR Clinique v1.0 · © 2026 Yacine Adaoun</div>", unsafe_allow_html=True)
