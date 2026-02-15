import streamlit as st
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import imutils
from PIL import Image
import io
import csv

# =========================================================
# PAGE CONFIG (doit être avant tout st.*)
# =========================================================
st.set_page_config(
    page_title="NEO PI-R — OMR Clinique",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLE (sobre, clinique)
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
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DONNÉES (COLLE ICI TES DICTIONNAIRES COMPLETS)
# =========================================================

# ---- scoring_key COMPLET (240 items) ----
# Colle ici ton scoring_key complet
scoring_key = {
    # ... COLLER TON scoring_key COMPLET ICI ...
}

# ---- item_to_facette COMPLET ----
# Colle ici ton item_to_facette complet
item_to_facette = {
    # ... COLLER TON item_to_facette COMPLET ICI ...
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

# =========================================================
# MÉTHODES SCIENTIFIQUES (OMR LETTRES ENTOURÉES)
# =========================================================
def preprocess_image(pil_img: Image.Image):
    """
    1) Conversion PIL -> OpenCV
    2) Détection contour du document + correction perspective
    3) Gray + threshold (noir = encre)
    """
    img = pil_img.convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    # fallback: si contour non détecté, on tente quand même (moins fiable)
    if docCnt is None:
        paper = img.copy()
        warped_gray = gray.copy()
    else:
        paper = four_point_transform(img, docCnt.reshape(4, 2))
        warped_gray = four_point_transform(gray, docCnt.reshape(4, 2))

    # threshold binaire inversé : pixels "encre" => 255
    thresh = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return paper, warped_gray, thresh


def read_answers_letters(
    warped_gray,
    thresh,
    grid_left=0.05, grid_right=0.95,
    grid_top=0.205, grid_bottom=0.86,
    rows=30, cols=8,
    option_centers=(0.12, 0.32, 0.52, 0.72, 0.90),
    box_w_ratio=0.16, box_h_ratio=0.60,
    min_mark_threshold=1200,
    ambiguity_gap=0.12,     # % gap relatif entre top1 et top2
    impute_blank_to_N=True
):
    """
    Lecture OMR sur FEUILLE SANS BULLES (lettres entourées).
    On mesure l'encrage dans 5 zones (FD,D,N,A,FA) par item.

    - min_mark_threshold : seuil minimal d'encrage pour déclarer "marqué"
    - ambiguity_gap : si (top1 - top2)/top1 < ambiguity_gap => ambiguïté
    """
    # Normalisation dimension pour stabilité
    H, W = warped_gray.shape[:2]
    target_w = 1700
    scale = target_w / float(W)
    warped_gray = cv2.resize(warped_gray, (target_w, int(H * scale)))
    thresh = cv2.resize(thresh, (target_w, int(H * scale)))
    H, W = warped_gray.shape[:2]

    # Zone grille
    x0 = int(grid_left * W);  x1 = int(grid_right * W)
    y0 = int(grid_top * H);   y1 = int(grid_bottom * H)

    grid_w = x1 - x0
    grid_h = y1 - y0
    cell_w = grid_w / cols
    cell_h = grid_h / rows

    responses = {}
    warnings = []

    blank_count = 0
    neutral_marked_count = 0
    neutral_imputed_count = 0
    low_mark_count = 0
    ambiguity_count = 0

    best_inks = []

    for r in range(rows):
        for c in range(cols):
            item_id = (r + 1) + 30 * c  # numérotation verticale par colonne

            # cellule item
            cx0 = int(x0 + c * cell_w)
            cy0 = int(y0 + r * cell_h)
            cw = int(cell_w)
            ch = int(cell_h)

            bw = max(8, int(cw * box_w_ratio))
            bh = max(8, int(ch * box_h_ratio))
            by = int(cy0 + (ch - bh) * 0.50)

            ink_values = []
            rois = []

            for oc in option_centers:
                bx = int(cx0 + oc * cw - bw // 2)
                bx = max(0, min(W - bw - 1, bx))
                by2 = max(0, min(H - bh - 1, by))

                roi = thresh[by2:by2 + bh, bx:bx + bw]
                ink = int(cv2.countNonZero(roi))
                ink_values.append(ink)
                rois.append((bx, by2, bw, bh))

            best_idx = int(np.argmax(ink_values))
            best_ink = int(ink_values[best_idx])
            best_inks.append(best_ink)

            # Ambiguïté
            sorted_inks = sorted(ink_values, reverse=True)
            if sorted_inks[0] > 0:
                rel_gap = (sorted_inks[0] - sorted_inks[1]) / float(sorted_inks[0])
            else:
                rel_gap = 1.0

            # Décision marqué vs vide
            if best_ink < min_mark_threshold:
                blank_count += 1
                if impute_blank_to_N:
                    responses[item_id] = 2  # N index 2
                    neutral_imputed_count += 1
                    warnings.append(f"Item {item_id}: non répondu → imputé à N (2 points)")
                else:
                    responses[item_id] = best_idx
                    warnings.append(f"Item {item_id}: non répondu")
            else:
                responses[item_id] = best_idx
                if best_idx == 2:
                    neutral_marked_count += 1

                # marquage faible (scientifique: encrage proche du seuil)
                if best_ink < int(min_mark_threshold * 1.35):
                    low_mark_count += 1
                    warnings.append(f"Item {item_id}: marquage faible (ink={best_ink})")

                if rel_gap < ambiguity_gap:
                    ambiguity_count += 1
                    warnings.append(f"Item {item_id}: ambiguïté (top1={sorted_inks[0]}, top2={sorted_inks[1]})")

    stats = {
        "total_items": rows * cols,
        "blank_count": blank_count,
        "neutral_marked_count": neutral_marked_count,
        "neutral_imputed_count": neutral_imputed_count,
        "neutral_total_count": neutral_marked_count + neutral_imputed_count,
        "low_mark_count": low_mark_count,
        "ambiguity_count": ambiguity_count,
        "best_ink_median": int(np.median(best_inks)) if best_inks else 0,
        "best_ink_p10": int(np.percentile(best_inks, 10)) if best_inks else 0,
        "best_ink_p90": int(np.percentile(best_inks, 90)) if best_inks else 0,
    }

    return responses, warnings, stats


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


def protocol_validity(stats, blank_invalid_threshold=15, neutral_invalid_threshold=42):
    reasons = []
    if stats["blank_count"] >= blank_invalid_threshold:
        reasons.append(f"Items non répondus: {stats['blank_count']} (seuil ≥ {blank_invalid_threshold}).")
    if stats["neutral_total_count"] >= neutral_invalid_threshold:
        reasons.append(f"Réponses Neutres (N) totales: {stats['neutral_total_count']} (seuil ≥ {neutral_invalid_threshold}).")
    return (len(reasons) == 0), reasons


# =========================================================
# UI — CLINIQUE
# =========================================================
title_col, meta_col = st.columns([0.75, 0.25], vertical_alignment="bottom")
with title_col:
    st.title("NEO PI-R — Analyse OMR (Feuille de réponses)")
    st.caption("Lecture scientifique des lettres entourées FD / D / N / A / FA — 240 items (8 colonnes × 30 lignes).")
with meta_col:
    st.write("")

with st.sidebar:
    st.markdown("## Acquisition (qualité)")
    st.caption("Recommandé : scan ou photo verticale, feuille à plat, lumière homogène, sans ombres.")

    st.markdown("---")
    st.markdown("## Paramètres de lecture")
    impute_blank_to_N = st.toggle("Imputer les items vides à N (2 points)", value=True)

    st.markdown("### Validité protocole")
    blank_invalid_threshold = st.number_input("Items vides (invalide si ≥)", 0, 240, 15, 1)
    neutral_invalid_threshold = st.number_input("N total (invalide si ≥)", 0, 240, 42, 1)

    st.markdown("### Zone de grille (à calibrer si nécessaire)")
    grid_left = st.slider("Grille — gauche", 0.00, 0.20, 0.05, 0.005)
    grid_right = st.slider("Grille — droite", 0.80, 1.00, 0.95, 0.005)
    grid_top = st.slider("Grille — haut", 0.10, 0.35, 0.205, 0.005)
    grid_bottom = st.slider("Grille — bas", 0.70, 0.95, 0.86, 0.005)

    st.markdown("### Seuils OMR (encrage)")
    min_mark_threshold = st.slider("Seuil marquage (ink)", 200, 4000, 1200, 50)
    ambiguity_gap = st.slider("Seuil ambiguïté (gap relatif)", 0.02, 0.40, 0.12, 0.01)

    st.markdown("---")
    st.markdown("## À propos")
    st.caption("NEO PI-R OMR — v1.0 (Clinique)")
    st.caption("© 2026 Yacine Adaoun — Tous droits réservés")

st.markdown("### Import")
c1, c2 = st.columns([0.65, 0.35], vertical_alignment="top")

with c1:
    uploaded_file = st.file_uploader("Charger une feuille (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Contrôles**")
    run = st.button("Analyser", type="primary", disabled=(uploaded_file is None))
    st.caption("Détecte la feuille, redresse, lit les réponses, calcule les scores et la validité.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RUN
# =========================================================
if uploaded_file and run:
    try:
        pil_img = Image.open(uploaded_file)
        paper, warped_gray, thresh = preprocess_image(pil_img)

        # Lecture des réponses (lettres)
        responses, warnings, stats = read_answers_letters(
            warped_gray,
            thresh,
            grid_left=grid_left, grid_right=grid_right,
            grid_top=grid_top, grid_bottom=grid_bottom,
            min_mark_threshold=min_mark_threshold,
            ambiguity_gap=ambiguity_gap,
            impute_blank_to_N=impute_blank_to_N
        )

        # Scores
        facette_scores, domain_scores = calculate_scores(responses)

        # Validité
        is_valid, reasons = protocol_validity(
            stats,
            blank_invalid_threshold=int(blank_invalid_threshold),
            neutral_invalid_threshold=int(neutral_invalid_threshold)
        )

        # Indicateurs “scientifiques”
        total_items = stats["total_items"]
        response_rate = 100 * (1 - stats["blank_count"] / max(1, total_items))

        # “Confidence” = pénalise ambiguïté + marquage faible
        penalty = (stats["ambiguity_count"] * 1.0 + stats["low_mark_count"] * 0.5) / max(1, total_items)
        confidence = max(0.0, 100 * (1 - penalty))

        st.markdown("### Synthèse")
        k1, k2, k3, k4 = st.columns(4)

        k1.markdown(
            f"<div class='card'><div class='label'>Validité</div><div class='value'>{'Valide' if is_valid else 'Invalide'}</div>"
            f"<div class='sub'>Règles: vides ≥ {blank_invalid_threshold} ; N ≥ {neutral_invalid_threshold}</div></div>",
            unsafe_allow_html=True
        )
        k2.markdown(
            f"<div class='card'><div class='label'>Taux de réponse</div><div class='value'>{response_rate:.1f}%</div>"
            f"<div class='sub'>Vides: {stats['blank_count']} / {total_items}</div></div>",
            unsafe_allow_html=True
        )
        k3.markdown(
            f"<div class='card'><div class='label'>N total</div><div class='value'>{stats['neutral_total_count']}</div>"
            f"<div class='sub'>N cochés: {stats['neutral_marked_count']} ; imputés: {stats['neutral_imputed_count']}</div></div>",
            unsafe_allow_html=True
        )
        k4.markdown(
            f"<div class='card'><div class='label'>Qualité de lecture</div><div class='value'>{confidence:.1f}%</div>"
            f"<div class='sub'>Ambiguïtés: {stats['ambiguity_count']} ; marquages faibles: {stats['low_mark_count']}</div></div>",
            unsafe_allow_html=True
        )

        st.subheader("Décision protocolaire")
        if is_valid:
            st.success("Protocole valide.")
        else:
            st.error("Protocole invalide.")
            for r in reasons:
                st.warning(r)

        tab1, tab2, tab3, tab4 = st.tabs(["Scores", "Qualité", "Avertissements", "Exports"])

        with tab1:
            fac_data = []
            for fac in sorted(facette_labels):
                items = [str(k) for k, v in item_to_facette.items() if v == fac]
                fac_data.append({
                    "Facette": facette_labels[fac],
                    "Items": ", ".join(items),
                    "Score brut": facette_scores[fac]
                })
            st.subheader("Scores par facette")
            st.dataframe(fac_data, use_container_width=True, hide_index=True)

            dom_data = [{"Domaine": domain_labels[d], "Score": domain_scores[d]} for d in sorted(domain_labels)]
            st.subheader("Totaux par domaine")
            st.dataframe(dom_data, use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("Statistiques d’encrage (contrôle scientifique)")
            st.write(
                {
                    "ink_median": stats["best_ink_median"],
                    "ink_p10": stats["best_ink_p10"],
                    "ink_p90": stats["best_ink_p90"],
                    "seuil_marque": int(min_mark_threshold)
                }
            )
            st.caption("Si ink_median est très bas, augmente la qualité photo/scan ou ajuste le seuil.")

            cimg1, cimg2 = st.columns(2)
            with cimg1:
                st.subheader("Original")
                st.image(pil_img, use_container_width=True)
            with cimg2:
                st.subheader("Redressé / Analyse")
                st.image(paper, channels="BGR", use_container_width=True)

        with tab3:
            st.subheader("Journal")
            if warnings:
                with st.expander("Afficher", expanded=True):
                    for w in warnings[:500]:
                        st.warning(w)
                if len(warnings) > 500:
                    st.info(f"{len(warnings)} avertissements au total. Affichage limité à 500.")
            else:
                st.success("Aucun avertissement.")

        with tab4:
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

            st.download_button("Télécharger CSV", output.getvalue(), "neo_pir_scores.csv", "text/csv")

            # Rapport TXT
            lines = []
            lines.append("RAPPORT NEO PI-R — OMR CLINIQUE")
            lines.append("")
            lines.append("VALIDITÉ")
            lines.append("Valide" if is_valid else "Invalide")
            for r in reasons:
                lines.append(f"- {r}")
            lines.append("")
            lines.append("INDICATEURS")
            lines.append(f"Taux de réponse: {response_rate:.1f}%")
            lines.append(f"Items vides: {stats['blank_count']}")
            lines.append(f"N cochés: {stats['neutral_marked_count']}")
            lines.append(f"N imputés: {stats['neutral_imputed_count']}")
            lines.append(f"N total: {stats['neutral_total_count']}")
            lines.append(f"Ambiguïtés: {stats['ambiguity_count']}")
            lines.append(f"Marquages faibles: {stats['low_mark_count']}")
            lines.append(f"Encrage médian: {stats['best_ink_median']}")
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

    except Exception as e:
        st.error(f"Erreur : {e}")

st.markdown("<div class='footer'>NEO PI-R — OMR Clinique v1.0 · © 2026 Yacine Adaoun</div>", unsafe_allow_html=True)
