import streamlit as st
import cv2
import numpy as np
from imutils import contours
from imutils.perspective import four_point_transform
import imutils
from PIL import Image
import io
import csv

# =========================================================
# CONFIG PAGE (doit être avant tout st.*)
# =========================================================
st.set_page_config(
    page_title="NEO PI-R Scanner",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS (interface pro)
# =========================================================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      h1, h2, h3 { letter-spacing: -0.2px; }

      div.stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
      }

      .kpi {
        border: 1px solid rgba(49, 51, 63, 0.15);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.6);
      }
      .kpi-label { font-size: 12px; color: rgba(49, 51, 63, 0.6); margin-bottom: 4px; }
      .kpi-value { font-size: 22px; font-weight: 700; }
      .kpi-sub { font-size: 12px; color: rgba(49, 51, 63, 0.6); margin-top: 4px; }

      .section {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 16px;
        padding: 16px;
        background: rgba(255,255,255,0.4);
      }

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

# =========================================================
# DONNÉES (COLLER TES TABLES ICI)
# =========================================================

# ---- 1) scoring_key COMPLET (240 items) ----
# COLLE ICI ton scoring_key complet
scoring_key = {
    # ... (COLLER TON DICTIONNAIRE COMPLET ICI) ...
}

# ---- 2) item_to_facette COMPLET ----
item_to_facette = {
    # ... (COLLER TON DICTIONNAIRE COMPLET ICI) ...
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
# FONCTIONS OMR
# =========================================================
def preprocess_image(image: Image.Image):
    """PIL -> OpenCV, détection document, redressement, binarisation."""
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
        raise ValueError("Contour du document non détecté. Photo trop inclinée/ombre/fond chargé.")

    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    thresh = cv2.adaptiveThreshold(
        warped, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return paper, thresh


def detect_bubbles(thresh, min_size=20, ar_min=0.85, ar_max=1.15, expected=1200, min_ratio=0.98):
    """
    Détecte les bulles imprimées.
    - Si > expected: garde les expected plus grandes (filtre bruit)
    - Si < expected*min_ratio: stop (grille non fiable)
    - Retourne EXACTEMENT 'expected' contours, triés top-to-bottom
    """
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    bubbleCnts = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ar = w / float(h)
        if w >= min_size and h >= min_size and (ar_min <= ar <= ar_max):
            bubbleCnts.append(c)

    if len(bubbleCnts) < int(expected * min_ratio):
        raise ValueError(
            f"Trop peu de bulles détectées ({len(bubbleCnts)}). "
            f"Attendu ~{expected}. Ajuste seuils / photo plus nette."
        )

    # si trop de contours: garder les plus grands
    if len(bubbleCnts) > expected:
        bubbleCnts = sorted(bubbleCnts, key=cv2.contourArea, reverse=True)[:expected]

    # tri vertical
    bubbleCnts = contours.sort_contours(bubbleCnts, method="top-to-bottom")[0]

    # sécurité finale
    if len(bubbleCnts) != expected:
        raise ValueError(
            f"Nombre de bulles final incorrect ({len(bubbleCnts)}). "
            "Réessaie avec un scan plus propre ou ajuste min_size/ar."
        )

    return bubbleCnts


def evaluate_responses_single_sheet(
    bubbleCnts,
    thresh,
    min_fill=5,
    weak_fill=30,
    ambiguity_diff=15,
    impute_blank_to_N=True
):
    """
    Feuille unique : 30 lignes, 8 items/ligne, 5 choix/item => 1200 bulles.
    Ordre des choix: FD, D, N, A, FA (N = index 2).

    Règle: item vide => imputer N (index 2 => 2 points).
    """
    responses = {}
    warnings = []

    rows = 30
    items_per_row = 8
    choices_per_item = 5
    bubbles_per_row = items_per_row * choices_per_item  # 40

    blank_count = 0
    neutral_marked_count = 0
    neutral_imputed_count = 0
    ambiguity_count = 0

    for r in range(rows):
        row_cnts = bubbleCnts[r * bubbles_per_row:(r + 1) * bubbles_per_row]
        row_cnts = contours.sort_contours(row_cnts, method="left-to-right")[0]

        for c in range(items_per_row):
            item_cnts = row_cnts[c * choices_per_item:(c + 1) * choices_per_item]
            item_cnts = contours.sort_contours(item_cnts, method="left-to-right")[0]

            bubbled = None
            fills = []

            for j, cnt in enumerate(item_cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                masked = cv2.bitwise_and(thresh, thresh, mask=mask)

                total = cv2.countNonZero(masked)
                area = cv2.contourArea(cnt)
                fill_percent = (total / area * 100) if area > 0 else 0

                fills.append(fill_percent)
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)

            item_id = (r + 1) + 30 * c
            chosen_idx = int(bubbled[1])
            max_fill = float(fills[chosen_idx])

            # qualité
            if max_fill < weak_fill:
                warnings.append(f"Item {item_id}: Détection faible ({max_fill:.1f}%)")

            sorted_fills = sorted(fills, reverse=True)
            if len(sorted_fills) > 1 and (sorted_fills[0] - sorted_fills[1]) < ambiguity_diff:
                ambiguity_count += 1
                warnings.append(f"Item {item_id}: Ambiguïté détectée")

            # vide => imputation N
            if max_fill < min_fill:
                blank_count += 1
                if impute_blank_to_N:
                    neutral_imputed_count += 1
                    responses[item_id] = 2
                    warnings.append(f"Item {item_id}: Non répondu → imputé à N (2 points)")
                else:
                    responses[item_id] = chosen_idx
                    warnings.append(f"Item {item_id}: Non répondu")
            else:
                responses[item_id] = chosen_idx
                if chosen_idx == 2:
                    neutral_marked_count += 1

    stats = {
        "total_items": rows * items_per_row,
        "blank_count": blank_count,
        "neutral_marked_count": neutral_marked_count,
        "neutral_imputed_count": neutral_imputed_count,
        "neutral_total_count": neutral_marked_count + neutral_imputed_count,
        "ambiguity_count": ambiguity_count
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
    """
    Règles:
    - items vides >= 15 => invalide
    - N total (cochés + imputés) >= 42 => invalide
    """
    reasons = []
    if stats["blank_count"] >= blank_invalid_threshold:
        reasons.append(f"Trop d'items non répondus: {stats['blank_count']} (seuil {blank_invalid_threshold}).")
    if stats["neutral_total_count"] >= neutral_invalid_threshold:
        reasons.append(f"Trop de réponses 'N' (total): {stats['neutral_total_count']} (seuil {neutral_invalid_threshold}).")
    return (len(reasons) == 0), reasons


# =========================================================
# INTERFACE
# =========================================================
left, right = st.columns([0.75, 0.25], vertical_alignment="bottom")
with left:
    st.title("NEO PI-R Scanner")
    st.caption("Feuille unique (240 items) — FD / D / N / A / FA")
with right:
    st.write("")

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
    st.markdown("### Règles protocole")
    blank_invalid_threshold = st.number_input("Seuil items vides (invalide si ≥)", min_value=0, max_value=240, value=15, step=1)
    neutral_invalid_threshold = st.number_input("Seuil 'N' total (invalide si ≥)", min_value=0, max_value=240, value=42, step=1)
    impute_blank_to_N = st.toggle("Imputer les items vides à N (2 points)", value=True)

    st.markdown("---")
    st.markdown("### À propos")
    st.caption("NEO PI-R Scanner — v1.0 (Essai)")
    st.caption("© 2026 Yacine Adaoun — Tous droits réservés")

st.markdown("### Import")
colA, colB = st.columns([0.65, 0.35], vertical_alignment="top")

with colA:
    uploaded_file = st.file_uploader(
        "Déposer la feuille de réponses (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    st.caption("Conseil: feuille à plat, bonne lumière, photo nette, sans ombres.")

with colB:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("**Contrôles**")
    run = st.button("Lancer l’analyse", type="primary", disabled=(uploaded_file is None))
    st.markdown("Détection, scoring, validité, exports.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RUN
# =========================================================
if uploaded_file and run:
    try:
        original = Image.open(uploaded_file)

        paper, thresh = preprocess_image(original)

        bubbleCnts = detect_bubbles(
            thresh,
            min_size=min_bubble_size,
            ar_min=ar_min,
            ar_max=ar_max,
            expected=240 * 5
        )

        responses, warnings, stats = evaluate_responses_single_sheet(
            bubbleCnts,
            thresh,
            min_fill=min_fill,
            weak_fill=weak_fill,
            ambiguity_diff=ambiguity_diff,
            impute_blank_to_N=impute_blank_to_N
        )

        # Dessiner bulles choisies en vert
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

        # Scores
        facette_scores, domain_scores = calculate_scores(responses)

        # Validité
        is_valid, reasons = protocol_validity(
            stats,
            blank_invalid_threshold=int(blank_invalid_threshold),
            neutral_invalid_threshold=int(neutral_invalid_threshold)
        )

        # KPI
        total_items = stats["total_items"]
        warning_count = len(warnings)
        conf = 100 * (1 - warning_count / max(1, total_items))

        st.markdown("### Résultats")
        k1, k2, k3 = st.columns(3)
        k1.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">Confiance (heuristique)</div>
              <div class="kpi-value">{conf:.1f}%</div>
              <div class="kpi-sub">Basée sur avertissements / {total_items} items</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        k2.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">Items vides</div>
              <div class="kpi-value">{stats["blank_count"]}</div>
              <div class="kpi-sub">Imputés à N: {stats["neutral_imputed_count"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        k3.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">N total</div>
              <div class="kpi-value">{stats["neutral_total_count"]}</div>
              <div class="kpi-sub">N cochés: {stats["neutral_marked_count"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Validité du protocole")
        if is_valid:
            st.success("Protocole valide.")
        else:
            st.error("Protocole invalide.")
            for r in reasons:
                st.warning(r)

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Scores", "Images", "Avertissements", "Exports"])

        with tab1:
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
            st.subheader("Journal")
            if warnings:
                with st.expander("Afficher les avertissements", expanded=True):
                    for w in warnings:
                        st.warning(w)
            else:
                st.success("Aucun avertissement.")

            if debug:
                st.code(
                    "\n".join([
                        f"bubbles={len(bubbleCnts)}",
                        f"blank_count={stats['blank_count']}",
                        f"neutral_marked={stats['neutral_marked_count']}",
                        f"neutral_imputed={stats['neutral_imputed_count']}",
                        f"neutral_total={stats['neutral_total_count']}",
                        f"ambiguity_count={stats['ambiguity_count']}",
                        f"warnings={len(warnings)}",
                        f"conf={conf:.2f}"
                    ]),
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
            report_lines.append("VALIDITÉ")
            report_lines.append("Valide" if is_valid else "Invalide")
            for r in reasons:
                report_lines.append(f"- {r}")
            report_lines.append("")

            report_lines.append("STATISTIQUES")
            report_lines.append(f"Items vides: {stats['blank_count']}")
            report_lines.append(f"N cochés: {stats['neutral_marked_count']}")
            report_lines.append(f"N imputés: {stats['neutral_imputed_count']}")
            report_lines.append(f"N total: {stats['neutral_total_count']}")
            report_lines.append("")

            report_lines.append("SCORES PAR FACETTE")
            for row in data:
                report_lines.append(f"{row['Facette']}: {row['Score brut']}")
            report_lines.append("")

            report_lines.append("TOTAUX DOMAINES")
            for row in dom_data:
                report_lines.append(f"{row['Domaine']}: {row['Score']}")
            report_lines.append("")

            report_lines.append("AVERTISSEMENTS")
            if warnings:
                report_lines.extend(warnings)
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
