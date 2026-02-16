import streamlit as st
from collections import defaultdict
import pandas as pd
import io
import csv

st.set_page_config(page_title="NEO PI-R Questionnaire + Calculateur", layout="wide")
st.title("🧠 NEO PI-R – Questionnaire complet avec boutons radio")

# ====================== TA CLÉ DE SCORING (copiée exactement de ton code) ======================
clé_de_score = { ... }  # ← Colle ici tout ton dictionnaire clé_de_score (240 lignes) du message précédent

# ====================== FACETTES PAR LIGNE (30 lignes de la grille) ======================
facet_per_row = [
    'N1','E1','O1','A1','C1','N2','E2','O2','A2','C2',
    'N3','E3','O3','A3','C3','N4','E4','O4','A4','C4',
    'N5','E5','O5','A5','C5','N6','E6','O6','A6','C6'
]

facette_labels = {  # exactement les tiens
    'N1': 'N1 - Anxiété', 'N2': 'N2 - Hostilité colérique', 'N3': 'N3 - Dépression',
    'N4': 'N4 - Timidité', 'N5': 'N5 - Impulsivité', 'N6': 'N6 - Vulnérabilité',
    'E1': 'E1 - Chaleur', 'E2': 'E2 - Grégarité', 'E3': 'E3 - Affirmation de soi',
    'E4': 'E4 - Activité', 'E5': "E5 - Recherche d'excitation", 'E6': 'E6 - Émotions positives',
    'O1': 'O1 - Imagination', 'O2': 'O2 - Esthétique', 'O3': 'O3 - Sentiments',
    'O4': 'O4 - Actions', 'O5': 'O5 - Idées', 'O6': 'O6 - Valeurs',
    'A1': 'A1 - Confiance', 'A2': 'A2 - Franchise', 'A3': 'A3 - Altruisme',
    'A4': 'A4 - Conformité', 'A5': 'A5 - Modestie', 'A6': 'A6 - Tendresse',
    'C1': 'C1 - Compétence', 'C2': 'C2 - Ordre', 'C3': 'C3 - Sens du devoir',
    'C4': 'C4 - Effort pour réussir', 'C5': 'C5 - Autodiscipline', 'C6': 'C6 - Délibération'
}

domain_labels = {'N': 'Névrosisme', 'E': 'Extraversion', 'O': 'Ouverture', 'A': 'Agréabilité', 'C': 'Conscience'}

# ====================== OPTIONS ======================
options = ["Fortement en désaccord (FD)", "Désaccord (D)", "Neutre (N)", "Accord (A)", "Fortement d'accord (FA)"]
option_map = {opt: idx for idx, opt in enumerate(options)}

# ====================== INITIALISATION ======================
if 'responses' not in st.session_state:
    st.session_state.responses = [None] * 240

# ====================== QUESTIONNAIRE (8 blocs) ======================
for bloc in range(8):
    start = bloc * 30
    with st.expander(f"🔹 Bloc {bloc+1} – Items {start+1} à {start+30}"):
        for j in range(30):
            i = start + j
            question_text = f"**Item {i+1}** : [Remplace par la vraie question du livret NEO PI-R ici]"
            st.session_state.responses[i] = st.radio(
                question_text,
                options,
                index=None,
                horizontal=True,
                key=f"q{i}"
            )

# ====================== CALCUL ======================
if st.button("🚀 Calculer les scores", type="primary"):
    responses = st.session_state.responses
    
    # Compter N et vides
    n_count = sum(1 for r in responses if r == "Neutre (N)")
    empty_count = sum(1 for r in responses if r is None)
    
    # Protocole
    raisons = []
    if n_count >= 42:
        raisons.append(f"Trop de réponses Neutre : {n_count} (≥ 42)")
    if empty_count >= 15:
        raisons.append(f"Trop d'items non répondus : {empty_count} (≥ 15)")
    
    valide = len(raisons) == 0
    imputation_points = 2 * empty_count if empty_count < 10 else 0
    
    # Calcul scores
    facet_scores = defaultdict(int)
    for item in range(1, 241):
        idx = item - 1
        if responses[idx] is not None:
            opt_idx = option_map[responses[idx]]
            score = clé_de_score[item][opt_idx]
            row = (item - 1) % 30
            fac = facet_per_row[row]
            facet_scores[fac] += score
    
    # Domaines
    domain_scores = {'N':0, 'E':0, 'O':0, 'A':0, 'C':0}
    for fac, sc in facet_scores.items():
        dom = fac[0]  # N/E/O/A/C
        domain_scores[dom] += sc
    
    total_brut = sum(domain_scores.values())
    total_avec_imp = total_brut + imputation_points
    
    # ====================== AFFICHAGE ======================
    st.success("✅ Calcul terminé")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Statut", "VALIDE" if valide else "INVALIDE", "⚠️" if not valide else "✅")
    col2.metric("Items vides", empty_count)
    col3.metric("Réponses Neutre", n_count)
    col4.metric("Imputation", f"+{imputation_points} pts")
    
    if not valide:
        for r in raisons:
            st.error(r)
    
    st.subheader("Scores bruts par facette")
    df_fac = pd.DataFrame([
        {"Facette": facette_labels.get(f, f), "Score brut": s}
        for f, s in sorted(facet_scores.items())
    ])
    st.dataframe(df_fac, use_container_width=True, hide_index=True)
    
    st.subheader("Scores bruts par domaine")
    df_dom = pd.DataFrame([
        {"Domaine": domain_labels[d], "Score brut": s}
        for d, s in domain_scores.items()
    ])
    st.dataframe(df_dom, use_container_width=True, hide_index=True)
    
    st.markdown(f"**Total brut** : {total_brut} / 960")
    st.markdown(f"**Total avec imputation** : {total_avec_imp} / 960")
    
    # Export
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Facette", "Score brut"])
    for f, s in facet_scores.items():
        writer.writerow([facette_labels.get(f, f), s])
    writer.writerow([])
    for d, s in domain_scores.items():
        writer.writerow([domain_labels[d], s])
    st.download_button("📥 Télécharger CSV", csv_buffer.getvalue(), "neo_pir_scores.csv", "text/csv")

st.caption("NEO PI-R Questionnaire + Calculateur • Boutons radio • Protocole respecté • Clé 100 % fidèle à ta grille")
