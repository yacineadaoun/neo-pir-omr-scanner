import streamlit as st
from collections import defaultdict
import pandas as pd
import io
import csv

st.set_page_config(page_title="NEO PI-R – Feuille de réponses numérique", layout="wide")

# ====================== TA CLÉ DE SCORING (colle ici ton dictionnaire complet) ======================
clé_de_score = {
    # ←←← COLLE ICI TOUT TON DICTIONNAIRE clé_de_score du message précédent (les 240 lignes)
    # Je l’ai laissé vide pour ne pas alourdir, mais tu le colles tel quel.
}

# ====================== FACETTES (exactement l’ordre de ta feuille) ======================
facet_per_row = [
    'N1','E1','O1','A1','C1','N2','E2','O2','A2','C2',
    'N3','E3','O3','A3','C3','N4','E4','O4','A4','C4',
    'N5','E5','O5','A5','C5','N6','E6','O6','A6','C6'
]

facette_labels = {  # tes labels
    'N1':'N1 - Anxiété', 'N2':'N2 - Hostilité colérique', 'N3':'N3 - Dépression',
    'N4':'N4 - Timidité', 'N5':'N5 - Impulsivité', 'N6':'N6 - Vulnérabilité',
    'E1':'E1 - Chaleur', 'E2':'E2 - Grégarité', 'E3':'E3 - Affirmation de soi',
    'E4':'E4 - Activité', 'E5':"E5 - Recherche d'excitation", 'E6':'E6 - Émotions positives',
    'O1':'O1 - Imagination', 'O2':'O2 - Esthétique', 'O3':'O3 - Sentiments',
    'O4':'O4 - Actions', 'O5':'O5 - Idées', 'O6':'O6 - Valeurs',
    'A1':'A1 - Confiance', 'A2':'A2 - Franchise', 'A3':'A3 - Altruisme',
    'A4':'A4 - Conformité', 'A5':'A5 - Modestie', 'A6':'A6 - Tendresse',
    'C1':'C1 - Compétence', 'C2':'C2 - Ordre', 'C3':'C3 - Sens du devoir',
    'C4':'C4 - Effort pour réussir', 'C5':'C5 - Autodiscipline', 'C6':'C6 - Délibération'
}

domain_labels = {'N':'Névrosisme', 'E':'Extraversion', 'O':'Ouverture', 'A':'Agréabilité', 'C':'Conscience'}

options = ["FD", "D", "N", "A", "FA"]

# ====================== SESSION ======================
if 'responses' not in st.session_state:
    st.session_state.responses = {i: None for i in range(1, 241)}

# ====================== EN-TÊTE (comme ta feuille) ======================
st.markdown("""
<div style="background:#0c4a6e; color:white; padding:15px; border-radius:8px; text-align:center; font-size:28px; font-weight:bold;">
    NEO PI-R™ – Feuille de réponses numérique
</div>
""", unsafe_allow_html=True)

st.markdown("**FD** = Fortement en désaccord **D** = Désaccord **N** = Neutre **A** = Accord **FA** = Fortement d’accord")

# ====================== GRILLE (exactement comme l’image) ======================
for bloc in range(6):
    with st.expander(f"📋 Bloc {bloc+1} – Lignes {(bloc*5)+1} à {(bloc+1)*5}", expanded=(bloc==0)):
        for r in range(bloc*5, (bloc+1)*5):
            cols = st.columns([0.8] + [1.8]*8 + [1.2])   # 0.8 = numéro ligne | 8 colonnes items | 1.2 = facette
            
            # Numéro de ligne
            with cols[0]:
                st.markdown(f"**{r+1}**")
            
            # 8 items de la ligne
            for c in range(8):
                item = (r + 1) + (c * 30)
                with cols[c+1]:
                    st.session_state.responses[item] = st.radio(
                        label="",
                        options=options,
                        horizontal=True,
                        key=f"item_{item}",
                        label_visibility="collapsed"
                    )
                    st.caption(f"**{item}**")
            
            # Étiquette facette (comme sur ta feuille)
            with cols[9]:
                st.markdown(f"**={facet_per_row[r]}**")

# ====================== CALCUL ======================
if st.button("🚀 CALCULER LES SCORES", type="primary", use_container_width=True):
    responses = st.session_state.responses
    
    # Comptage N et vides
    n_count = sum(1 for v in responses.values() if v == "N")
    empty_count = sum(1 for v in responses.values() if v is None)
    
    # Protocole
    raisons = []
    if n_count >= 42: raisons.append(f"Trop de 'N' : {n_count} (≥ 42)")
    if empty_count >= 15: raisons.append(f"Trop de cases vides : {empty_count} (≥ 15)")
    valide = len(raisons) == 0
    imputation_points = 2 * empty_count if empty_count < 10 else 0
    
    # Scores
    facet_scores = [0] * 30
    for r in range(30):
        for c in range(8):
            item = (r + 1) + (c * 30)
            choix = responses[item]
            if choix is not None:
                idx = options.index(choix)
                score = clé_de_score[item][idx]
                facet_scores[r] += score
    
    # Domaines
    domain_scores = {'N':0, 'E':0, 'O':0, 'A':0, 'C':0}
    for i, fac in enumerate(facet_per_row):
        dom = fac[0]
        domain_scores[dom] += facet_scores[i]
    
    total_brut = sum(domain_scores.values())
    total_avec_imp = total_brut + imputation_points
    
    # ====================== RÉSULTATS ======================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Statut", "✅ VALIDE" if valide else "❌ INVALIDE")
    col2.metric("Cases vides", empty_count)
    col3.metric("Réponses N", n_count)
    col4.metric("Imputation", f"+{imputation_points} pts")
    
    if not valide:
        for r in raisons:
            st.error("• " + r)
    
    st.subheader("Scores bruts par facette")
    df_fac = pd.DataFrame([
        {"Facette": facette_labels[f], "Score brut": facet_scores[i]}
        for i, f in enumerate(facet_per_row)
    ])
    st.dataframe(df_fac, use_container_width=True, hide_index=True)
    
    st.subheader("Scores bruts par domaine")
    df_dom = pd.DataFrame([
        {"Domaine": domain_labels[d], "Score brut": s} for d, s in domain_scores.items()
    ])
    st.dataframe(df_dom, use_container_width=True, hide_index=True)
    
    st.markdown(f"**Total brut** : {total_brut} / 960")
    st.markdown(f"**Total avec imputation** : {total_avec_imp} / 960")
    
    # Export
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Facette", "Score brut"])
    for i, f in enumerate(facet_per_row):
        writer.writerow([facette_labels[f], facet_scores[i]])
    writer.writerow([])
    for d, s in domain_scores.items():
        writer.writerow([domain_labels[d], s])
    st.download_button("📥 Télécharger CSV", csv_buffer.getvalue(), "neo_pir_feuille_reponses.csv", "text/csv")

st.caption("Interface 100 % fidèle à ta feuille papier • Calcul scientifique avec ta clé • Protocole respecté")
