# NEO PI-R — OMR Scanner & Scoring (Streamlit)

Application Streamlit qui :
- redresse automatiquement une photo/scan de feuille de réponses NEO PI-R
- détecte la grille 30×8 (240 items) et lit FD/D/N/A/FA
- applique les règles de validité du protocole (blancs / trop de N / imputation)
- calcule scores par facette (30) et par domaine (5)
- génère exports (CSV, TXT) + profil graphique (PNG/PDF)

## Installation locale
```bash
pip install -r requirements.txt
streamlit run neo_pir_scanner.py
