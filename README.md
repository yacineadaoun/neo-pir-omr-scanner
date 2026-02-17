# NEO PI-R OMR Scanner (Streamlit)

## Features
- Upload photo/scan
- Auto rotation (0/90/180/270)
- Document warp (perspective)
- Robust table bbox detection (grid-lines + fallback)
- Flexible grid (real line detection) + per-cell micro-alignment
- Mark detection (FD/D/N/A/FA), ambiguity + multi-mark
- Protocol validity rules + imputation
- Raw scores (facets + domains)
- Optional T-scores + percentiles via `norms.csv` (no proprietary data embedded)
- Charts: bars + radar
- Exports: CSV + TXT + PDF (professional)

## Deploy (Streamlit Cloud)
- requirements.txt (opencv-python-headless)
- packages.txt (libgl1, libglib2.0-0)
- runtime.txt (python-3.11)

## Norms
Provide your own `norms.csv` using the template. The app will compute:
T = 50 + 10*(raw-mean)/sd
Percentile via normal CDF approximation.
