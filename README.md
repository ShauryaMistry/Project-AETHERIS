Markdown
# Aetheris: Automated Exoplanet Detection & ML Vetting Pipeline

Aetheris is a Python-based transit-search and machine learning vetting pipeline designed to process high-cadence space telescope data (such as TESS), detect planetary transit candidates, and cross-match stellar parameters to classify signals while filtering out instrumental noise and stellar activity.

## Features
- **Data Ingestion:** Downloads and processes light curve products automatically.
- **Advanced Conditioning:** Employs defensive sigma-clipping to isolate structural transit signatures from massive stellar flares.
- **High-Resolution Transit Search:** Utilizes a custom high-density Box Least Squares (BLS) mesh grid optimized to detect short-to-medium period planets while bypassing low-frequency aliasing and noise walls.
- **Stellar Cross-Matching:** Resolves target identities and physical stellar characteristics via live SIMBAD metadata lookup, featuring automatic API case-normalization.
- **Machine Learning Vetting:** Runs candidates through an integrated ML vetting bundle (`aetheris_classifier.pkl`) to generate a definitive candidate vs. false positive classification verdict with anomaly detection scoring.
- **Diagnostic Dashboard:** Generates full-bleed, multi-panel visual data validation reports showing detrended light curves, BLS power spectrums, and folded transit models.

## Quick Start

### Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install numpy matplotlib lightkurve astroquery astropy scikit-learn
Running the Pipeline
Open the primary Jupyter notebook and execute the core blocks:

Cell 1: Loads the environmental layout overrides and imports pipeline modules.

Cell 2: Define your system target name (e.g., target_name = "TRAPPIST-1").

Cell 3: Runs the automated end-to-end cleaning, BLS processing, ML vetting loop, and displays the diagnostic dashboard inline.

Pipeline Architecture
main.py: Core application logic containing the classifier loaders and candidate vetting routines.

modules/downloader.py: Data fetching engine.

modules/bls.py: Transit search utilities.

modules/gaia.py: Stellar property query engine.
```
Validation & Null Result: While the pipeline effectively bypassed the high-frequency 0.8-day noise wall and the ML classifier flagged a weak candidate at 8.3987 days (60.86% confidence), visual inspection of the binned phase-fold confirms a flat transit profile. The BLS engine latched onto random red-noise fluctuations within the active M-dwarf environment, demonstrating the vital necessity of human-in-the-loop vetting for low-confidence ML outputs.

modules/plots.py: Custom dashboard canvas layout engine.

License
This project is licensed under the MIT License - see the LICENSE file for details.
