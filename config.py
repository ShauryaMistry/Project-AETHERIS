"""
==========================================================
AETHERIS
Configuration File

All project settings are centralized here.
==========================================================
"""

from pathlib import Path

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Aetheris"
VERSION = "1.1.0"
AUTHOR = "Shaurya"

DEFAULT_TARGET = "TOI-6958"

# ==========================================================
# DIRECTORIES
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"

CACHE_DIR = DATA_DIR / "cache"
TESS_DIR = DATA_DIR / "tess"
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_DIR = DATA_DIR / "temp"

OUTPUT_DIR = ROOT_DIR / "outputs"

CSV_DIR = OUTPUT_DIR / "csv"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

for folder in [
    CACHE_DIR, TESS_DIR, PROCESSED_DIR, TEMP_DIR,
    OUTPUT_DIR, CSV_DIR, FIGURE_DIR, REPORT_DIR, LOG_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================================
# DOWNLOAD SETTINGS
# ==========================================================

MISSION = "TESS"
AUTHOR_NAME = "SPOC"
QUALITY_BITMASK = "default"
DOWNLOAD_ALL_SECTORS = False
USE_DOWNLOAD_CACHE = True
OVERWRITE_DOWNLOAD = False

# ==========================================================
# PREPROCESSING
# ==========================================================

REMOVE_NANS = True
REMOVE_OUTLIERS = True
OUTLIER_SIGMA = 5
NORMALIZE = True
FLATTEN = True
FLATTEN_WINDOW = 401
SAVE_PROCESSED = True
USE_PROCESSED_CACHE = True

# ==========================================================
# BLS SETTINGS
# ==========================================================

MIN_PERIOD = 0.5
MAX_PERIOD = 30.0
N_PERIODS = 10000
MIN_DURATION = 0.05
MAX_DURATION = 0.30
N_DURATIONS = 10
OVERSAMPLE = 10

# ==========================================================
# VETTING
# ==========================================================

MIN_DEPTH = 100e-6      # 100 ppm
MIN_SNR = 7
MIN_TRANSITS = 2

# ==========================================================
# DIAGNOSTICS
# ==========================================================

# Odd/even transit depth difference above this many sigma
# is flagged as a likely eclipsing binary, not a planet.
ODD_EVEN_SIGMA_THRESHOLD = 3.0

# ==========================================================
# STELLAR DEFAULTS
# Used when Gaia astrophysical parameters are unavailable.
# ==========================================================

DEFAULT_ST_RADIUS = 1.0     # solar radii
DEFAULT_ST_TEFF = 5500.0    # kelvin
DEFAULT_ST_LOGG = 4.4       # cgs

# ==========================================================
# MACHINE LEARNING
# ==========================================================

ML_MODEL_PATH = ROOT_DIR / "aetheris_classifier.pkl"

USE_ML_CLASSIFIER = True

# ==========================================================
# PLOTTING
# ==========================================================

SAVE_PLOTS = True
FIG_WIDTH = 12
FIG_HEIGHT = 5
FIG_DPI = 300
FIG_FORMAT = "png"

# ==========================================================
# OUTPUT
# ==========================================================

SAVE_CANDIDATE_JSON = True
SAVE_CSV = True

# ==========================================================
# REPORT
# ==========================================================

SAVE_REPORT = True
REPORT_TITLE = "Aetheris Exoplanet Detection Report"

# ==========================================================
# GUI (Future)
# ==========================================================

THEME = "dark"
ACCENT_COLOR = "#00C8FF"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL = "INFO"
LOG_FILE = LOG_DIR / "aetheris.log"
