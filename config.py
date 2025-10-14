from pathlib import Path

# Get the base directory (directory where this file resides)
BASE_DIR = Path(__file__).parent

# Paths to various resources
DATA_DIR = BASE_DIR / 'data'

PATH_TO_SQLITE = DATA_DIR / "sqlite/belgium.sqlite" # Path where the sqlite database is stored
L_SPACE_PATH = DATA_DIR / "pkl/belgium_routesCleaned.pkl"  # Path where the clean L-space graph was stored (cleaned)
P_SPACE_PATH = DATA_DIR / "pkl/belgium_P.pkl" # Path where the clean P-space graph was stored (cleaned)