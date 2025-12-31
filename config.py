# config.py
# Konfigurasi bobot dan parameter untuk sistem pencarian

# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Haversine distance parameters
MAX_DISTANCE_KM = 10  # Maximum distance in kilometers for relevance

# Ranking weight configuration
WEIGHTS = {
    'bm25_score': 0.4,
    'distance_score': 0.3,
    'rating_score': 0.2,
    'popularity_score': 0.1
}

# Evaluation parameters
DEFAULT_K = 10  # Default K for Precision@K and Recall@K

# Store types
STORE_TYPES = ['Alfamart', 'Indomaret']

# Data file path
DATA_FILE = 'data/Data_Alfamart Indomaret_South Jakarta.csv'

# Map configuration
MAP_CENTER = [-6.28, 106.80]  # Jakarta Selatan
MAP_ZOOM = 12
