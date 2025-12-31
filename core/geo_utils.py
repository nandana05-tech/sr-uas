# core/geo_utils.py
"""
Utilitas geolokasi termasuk kalkulasi jarak Haversine.
"""

import math
from typing import Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAX_DISTANCE_KM

# Radius bumi dalam kilometer
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Menghitung jarak antara dua titik koordinat menggunakan formula Haversine.
    
    Formula Haversine memberikan jarak great-circle antara dua titik
    pada permukaan bola (bumi) berdasarkan latitude dan longitude.
    
    Args:
        lat1: Latitude titik pertama (dalam derajat)
        lon1: Longitude titik pertama (dalam derajat)
        lat2: Latitude titik kedua (dalam derajat)
        lon2: Longitude titik kedua (dalam derajat)
        
    Returns:
        Jarak dalam kilometer
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    # Calculate distance
    distance = EARTH_RADIUS_KM * c
    
    return distance


def calculate_distance_score(distance: float, max_distance: float = None) -> float:
    """
    Mengkonversi jarak menjadi skor (0-1).
    Semakin dekat, semakin tinggi skornya.
    
    Args:
        distance: Jarak dalam kilometer
        max_distance: Jarak maksimum untuk normalisasi
        
    Returns:
        Score antara 0-1
    """
    if max_distance is None:
        max_distance = MAX_DISTANCE_KM
    
    if distance >= max_distance:
        return 0.0
    
    # Linear decay: score = 1 - (distance / max_distance)
    score = 1.0 - (distance / max_distance)
    
    return max(0.0, min(1.0, score))


def calculate_distance_score_exponential(distance: float, decay_rate: float = 0.5) -> float:
    """
    Menghitung skor jarak dengan exponential decay.
    Memberikan penalti lebih besar untuk jarak yang jauh.
    
    Args:
        distance: Jarak dalam kilometer
        decay_rate: Rate of decay (semakin besar, semakin cepat turun)
        
    Returns:
        Score antara 0-1
    """
    return math.exp(-decay_rate * distance)


def get_bounding_box(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    """
    Mendapatkan bounding box untuk filtering awal berdasarkan jarak.
    
    Args:
        lat: Latitude titik pusat
        lon: Longitude titik pusat
        radius_km: Radius dalam kilometer
        
    Returns:
        Tuple (min_lat, max_lat, min_lon, max_lon)
    """
    # Approximate degrees per km at equator
    lat_diff = radius_km / 111.0  # ~111 km per degree latitude
    lon_diff = radius_km / (111.0 * math.cos(math.radians(lat)))  # varies by latitude
    
    min_lat = lat - lat_diff
    max_lat = lat + lat_diff
    min_lon = lon - lon_diff
    max_lon = lon + lon_diff
    
    return (min_lat, max_lat, min_lon, max_lon)


def is_within_bounding_box(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    """
    Memeriksa apakah koordinat berada dalam bounding box.
    
    Args:
        lat: Latitude
        lon: Longitude
        bbox: Tuple (min_lat, max_lat, min_lon, max_lon)
        
    Returns:
        True jika dalam bounding box
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)


def format_distance(distance_km: float) -> str:
    """
    Format jarak untuk display.
    
    Args:
        distance_km: Jarak dalam kilometer
        
    Returns:
        Formatted string
    """
    if distance_km < 1:
        return f"{distance_km * 1000:.0f} m"
    else:
        return f"{distance_km:.2f} km"
