# core/ranking.py
"""
Final heuristic ranking yang menggabungkan berbagai faktor.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WEIGHTS, MAX_DISTANCE_KM
from .preprocessing import preprocess_text, preprocess_query, create_document_field
from .bm25_engine import BM25Engine
from .geo_utils import haversine_distance, calculate_distance_score


class HeuristicRanker:
    """
    Ranker yang menggabungkan beberapa faktor:
    - BM25 text similarity
    - Distance score (Haversine)
    - Rating score
    - Popularity score (user_ratings_total)
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Inisialisasi ranker.
        
        Args:
            weights: Dictionary bobot untuk setiap faktor
        """
        self.weights = weights if weights is not None else WEIGHTS
        self.bm25 = BM25Engine()
        self.df = None
        self.corpus = []
    
    def fit(self, df: pd.DataFrame):
        """
        Fit ranker dengan dataframe.
        
        Args:
            df: DataFrame dengan data lokasi minimarket
        """
        self.df = df.copy()
        
        # Create document field dan preprocess
        self.corpus = []
        for idx, row in df.iterrows():
            doc_text = create_document_field(row)
            tokens = preprocess_text(doc_text)
            self.corpus.append(tokens)
        
        # Fit BM25
        self.bm25.fit(self.corpus)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalisasi scores ke range 0-1.
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def rank(self, query: str, user_lat: float = None, user_lon: float = None,
             store_filter: str = None, top_k: int = 50, require_text_match: bool = True) -> pd.DataFrame:
        """
        Melakukan ranking berdasarkan query dan lokasi user.
        
        Args:
            query: Query pencarian
            user_lat: Latitude user (optional)
            user_lon: Longitude user (optional)
            store_filter: Filter tipe store (Alfamart/Indomaret)
            top_k: Jumlah hasil maksimum
            require_text_match: Jika True, hanya tampilkan hasil dengan BM25 > 0
            
        Returns:
            DataFrame hasil ranking dengan skor
        """
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()
        
        # Preprocess query
        query_tokens = preprocess_query(query)
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_all_scores(query_tokens)
        
        # Create result dataframe
        result_df = self.df.copy()
        result_df['bm25_score'] = bm25_scores
        
        # Normalize BM25 scores
        result_df['bm25_score_norm'] = self._normalize_scores(bm25_scores)
        
        # Calculate distance scores if user location provided
        if user_lat is not None and user_lon is not None:
            distances = []
            for idx, row in result_df.iterrows():
                dist = haversine_distance(
                    user_lat, user_lon,
                    row['latitude'], row['longitude']
                )
                distances.append(dist)
            
            result_df['distance_km'] = distances
            result_df['distance_score'] = [
                calculate_distance_score(d) for d in distances
            ]
        else:
            result_df['distance_km'] = 0
            result_df['distance_score'] = 0.5  # neutral score
        
        # Rating score (normalize 0-5 to 0-1)
        result_df['rating_score'] = result_df['rating_tempat'].fillna(0) / 5.0
        
        # Popularity score (normalize by max)
        max_ratings = result_df['user_ratings_total'].max()
        if max_ratings > 0:
            result_df['popularity_score'] = result_df['user_ratings_total'].fillna(0) / max_ratings
        else:
            result_df['popularity_score'] = 0
        
        # Calculate final score
        result_df['final_score'] = (
            self.weights['bm25_score'] * result_df['bm25_score_norm'] +
            self.weights['distance_score'] * result_df['distance_score'] +
            self.weights['rating_score'] * result_df['rating_score'] +
            self.weights['popularity_score'] * result_df['popularity_score']
        )
        
        # Apply store filter
        if store_filter and store_filter != 'Semua':
            result_df = result_df[result_df['store'] == store_filter]
        
        # Filter by BM25 > 0 only if text match is required
        if require_text_match:
            result_df = result_df[result_df['bm25_score'] > 0]
        
        # Sort by final score
        result_df = result_df.sort_values('final_score', ascending=False)
        
        # Get top k
        result_df = result_df.head(top_k).reset_index(drop=True)
        
        # Add rank column
        result_df['rank'] = range(1, len(result_df) + 1)
        
        return result_df
    
    def determine_relevance(self, result_df: pd.DataFrame, query: str,
                           user_lat: float = None, user_lon: float = None,
                           max_distance: float = None) -> List[bool]:
        """
        Menentukan ground truth relevance berdasarkan heuristic rules.
        
        Rules for relevance:
        1. BM25 score > threshold (textual relevance) - only if query provided
        2. Distance < max_distance (jika lokasi tersedia)
        3. Store type match (jika query menyebutkan)
        
        Args:
            result_df: DataFrame hasil ranking
            query: Query original
            user_lat: User latitude
            user_lon: User longitude
            max_distance: Maximum distance untuk relevan
            
        Returns:
            List of boolean indicating relevance
        """
        if max_distance is None:
            max_distance = MAX_DISTANCE_KM
        
        query_lower = query.lower() if query else ""
        has_query = bool(query and query.strip())
        relevance = []
        
        # Determine if query mentions specific store
        mentions_alfamart = 'alfamart' in query_lower or 'alfa' in query_lower
        mentions_indomaret = 'indomaret' in query_lower or 'indo' in query_lower
        
        # Calculate BM25 threshold (median) for more meaningful evaluation
        bm25_scores = result_df['bm25_score'].tolist()
        if bm25_scores and has_query:
            bm25_threshold = np.median(bm25_scores)
        else:
            bm25_threshold = 0
        
        for idx, row in result_df.iterrows():
            is_relevant = True
            
            # Rule 1: BM25 score must be above median (only if query provided)
            if has_query and row.get('bm25_score', 0) <= bm25_threshold:
                is_relevant = False
            
            # Rule 2: Distance check (if available)
            if user_lat is not None and user_lon is not None:
                if row.get('distance_km', float('inf')) > max_distance:
                    is_relevant = False
            
            # Rule 3: Store type match
            store_type = str(row.get('store', '')).lower()
            if mentions_alfamart and 'alfamart' not in store_type:
                is_relevant = False
            if mentions_indomaret and 'indomaret' not in store_type:
                is_relevant = False
            
            relevance.append(is_relevant)
        
        return relevance
