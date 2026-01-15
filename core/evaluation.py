# core/evaluation.py
"""
Modul evaluasi untuk menghitung Precision, Recall, dan MAP.
"""

from typing import List, Dict
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_K


class Evaluator:
    """
    Evaluator untuk mengukur kualitas hasil pencarian.
    
    Metrics:
    - Precision@K
    - Recall@K
    - Average Precision (AP)
    - Mean Average Precision (MAP)
    """
    
    def __init__(self, k: int = None):
        """
        Inisialisasi evaluator.
        
        Args:
            k: Default K untuk Precision@K dan Recall@K
        """
        self.k = k if k is not None else DEFAULT_K
    
    def precision_at_k(self, relevance: List[bool], k: int = None) -> float:
        """
        Menghitung Precision@K.
        
        Precision@K = |Relevant items in top-K| / K
        
        Kemampuan model untuk mengidentifikasi hanya objek yang relevan
        (persentase prediksi positif yang benar).
        
        Args:
            relevance: List of boolean (True = relevant)
            k: Number of items to consider
            
        Returns:
            Precision score (0-1)
        """
        if k is None:
            k = self.k
        
        if k <= 0 or len(relevance) == 0:
            return 0.0
        
        # Take only top-k items
        top_k_relevance = relevance[:k]
        
        # Count relevant items
        relevant_count = sum(1 for r in top_k_relevance if r)
        
        return relevant_count / k
    
    def recall_at_k(self, relevance: List[bool], k: int = None) -> float:
        """
        Menghitung Recall@K.
        
        Recall@K = |Relevant items in top-K| / |Total relevant items|
        
        Kemampuan model untuk menemukan semua kasus yang relevan
        (semua ground-truth). Ini adalah persentase prediksi positif yang benar
        di antara semua ground-truth yang diberikan.
        
        Args:
            relevance: List of boolean for ALL items (True = relevant)
            k: Number of items to consider
            
        Returns:
            Recall score (0-1)
        """
        if k is None:
            k = self.k
        
        if k <= 0 or len(relevance) == 0:
            return 0.0
        
        # Count total relevant items
        total_relevant = sum(1 for r in relevance if r)
        
        if total_relevant == 0:
            return 0.0
        
        # Count relevant in top-k
        top_k_relevance = relevance[:k]
        relevant_in_top_k = sum(1 for r in top_k_relevance if r)
        
        return relevant_in_top_k / total_relevant
    
    def average_precision(self, relevance: List[bool]) -> float:
        """
        Menghitung Average Precision (AP).
        
        AP = (1/|R|) * Σ(P@k * rel(k))
        
        Metrik berdasarkan area di bawah kurva Precision × Recall
        yang telah diproses untuk menghilangkan perilaku zig-zag.
        
        Args:
            relevance: List of boolean (True = relevant)
            
        Returns:
            Average Precision score (0-1)
        """
        if not relevance:
            return 0.0
        
        # Count total relevant
        total_relevant = sum(1 for r in relevance if r)
        
        if total_relevant == 0:
            return 0.0
        
        ap_sum = 0.0
        relevant_count = 0
        
        for i, is_relevant in enumerate(relevance):
            if is_relevant:
                relevant_count += 1
                # Precision at position i+1
                precision_at_i = relevant_count / (i + 1)
                ap_sum += precision_at_i
        
        return ap_sum / total_relevant
    
    def mean_average_precision(self, all_relevances: List[List[bool]]) -> float:
        """
        Menghitung Mean Average Precision (MAP).
        
        MAP = (1/|Q|) * Σ AP(q)
        
        Rata-rata AP dari banyak query.
        
        Args:
            all_relevances: List of relevance lists (satu per query)
            
        Returns:
            MAP score (0-1)
        """
        if not all_relevances:
            return 0.0
        
        ap_scores = [self.average_precision(rel) for rel in all_relevances]
        
        return sum(ap_scores) / len(ap_scores)
    
    def evaluate(self, relevance: List[bool], k: int = None) -> Dict[str, float]:
        """
        Menghitung semua metrics sekaligus.
        
        Args:
            relevance: List of boolean (True = relevant)
            k: K untuk Precision@K dan Recall@K
            
        Returns:
            Dictionary dengan semua metrics
        """
        if k is None:
            k = self.k
        
        return {
            'precision_k': self.precision_at_k(relevance, k),
            'recall_k': self.recall_at_k(relevance, k),
            'average_precision': self.average_precision(relevance),
            'k': k
        }
    
    def get_formatted_results(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Format metrics untuk display.
        
        Args:
            metrics: Dictionary dengan metrics
            
        Returns:
            Dictionary dengan formatted strings
        """
        k = metrics.get('k', self.k)
        
        return {
            f'Precision@{k}': f"{metrics['precision_k']:.2%}",
            f'Recall@{k}': f"{metrics['recall_k']:.2%}",
            'Average Precision': f"{metrics['average_precision']:.2%}"
        }
    
    def explain_metrics(self) -> Dict[str, str]:
        """
        Penjelasan untuk setiap metric.
        
        Returns:
            Dictionary dengan penjelasan
        """
        return {
            'Precision@K': (
                "Kemampuan model untuk mengidentifikasi hanya objek yang relevan "
                "(persentase prediksi positif yang benar)."
            ),
            'Recall@K': (
                "Kemampuan model untuk menemukan semua kasus yang relevan "
                "(persentase ground-truth yang berhasil ditemukan di top-K)."
            ),
            'Average Precision': (
                "Metrik berdasarkan area di bawah kurva Precision × Recall "
                "yang telah diproses untuk menghilangkan perilaku zig-zag."
            ),
            'MAP': (
                "Mean Average Precision: "
                "Rata-rata AP dari banyak query, menunjukkan performa sistem secara umum."
            )
        }


class ComponentEvaluator:
    """
    Evaluator untuk mengevaluasi komponen BM25 dan Haversine secara terpisah.
    
    Membantu menganalisis kontribusi masing-masing komponen terhadap kualitas ranking.
    """
    
    def __init__(self, k: int = None):
        """
        Inisialisasi ComponentEvaluator.
        
        Args:
            k: Default K untuk evaluasi
        """
        self.k = k if k is not None else DEFAULT_K
        self.evaluator = Evaluator(k=self.k)
    
    def evaluate_bm25(self, bm25_scores: List[float], relevance: List[bool], 
                      query_match_stats: dict = None) -> Dict[str, float]:
        """
        Evaluasi performa komponen BM25.
        
        Args:
            bm25_scores: List skor BM25 untuk setiap hasil
            relevance: List boolean relevansi
            query_match_stats: Statistik kecocokan query (dari BM25Engine.get_query_match_stats)
            
        Returns:
            Dictionary dengan metrik BM25
        """
        if not bm25_scores:
            return {
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'score_std': 0.0,
                'nonzero_count': 0,
                'nonzero_rate': 0.0,
                'term_match_rate': 0.0,
                'bm25_precision': 0.0,
                'bm25_recall': 0.0
            }
        
        scores = np.array(bm25_scores)
        nonzero_scores = scores[scores > 0]
        
        # Calculate BM25-only relevance (items with BM25 > median are considered relevant)
        median_score = np.median(scores[scores > 0]) if len(nonzero_scores) > 0 else 0
        bm25_relevance = [s > median_score for s in bm25_scores]
        
        # Calculate precision/recall for BM25-based ranking
        k = min(self.k, len(bm25_scores))
        bm25_precision = sum(1 for i in range(k) if bm25_relevance[i]) / k if k > 0 else 0
        
        total_bm25_relevant = sum(1 for r in bm25_relevance if r)
        bm25_recall = sum(1 for i in range(k) if bm25_relevance[i]) / total_bm25_relevant if total_bm25_relevant > 0 else 0
        
        result = {
            'avg_score': round(float(np.mean(scores)), 4),
            'max_score': round(float(np.max(scores)), 4),
            'min_score': round(float(np.min(scores)), 4),
            'score_std': round(float(np.std(scores)), 4),
            'nonzero_count': int(len(nonzero_scores)),
            'nonzero_rate': round(len(nonzero_scores) / len(scores), 4) if scores.size > 0 else 0.0,
            'bm25_precision': round(bm25_precision, 4),
            'bm25_recall': round(bm25_recall, 4)
        }
        
        # Add query match stats if provided
        if query_match_stats:
            result['term_match_rate'] = query_match_stats.get('term_match_rate', 0.0)
            result['matched_terms'] = query_match_stats.get('matched_terms', 0)
            result['query_terms'] = query_match_stats.get('query_terms', 0)
            result['avg_idf'] = query_match_stats.get('avg_idf', 0.0)
        
        return result
    
    def evaluate_distance(self, distances: List[float], relevance: List[bool],
                          max_relevant_distance: float = 5.0) -> Dict[str, float]:
        """
        Evaluasi performa komponen Haversine (distance).
        
        Args:
            distances: List jarak dalam km
            relevance: List boolean relevansi
            max_relevant_distance: Jarak maksimum untuk dianggap relevan
            
        Returns:
            Dictionary dengan metrik distance
        """
        if not distances:
            return {
                'avg_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'median_distance': 0.0,
                'within_1km': 0,
                'within_5km': 0,
                'within_10km': 0,
                'distance_precision': 0.0,
                'spatial_efficiency': 0.0
            }
        
        from .geo_utils import get_distance_statistics
        
        # Get distance statistics
        dist_stats = get_distance_statistics(distances)
        
        # Calculate distance-based precision (within max_relevant_distance)
        k = min(self.k, len(distances))
        valid_distances = [d for d in distances[:k] if d is not None and d >= 0]
        
        if valid_distances:
            distance_precision = sum(1 for d in valid_distances if d <= max_relevant_distance) / len(valid_distances)
        else:
            distance_precision = 0.0
        
        return {
            'avg_distance': dist_stats.get('avg_distance', 0.0),
            'min_distance': dist_stats.get('min_distance', 0.0),
            'max_distance': dist_stats.get('max_distance', 0.0),
            'median_distance': dist_stats.get('median_distance', 0.0),
            'within_1km': dist_stats.get('coverage', {}).get('within_1km', 0),
            'within_5km': dist_stats.get('coverage', {}).get('within_5km', 0),
            'within_10km': dist_stats.get('coverage', {}).get('within_10km', 0),
            'distance_precision': round(distance_precision, 4),
            'spatial_efficiency': dist_stats.get('spatial_efficiency', 0.0)
        }
    
    def evaluate_components(self, result_df, query_tokens: List[str] = None,
                           bm25_engine = None) -> Dict[str, dict]:
        """
        Evaluasi lengkap kedua komponen.
        
        Args:
            result_df: DataFrame hasil ranking
            query_tokens: List token query
            bm25_engine: Instance BM25Engine untuk mendapatkan statistik
            
        Returns:
            Dictionary dengan evaluasi BM25 dan Distance
        """
        bm25_scores = result_df.get('bm25_score', []).tolist() if 'bm25_score' in result_df.columns else []
        distances = result_df.get('distance_km', []).tolist() if 'distance_km' in result_df.columns else []
        relevance = [True] * len(result_df)  # Default all relevant if not specified
        
        # Get query match stats if bm25_engine provided
        query_match_stats = None
        if bm25_engine and query_tokens:
            query_match_stats = bm25_engine.get_query_match_stats(query_tokens)
        
        bm25_eval = self.evaluate_bm25(bm25_scores, relevance, query_match_stats)
        distance_eval = self.evaluate_distance(distances, relevance)
        
        return {
            'bm25': bm25_eval,
            'distance': distance_eval
        }
