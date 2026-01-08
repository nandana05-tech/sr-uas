# core/evaluation.py
"""
Modul evaluasi untuk menghitung Precision, Recall, dan MAP.
"""

from typing import List, Dict
import sys
import os

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
