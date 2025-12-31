# core/bm25_engine.py
"""
BM25 scoring engine untuk information retrieval.
"""

import math
from collections import Counter
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BM25_K1, BM25_B


class BM25Engine:
    """
    Implementasi algoritma BM25 (Best Matching 25).
    
    BM25 adalah fungsi ranking yang digunakan oleh search engine
    untuk mengestimasi relevansi dokumen terhadap query.
    """
    
    def __init__(self, k1: float = None, b: float = None):
        """
        Inisialisasi BM25 Engine.
        
        Args:
            k1: Parameter k1 (default dari config)
            b: Parameter b (default dari config)
        """
        self.k1 = k1 if k1 is not None else BM25_K1
        self.b = b if b is not None else BM25_B
        
        self.corpus = []
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = {}  # term -> jumlah dokumen yang mengandung term
        self.idf = {}  # term -> idf score
        self.doc_term_freqs = []  # list of Counter untuk setiap dokumen
        self.N = 0  # jumlah dokumen
    
    def fit(self, corpus: List[List[str]]):
        """
        Fit/index corpus dokumen.
        
        Args:
            corpus: List of tokenized documents (list of list of strings)
        """
        self.corpus = corpus
        self.N = len(corpus)
        
        if self.N == 0:
            return
        
        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.N
        
        # Calculate document frequencies
        self.doc_freqs = {}
        self.doc_term_freqs = []
        
        for doc in corpus:
            term_freq = Counter(doc)
            self.doc_term_freqs.append(term_freq)
            
            # Count unique terms per document for df
            for term in set(doc):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # Calculate IDF for each term
        self._calculate_idf()
    
    def _calculate_idf(self):
        """
        Menghitung IDF (Inverse Document Frequency) untuk setiap term.
        
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
        """
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # BM25 IDF formula
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = idf
    
    def score(self, query: List[str], doc_idx: int) -> float:
        """
        Menghitung BM25 score untuk satu dokumen terhadap query.
        
        Args:
            query: List of query tokens
            doc_idx: Index dokumen dalam corpus
            
        Returns:
            BM25 score
        """
        if doc_idx >= self.N:
            return 0.0
        
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        for term in query:
            if term not in self.idf:
                continue
            
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            
            idf = self.idf[term]
            
            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Mencari dokumen yang paling relevan dengan query.
        
        Args:
            query: List of query tokens
            top_k: Jumlah hasil yang dikembalikan
            
        Returns:
            List of (doc_idx, score) tuples, sorted by score descending
        """
        scores = []
        
        for idx in range(self.N):
            score = self.score(query, idx)
            if score > 0:
                scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def get_all_scores(self, query: List[str]) -> List[float]:
        """
        Mendapatkan BM25 scores untuk semua dokumen.
        
        Args:
            query: List of query tokens
            
        Returns:
            List of scores untuk semua dokumen
        """
        return [self.score(query, idx) for idx in range(self.N)]
