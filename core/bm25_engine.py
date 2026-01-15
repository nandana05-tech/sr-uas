# core/bm25_engine.py

"""
BM25 (Best Matching 25) - Algoritma ranking
Digunakan untuk mencari dokumen yang paling relevan
terhadap query (kata kunci) dari user.
"""

import math                      # Untuk fungsi matematika (log)
from collections import Counter  # Untuk menghitung frekuensi kata
import sys                       # Untuk manipulasi path sistem
import os                        # Untuk operasi path file

# Menambahkan root project ke sys.path agar bisa import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import parameter BM25 dari file config
from config import BM25_K1, BM25_B


class BM25Engine:
    """
    Kelas BM25Engine mengimplementasikan algoritma BM25
    untuk menghitung tingkat relevansi dokumen terhadap query.
    """
    
    def __init__(self, k1=None, b=None):
        # Parameter k1: mengontrol pengaruh frekuensi kata (TF)
        self.k1 = k1 if k1 is not None else BM25_K1
        
        # Parameter b: mengontrol normalisasi panjang dokumen
        self.b = b if b is not None else BM25_B
        
        # Variabel yang akan diisi saat proses fit()
        self.corpus = []           # Menyimpan semua dokumen
        self.doc_lengths = []      # Panjang masing-masing dokumen
        self.avgdl = 0             # Rata-rata panjang dokumen
        self.N = 0                 # Jumlah total dokumen
        self.doc_freqs = {}        # Document Frequency tiap kata
        self.idf = {}              # Nilai IDF tiap kata
        self.doc_term_freqs = []   # Term Frequency per dokumen
    
    def fit(self, corpus):
        """
        Mempelajari corpus (dataset dokumen).
        corpus: list dokumen, setiap dokumen berupa list kata.
        Contoh:
        [
            ["cipete", "jagakarsa", "cilandak"],
            ["cipete", "ragunan"]
        ]
        """
        self.corpus = corpus            # Simpan corpus
        self.N = len(corpus)            # Hitung jumlah dokumen
        
        # Jika corpus kosong, hentikan proses
        if self.N == 0:
            return
        
        # Hitung panjang setiap dokumen
        self.doc_lengths = [len(doc) for doc in corpus]
        
        # Hitung rata-rata panjang dokumen
        self.avgdl = sum(self.doc_lengths) / self.N
        
        # Reset data frekuensi
        self.doc_freqs = {}
        self.doc_term_freqs = []
        
        # Loop setiap dokumen
        for doc in corpus:
            # Hitung frekuensi kata dalam satu dokumen
            term_freq = Counter(doc)
            self.doc_term_freqs.append(term_freq)
            
            # Hitung berapa dokumen yang mengandung kata tertentu
            for term in set(doc):  # set() agar dihitung sekali per dokumen
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1 # ← df (document frequency) dihitung 
        
        # Hitung nilai IDF (inverse document frequency) untuk setiap kata
        self._calculate_idf()
    
    def _calculate_idf(self):
        """
        Menghitung IDF (Inverse Document Frequency)
        - Kata jarang → IDF tinggi (lebih penting)
        - Kata sering → IDF rendah (kurang penting)
        """
        self.idf = {}
        
        # Loop setiap kata dan document frequency-nya
        for term, df in self.doc_freqs.items():
            # Rumus IDF versi BM25
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query, doc_idx):
        """
        Menghitung skor BM25 untuk satu dokumen tertentu.
        query   : list kata kunci
        doc_idx : index dokumen
        """
        # Jika index dokumen tidak valid
        if doc_idx >= self.N:
            return 0.0
        
        score = 0.0                               # Inisialisasi skor
        doc_len = self.doc_lengths[doc_idx]      # Panjang dokumen
        term_freqs = self.doc_term_freqs[doc_idx]# TF (pembobotan pada kata sering muncul) dokumen
        
        # Loop setiap kata pada query
        for term in query:
            # Jika kata tidak pernah muncul di corpus
            if term not in self.idf:
                continue
            
            # Ambil frekuensi kata di dokumen
            tf = term_freqs.get(term, 0)
            
            # Jika kata tidak ada di dokumen
            if tf == 0:
                continue
            
            idf = self.idf[term]  # Ambil nilai IDF (Inverse Document Frequency)
            
            # Bagian atas rumus BM25
            numerator = tf * (self.k1 + 1)
            
            # Bagian bawah rumus BM25 (normalisasi panjang dokumen)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            # Tambahkan skor kata ke total skor dokumen
            score += idf * numerator / denominator
        
        return score
    
    def search(self, query, top_k=10):
        """
        Mencari dokumen paling relevan dengan query.
        Return: list tuple (index_dokumen, skor)
        """
        scores = []
        
        # Hitung skor untuk setiap dokumen
        for idx in range(self.N):
            s = self.score(query, idx)
            if s > 0:
                scores.append((idx, s))
        
        # Urutkan berdasarkan skor tertinggi
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil top_k hasil terbaik
        return scores[:top_k]
    
    def get_all_scores(self, query):
        """
        Mengembalikan skor BM25 untuk semua dokumen.
        """
        return [self.score(query, idx) for idx in range(self.N)]
    
    def get_statistics(self) -> dict:
        """
        Mengembalikan statistik corpus untuk evaluasi.
        
        Returns:
            Dictionary dengan statistik corpus
        """
        return {
            'num_documents': self.N,
            'avg_doc_length': self.avgdl,
            'vocabulary_size': len(self.doc_freqs),
            'min_doc_length': min(self.doc_lengths) if self.doc_lengths else 0,
            'max_doc_length': max(self.doc_lengths) if self.doc_lengths else 0,
            'k1': self.k1,
            'b': self.b
        }
    
    def get_query_match_stats(self, query: list) -> dict:
        """
        Mengembalikan statistik kecocokan query dengan corpus.
        
        Args:
            query: List of query tokens
            
        Returns:
            Dictionary dengan statistik kecocokan
        """
        if not query:
            return {
                'query_terms': 0,
                'matched_terms': 0,
                'term_match_rate': 0.0,
                'matched_term_list': [],
                'unmatched_term_list': []
            }
        
        matched_terms = [t for t in query if t in self.doc_freqs]
        unmatched_terms = [t for t in query if t not in self.doc_freqs]
        
        # Calculate average IDF for matched terms
        avg_idf = 0.0
        if matched_terms:
            avg_idf = sum(self.idf.get(t, 0) for t in matched_terms) / len(matched_terms)
        
        return {
            'query_terms': len(query),
            'matched_terms': len(matched_terms),
            'term_match_rate': len(matched_terms) / len(query) if query else 0.0,
            'matched_term_list': matched_terms,
            'unmatched_term_list': unmatched_terms,
            'avg_idf': avg_idf
        }
