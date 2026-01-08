# ui/layout.py
"""
Layout utama aplikasi tanpa sidebar.
"""

import streamlit as st
import pandas as pd
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STORE_TYPES, MAP_CENTER


def render_header():
    """Render header aplikasi."""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2rem;
        }
        .main-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
    </style>
    <div class="main-header">
        <h1>Sistem Pencarian Lokasi Minimarket</h1>
        <p>Pencarian dengan BM25 Ranking dan Evaluasi Information Retrieval</p>
    </div>
    """, unsafe_allow_html=True)


def render_search_input() -> Tuple[str, str, Optional[float], Optional[float]]:
    """
    Render input pencarian.
    
    Returns:
        Tuple (query, store_filter, user_lat, user_lon)
    """
    st.markdown("### Cari Lokasi Minimarket")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Masukkan query pencarian",
            placeholder="Contoh: Indomaret Cilandak, Alfamart dekat stasiun, dll.",
            help="Ketik nama tempat, alamat, kelurahan, atau kecamatan"
        )
    
    with col2:
        store_options = ['Semua'] + STORE_TYPES
        store_filter = st.selectbox(
            "Filter Toko",
            options=store_options,
            help="Filter berdasarkan jenis minimarket"
        )
    
    # Location input
    use_location = st.checkbox("Gunakan lokasi referensi untuk sorting berdasarkan jarak", value=False)
    
    user_lat = None
    user_lon = None
    
    if use_location:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            user_lat = st.number_input(
                "Latitude",
                value=MAP_CENTER[0],
                min_value=-90.0,
                max_value=90.0,
                format="%.6f",
                help="Latitude lokasi Anda (contoh: -6.2088)"
            )
        with col_lon:
            user_lon = st.number_input(
                "Longitude",
                value=MAP_CENTER[1],
                min_value=-180.0,
                max_value=180.0,
                format="%.6f",
                help="Longitude lokasi Anda (contoh: 106.8456)"
            )
    
    return query, store_filter, user_lat, user_lon


def render_main_layout():
    """
    Render layout utama aplikasi.
    Setting page config dan styling.
    """
    # Page config sudah di app.py
    
    # Custom CSS with improved text contrast
    st.markdown("""
    <style>
        /* Improve metric label visibility */
        .stMetric {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        
        .stMetric label {
            color: #1a1a2e !important;
            font-weight: 600 !important;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: #2d3436 !important;
            font-weight: 700 !important;
        }
        
        .stMetric [data-testid="stMetricLabel"] {
            color: #2d3436 !important;
            font-weight: 600 !important;
        }
        
        .result-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        
        .info-box {
            background-color: #e8f4f8;
            border-left: 4px solid #17a2b8;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
            color: #333;
        }
        
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Render header
    render_header()


def render_no_results():
    """Render message ketika tidak ada hasil."""
    st.markdown("""
    <div class="info-box">
        <h4>Tidak ada hasil ditemukan</h4>
        <p>Coba gunakan kata kunci yang berbeda atau kurangi filter.</p>
        <ul>
            <li>Gunakan nama kelurahan atau kecamatan</li>
            <li>Coba "Indomaret" atau "Alfamart"</li>
            <li>Gunakan nama jalan atau landmark terdekat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def render_instructions():
    """Render petunjuk penggunaan."""
    with st.expander("Petunjuk Penggunaan", expanded=False):
        st.markdown("""
        ### Cara Menggunakan Sistem
        
        1. **Masukkan Query**: Ketik nama tempat, alamat, atau area yang ingin dicari
        2. **Filter Toko**: Pilih jenis minimarket (Alfamart/Indomaret) atau semua
        3. **Lokasi Referensi**: Aktifkan untuk sorting berdasarkan jarak dari posisi Anda
        4. **Lihat Hasil**: 
           - Evaluation Summary menunjukkan kualitas hasil pencarian
           - Peta interaktif menampilkan lokasi
           - Tabel ranking menampilkan detail setiap lokasi
        
        ### Tentang Evaluasi
        
        - **Precision@K**: Seberapa tepat hasil (dari K hasil, berapa yang relevan)
        - **Recall@K**: Seberapa lengkap hasil (dari semua yang relevan, berapa yang ditemukan)
        - **Average Precision (AP)**: Kualitas keseluruhan ranking
        
        *Evaluasi menggunakan pseudo-relevance judgment berdasarkan:*
        - *Kecocokan teks dengan query*
        - *Jarak dari lokasi referensi (jika diaktifkan)*
        - *Kesesuaian tipe toko*
        """)
