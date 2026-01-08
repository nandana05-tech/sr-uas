# app.py
"""
Entry point untuk aplikasi Streamlit
Sistem Pencarian Lokasi Minimarket dengan Evaluasi MAP, Precision, dan Recall
"""

import streamlit as st
import pandas as pd
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Sistem Pencarian Minimarket",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config import DATA_FILE, DEFAULT_K
from core.ranking import HeuristicRanker
from core.evaluation import Evaluator
from ui.layout import render_main_layout, render_search_input, render_no_results, render_instructions
from ui.result_table import render_result_table
from ui.map_view import render_map
from ui.metrics_view import render_metrics, render_metrics_explanation


@st.cache_data
def load_data():
    """Load dan cache data minimarket."""
    try:
        # Load from data folder
        data_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return df
        elif os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            return df
        
        st.error(f"File data tidak ditemukan. Pastikan file CSV ada di lokasi: {DATA_FILE}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None



@st.cache_resource
def initialize_ranker(data_hash):
    """Initialize dan cache ranker."""
    ranker = HeuristicRanker()
    return ranker


def main():
    """Main application function."""
    
    # Render main layout
    render_main_layout()
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Gagal memuat data. Silakan periksa file CSV.")
        return
    
    # Show data info
    with st.expander("Info Dataset", expanded=False):
        st.write(f"**Total lokasi:** {len(df)}")
        st.write(f"**Kolom:** {', '.join(df.columns.tolist())}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indomaret", len(df[df['store'] == 'Indomaret']))
        with col2:
            st.metric("Alfamart", len(df[df['store'] == 'Alfamart']))
    
    # Render instructions
    render_instructions()
    
    # Get search input
    query, store_filter, user_lat, user_lon = render_search_input()
    
    # Search button
    search_clicked = st.button("Cari Lokasi", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Perform search if query provided
    if search_clicked and query:
        with st.spinner("Mencari lokasi minimarket..."):
            # Initialize ranker
            ranker = HeuristicRanker()
            ranker.fit(df)
            
            # Perform ranking
            results = ranker.rank(
                query=query,
                user_lat=user_lat,
                user_lon=user_lon,
                store_filter=store_filter,
                top_k=50
            )
        
        if len(results) == 0:
            render_no_results()
        else:
            # Determine relevance
            relevance = ranker.determine_relevance(
                results, query, user_lat, user_lon
            )
            
            # Calculate evaluation metrics
            evaluator = Evaluator(k=DEFAULT_K)
            metrics = evaluator.evaluate(relevance, k=min(DEFAULT_K, len(results)))
            
            # LAYOUT: Evaluation -> Map -> Table
            
            # 1. Evaluation Summary
            render_metrics(metrics, k=min(DEFAULT_K, len(results)))
            render_metrics_explanation()
            
            st.markdown("---")
            
            # 2. Interactive Map
            user_location = (user_lat, user_lon) if user_lat and user_lon else None
            render_map(results, user_location=user_location, relevance=relevance)
            
            st.markdown("---")
            
            # 3. Result Table
            render_result_table(results, relevance=relevance)
            
            # Download results
            st.markdown("---")
            st.markdown("### Download Hasil")
            
            # Prepare download data
            download_cols = [
                'rank', 'nama_tempat', 'store', 'rating_tempat',
                'user_ratings_total', 'alamat_tempat', 'nama_kelurahan',
                'nama_kecamatan', 'latitude', 'longitude'
            ]
            
            if 'distance_km' in results.columns:
                download_cols.insert(5, 'distance_km')
            
            download_df = results[[c for c in download_cols if c in results.columns]]
            
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"hasil_pencarian_{query.replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    elif search_clicked and not query:
        st.warning("Silakan masukkan query pencarian terlebih dahulu.")
    
    else:
        # Show sample data
        st.markdown("### Contoh Data")
        st.info("Masukkan query dan klik 'Cari Lokasi' untuk memulai pencarian.")
        
        sample_df = df.head(10)[['nama_tempat', 'store', 'rating_tempat', 'nama_kelurahan', 'nama_kecamatan']]
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        
        # Example queries
        st.markdown("### Contoh Query")
        example_queries = [
            "Indomaret Cilandak",
            "Alfamart dekat Jagakarsa",
            "minimarket Kebayoran",
            "Indomaret Fatmawati",
            "Alfamart Pasar Minggu"
        ]
        
        cols = st.columns(len(example_queries))
        for i, example in enumerate(example_queries):
            with cols[i]:
                st.code(example, language=None)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px; padding: 1rem;">
        <p>Sistem Pencarian Lokasi Minimarket dengan Evaluasi IR</p>
        <p>Menggunakan BM25 Ranking + Haversine Distance + Precision/Recall/AP</p>
        <p>Data: Alfamart & Indomaret - Jakarta Selatan</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
