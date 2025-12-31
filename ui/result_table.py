# ui/result_table.py
"""
Komponen tabel hasil ranking.
"""

import streamlit as st
import pandas as pd
from typing import List


def render_result_table(results: pd.DataFrame, relevance: List[bool] = None):
    """
    Render tabel hasil ranking.
    
    Args:
        results: DataFrame dengan hasil ranking
        relevance: List boolean untuk menandai item relevan (optional)
    """
    if results is None or len(results) == 0:
        st.info("Tidak ada hasil untuk ditampilkan.")
        return
    
    st.markdown("### Tabel Ranking Hasil Pencarian")
    
    # Prepare display columns
    display_cols = [
        'rank', 'nama_tempat', 'store', 'rating_tempat',
        'alamat_tempat', 'nama_kelurahan', 'nama_kecamatan'
    ]
    
    # Add distance if available
    if 'distance_km' in results.columns and results['distance_km'].sum() > 0:
        display_cols.insert(4, 'distance_km')
    
    # Add final score
    if 'final_score' in results.columns:
        display_cols.append('final_score')
    
    # Filter columns that exist
    display_cols = [col for col in display_cols if col in results.columns]
    
    # Create display dataframe
    display_df = results[display_cols].copy()
    
    # Rename columns for display
    column_names = {
        'rank': 'Rank',
        'nama_tempat': 'Nama Tempat',
        'store': 'Toko',
        'rating_tempat': 'Rating',
        'distance_km': 'Jarak (km)',
        'alamat_tempat': 'Alamat',
        'nama_kelurahan': 'Kelurahan',
        'nama_kecamatan': 'Kecamatan',
        'final_score': 'Skor'
    }
    
    display_df = display_df.rename(columns=column_names)
    
    # Format numeric columns
    if 'Rating' in display_df.columns:
        display_df['Rating'] = display_df['Rating'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else "-"
        )
    
    if 'Jarak (km)' in display_df.columns:
        display_df['Jarak (km)'] = display_df['Jarak (km)'].apply(
            lambda x: f"{x:.2f}" if x > 0 else "-"
        )
    
    if 'Skor' in display_df.columns:
        display_df['Skor'] = display_df['Skor'].apply(lambda x: f"{x:.3f}")
    
    # Add relevance indicator if provided
    if relevance is not None and len(relevance) == len(display_df):
        display_df.insert(1, 'Rel', ['Ya' if r else 'Tidak' for r in relevance])
    
    # Shorten address if too long
    if 'Alamat' in display_df.columns:
        display_df['Alamat'] = display_df['Alamat'].apply(
            lambda x: (x[:60] + '...') if isinstance(x, str) and len(x) > 60 else x
        )
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(len(display_df) * 40 + 40, 500)
    )
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Hasil", len(results))
    
    with col2:
        if relevance:
            relevant_count = sum(1 for r in relevance if r)
            st.metric("Hasil Relevan", f"{relevant_count} ({relevant_count/len(relevance)*100:.0f}%)")
    
    with col3:
        if 'rating_tempat' in results.columns:
            avg_rating = results['rating_tempat'].mean()
            st.metric("Rata-rata Rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "-")


def render_detail_card(row: pd.Series, rank: int):
    """
    Render card detail untuk satu hasil.
    
    Args:
        row: Series data satu baris
        rank: Ranking position
    """
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {'#0066b2' if row.get('store') == 'Indomaret' else '#ed1c24'};
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin:0;">#{rank} {row.get('nama_tempat', 'N/A')}</h4>
            <span style="
                background: {'#0066b2' if row.get('store') == 'Indomaret' else '#ed1c24'};
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.8rem;
            ">{row.get('store', 'N/A')}</span>
        </div>
        <p style="color: #666; margin: 0.5rem 0;">{row.get('alamat_tempat', 'N/A')}</p>
        <div style="display: flex; gap: 1rem; font-size: 0.9rem; color: #888;">
            <span>Rating: {row.get('rating_tempat', 0):.1f}</span>
            <span>Ulasan: {row.get('user_ratings_total', 0)}</span>
            <span>{row.get('nama_kelurahan', 'N/A')}, {row.get('nama_kecamatan', 'N/A')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_detailed_results(results: pd.DataFrame, max_items: int = 5):
    """
    Render hasil dalam format card detail.
    
    Args:
        results: DataFrame hasil
        max_items: Jumlah maksimum item yang ditampilkan
    """
    st.markdown("### Detail Hasil Teratas")
    
    for idx, row in results.head(max_items).iterrows():
        rank = row.get('rank', idx + 1)
        render_detail_card(row, rank)
