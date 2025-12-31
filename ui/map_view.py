# ui/map_view.py
"""
Komponen visualisasi peta menggunakan Folium.
"""

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAP_CENTER, MAP_ZOOM


def get_store_color(store_type: str) -> str:
    """
    Mendapatkan warna marker berdasarkan tipe toko.
    
    Args:
        store_type: Tipe toko (Alfamart/Indomaret)
        
    Returns:
        Warna untuk marker
    """
    if isinstance(store_type, str):
        if 'indomaret' in store_type.lower():
            return 'blue'
        elif 'alfamart' in store_type.lower():
            return 'red'
    return 'gray'


def get_store_icon(store_type: str) -> str:
    """
    Mendapatkan icon berdasarkan tipe toko.
    
    Args:
        store_type: Tipe toko
        
    Returns:
        Icon name
    """
    return 'shopping-cart'


def create_popup_html(row: pd.Series) -> str:
    """
    Membuat HTML untuk popup marker.
    
    Args:
        row: Data satu lokasi
        
    Returns:
        HTML string
    """
    store_color = '#0066b2' if row.get('store', '').lower() == 'indomaret' else '#ed1c24'
    
    html = f"""
    <div style="width: 250px; font-family: Arial, sans-serif;">
        <h4 style="margin: 0 0 8px 0; color: {store_color};">
            {row.get('nama_tempat', 'N/A')}
        </h4>
        <p style="font-size: 12px; color: #666; margin: 0 0 8px 0;">
            {row.get('alamat_tempat', 'N/A')}
        </p>
        <div style="display: flex; gap: 10px; font-size: 11px; color: #888;">
            <span>Rating: {row.get('rating_tempat', 0):.1f}</span>
            <span>Ulasan: {row.get('user_ratings_total', 0)}</span>
        </div>
        <div style="font-size: 11px; color: #888; margin-top: 5px;">
            {row.get('nama_kelurahan', '')}, {row.get('nama_kecamatan', '')}
        </div>
    </div>
    """
    return html


def render_map(
    results: pd.DataFrame,
    user_location: Tuple[float, float] = None,
    relevance: List[bool] = None,
    height: int = 500
):
    """
    Render peta interaktif dengan hasil pencarian.
    
    Args:
        results: DataFrame dengan hasil
        user_location: Tuple (lat, lon) lokasi user
        relevance: List boolean untuk marking
        height: Height of the map in pixels
    """
    if results is None or len(results) == 0:
        st.info("Tidak ada lokasi untuk ditampilkan di peta.")
        return
    
    st.markdown("### Peta Lokasi Minimarket")
    
    # Calculate map center
    if user_location:
        center = list(user_location)
    else:
        center = [
            results['latitude'].mean(),
            results['longitude'].mean()
        ]
    
    # Create map
    m = folium.Map(
        location=center,
        zoom_start=MAP_ZOOM if not user_location else 14,
        tiles='OpenStreetMap'
    )
    
    # Add user location marker if available
    if user_location:
        folium.Marker(
            location=user_location,
            popup="Lokasi Anda",
            icon=folium.Icon(color='green', icon='user', prefix='fa'),
            tooltip="Lokasi Referensi"
        ).add_to(m)
    
    # Use MarkerCluster for better performance
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for results
    for idx, row in results.head(50).iterrows():  # Limit to 50 for performance
        # Get position from results dataframe
        pos = list(results.index).index(idx) if idx in results.index else 0
        
        # Determine marker properties
        color = get_store_color(row.get('store', ''))
        
        # Check relevance for opacity
        if relevance and pos < len(relevance):
            opacity = 1.0 if relevance[pos] else 0.5
        else:
            opacity = 1.0
        
        # Create popup
        popup_html = create_popup_html(row)
        popup = folium.Popup(popup_html, max_width=300)
        
        # Rank for tooltip
        rank = row.get('rank', pos + 1)
        tooltip = f"#{rank} {row.get('nama_tempat', 'N/A')}"
        
        # Create marker
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            tooltip=tooltip,
            icon=folium.Icon(
                color=color,
                icon=get_store_icon(row.get('store', '')),
                prefix='fa'
            )
        ).add_to(marker_cluster)
    
    # Add legend with better contrast
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background-color: white;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        font-size: 14px;
        font-family: Arial, sans-serif;
        color: #333;
        line-height: 1.6;
    ">
        <b style="color: #1a1a2e; font-size: 14px;">Legenda:</b><br>
        <span style="color: #0066b2; font-weight: 600;">●</span> <span style="color: #333;">Indomaret</span><br>
        <span style="color: #ed1c24; font-weight: 600;">●</span> <span style="color: #333;">Alfamart</span><br>
        <span style="color: #28a745; font-weight: 600;">●</span> <span style="color: #333;">Lokasi Anda</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map
    st_folium(m, width='100%', height=height, returned_objects=[])
    
    # Map stats
    col1, col2 = st.columns(2)
    with col1:
        indomaret_count = len(results[results['store'] == 'Indomaret'])
        st.metric("Indomaret", indomaret_count)
    with col2:
        alfamart_count = len(results[results['store'] == 'Alfamart'])
        st.metric("Alfamart", alfamart_count)
