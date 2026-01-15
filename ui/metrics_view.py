# ui/metrics_view.py
"""
Komponen tampilan metrik evaluasi.
"""

import streamlit as st
from typing import Dict, List


def render_metric_card(title: str, value: str, description: str = None, color: str = "#667eea"):
    """
    Render single metric card.
    
    Args:
        title: Judul metric
        value: Nilai metric
        description: Deskripsi (optional)
        color: Warna accent
    """
    # Use bright white/light colors for dark mode visibility
    desc_html = f'<p style="font-size: 12px; color: #e0e0e0; margin: 0; font-weight: 500;">{description}</p>' if description else ''
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}40 0%, {color}25 100%);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        text-align: center;
        border: 1px solid {color}50;
    ">
        <p style="margin: 0; font-size: 13px; color: #ffffff; text-transform: uppercase; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{title}</p>
        <h2 style="margin: 0.5rem 0; color: {color}; font-weight: 700; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">{value}</h2>
        {desc_html}
    </div>
    """, unsafe_allow_html=True)


def render_metrics(metrics: Dict[str, float], k: int = 10):
    """
    Render evaluation metrics summary.
    
    Args:
        metrics: Dictionary dengan metrics (precision_k, recall_k, average_precision)
        k: Value of K used
    """
    st.markdown("### Evaluasi Kualitas Hasil Pencarian")
    
    # Info box with colors visible on dark mode
    st.markdown("""
    <div style="
        background: rgba(23, 162, 184, 0.2);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 13px;
        color: #7dd3e8;
        border: 1px solid #17a2b8;
    ">
        <b style="color: #7dd3e8;">Evaluasi berdasarkan relevansi heuristik</b> - 
        Menunjukkan kualitas hasil pencarian berdasarkan kecocokan teks, jarak, dan tipe toko.
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        precision_val = metrics.get('precision_k', 0)
        render_metric_card(
            f"Precision@{k}",
            f"{precision_val:.1%}",
            "Persentase prediksi positif yang benar",
            "#28a745"
        )
    
    with col2:
        recall_val = metrics.get('recall_k', 0)
        render_metric_card(
            f"Recall@{k}",
            f"{recall_val:.1%}",
            "Kemampuan menemukan semua kasus relevan",
            "#17a2b8"
        )
    
    with col3:
        ap_val = metrics.get('average_precision', 0)
        render_metric_card(
            "Average Precision",
            f"{ap_val:.1%}",
            "Area di bawah kurva Precision × Recall",
            "#6f42c1"
        )
    
    # Overall assessment
    avg_score = (precision_val + recall_val + ap_val) / 3
    
    if avg_score >= 0.7:
        assessment = ("Sangat Baik", "#28a745", 
                     "Hasil pencarian memiliki relevansi tinggi dengan query.")
    elif avg_score >= 0.5:
        assessment = ("Baik", "#ffc107", 
                     "Hasil pencarian cukup relevan dengan query.")
    elif avg_score >= 0.3:
        assessment = ("Cukup", "#fd7e14", 
                     "Hasil pencarian memiliki beberapa item yang relevan.")
    else:
        assessment = ("Perlu Perbaikan", "#dc3545", 
                     "Hasil pencarian kurang relevan. Coba query yang lebih spesifik.")
    
    st.markdown(f"""
    <div style="
        background: {assessment[1]}35;
        border: 2px solid {assessment[1]};
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        text-align: center;
    ">
        <span style="font-size: 1.2rem; color: #ffffff; font-weight: 600;">{assessment[0]}</span>
        <p style="margin: 0.5rem 0 0 0; font-size: 13px; color: #e0e0e0;">{assessment[2]}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_explanation():
    """Render penjelasan metrik."""
    # Add top margin/spacing before expander
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    with st.expander("Penjelasan Metrik Evaluasi", expanded=False):
        st.markdown("""
        ### Metrik Information Retrieval
        
        #### Precision@K
        Kemampuan model untuk mengidentifikasi **hanya objek yang relevan** 
        (persentase prediksi positif yang benar).
        
        $$Precision@K = \\frac{\\text{Jumlah hasil relevan di top-K}}{K}$$
        
        *"Dari K prediksi yang diberikan, berapa persen yang benar-benar relevan?"*
        
        ---
        
        #### Recall@K
        Kemampuan model untuk **menemukan semua kasus yang relevan** 
        (semua ground-truth). Ini adalah persentase prediksi positif yang benar 
        di antara semua ground-truth yang diberikan.
        
        $$Recall@K = \\frac{\\text{Jumlah hasil relevan di top-K}}{\\text{Total ground-truth relevan}}$$
        
        *"Dari semua lokasi yang seharusnya ditemukan, berapa persen yang berhasil diidentifikasi?"*
        
        ---
        
        #### Average Precision (AP)
        Metrik berdasarkan **area di bawah kurva Precision × Recall** yang telah 
        diproses untuk menghilangkan perilaku zig-zag.
        
        $$AP = \\frac{1}{|R|} \\sum_{k=1}^{n} P(k) \\times rel(k)$$
        
        *"Seberapa baik sistem menempatkan hasil relevan di posisi atas dalam ranking?"*
        
        ---
        
        ### Ground Truth (Relevansi)
        
        Karena dataset tidak memiliki label relevansi bawaan, sistem menggunakan 
        **rule-based relevance labeling**:
        
        #### Pencarian dengan Query Teks:
        - **Relevan jika:**
            - Skor BM25 > median (ada kecocokan teks yang signifikan)
            - Jarak < 10 km (jika lokasi diaktifkan)
            - Tipe toko sesuai query (jika disebutkan)
        
        #### Pencarian Tanpa Query (Filter/Lokasi saja):
        - **Relevan jika memenuhi minimal 2 dari 3 kriteria:**
            - Berada di top 25% berdasarkan skor akhir
            - Rating tempat >= 4.0
            - Popularitas (jumlah review) di atas median
        - **Kriteria tambahan jika lokasi aktif:**
            - Jarak <= 5 km dari lokasi referensi
        
        *Pendekatan ini sah secara akademis sebagai **pseudo-relevance judgment**.*
        """)


def render_metrics_bar(metrics: Dict[str, float], k: int = 10):
    """
    Render metrics sebagai progress bars.
    
    Args:
        metrics: Dictionary metrics
        k: Value of K
    """
    st.markdown(f"### Skor Evaluasi (K={k})")
    
    # Precision
    precision_val = metrics.get('precision_k', 0)
    st.markdown(f"**Precision@{k}**")
    st.progress(precision_val)
    st.caption(f"{precision_val:.1%} - Persentase prediksi positif yang benar")
    
    # Recall
    recall_val = metrics.get('recall_k', 0)
    st.markdown(f"**Recall@{k}**")
    st.progress(recall_val)
    st.caption(f"{recall_val:.1%} - Kemampuan menemukan semua kasus relevan")
    
    # Average Precision
    ap_val = metrics.get('average_precision', 0)
    st.markdown("**Average Precision**")
    st.progress(ap_val)
    st.caption(f"{ap_val:.1%} - Area di bawah kurva Precision × Recall")


def render_component_evaluation(bm25_metrics: Dict, distance_metrics: Dict, 
                                 weights: Dict = None, has_location: bool = False):
    """
    Render evaluasi komponen BM25 dan Haversine secara terpisah.
    
    Args:
        bm25_metrics: Dictionary metrik BM25 dari ComponentEvaluator
        distance_metrics: Dictionary metrik Distance dari ComponentEvaluator
        weights: Dictionary bobot (bm25_score, distance_score, etc)
        has_location: Boolean apakah lokasi user tersedia
    """
    if weights is None:
        weights = {'bm25_score': 0.4, 'distance_score': 0.3, 'rating_score': 0.2, 'popularity_score': 0.1}
    
    # Add spacing
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    
    with st.expander("Evaluasi Komponen Ranking (BM25 & Haversine)", expanded=False):
        st.markdown("""
        <div style="
            background: rgba(111, 66, 193, 0.15);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 13px;
            color: #c4a7e7;
            border: 1px solid #6f42c1;
        ">
            <b>Analisis kontribusi masing-masing komponen</b> - 
            Menunjukkan performa BM25 (text relevance) dan Haversine (distance) secara terpisah.
        </div>
        """, unsafe_allow_html=True)
        
        # Two columns for BM25 and Distance
        col1, col2 = st.columns(2)
        
        # BM25 Metrics
        with col1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #28a74540 0%, #28a74525 100%);
                border-left: 4px solid #28a745;
                padding: 1rem;
                border-radius: 0 10px 10px 0;
                margin-bottom: 1rem;
                border: 1px solid #28a74550;
            ">
                <h4 style="margin: 0 0 0.75rem 0; color: #ffffff;">BM25 (Text Relevance)</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Term Match Rate
            term_match_rate = bm25_metrics.get('term_match_rate', 0)
            st.metric(
                "Term Match Rate",
                f"{term_match_rate:.0%}",
                help="Persentase kata kunci query yang ditemukan di corpus"
            )
            
            # BM25 Score Stats
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric(
                    "Avg BM25 Score",
                    f"{bm25_metrics.get('avg_score', 0):.2f}",
                    help="Rata-rata skor BM25 hasil pencarian"
                )
            with col1b:
                st.metric(
                    "Max BM25 Score",
                    f"{bm25_metrics.get('max_score', 0):.2f}",
                    help="Skor BM25 tertinggi"
                )
            
            # Nonzero Results
            nonzero_count = bm25_metrics.get('nonzero_count', 0)
            nonzero_rate = bm25_metrics.get('nonzero_rate', 0)
            st.metric(
                "Hasil dengan Kecocokan Teks",
                f"{nonzero_count} ({nonzero_rate:.0%})",
                help="Jumlah hasil yang memiliki skor BM25 > 0"
            )
            
            # BM25 Precision
            bm25_precision = bm25_metrics.get('bm25_precision', 0)
            st.progress(bm25_precision)
            st.caption(f"BM25 Precision@K: {bm25_precision:.1%}")
        
        # Distance Metrics
        with col2:
            if has_location:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #17a2b840 0%, #17a2b825 100%);
                    border-left: 4px solid #17a2b8;
                    padding: 1rem;
                    border-radius: 0 10px 10px 0;
                    margin-bottom: 1rem;
                    border: 1px solid #17a2b850;
                ">
                    <h4 style="margin: 0 0 0.75rem 0; color: #ffffff;">Haversine (Distance)</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Average Distance
                avg_dist = distance_metrics.get('avg_distance', 0)
                st.metric(
                    "Rata-rata Jarak",
                    f"{avg_dist:.2f} km",
                    help="Rata-rata jarak hasil dari lokasi Anda"
                )
                
                # Distance Range
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric(
                        "Jarak Terdekat",
                        f"{distance_metrics.get('min_distance', 0):.2f} km"
                    )
                with col2b:
                    st.metric(
                        "Jarak Terjauh",
                        f"{distance_metrics.get('max_distance', 0):.2f} km"
                    )
                
                # Coverage
                within_5km = distance_metrics.get('within_5km', 0)
                total = distance_metrics.get('within_10km', 0) or 1
                st.metric(
                    "Hasil dalam 5 km",
                    f"{within_5km}",
                    help="Jumlah hasil dalam radius 5 km"
                )
                
                # Distance Precision
                dist_precision = distance_metrics.get('distance_precision', 0)
                st.progress(dist_precision)
                st.caption(f"Distance Precision (≤5km): {dist_precision:.1%}")
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #6c757d40 0%, #6c757d25 100%);
                    border-left: 4px solid #6c757d;
                    padding: 1rem;
                    border-radius: 0 10px 10px 0;
                    margin-bottom: 1rem;
                    border: 1px solid #6c757d50;
                ">
                    <h4 style="margin: 0 0 0.75rem 0; color: #adb5bd;">Haversine (Distance)</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("Aktifkan lokasi referensi untuk melihat evaluasi jarak.")
        
        # Weight Contribution Visualization
        st.markdown("---")
        st.markdown("#### Bobot Kontribusi Komponen")
        
        weight_data = {
            'Komponen': ['BM25 (Text)', 'Distance', 'Rating', 'Popularity'],
            'Bobot': [
                weights.get('bm25_score', 0.4),
                weights.get('distance_score', 0.3),
                weights.get('rating_score', 0.2),
                weights.get('popularity_score', 0.1)
            ]
        }
        
        # Create progress bars for weights
        for i, (komponen, bobot) in enumerate(zip(weight_data['Komponen'], weight_data['Bobot'])):
            colors = ['#28a745', '#17a2b8', '#ffc107', '#6f42c1']
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="width: 120px; color: #e0e0e0; font-size: 13px;">{komponen}</span>
                <div style="flex: 1; background: #333; border-radius: 4px; height: 20px; margin: 0 10px;">
                    <div style="width: {bobot*100}%; background: {colors[i]}; height: 100%; border-radius: 4px;"></div>
                </div>
                <span style="width: 50px; color: #e0e0e0; font-size: 13px; text-align: right;">{bobot:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
