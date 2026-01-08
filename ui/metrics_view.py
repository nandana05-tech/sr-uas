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
        
        - **Relevan jika:**
            - Skor BM25 > median (ada kecocokan teks yang signifikan)
            - Jarak < 10 km (jika lokasi diaktifkan)
            - Tipe toko sesuai query (jika disebutkan)
        
        - **Tidak relevan jika:**
            - Skor BM25 di bawah threshold
            - Jarak terlalu jauh dari lokasi referensi
            - Tipe toko tidak sesuai yang dicari
        
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
