# core/preprocessing.py
"""
Text preprocessing dan normalization untuk sistem pencarian.
"""

import re
import string

# Stopwords bahasa Indonesia
STOPWORDS_ID = {
    'yang', 'di', 'dan', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'sebagai', 'dalam', 'tidak', 'akan', 'juga', 'atau',
    'ada', 'mereka', 'sudah', 'saya', 'seperti', 'dapat', 'jika', 'hanya',
    'oleh', 'saat', 'harus', 'antara', 'setelah', 'belum', 'atas', 'bawah',
    'rt', 'rw', 'no', 'jl', 'jalan', 'kel', 'kota', 'kec'
}

def clean_text(text: str) -> str:
    """
    Membersihkan teks dari karakter khusus dan normalisasi.
    
    Args:
        text: Teks input
        
    Returns:
        Teks yang sudah dibersihkan
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation kecuali beberapa karakter
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def tokenize(text: str) -> list:
    """
    Memecah teks menjadi token-token.
    
    Args:
        text: Teks input
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    return text.split()


def remove_stopwords(tokens: list) -> list:
    """
    Menghapus stopwords dari list token.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of tokens tanpa stopwords
    """
    return [token for token in tokens if token not in STOPWORDS_ID and len(token) > 1]


def preprocess_text(text: str, remove_stop: bool = True) -> list:
    """
    Pipeline lengkap preprocessing teks.
    
    Args:
        text: Teks input
        remove_stop: Apakah menghapus stopwords
        
    Returns:
        List of preprocessed tokens
    """
    # Clean
    cleaned = clean_text(text)
    
    # Tokenize
    tokens = tokenize(cleaned)
    
    # Remove stopwords
    if remove_stop:
        tokens = remove_stopwords(tokens)
    
    return tokens


def preprocess_query(query: str) -> list:
    """
    Preprocessing khusus untuk query pencarian.
    Lebih permisif, tidak menghapus semua stopwords.
    
    Args:
        query: Query pencarian
        
    Returns:
        List of preprocessed tokens
    """
    # Clean
    cleaned = clean_text(query)
    
    # Tokenize
    tokens = tokenize(cleaned)
    
    # Hapus stopwords yang sangat umum saja
    very_common = {'yang', 'di', 'dan', 'ke', 'dari', 'ini', 'itu', 'dengan'}
    tokens = [t for t in tokens if t not in very_common and len(t) > 1]
    
    return tokens


def create_document_field(row) -> str:
    """
    Membuat field dokumen gabungan dari row dataframe.
    
    Args:
        row: Pandas Series dari satu baris data
        
    Returns:
        Gabungan teks untuk diindex
    """
    fields = []
    
    if 'nama_tempat' in row.index:
        fields.append(str(row['nama_tempat']))
    
    if 'alamat_tempat' in row.index:
        fields.append(str(row['alamat_tempat']))
    
    if 'nama_kelurahan' in row.index:
        fields.append(str(row['nama_kelurahan']))
    
    if 'nama_kecamatan' in row.index:
        fields.append(str(row['nama_kecamatan']))
    
    if 'store' in row.index:
        fields.append(str(row['store']))
    
    return ' '.join(fields)
