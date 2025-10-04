import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gamma, chi2
from scipy.special import gamma as gammascipy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import networkx as nx
from math import gcd
from functools import reduce

# Atur konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Sistem Bonus-Malus Ideal", page_icon="📊")

# Tingkatkan batas elemen untuk styling Pandas
pd.set_option("styler.render.max_elements", 746416)
st.sidebar.image("logougm 2.png", use_container_width=True)

st.markdown(
    """
    <style>
        /* Background utama jadi hitam dengan gradient halus */
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
            color: white;
        }

        /* Header styling untuk lebih menonjol */
        .stTitle {
            color: #ffffff;
            font-size: 2.5em;
            font-weight: bold;
        }

        /* Table/Dataframe styling dengan border halus dan hover effect */
        table.dataframe, .dataframe th, .dataframe td {
            border: 1px solid #333 !important;
            color: white !important;
            background-color: #111 !important;
            font-size: 14px !important;
            -webkit-font-smoothing: antialiased !important;
            -moz-osx-font-smoothing: grayscale !important;
            text-shadow: none !important;
        }
        .dataframe th {
            background-color: #222 !important;
            color: #fff !important;
            font-weight: bold !important;
        }
        .dataframe tbody tr:hover {
            background-color: #333 !important;
        }

        /* Hilangkan striping default, gunakan hover saja */
        .dataframe tbody tr:nth-child(odd) {
            background-color: #111 !important;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #111 !important;
        }

        /* Scrollbar yang lebih stylish */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#555, #777);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-track {
            background: #222;
            border-radius: 4px;
        }

        /* Metric styling untuk tema gelap */
        .stMetric > label {
            color: #aaa;
        }
        .stMetric > .stMetricValue {
            color: #fff;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            color: #fff;
            background-color: #222;
        }
        .streamlit-expanderContent {
            background-color: #111;
        }

        /* Button styling untuk lebih menarik */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }

        /* Progress bar styling */
        .stProgress > div > div > div {
            background-color: #4CAF50;
        }

        /* Improved input styling */
        .stNumberInput > div > div > input {
            background-color: #222;
            color: white;
            border: 1px solid #444;
            border-radius: 6px;
        }

        /* Slider styling */
        .stSlider > div > div > div > div {
            background-color: #4CAF50;
        }

        /* Radio and selectbox styling */
        .stRadio > div, .stSelectbox > div {
            background-color: #222;
            border-radius: 6px;
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Fungsi untuk mencari akar lambda dari persamaan kubik (Fourth-Degree Loss)
def solve_lambda_cubic(alpha, beta):
    E1 = alpha / beta
    E2 = alpha * (alpha + 1) / (beta**2)
    E3 = alpha * (alpha + 1) * (alpha + 2) / (beta**3)
    coefs = [1.0, -3.0*E1, 3.0*E2, -E3]
    r = np.roots(coefs)
    real_roots = [np.real(x) for x in r if abs(np.imag(x)) < 1e-8]
    pos_real = [x for x in real_roots if x > 0]
    if len(pos_real) == 1:
        return float(pos_real[0])
    if len(pos_real) > 1:
        cand = min(pos_real, key=lambda x: abs(x - E1))
        return float(cand)
    r_sorted = sorted(r, key=lambda z: (abs(np.imag(z)), -np.real(z)))
    return float(np.real(r_sorted[0]))

# Fungsi predictive PDF
def pred_pdf(y, a, k, tau, t):
    comb = gammascipy(y + a + k) / (gammascipy(y + 1) * gammascipy(a + k))
    term1 = ((tau + t) / (tau + t + 1)) ** (a + k)
    term2 = (1 / (tau + t + 1)) ** y
    return comb * term1 * term2

# Fungsi untuk compute baseline probabilities
def compute_baseline_probs(a, tau, k_use=0, t_use=0, max_y=5):
    pred_baseline = np.array([pred_pdf(y, a, k_use, tau, t_use) for y in range(max_y + 1)])
    bsly = pred_baseline  # bsly0 to bsly5
    cumsums = np.cumsum(pred_baseline[:-1])  # cum up to y=0 to 4
    bsl_ngty_full = 1 - np.append([0], cumsums)  # ngty0=1, ngty1=1-bsly0, ..., ngty5=1-sum0-4
    bsl_ngty = bsl_ngty_full[1:6]  # ngty1 to ngty5
    return tuple(np.concatenate((bsly, bsl_ngty)).tolist())

def simulate_bms(P_matrix, ncd_vec, n_classes, n_years=100, tol=1e-6, country_name="Unknown", pi_init=None, suppress_output=False):
    if P_matrix.shape != (n_classes, n_classes):
        raise ValueError("P_matrix must be n_classes x n_classes.")
    if len(ncd_vec) != n_classes:
        raise ValueError("ncd_vec must have length n_classes.")

    premium_vec = (1 - np.array(ncd_vec))
    if pi_init is None:
        pi_iter = np.ones(n_classes) / n_classes
    else:
        pi_iter = np.array(pi_init) / np.sum(pi_init)

    prem_list = np.zeros(n_years)
    pi_list = [pi_iter.copy()]

    for n in range(n_years):
        pi_iter = pi_iter @ P_matrix
        pi_iter = pi_iter / np.sum(pi_iter)
        prem_list[n] = np.sum(pi_iter * premium_vec)
        pi_list.append(pi_iter.copy())

    SP = pi_list[-1]
    TV = np.array([np.sum(np.abs(pi_list[n] - SP)) for n in range(n_years)])

    thn_stabil_idx = np.where(TV < tol)[0]
    if len(thn_stabil_idx) == 0:
        if not suppress_output:
            st.warning(f"Konvergensi (TV < tol) tidak tercapai untuk {country_name} dalam {n_years} tahun.")
        thn_stabil = np.nan
        pstasioner = np.nan
    else:
        thn_stabil = thn_stabil_idx[0]
        pstasioner = prem_list[thn_stabil - 1]
        if not suppress_output:
            st.success(f"Premi stasioner untuk {country_name} = {pstasioner:.5f} pada tahun ke-{thn_stabil}")

    max_check = min(100, n_years)
    df_conv = pd.DataFrame({
        'Tahun': range(1, max_check + 1),
        'Premi': prem_list[:max_check].round(5),
        'Total_Variation': TV[:max_check].round(6)
    })

    return {
        'df_conv': df_conv,
        'SP': SP,
        'thn_stabil': thn_stabil,
        'TV': TV,
        'pstasioner': pstasioner,
        'prem_list': prem_list
    }

# Fungsi untuk load dan validasi file, pakai cache agar cepat
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    if not isinstance(df, pd.DataFrame):
        st.error(f"❌ File tidak menghasilkan DataFrame yang valid. Tipe data: {type(df)}")
        return None
    
    if df.empty or len(df.columns) == 0:
        st.error("❌ DataFrame kosong atau tidak memiliki kolom. Pastikan file berisi data tabular.")
        return None
    
    return df

# Define countries and P_builder functions globally
countries = {
    'Malaysia': {'ncd': [0.55, 0.45, 0.3833, 0.3, 0.25, 0], 'n_classes': 6},
    'Thailand': {'ncd': [0.4, 0.3, 0.2, 0, -0.2, -0.3, -0.4], 'n_classes': 7},
    'Denmark': {'ncd': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.2, -0.5], 'n_classes': 10},
    'British': {'ncd': [0.77, 0.6, 0.55, 0.45, 0.35, 0.25, 0], 'n_classes': 7},
    'Kenya': {'ncd': [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0], 'n_classes': 7},
    'Hong Kong': {'ncd': [0.6, 0.6, 0.4, 0.3, 0.2, 0], 'n_classes': 6},
    'Swedia': {'ncd': [0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0], 'n_classes': 7}
}

# Define P_builder functions
def P_malaysia(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, 0, 0, ngty1,
        bsly0, 0, 0, 0, 0, ngty1,
        0, bsly0, 0, 0, 0, ngty1,
        0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(6, 6)

def P_thailand(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, bsly1, ngty2, 0, 0,
        bsly0, 0, 0, bsly1, ngty2, 0, 0,
        0, bsly0, 0, bsly1, ngty2, 0, 0,
        0, 0, bsly0, bsly1, ngty2, 0, 0,
        0, 0, bsly0, bsly1, 0, ngty2, 0,
        0, 0, bsly0, bsly1, 0, 0, ngty2,
        0, 0, bsly0, bsly1, 0, 0, ngty2
    ]
    return np.array(data).reshape(7, 7)

def P_denmark(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, bsly1, 0, bsly2, 0, bsly3, 0, bsly4, ngty5,
        bsly0, 0, 0, bsly1, 0, bsly2, 0, bsly3, 0, ngty4,
        0, bsly0, 0, 0, bsly1, 0, bsly2, 0, bsly3, ngty4,
        0, 0, bsly0, 0, 0, bsly1, 0, bsly2, 0, ngty3,
        0, 0, 0, bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        0, 0, 0, 0, bsly0, 0, 0, bsly1, 0, ngty2,
        0, 0, 0, 0, 0, bsly0, 0, 0, bsly1, ngty2,
        0, 0, 0, 0, 0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, 0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(10, 10)

def P_british(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        0, bsly0, 0, 0, bsly1, 0, ngty2,
        0, 0, bsly0, 0, bsly1, 0, ngty2,
        0, 0, 0, bsly0, 0, bsly1, ngty2,
        0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(7, 7)

def P_kenya(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, 0, 0, 0, ngty1,
        bsly0, 0, 0, 0, 0, 0, ngty1,
        0, bsly0, 0, 0, 0, 0, ngty1,
        0, 0, bsly0, 0, 0, 0, ngty1,
        0, 0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(7, 7)

def P_hongkong(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, bsly1, 0, 0, ngty2,
        bsly0, 0, 0, bsly1, 0, ngty2,
        0, bsly0, 0, 0, 0, ngty1,
        0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(6, 6)

def P_swedia(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, bsly1, 0, bsly2, 0, ngty3,
        bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        0, bsly0, 0, 0, bsly1, 0, ngty2,
        0, 0, bsly0, 0, 0, bsly1, ngty2,
        0, 0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(7, 7)

# Assign P_builder to countries
countries['Malaysia']['P_builder'] = P_malaysia
countries['Thailand']['P_builder'] = P_thailand
countries['Denmark']['P_builder'] = P_denmark
countries['British']['P_builder'] = P_british
countries['Kenya']['P_builder'] = P_kenya
countries['Hong Kong']['P_builder'] = P_hongkong
countries['Swedia']['P_builder'] = P_swedia

# Fungsi pengecekan distribusi stasioner (terjemahan dari R ke Python)
def check_stationary_distribution(transition_matrix):
    if not isinstance(transition_matrix, np.ndarray) or transition_matrix.ndim != 2:
        st.error("Error: Matriks transisi harus berupa matriks")
        return None
    
    n = transition_matrix.shape[0]
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        st.error("Error: Matriks transisi harus persegi")
        return None
    
    row_sums = np.sum(transition_matrix, axis=1)
    if np.any(np.abs(row_sums - 1) > 1e-8):
        st.error("Error: Setiap baris matriks transisi harus berjumlah 1")
        return None
    
    if np.any(transition_matrix < 0):
        st.error("Error: Probabilitas tidak boleh negatif")
        return None
    
    result = {}
    
    # 1. Finite states
    st.markdown("### 1. Jumlah Status Berhingga")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Jumlah Status", n)
    result['finite_states'] = n < np.inf
    if result['finite_states']:
        st.success("✅ Kriteria terpenuhi: Status berhingga.")
    else:
        st.error("❌ Gagal: Jumlah status tidak berhingga.")
    
    # 2. Irreducible (strongly connected)
    st.markdown("### 2. Rantai Markov: Accessible dan Irreducible")
    G = nx.from_numpy_array(transition_matrix > 0, create_using=nx.DiGraph)
    num_scc = nx.number_strongly_connected_components(G)
    result['accessible_irreducible'] = num_scc == 1
    col_ir1, col_ir2 = st.columns(2)
    with col_ir1:
        st.metric("Komponen Kuat", num_scc)
    if result['accessible_irreducible']:
        st.success("✅ Kriteria terpenuhi: Rantai irreducible (1 komponen kuat terhubung). Semua status dapat diakses dari status lain.")
    else:
        st.error(f"❌ Gagal: Rantai tidak irreducible ({num_scc} komponen kuat).")
        # Show components if failed
        sccs = list(nx.strongly_connected_components(G))
        for i, scc in enumerate(sccs):
            st.warning(f"Komponen {i+1}: Status {sorted([x+1 for x in scc])}")
    
    # 3. Positive recurrent
    st.markdown("### 3. Positive Recurrent")
    result['positive_recurrent'] = result['accessible_irreducible']
    if result['positive_recurrent']:
        st.success("✅ Kriteria terpenuhi: Karena rantai irreducible dan berhingga, semua status positive recurrent.")
    else:
        st.error("❌ Gagal: Rantai tidak positive recurrent karena tidak irreducible.")
    
    # 4. Non-absorbing
    st.markdown("### 4. Non-Absorbing States")
    diag = np.diag(transition_matrix)
    off_diag = transition_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    absorbing_states = np.where((diag == 1) & (np.sum(off_diag, axis=1) == 0))[0]
    result['non_absorbing'] = len(absorbing_states) == 0
    col_abs1, col_abs2 = st.columns(2)
    with col_abs1:
        st.metric("Status Absorbing", len(absorbing_states))
    if result['non_absorbing']:
        st.success("✅ Kriteria terpenuhi: Tidak ada status absorbing.")
    else:
        st.error(f"❌ Gagal: Terdapat {len(absorbing_states)} status absorbing di indeks: {[(x+1) for x in absorbing_states.tolist()]}")
    
    # 5. Aperiodic
    st.markdown("### 5. Aperiodic")
    has_self_loops = np.any(np.diag(transition_matrix) > 0)
    if has_self_loops:
        result['aperiodic'] = True
        st.success("✅ Kriteria terpenuhi: Aperiodic (terdapat self-loop pada diagonal positif).")
        st.info("Self-loops ditemukan di: " + ", ".join([f"status {i+1}" for i in np.where(np.diag(transition_matrix) > 0)[0]]))
    else:
        cycles = []
        try:
            cycles = list(nx.simple_cycles(G))
            cycle_lengths = [len(cycle) for cycle in cycles]
            unique_cycles = sorted(set(cycle_lengths))
            if len(cycles) == 0:
                result['aperiodic'] = True
                st.success("✅ Kriteria terpenuhi: Aperiodic (tidak ada siklus terdeteksi).")
            else:
                def compute_gcd(numbers):
                    return reduce(gcd, numbers)
                period = compute_gcd(cycle_lengths)
                result['aperiodic'] = period == 1
                col_per1, col_per2 = st.columns(2)
                with col_per1:
                    st.metric("Periode (GCD)", period)
                with col_per2:
                    st.metric("Panjang Siklus Unik", len(unique_cycles))
                if result['aperiodic']:
                    st.success("✅ Kriteria terpenuhi: Aperiodic (GCD panjang siklus = 1).")
                else:
                    st.error(f"❌ Gagal: Rantai periodik dengan periode {period}.")
                st.info(f"Siklus ditemukan dengan panjang: {unique_cycles[:5]}{'...' if len(unique_cycles)>5 else ''}")
        except Exception as e:
            result['aperiodic'] = False
            st.error(f"❌ Gagal: Tidak dapat menentukan periodisitas: {e}")
    
    # Kesimpulan
    st.markdown("### Kesimpulan")
    result['has_stationary_distribution'] = all(result.values())
    passed_count = sum(result.values())
    col_con1, col_con2 = st.columns(2)
    with col_con1:
        st.metric("Kriteria Terpenuhi", passed_count, delta=None)
    if result['has_stationary_distribution']:
        st.success("🎉 Rantai Markov memiliki distribusi stasioner karena semua kriteria terpenuhi!")
    else:
        st.error("⚠️ Rantai Markov TIDAK memiliki distribusi stasioner. Perbaiki matriks transisi.")
    
    return result

# Sidebar untuk unggah file dan pengaturan global
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("📂 Unggah file .csv/.xlsx", type=["csv", "xlsx"], help="Unggah file data klaim (.csv atau .xlsx) untuk analisis.")
    st.subheader("💰 Premi Dasar")
    premium_value = st.number_input("Masukkan premi dasar:", min_value=0.0, value=100.0, step=1.0, help="Premi dasar yang digunakan untuk menghitung premi optimal")
    st.info("👈 Gunakan tab di atas untuk navigasi. Unggah data terlebih dahulu untuk memulai analisis.")

# Judul utama aplikasi dengan deskripsi singkat
st.title("📊 Perancangan Sistem Bonus-Malus Ideal")
st.markdown("**Deskripsi:** Aplikasi ini dapat membantu perusahaan dalam merancang premi ideal menggunakan pendekatan Markovian dan Bayesian. Unggah data klaim dan eksplorasi berbagai skenario untuk pengambilan keputusan yang lebih baik.")

# Konten utama dengan tabs untuk navigasi yang lebih baik
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Data & Estimasi", "⚖️ Loss Function", "🏆 Premi Optimal", "🔄 Simulasi", "🏭 Premi Stasioner", "📊 Sensitivitas"])

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        if df is None:
            st.stop()

        # Bagian 1: Data Upload & Estimasi Parameter (Tab 1)
        with tab1:
            st.header("📋 Data, Fitting Distribusi dan Estimasi Parameter")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df.head(100).style.highlight_null(props='background-color:#ff4444'), use_container_width=True, hide_index=True)
            with col2:
                st.metric("Total Baris", len(df))
                st.metric("Total Kolom", len(df.columns))
                csv = df.to_csv(index=False)
                st.download_button("📥 Unduh dalam .csv", csv, "data_full.csv", "text/csv", use_container_width=True)
            
            st.subheader("🔍 Pilih Kolom Frekuensi Klaim")
            freq_column = st.selectbox("Kolom frekuensi klaim:", df.columns, help="Pilih kolom yang berisi jumlah klaim per polis.")
            
            if freq_column:
                freq_data = df[freq_column]
                if freq_data.empty:
                    st.error("❌ Kolom frekuensi kosong.")
                    st.stop()

                xbar = freq_data.mean()
                skuadrat = freq_data.var()
                tau = xbar / (skuadrat - xbar)
                aa = (xbar ** 2) / (skuadrat - xbar)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rata-rata (x̄)", f"{xbar:.4f}")
                col2.metric("Variansi (s²)", f"{skuadrat:.4f}")
                col3.metric("Tau (τ)", f"{tau:.4f}")
                col4.metric("Alpha (α)", f"{aa:.4f}")

                with st.expander("🔬 Uji Hipotesis Chi-Square (Goodness of Fit)", expanded=False):
                    unique_categories = sorted(freq_data.unique())
                    n_categories = len(unique_categories)
                    observed = freq_data.value_counts().sort_index().reindex(unique_categories, fill_value=0).values

                    P = np.zeros(n_categories)
                    P[0] = (tau / (1 + tau)) ** aa
                    for k in range(n_categories - 1):
                        P[k + 1] = ((k + aa) / ((k + 1) * (1 + tau))) * P[k]
                    P = P / np.sum(P)

                    n = len(freq_data)
                    expected = P * n
                    chisquare = np.sum((observed - expected) ** 2 / expected)
                    df_chi = n_categories - 1 - 2
                    critical_value = chi2.ppf(1 - 0.05, df_chi)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chi-Square", f"{chisquare:.4f}")
                    col2.metric("Degrees of Freedom", df_chi)
                    col3.metric("Nilai Kritis (α=0.05)", f"{critical_value:.4f}")

                    if chisquare < critical_value:
                        st.success("✅ Data cocok dengan distribusi binomial negatif.")
                    else:
                        st.warning("⚠️ Data tidak sepenuhnya cocok. Pertimbangkan distribusi alternatif?")

        # Bagian 2: Loss Function (Tab 2)
        with tab2:
            st.header("⚖️ Pemilihan Loss Function")
            st.markdown("**Pilihan:** Pilih fungsi kerugian untuk estimasi prior/posterior mean.")
            loss_function = st.radio("Jenis loss function:", ("Squared-Error Loss", "Absolute Loss Function", "Fourth-Degree Loss"), horizontal=True, help="Squared-Error: Mean. Absolute: Median. Fourth-Degree: Robust terhadap outlier.")
            
            # Hitung prior_val di sini untuk digunakan di tab lain (cache)
            @st.cache_data
            def compute_prior(loss_func, a, t):
                if loss_func == "Squared-Error Loss":
                    return a / t
                elif loss_func == "Absolute Loss Function":
                    return gamma.ppf(0.5, a=a, scale=1.0/t)
                else: 
                    return solve_lambda_cubic(a, t)
            
            prior_val = compute_prior(loss_function, aa, tau)
            st.metric("Nilai Prior (λ₀)", f"{prior_val:.4f}", help="Nilai prior berdasarkan loss function yang dipilih.")

        # Bagian 3: Tabel Premi Optimal (Tab 3)
        with tab3:
            st.header("🏆 Tabel Premi Optimal Sistem Bonus-Malus")
            col1, col2 = st.columns(2)
            with col1:
                max_k_opt = st.number_input("Maksimum k:", min_value=0, max_value=15, step=1, value=4, help="Batas maksimum klaim yang dilakukan pemegang polis.")
            with col2:
                max_t_opt = st.number_input("Maksimum t:", min_value=0, max_value=15, step=1, value=7, help="Batas maksimum tahun pertanggungan.")

            if st.button("💡 Hitung Tabel Premi", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                t_vals = np.arange(0, max_t_opt + 1)
                k_vals = np.arange(0, max_k_opt + 1)
                result = np.full((len(t_vals), len(k_vals)), np.nan)
                
                for t_idx, t in enumerate(t_vals):
                    for k_idx, k in enumerate(k_vals):
                        status_text.text(f"Menghitung t={t}, k={k}...")
                        progress_bar.progress((t_idx * len(k_vals) + k_idx + 1) / (len(t_vals) * len(k_vals)))
                        
                        if t == 0 and k > 0:
                            continue
                        
                        alpha = aa + k
                        rate = tau + t
                        if k == 0:
                            post_val = compute_prior(loss_function, alpha, rate)
                            factor = post_val / prior_val
                        else:
                            post_val = compute_prior(loss_function, alpha, rate)
                            factor = 1 + post_val
                        result[t_idx, k_idx] = premium_value * factor
                
                progress_bar.progress(1.0)
                status_text.text("Berhasil dihitung!")
                
                result_df = pd.DataFrame(result, index=[f"t={t}" for t in t_vals], columns=[f"k={k}" for k in k_vals]).round(2)
                st.dataframe(result_df.style.format("{:.2f}").background_gradient(cmap='Blues_r', axis=None).set_caption("Tabel Premi Bonus-Malus Optimal"), use_container_width=True, hide_index=False)
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=True, sheet_name='Tabel_Premi')
                xlsx = output.getvalue()
                st.download_button(
                    label="📥 Download Tabel Premi (.xlsx)",
                    data=xlsx,
                    file_name="tabel_premi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # Bagian 4: Simulasi (Tab 4) - Fixed logic for cumulative claims
        with tab4:
            st.header("🔄 Simulasi Premi Ideal")
            simulation_mode = st.radio("Mode Simulasi:", ("Manual Input", "Berdasarkan Matriks Negara"))
            
            if simulation_mode == "Manual Input":
                st.markdown("**Instruksi:** Masukkan jumlah klaim per tahun (maksimum sesuai slider). Premi dihitung independen per tahun berdasarkan klaim tahun tersebut.")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_years_opt = st.slider("Jumlah tahun pertanggungan:", min_value=1, max_value=10, value=7, help="Jumlah tahun untuk simulasi.")
                with col2:
                    max_k_sim_opt = st.slider("Batas maksimum klaim per tahun:", min_value=0, max_value=50, value=4, help="Batas klaim maksimal per tahun dalam simulasi.")
                
                simulation_data = []
                
                for year in range(1, num_years_opt + 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        claims_this_year = st.number_input(f"Klaim tahun ke-{year}:", min_value=0, max_value=max_k_sim_opt, step=1, key=f"claims_opt_{year}")
                    k = claims_this_year
                    t = year
                    alpha = aa + k
                    rate = tau + t
                    post_val = compute_prior(loss_function, alpha, rate)
                    factor = post_val / prior_val if k == 0 else 1 + post_val
                    premium = premium_value * factor
                    simulation_data.append({"Tahun": year, "Klaim Tahun Ini (k)": k, "Premi": premium})
                    
                    col2.metric(f"Premi Tahun {year}", f"Rp {premium:,.2f}")
                
                if simulation_data:
                    sim_df = pd.DataFrame(simulation_data)
                    st.dataframe(sim_df.style.format({"Premi": "{:,.2f}"}), use_container_width=True, hide_index=True)
                    avg_premium = np.mean([row["Premi"] for row in simulation_data])
                    st.metric("Rata-Rata Premi", f"Rp {avg_premium:,.2f}", help="Premi rata-rata selama periode simulasi.")
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        sim_df.to_excel(writer, index=False, sheet_name='Simulasi_Manual')
                    xlsx = output.getvalue()
                    st.download_button(
                        label="📥 Download Simulasi Manual (.xlsx)",
                        data=xlsx,
                        file_name="simulasi_manual.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            else:
                st.markdown("**Instruksi:** Simulasi berdasarkan matriks negara. Premi mengikuti optimal dengan batas t dan k dari sistem negara.")
                
                country_sim = st.selectbox("Pilih Negara:", list(countries.keys()))
                k_max_dict = {
                    'Malaysia': 1,
                    'Thailand': 2,
                    'Denmark': 5,
                    'British': 3,
                    'Kenya': 1,
                    'Hong Kong': 2,
                    'Swedia': 3
                }
                max_k_sim_opt = k_max_dict[country_sim]
                
                if st.button("Jalankan Simulasi BMS Ideal", type="primary"):
                    # Hitung P_matrix baseline untuk konvergensi
                    probs_baseline = compute_baseline_probs(aa, tau)
                    P_matrix_sim = countries[country_sim]['P_builder'](*probs_baseline)
                    ncd_vec_sim = countries[country_sim]['ncd']
                    n_classes_sim = countries[country_sim]['n_classes']
                    
                    result_conv = simulate_bms(P_matrix_sim, ncd_vec_sim, n_classes_sim, n_years=100, tol=1e-6, country_name=country_sim, suppress_output=True)
                    num_years_opt = int(result_conv['thn_stabil']) if not np.isnan(result_conv['thn_stabil']) else 7
                    
                    st.info(f"t maksimum berdasarkan konvergensi: {num_years_opt} tahun. k maksimum: {max_k_sim_opt}")
                    
                    simulation_data = []
                    cumulative_k = 0
                    
                    for year in range(1, num_years_opt + 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            remaining_capacity = max_k_sim_opt - cumulative_k
                            claims_this_year_input = st.number_input(f"Klaim tahun ke-{year}:", min_value=0, max_value=None, step=1, key=f"claims_opt_matrix_{year}")
                            claims_this_year = min(claims_this_year_input, remaining_capacity)
                            if claims_this_year_input > remaining_capacity:
                                st.warning(f"Klaim tahun {year} dibatasi ke {remaining_capacity} karena melebihi batas kumulatif {max_k_sim_opt}.")
                        cumulative_k += claims_this_year
                        k = cumulative_k
                        t = year
                        alpha = aa + k
                        rate = tau + t
                        post_val = compute_prior(loss_function, alpha, rate)
                        factor = post_val / prior_val if k == 0 else 1 + post_val
                        premium = premium_value * factor
                        simulation_data.append({"Tahun": year, "Klaim Tahun Ini": claims_this_year, "Banyak Klaim (k)": k, "Premi": premium})
                        
                        col2.metric(f"Premi Tahun {year}", f"Rp {premium:,.2f}")
                    
                    if simulation_data:
                        sim_df = pd.DataFrame(simulation_data)
                        st.dataframe(sim_df.style.format({"Premi": "{:,.2f}"}), use_container_width=True, hide_index=True)
                        avg_premium = np.mean([row["Premi"] for row in simulation_data])
                        st.metric("Rata-Rata Premi", f"Rp {avg_premium:,.2f}", help="Premi rata-rata selama periode simulasi.")
                        
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            sim_df.to_excel(writer, index=False, sheet_name=f'Simulasi_{country_sim}')
                        xlsx = output.getvalue()
                        st.download_button(
                            label=f"📥 Download Simulasi {country_sim} (.xlsx)",
                            data=xlsx,
                            file_name=f"Simulasi_{country_sim}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

        # Bagian 5: Premi Stasioner (Tab 5) - Modifikasi dengan opsi matrix
        with tab5:
            st.header("🏭 Perhitungan Premi Stasioner")

            matrix_mode = st.radio("Pilih Mode Matriks:", ("Sistem Negara", "Definisikan Matriks"), horizontal=True)

            if matrix_mode == "Sistem Negara":
                col1, col2 = st.columns(2)
                with col1:
                    country = st.selectbox("Pilih Sistem/Negara:", list(countries.keys()), help="Pilih negara untuk simulasi premi stasioner.")
                with col2:
                    n_years_stat = st.slider("Jumlah tahun simulasi:", min_value=50, max_value=1000, value=100, help="Lebih banyak tahun untuk konvergensi yang lebih akurat untuk sistem dengan kelas yang rumit.")
                tol_stat = st.number_input("Nilai Toleransi (tol):", min_value=1e-10, max_value=1e-3, value=1e-6, step=1e-7, format="%.0e")

                use_baseline = st.checkbox("Gunakan Baseline (k = 0, t = 0)", value=True, help="Gunakan probabilitas baseline atau sesuaikan t dan k.")
                k_stat = 0
                t_stat = 0
                if not use_baseline:
                    col_k, col_t = st.columns(2)
                    with col_k:
                        k_stat = st.number_input("Nilai k:", min_value=0, max_value=10, step=1, value=0)
                    with col_t:
                        t_stat = st.number_input("Nilai t:", min_value=0, max_value=10, step=1, value=0)

                if st.button("🚀 Jalankan Simulasi Stasioner", type="primary"):
                    progress_bar = st.progress(0)
                    probs = compute_baseline_probs(aa, tau, k_stat, t_stat)
                    P_matrix = countries[country]['P_builder'](*probs)
                    ncd_vec = countries[country]['ncd']
                    n_classes = countries[country]['n_classes']

                    with st.spinner("Menghitung premi stasioner..."):
                        result_stat = simulate_bms(P_matrix, ncd_vec, n_classes, n_years=n_years_stat, tol=tol_stat, country_name=country, suppress_output=False)
                        progress_bar.progress(1.0)

                    st.dataframe(result_stat['df_conv'].style.format({"Premi": "{:.5f}", "Total_Variation": "{:.6g}"}), use_container_width=True, hide_index=True)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_stat['df_conv'].to_excel(writer, index=False, sheet_name=f'Premi_Stasioner_{country}')
                    xlsx = output.getvalue()
                    st.download_button(
                        label=f"📥 Download Premi Stasioner {country} (.xlsx)",
                        data=xlsx,
                        file_name=f"premi_stasioner_{country}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    if not np.isnan(result_stat['pstasioner']):
                        st.metric("Premi Stasioner Akhir", f"{result_stat['pstasioner']:,.5f}")

                    if st.checkbox("📈 Tampilkan Grafik Konvergensi", value=True):
                        fig_stat = px.line(x=np.arange(1, min(51, n_years_stat + 1)), y=result_stat['prem_list'][:50],
                                           title=f"Konvergensi Premi BMS - {country}",
                                           labels={'x': 'Tahun', 'y': 'Premi'})
                        fig_stat.update_layout(plot_bgcolor='#111', paper_bgcolor='#000', font_color='white', 
                                               xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
                        st.plotly_chart(fig_stat, use_container_width=True)

            else:  # Matriks Custom
                st.subheader("🔧 Konfigurasi Matriks")
                n_classes_custom = st.number_input("Jumlah kelas:", min_value=2, max_value=20, value=3, help="Ukuran matriks n x n.")
                
                st.markdown("**Input Matriks:** Gunakan editor di bawah untuk mengisi probabilitas transisi (baris harus sum=1).")
                default_matrix = np.full((n_classes_custom, n_classes_custom), 1.0 / n_classes_custom)
                df_matrix = pd.DataFrame(default_matrix, columns=[f"To {j+1}" for j in range(n_classes_custom)], index=[f"From {i+1}" for i in range(n_classes_custom)])
                edited_df = st.data_editor(
                    df_matrix,
                    num_rows="fixed",
                    column_config={col: st.column_config.NumberColumn(col, min_value=0.0, max_value=1.0, step=0.01, format="%.4f") for col in df_matrix.columns},
                    use_container_width=True,
                    hide_index=False
                )
                P_matrix_custom = edited_df.values
                
                # Input NCD vector
                st.subheader("Input Vektor NCD (No-Claim Discount)")
                available_ncd = [c for c in countries if len(countries[c]['ncd']) == n_classes_custom]
                if len(available_ncd) > 0:
                    selected_ncd_country = st.selectbox("Pilih vektor NCD:", ["Manual"] + available_ncd)
                    if selected_ncd_country == "Manual":
                        st.markdown("**Input Manual:** Masukkan nilai NCD untuk setiap kelas.")
                        ncd_vec_custom = []
                        for i in range(n_classes_custom):
                            ncd_i = st.number_input(f"NCD untuk kelas {i+1}:", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key=f"ncd_custom_{i}")
                            ncd_vec_custom.append(ncd_i)
                    else:
                        ncd_vec_custom = countries[selected_ncd_country]['ncd']
                        st.info(f"Menggunakan NCD dari {selected_ncd_country}: {ncd_vec_custom}")
                else:
                    st.markdown("**Input Manual:** Masukkan nilai NCD untuk setiap kelas.")
                    ncd_vec_custom = []
                    for i in range(n_classes_custom):
                        ncd_i = st.number_input(f"NCD untuk kelas {i+1}:", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key=f"ncd_custom_{i}")
                        ncd_vec_custom.append(ncd_i)
                
                col1, col2 = st.columns(2)
                with col1:
                    n_years_stat_custom = st.slider("Jumlah tahun simulasi:", min_value=50, max_value=1000, value=100, help="Lebih banyak tahun untuk konvergensi yang lebih akurat.")
                with col2:
                    tol_stat_custom = st.number_input("Nilai Toleransi (tol):", min_value=1e-10, max_value=1e-3, value=1e-6, step=1e-7, format="%.0e")
                
                if st.button("🔍 Cek Distribusi Stasioner", type="secondary"):
                    check_result = check_stationary_distribution(P_matrix_custom)
                    if check_result and check_result.get('has_stationary_distribution', False):
                        st.session_state.custom_matrix_valid = True
                        st.session_state.P_matrix_custom = P_matrix_custom
                        st.session_state.ncd_vec_custom = ncd_vec_custom
                        st.session_state.n_classes_custom = n_classes_custom
                    else:
                        st.session_state.custom_matrix_valid = False
                        st.warning("Matriks tidak valid untuk distribusi stasioner. Perbaiki dan cek ulang.")
                
                if st.button("🚀 Jalankan Simulasi Stasioner", type="primary", disabled=not hasattr(st.session_state, 'custom_matrix_valid') or not st.session_state.custom_matrix_valid):
                    if 'P_matrix_custom' in st.session_state and 'ncd_vec_custom' in st.session_state:
                        progress_bar = st.progress(0)
                        with st.spinner("Menghitung premi stasioner..."):
                            result_stat_custom = simulate_bms(st.session_state.P_matrix_custom, st.session_state.ncd_vec_custom, st.session_state.n_classes_custom, n_years=n_years_stat_custom, tol=tol_stat_custom, country_name="Custom", suppress_output=False)
                            progress_bar.progress(1.0)

                        st.dataframe(result_stat_custom['df_conv'].style.format({"Premi": "{:.5f}", "Total_Variation": "{:.6g}"}), use_container_width=True, hide_index=True)
                        
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            result_stat_custom['df_conv'].to_excel(writer, index=False, sheet_name='Premi_Stasioner')
                        xlsx = output.getvalue()
                        st.download_button(
                            label="📥 Download Premi Stasioner (.xlsx)",
                            data=xlsx,
                            file_name="premi_stasioner.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                        if not np.isnan(result_stat_custom['pstasioner']):
                            st.metric("Premi Stasioner Akhir", f"{result_stat_custom['pstasioner']:,.5f}")

                        if st.checkbox("📈 Tampilkan Grafik Konvergensi", value=True):
                            fig_stat_custom = px.line(x=np.arange(1, min(51, n_years_stat_custom + 1)), y=result_stat_custom['prem_list'][:50],
                                                       title="Konvergensi Premi BMS - Custom",
                                                       labels={'x': 'Tahun', 'y': 'Premi'})
                            fig_stat_custom.update_layout(plot_bgcolor='#111', paper_bgcolor='#000', font_color='white', 
                                                           xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
                            st.plotly_chart(fig_stat_custom, use_container_width=True)

        # Bagian 6: Analisis Sensitivitas (Tab 6) - Implementasi lengkap
        with tab6:
            st.header("📊 Analisis Sensitivitas")
            st.markdown("**Catatan:** Analisis ini digunakan untuk eksplorasi bagaimana perubahan parameter distribusi dan historis dapat mempengaruhi besar kecilnya premi. Pilih opsi di bawah untuk menghitung.")

            sens_type = st.selectbox("Jenis Sensitivitas:", ["Terhadap a", "Terhadap τ", "Peluang Prediktif (t & k)", "Distribusi Stasioner setiap Negara"])

            if sens_type == "Terhadap a":
                st.subheader("Sensitivitas terhadap a")
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    a_min = st.number_input("Nilai a minimum:", value=1.0, step=0.5)
                with col_a2:
                    a_max = st.number_input("Nilai a maksimum:", value=9.0, step=0.5)
                with col_a3:
                    step_a = st.number_input("Step a:", value=1.0, step=0.01, min_value=0.01)
                a_vals_sens = np.arange(a_min, a_max + step_a, step_a)
                tau_fixed_sens = tau
                selected_countries_a = st.multiselect("Pilih negara untuk sensitivitas a:", list(countries.keys()), default=['Malaysia', 'Thailand', 'Denmark'])
                if st.button("Hitung Sensitivitas a", type="primary"):
                    with st.spinner("Menghitung..."):
                        results_a = []
                        for country_name in selected_countries_a:
                            config = countries[country_name]
                            ncd_vec = config['ncd']
                            n_classes = config['n_classes']
                            P_builder = config['P_builder']
                            for a_val in a_vals_sens:
                                probs = compute_baseline_probs(a_val, tau_fixed_sens)
                                P_matrix = P_builder(*probs)
                                sim_result = simulate_bms(P_matrix, ncd_vec, n_classes, n_years=100, tol=1e-6, country_name=country_name, suppress_output=True)
                                results_a.append({'a': a_val, 'Premi_Stasioner': sim_result['pstasioner'], 'Negara': country_name})
                        df_a = pd.DataFrame(results_a)
                        st.dataframe(df_a.round(5), use_container_width=True)
                        
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_a.to_excel(writer, index=False, sheet_name='Sensitivitas_a')
                        xlsx = output.getvalue()
                        st.download_button(
                            label="📥 Download Sensitivitas a (.xlsx)",
                            data=xlsx,
                            file_name="sensitivitas_a.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        fig_a = px.line(df_a, x='a', y='Premi_Stasioner', color='Negara', markers=True, title='Sensitivitas terhadap a')
                        fig_a.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
                        st.plotly_chart(fig_a, use_container_width=True)

            elif sens_type == "Terhadap τ":
                st.subheader("Sensitivitas terhadap τ")
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    tau_min = st.number_input("Min τ:", value=15.0, step=1.0)
                with col_t2:
                    tau_max = st.number_input("Max τ:", value=23.0, step=1.0)
                with col_t3:
                    step_tau = st.number_input("Step τ:", value=1.0, step=0.01, min_value=0.01)
                tau_vals_sens = np.arange(tau_min, tau_max + step_tau, step_tau)
                a_fixed_sens = aa
                selected_countries_tau = st.multiselect("Pilih negara untuk sensitivitas τ:", list(countries.keys()), default=['Malaysia', 'Thailand', 'Denmark'])
                if st.button("Hitung Sensitivitas τ", type="primary"):
                    with st.spinner("Menghitung..."):
                        results_tau = []
                        for country_name in selected_countries_tau:
                            config = countries[country_name]
                            ncd_vec = config['ncd']
                            n_classes = config['n_classes']
                            P_builder = config['P_builder']
                            for tau_val in tau_vals_sens:
                                probs = compute_baseline_probs(a_fixed_sens, tau_val)
                                P_matrix = P_builder(*probs)
                                sim_result = simulate_bms(P_matrix, ncd_vec, n_classes, n_years=100, tol=1e-6, country_name=country_name, suppress_output=True)
                                results_tau.append({'τ': tau_val, 'Premi_Stasioner': sim_result['pstasioner'], 'Negara': country_name})
                        df_tau = pd.DataFrame(results_tau)
                        st.dataframe(df_tau.round(5), use_container_width=True)
                        
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_tau.to_excel(writer, index=False, sheet_name='Sensitivitas_tau')
                        xlsx = output.getvalue()
                        st.download_button(
                            label="📥 Download Sensitivitas τ (.xlsx)",
                            data=xlsx,
                            file_name="sensitivitas_tau.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        fig_tau = px.line(df_tau, x='τ', y='Premi_Stasioner', color='Negara', markers=True, title='Sensitivitas terhadap τ')
                        fig_tau.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
                        st.plotly_chart(fig_tau, use_container_width=True)

            elif sens_type == "Peluang Prediktif (t & k)":
                st.subheader("Sensitivitas Peluang Prediktif terhadap t dan k")
                col_pk1, col_pk2 = st.columns(2)
                with col_pk1:
                    t_min_pk = st.number_input("Min t (tahun):", value=1, step=1)
                    t_max_pk = st.number_input("Max t (tahun):", value=2, step=1)
                with col_pk2:
                    k_min_pk = st.number_input("Min k (banyak klaim):", value=0, step=1)
                    k_max_pk = st.number_input("Max k (banyak klaim):", value=5, step=1)
                if st.button("Hitung Peluang Prediktif", type="primary"):
                    with st.spinner("Menghitung..."):
                        results_tk = []
                        for t_val in range(t_min_pk, t_max_pk + 1):
                            for k_val in range(k_min_pk, k_max_pk + 1):
                                probs = [pred_pdf(y, aa, k_val, tau, t_val) for y in range(5)]  # p0 to p4
                                results_tk.append({'t': t_val, 'k': k_val, 'p0': probs[0], 'p1': probs[1], 'p2': probs[2], 'p3': probs[3], 'p4': probs[4]})
                        df_tk = pd.DataFrame(results_tk)
                        st.dataframe(df_tk.round(5), use_container_width=True)
                        
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_tk.to_excel(writer, index=False, sheet_name='Peluang_Prediktif')
                        xlsx = output.getvalue()
                        st.download_button(
                            label="📥 Download Peluang Prediktif (.xlsx)",
                            data=xlsx,
                            file_name="peluang_prediktif.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        df_long = df_tk.melt(id_vars=['t', 'k'], value_vars=['p0', 'p1', 'p2', 'p3', 'p4'], var_name='', value_name='Peluang Prediktif')
                        fig_tk = px.line(df_long, x='', y='Peluang Prediktif', color='k', facet_col='t', markers=True, title='Sensitivitas Peluang Prediktif')
                        fig_tk.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
                        st.plotly_chart(fig_tk, use_container_width=True)

            elif sens_type == "Distribusi Stasioner setiap Negara":
                st.subheader("Sensitivitas Distribusi Stasioner dan Premi setiap Negara")
                col_dist1, col_dist2 = st.columns(2)
                with col_dist1:
                    t_min_dist = st.number_input("Min t (tahun):", value=1, step=1)
                    t_max_dist = st.number_input("Max t (tahun):", value=5, step=1)
                with col_dist2:
                    k_min_dist = st.number_input("Min k (banyak klaim):", value=0, step=1)
                    k_max_dist = st.number_input("Max k (banyak klaim):", value=5, step=1)
                selected_countries_dist = st.multiselect("Pilih negara untuk distribusi:", list(countries.keys()), default=list(countries.keys()))
                if st.button("Hitung Sensitivitas Distribusi", type="primary"):
                    with st.spinner("Menghitung untuk semua negara..."):
                        all_results = {}
                        all_premium_tables = {}
                        for country_name in selected_countries_dist:
                            config = countries[country_name]
                            n_classes = config['n_classes']
                            ncd_vec = config['ncd']
                            P_builder = config['P_builder']
                            results = pd.DataFrame(columns=['t', 'k'] + [f'pi{i+1}' for i in range(n_classes)])
                            premium_table = pd.DataFrame(columns=['t', 'k', 'premium'])
                            for t in range(t_min_dist, t_max_dist + 1):
                                for k in range(k_min_dist, k_max_dist + 1):
                                    probs = compute_baseline_probs(aa, tau, k, t)
                                    P_matrix = P_builder(*probs)
                                    sim_result = simulate_bms(P_matrix, ncd_vec, n_classes, n_years=100, tol=1e-6, country_name=f"{country_name} t={t} k={k}", suppress_output=True)
                                    sp = sim_result['SP']
                                    row_pi = {'t': t, 'k': k}
                                    for i in range(n_classes):
                                        row_pi[f'pi{i+1}'] = sp[i]
                                    results = pd.concat([results, pd.DataFrame([row_pi])], ignore_index=True)
                                    row_prem = {'t': t, 'k': k, 'premium': sim_result['pstasioner']}
                                    premium_table = pd.concat([premium_table, pd.DataFrame([row_prem])], ignore_index=True)
                            all_results[country_name] = results
                            all_premium_tables[country_name] = premium_table

                            sp_sens = results[results['t'] == t_min_dist]
                            if not sp_sens.empty:
                                pi_cols = [f'pi{i+1}' for i in range(n_classes)]
                                sp_long = sp_sens.melt(id_vars=['t', 'k'], value_vars=pi_cols, var_name='Kelas', value_name='Distribusi Stasioner')
                                fig_sp = px.line(sp_long, x='Kelas', y='Distribusi Stasioner', color='k', markers=True, title=f"{country_name} - Distribusi Stasioner (t={t_min_dist}-{t_max_dist})")
                                fig_sp.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
                                st.plotly_chart(fig_sp, use_container_width=True)

                            st.write(f"**Tabel Premium {country_name}:**")
                            st.dataframe(premium_table.round(5), use_container_width=True)
                            
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                premium_table.to_excel(writer, index=False, sheet_name=f'Premium_{country_name}')
                            xlsx = output.getvalue()
                            st.download_button(
                                label=f"📥 Download Premium {country_name} (.xlsx)",
                                data=xlsx,
                                file_name=f"premium_{country_name}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        # Combined premium plot
                        sens_prem = pd.DataFrame()
                        for country_name, prem_table in all_premium_tables.items():
                            prem_table_copy = prem_table.copy()
                            prem_table_copy['Negara'] = country_name
                            sens_prem = pd.concat([sens_prem, prem_table_copy], ignore_index=True)
                        if not sens_prem.empty:
                            st.write("**Combined Premium Data:**")
                            st.dataframe(sens_prem.round(5), use_container_width=True)
                            
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                sens_prem.to_excel(writer, index=False, sheet_name='Combined_Premium')
                            xlsx = output.getvalue()
                            st.download_button(
                                label="📥 Download Combined Premium (.xlsx)",
                                data=xlsx,
                                file_name="combined_premium.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            fig_combined = px.line(sens_prem, x='t', y='premium', color='Negara', facet_col='k', markers=True, title='Sensitivitas Premi Stasioner per Negara berdasarkan Parameter Historis')
                            fig_combined.update_layout(plot_bgcolor='#000000', paper_bgcolor='#000000', font_color='white')
                            st.plotly_chart(fig_combined, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {str(e)}")
        st.info("Coba periksa file data atau parameter input.")
else:
    st.info("⬆️ Silakan unggah file .csv atau .xlsx di sidebar untuk memulai analisis. Aplikasi siap membantu Anda menghitung premi bonus-malus optimal!")
