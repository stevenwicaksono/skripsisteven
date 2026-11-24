# ==============================================================================
# Kode Program Import Library
# ==============================================================================
import os
import warnings
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time as _time
import sys as _sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    KFold,
    cross_val_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
matplotlib.use("Agg")
warnings.filterwarnings("ignore")
plt.style.use("default")

# ==============================================================================
# Kode Program Konfigurasi Variabel
# ==============================================================================
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PADI = os.path.join(BASE_DIR, "data", "Data_Padi_Indonesia.csv")
DATA_CUACA = os.path.join(BASE_DIR, "data", "Data_Cuaca_Provinsi_Tahun - Copy.csv")
TAHUN_MIN_DATA = 2018
TAHUN_MAX_DATA = 2024
VIS_DIR   = os.path.join(BASE_DIR, "visualizations")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR  = os.path.join(BASE_DIR, "temp_joblib")

for d in [VIS_DIR, MODEL_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

os.environ["JOBLIB_TEMP_FOLDER"] = TEMP_DIR

# ==============================================================================
# Kode Program Kelas Progress Pipeline
# ==============================================================================
class _PipelineProgress:
    SECTIONS = [
        ("1. Load Dataset",                      13),
        ("2. Preprocessing",                      8),
        ("3. Split Train-Test",                   4),
        ("4. Seleksi Fitur",                      6),
        ("5. Baseline Linear Regression",         3),
        ("6. GridSearchCV Regresi",               4),
        ("7. Evaluasi Regresi",                   7),
        ("8. Evaluasi Klasifikasi (4 model)",     7),
        ("9. Visualisasi Evaluasi Regresi",       6),
        ("10. Simpan Model",                      3),
        ("11. Verifikasi Prediksi vs Aktual",     4),
        ("12. Visualisasi Per Provinsi",          3),
    ]
    _TOTAL_STEPS = sum(w for _, w in SECTIONS)

    def __init__(self):
        self._done      = 0
        self._cur       = -1
        self._start     = _time.time()
        self._width     = 40

    def next_section(self, title: str) -> None:
        if self._cur >= 0:
            self._done += self.SECTIONS[self._cur][1]
        self._cur += 1
        pct    = self._done / self._TOTAL_STEPS
        filled = int(self._width * pct)
        bar    = "█" * filled + "░" * (self._width - filled)
        elapsed = _time.time() - self._start
        eta     = (elapsed / pct * (1 - pct)) if pct > 0 else 0
        label   = f"[{self._cur + 1}/{len(self.SECTIONS)}]"
        _sys.stdout.write(
            f"\n{SEP}\n"
            f"  {label} {title}\n"
            f"  [{bar}] {pct*100:5.1f}%"
            f"  | Berlalu: {self._fmt(elapsed)}"
            f"  ETA: {self._fmt(eta)}\n"
            f"{SEP}\n"
        )
        _sys.stdout.flush()

    def finish(self) -> None:
        bar     = "█" * self._width
        elapsed = _time.time() - self._start
        _sys.stdout.write(
            f"\n{SEP}\n"
            f"  [✓] SELESAI\n"
            f"  [{bar}] 100.0%  | Total: {self._fmt(elapsed)}\n"
            f"{SEP}\n"
        )
        _sys.stdout.flush()

    @staticmethod
    def _fmt(secs: float) -> str:
        secs = max(0, secs)
        m, s = divmod(int(secs), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

progress = _PipelineProgress()

N_SPLITS    = 5
THRESHOLD   = 0.000
MIN_FEATURES = 10
SKLEARN_N_JOBS = 1   

PROVINSI_BARU = [
    "Papua Barat Daya",
    "Papua Pegunungan",
    "Papua Selatan",
    "Papua Tengah",
]

SEP = "=" * 72

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ==============================================================================
# Kode Program Fungsi Parsing dan Normalisasi Data
# ==============================================================================
def parse_id(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return np.nan

    s = s.replace(".", "").replace(",", ".") if "," in s else s.replace(" ", "")

    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_provinsi(series: pd.Series) -> pd.Series:
    mapping = {
        "jambi":          "Jambi",
        "sumatra utara":  "Sumatera Utara",
        "sumatra barat":  "Sumatera Barat",
        "sumatra selatan":"Sumatera Selatan",
        "sumatra":        "Sumatera",
    }

    def fix(x):
        if pd.isna(x):
            return x
        s = str(x).strip()
        if s.lower() in mapping:
            return mapping[s.lower()]
        return s.replace("Sumatra", "Sumatera").replace("sumatra", "Sumatera")
    return series.apply(fix)

# ==============================================================================
# Kode Program Fungsi Perhitungan Metrik Evaluasi
# ==============================================================================
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom  = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom  = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom  = np.where(denom < 1e-9, np.nan, denom)
    return np.nanmean(np.abs(y_true - y_pred) / denom) * 100

def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def status_error(error_percent):
    if pd.isna(error_percent):
        return "Tidak Terhitung"
    if error_percent < 5:
        return "Error Sangat Rendah"
    if error_percent < 10:
        return "Error Rendah"
    return "Error Tinggi"

def status_r2_text(r2):
    if r2 >= 0.90: return "Sangat Baik"
    if r2 >= 0.80: return "Baik"
    if r2 >= 0.70: return "Cukup"
    if r2 >= 0.50: return "Moderat"
    return "Perlu Perbaikan"

def regression_metrics(y_true, y_pred, label=""):
    mse      = mean_squared_error(y_true, y_pred)
    r2       = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "Label": label,
        "R2":    r2,
        "RMSE":  np.sqrt(mse),
        "MAE":   mean_absolute_error(y_true, y_pred),
        "MSE":   mse,
        "MAPE":  safe_mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }

def print_classification_report_persen(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n  {'Kelas':<14} {'Presisi':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"  {'-' * 56}")
    for i, cls in enumerate(class_names):
        print(
            f"  {cls:<14} {precision[i]*100:>9.2f}% {recall[i]*100:>9.2f}% "
            f"{f1[i]*100:>9.2f}% {int(support[i]):>10}"
        )
    print(f"  {'-' * 56}")

    mac_p = precision.mean() * 100
    mac_r = recall.mean() * 100
    mac_f = f1.mean() * 100
    print(f"  {'macro avg':<14} {mac_p:>9.2f}% {mac_r:>9.2f}% {mac_f:>9.2f}% {int(sum(support)):>10}")

    if support.sum() > 0:
        w = support / support.sum()
        print(
            f"  {'weighted avg':<14} {(precision*w).sum()*100:>9.2f}% "
            f"{(recall*w).sum()*100:>9.2f}% {(f1*w).sum()*100:>9.2f}% "
            f"{int(sum(support)):>10}"
        )
    print(f"\n  Accuracy overall : {accuracy*100:.2f}%")

# ==============================================================================
# Kode Program Memuat Dataset Padi dan Cuaca
# ==============================================================================
progress.next_section("1. LOAD DATASET")

df_padi = pd.read_csv(DATA_PADI, encoding="utf-8-sig")
df_padi = df_padi[[
    "Provinsi", "Tahun",
    "Luas_Panen(ha)", "Produktivitas(ku/ha)", "Produksi(ton)",
]].copy()

for col in ["Luas_Panen(ha)", "Produktivitas(ku/ha)", "Produksi(ton)"]:
    df_padi[col] = df_padi[col].apply(parse_id)

df_padi.rename(columns={
    "Luas_Panen(ha)":     "Luas Panen",
    "Produktivitas(ku/ha)": "Produktivitas",
    "Produksi(ton)":      "Produksi",
}, inplace=True)

df_padi["Provinsi"] = normalize_provinsi(df_padi["Provinsi"])
df_padi["Tahun"]    = pd.to_numeric(df_padi["Tahun"], errors="coerce").astype("Int64")
df_padi             = df_padi.dropna(
    subset=["Provinsi", "Tahun", "Luas Panen", "Produktivitas", "Produksi"]
).copy()
df_padi["Tahun"] = df_padi["Tahun"].astype(int)

df_padi = df_padi[
    (df_padi["Tahun"] >= TAHUN_MIN_DATA) &
    (df_padi["Tahun"] <= TAHUN_MAX_DATA)
].copy()

print(f"\n  Filter tahun padi: {TAHUN_MIN_DATA}–{TAHUN_MAX_DATA}")
print(f"  Data padi       : {df_padi.shape[0]} baris × {df_padi.shape[1]} kolom")
print(f"  Provinsi padi   : {df_padi['Provinsi'].nunique()} provinsi")
print(f"  Tahun padi      : {df_padi['Tahun'].min()} – {df_padi['Tahun'].max()}")

df_cuaca = pd.read_csv(DATA_CUACA, encoding="utf-8-sig")
df_cuaca = df_cuaca.dropna(how="all").copy()

df_cuaca["datetime_parsed"] = pd.to_datetime(df_cuaca["datetime"], errors="coerce")
if df_cuaca["datetime_parsed"].isna().mean() > 0.5:
    df_cuaca["datetime_parsed"] = pd.to_datetime(
        df_cuaca["datetime"], format="%m/%d/%Y", errors="coerce"
    )

df_cuaca = df_cuaca.dropna(subset=["datetime_parsed", "name"]).copy()
df_cuaca["Tahun"]    = df_cuaca["datetime_parsed"].dt.year.astype(int)
df_cuaca["Provinsi"] = normalize_provinsi(df_cuaca["name"])

df_cuaca = df_cuaca[
    (df_cuaca["Tahun"] >= TAHUN_MIN_DATA) &
    (df_cuaca["Tahun"] <= TAHUN_MAX_DATA)
].copy()

kolom_cuaca_numerik = [
    "temp", "humidity", "precip", "tempmax", "tempmin", "feelslike",
    "feelslikemax", "feelslikemin", "windspeed", "windgust", "winddir",
    "solarradiation", "solarenergy", "uvindex", "dew", "sealevelpressure",
    "cloudcover", "visibility", "precipprob", "precipcover", "moonphase",
]
for col in kolom_cuaca_numerik:
    if col in df_cuaca.columns:
        df_cuaca[col] = pd.to_numeric(df_cuaca[col], errors="coerce")

if "temp" in df_cuaca.columns:
    df_cuaca = df_cuaca[(df_cuaca["temp"].isna()) | (df_cuaca["temp"] < 50)].copy()

if {"sunrise", "sunset"}.issubset(df_cuaca.columns):
    df_cuaca["sunrise_dt"]      = pd.to_datetime(df_cuaca["sunrise"], errors="coerce")
    df_cuaca["sunset_dt"]       = pd.to_datetime(df_cuaca["sunset"],  errors="coerce")
    df_cuaca["daylight_minutes"] = (
        df_cuaca["sunset_dt"] - df_cuaca["sunrise_dt"]
    ).dt.total_seconds() / 60
else:
    df_cuaca["daylight_minutes"] = np.nan

le_icon = LabelEncoder()
if "icon" in df_cuaca.columns:
    df_cuaca["icon_encoded"] = le_icon.fit_transform(df_cuaca["icon"].fillna("unknown"))
    print("\n  Mapping icon encoding:")
    for cls, code in zip(le_icon.classes_, le_icon.transform(le_icon.classes_)):
        print(f"    {cls:<30} → {code}")
else:
    df_cuaca["icon_encoded"] = 0
    print("\n    Kolom icon tidak ditemukan → icon_encoded dibuat 0")

if "conditions" in df_cuaca.columns:
    conditions_dummies = df_cuaca["conditions"].fillna("").str.get_dummies(sep=", ")
    df_cuaca           = pd.concat([df_cuaca, conditions_dummies], axis=1)
    condition_cols     = conditions_dummies.columns.tolist()
else:
    condition_cols = []

print(f"\n  Kondisi cuaca unik (conditions): {condition_cols}")

agg_dict = {
    "Curah_hujan":      ("precip",           "sum"),
    "Kelembapan":       ("humidity",         "mean"),
    "Suhu_rata_rata":   ("temp",             "mean"),
    "suhu_max_mean":    ("tempmax",          "mean"),
    "suhu_min_mean":    ("tempmin",          "mean"),
    "feelslike_mean":   ("feelslike",        "mean"),
    "feelslikemax_mean":("feelslikemax",     "mean"),
    "feelslikemin_mean":("feelslikemin",     "mean"),
    "windspeed_mean":   ("windspeed",        "mean"),
    "windgust_mean":    ("windgust",         "mean"),
    "winddir_mean":     ("winddir",          "mean"),
    "solar_mean":       ("solarradiation",   "mean"),
    "solarenergy_mean": ("solarenergy",      "mean"),
    "uvindex_mean":     ("uvindex",          "mean"),
    "dew_mean":         ("dew",              "mean"),
    "pressure_mean":    ("sealevelpressure", "mean"),
    "cloudcover_mean":  ("cloudcover",       "mean"),
    "visibility_mean":  ("visibility",       "mean"),
    "precipprob_mean":  ("precipprob",       "mean"),
    "precipcover_mean": ("precipcover",      "mean"),
    "moonphase_mean":   ("moonphase",        "mean"),
    "daylight_mean":    ("daylight_minutes", "mean"),
    "icon_encoded_mean":("icon_encoded",     "mean"),
}

agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df_cuaca.columns}

for col in condition_cols:
    agg_dict[f"cond_{col}"] = (col, "mean")

cuaca_yr = (
    df_cuaca.groupby(["Provinsi", "Tahun"])
    .agg(**agg_dict)
    .round(3)
    .reset_index()
)
cuaca_yr.rename(columns={
    "Curah_hujan":    "Curah hujan",
    "Kelembapan":     "Kelembapan",
    "Suhu_rata_rata": "Suhu rata-rata",
}, inplace=True)

print(f"\n  Data cuaca harian       : {df_cuaca.shape[0]} baris")
print(f"  Data cuaca tahunan      : {cuaca_yr.shape[0]} baris")
print(f"  Total fitur cuaca       : {cuaca_yr.shape[1] - 2} fitur")
print(f"  Provinsi cuaca tahunan  : {cuaca_yr['Provinsi'].nunique()} provinsi")
print(f"  Tahun cuaca             : {cuaca_yr['Tahun'].min()} – {cuaca_yr['Tahun'].max()}")

# ==============================================================================
# Kode Program Penggabungan Dataset Padi dan Cuaca
# ==============================================================================
df = df_padi.merge(cuaca_yr, on=["Provinsi", "Tahun"], how="inner")
df = df.sort_values(["Tahun", "Provinsi"]).reset_index(drop=True)

cek_merge   = df_padi.merge(cuaca_yr[["Provinsi", "Tahun"]], on=["Provinsi", "Tahun"], how="left", indicator=True)
tidak_match = cek_merge[cek_merge["_merge"] == "left_only"]
if not tidak_match.empty:
    tidak_match[["Provinsi", "Tahun"]].to_csv(
        os.path.join(VIS_DIR, "data_padi_tidak_punya_cuaca.csv"), index=False
    )

print(f"\n  Jumlah data hasil merge : {df.shape[0]} baris × {df.shape[1]} kolom")
print(f"  Provinsi hasil merge    : {df['Provinsi'].nunique()} provinsi")
print(f"  Rentang tahun merge     : {df['Tahun'].min()} – {df['Tahun'].max()}")
print(f"  Data padi tidak match   : {len(tidak_match)} baris")
if len(tidak_match) > 0:
    print("  CSV disimpan            : data_padi_tidak_punya_cuaca.csv")

print(f"\n  {'Kolom':<28} {'Missing':>8}  Status")
print(f"  {'-' * 46}")
for col, val in df.isnull().sum().items():
    print(f"  {col:<28} {val:>8}  {'OK' if val == 0 else 'MISSING'}")

df.to_csv(os.path.join(VIS_DIR, "dataset_hasil_merge_padi_cuaca.csv"), index=False)
print("\n  CSV disimpan: dataset_hasil_merge_padi_cuaca.csv")
# ==============================================================================
# Kode Program Preprocessing Data
# ==============================================================================
section("2. PREPROCESSING")

le = LabelEncoder()
df["Provinsi_Encoded"] = le.fit_transform(df["Provinsi"]) + 1

df["Provinsi_Baru"] = df["Provinsi"].isin(PROVINSI_BARU).astype(int)

print("\n  Mapping label encoding provinsi:")
print(f"  {'Provinsi':<28} {'Kode':>5} {'Prov_Baru':>10}")
print(f"  {'-' * 48}")
for prov, code in zip(le.classes_, le.transform(le.classes_) + 1):
    flag = 1 if prov in PROVINSI_BARU else 0
    print(f"  {prov:<28} {code:>5} {flag:>10}")

kelengkapan = (
    df.groupby("Provinsi")["Tahun"]
    .agg(
        Jumlah_Data="count", Jumlah_Tahun="nunique",
        Tahun_Min="min", Tahun_Max="max",
        Daftar_Tahun=lambda x: ", ".join(map(str, sorted(x.unique())))
    )
    .reset_index()
    .sort_values(["Jumlah_Tahun", "Provinsi"])
)
kelengkapan.to_csv(os.path.join(VIS_DIR, "cek_kelengkapan_tahun_per_provinsi.csv"), index=False)
print("\n  CSV disimpan: cek_kelengkapan_tahun_per_provinsi.csv")
print("\n  Provinsi dengan jumlah tahun paling sedikit:")
print(kelengkapan.head(10).to_string(index=False))

print("\n  [2A] Tidak ada penghapusan outlier — semua data dipertahankan")
df_clean  = df.copy()
n_iqr_out = 0
print(f"  Data awal: {df.shape[0]}  |  Dihapus: {n_iqr_out}  |  Semua data dipakai")

print("\n  [2B] Penghapusan data anomali (produktivitas < 0.5 atau > 15 ton/ha)")
df_clean["_prod_ratio"] = df_clean["Produksi"] / df_clean["Luas Panen"]

anomali = df_clean[
    (df_clean["_prod_ratio"] < 0.5) | (df_clean["_prod_ratio"] > 15.0)
][["Provinsi", "Tahun", "Produksi", "Luas Panen", "_prod_ratio"]].copy()
anomali.rename(columns={"_prod_ratio": "Produktivitas_ratio_ton_per_ha"}, inplace=True)

if not anomali.empty:
    anomali.to_csv(os.path.join(VIS_DIR, "data_anomali_produktivitas.csv"), index=False)
    print("  CSV disimpan: data_anomali_produktivitas.csv")
    print(anomali.to_string(index=False))
else:
    print("  Tidak ada anomali produktivitas ditemukan.")

df_clean   = df_clean[(df_clean["_prod_ratio"] >= 0.5) & (df_clean["_prod_ratio"] <= 15.0)]
df_clean   = df_clean.drop(columns=["_prod_ratio"])
n_anomali  = len(anomali)
df_clean   = df_clean.sort_values(["Tahun", "Provinsi"]).reset_index(drop=True)

for col in df_clean.select_dtypes(include=[np.number]).columns:
    if df_clean[col].isna().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

median_prod = df_clean["Produksi"].median()
print(f"\n  [2C] Kategori produksi  |  Median: {median_prod:,.0f} ton")

df_clean["Kategori_Produksi"] = df_clean["Produksi"].apply(
    lambda x: "Rendah" if x < median_prod else "Tinggi"
)
le_target = LabelEncoder()
df_clean["Kategori_Label"] = le_target.fit_transform(df_clean["Kategori_Produksi"])

for cat in sorted(df_clean["Kategori_Produksi"].unique()):
    n = (df_clean["Kategori_Produksi"] == cat).sum()
    print(f"  {cat:<12}: {n:>5} ({n / len(df_clean) * 100:.1f}%)")

print(f"\n  Awal: {df.shape[0]}  |  IQR: {n_iqr_out}  |  Anomali: {n_anomali}  |  Bersih: {df_clean.shape[0]}")

CUACA_FEATURES = [col for col in cuaca_yr.columns if col not in ["Provinsi", "Tahun"]]

df_clean["temp_range"]     = df_clean["suhu_max_mean"] - df_clean["suhu_min_mean"]
df_clean["hujan_per_hari"] = df_clean["Curah hujan"] / 365
df_clean["heat_index"]     = df_clean["Suhu rata-rata"] * df_clean["Kelembapan"] / 100

ALL_FEATURES = ["Provinsi_Encoded", "Provinsi_Baru",
                "temp_range", "hujan_per_hari", "heat_index"] + CUACA_FEATURES
TARGET_PRODUKTIVITAS = "Produktivitas"
TARGET_PRODUKSI      = "Produksi"

X_all        = df_clean[ALL_FEATURES].copy()
y_prod_all   = df_clean[TARGET_PRODUKTIVITAS].copy()
y_produksi_all = df_clean[TARGET_PRODUKSI].copy()
y_cls_all    = df_clean["Kategori_Label"].copy()

print(f"\n  Total ALL_FEATURES: {len(ALL_FEATURES)} fitur")
print(f"  Fitur: {ALL_FEATURES}")
print(f"\n  Target utama model : Produktivitas (ku/ha)")
print(f"  Validasi akhir     : Produksi Prediksi dibandingkan Produksi Aktual")
print(f"  Semua data dipakai : {len(df_clean)} baris")
print(
    f"\n  Statistik Produktivitas — mean: {y_prod_all.mean():,.2f}  "
    f"std: {y_prod_all.std():,.2f}  min: {y_prod_all.min():,.2f}  max: {y_prod_all.max():,.2f}"
)
print(
    f"  Statistik Produksi      — mean: {y_produksi_all.mean():,.0f}  "
    f"std: {y_produksi_all.std():,.0f}  min: {y_produksi_all.min():,.0f}  max: {y_produksi_all.max():,.0f}"
)

# ==============================================================================
# Kode Program Pembagian Data Train dan Test
# ==============================================================================
section("3. PEMBAGIAN DATA TRAIN-TEST 69:31 BERDASARKAN TAHUN")
tahun_urut     = sorted(df_clean["Tahun"].unique())
n_tahun        = len(tahun_urut)
n_train_tahun  = max(1, int(np.floor(n_tahun * 0.8)))
n_test_tahun   = n_tahun - n_train_tahun

tahun_train    = tahun_urut[:n_train_tahun]
tahun_test     = tahun_urut[n_train_tahun:]

TEST_YEAR_START = tahun_test[0]
TEST_YEAR_END   = tahun_test[-1]
TEST_YEAR       = TEST_YEAR_END

train_mask = df_clean["Tahun"].isin(tahun_train)
test_mask  = df_clean["Tahun"].isin(tahun_test)

X_train_all = X_all.loc[train_mask].copy()
X_test_all  = X_all.loc[test_mask].copy()
y_train     = y_prod_all.loc[train_mask].copy()
y_test      = y_prod_all.loc[test_mask].copy()

df_train = df_clean.loc[train_mask].copy()
df_test  = df_clean.loc[test_mask].copy()

print(f"\n  Total tahun         : {n_tahun} tahun ({tahun_urut[0]} – {tahun_urut[-1]})")
print(f"  Proporsi train      : 80% → {n_train_tahun} tahun")
print(f"  Proporsi test       : 20% → {n_test_tahun} tahun")
print(f"\n  Tahun train         : {tahun_train[0]} – {tahun_train[-1]}")
print(f"  Tahun test          : {tahun_test[0]} – {tahun_test[-1]}")
print(f"\n  Data train          : {len(df_train)} baris")
print(f"  Data test           : {len(df_test)} baris")

if len(df_train) < 20:
    print("\n  [PERINGATAN] Data train sangat sedikit. Hasil model mungkin kurang stabil.")

# ==============================================================================
# Kode Program Seleksi Fitur Berbasis Feature Importance
# ==============================================================================
section("4. SELEKSI FITUR BERBASIS FEATURE IMPORTANCE (CV-Based)")

rf_init = RandomForestRegressor(
    n_estimators=200, max_depth=6, min_samples_leaf=2,
    random_state=42, n_jobs=SKLEARN_N_JOBS,
)
rf_init.fit(X_train_all, y_train)

feat_imp = (
    pd.DataFrame({"Fitur": ALL_FEATURES, "Importance": rf_init.feature_importances_})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
feat_imp["Persen"] = feat_imp["Importance"] * 100
feat_imp.to_csv(os.path.join(VIS_DIR, "feature_importance_awal.csv"), index=False)

print(f"\n  {'Rank':<5} {'Fitur':<34} {'Importance':>12} {'Persen':>10}")
print(f"  {'-' * 67}")
for i, row in feat_imp.iterrows():
    print(f"  {i+1:<5} {row['Fitur']:<34} {row['Importance']:>12.6f} {row['Persen']:>9.4f}%")

cv_splits = min(N_SPLITS, max(2, len(X_train_all) // 20))
cv_splits = min(cv_splits, len(X_train_all) - 1)
cv_ts     = TimeSeriesSplit(n_splits=cv_splits)

cv_results = []
for n_features in range(2, min(len(ALL_FEATURES) + 1, 21)):
    top_features = feat_imp.head(n_features)["Fitur"].tolist()
    X_temp = X_train_all[top_features]
    try:
        scores = cross_val_score(
            RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=SKLEARN_N_JOBS),
            X_temp, y_train, cv=cv_ts, scoring="r2", n_jobs=SKLEARN_N_JOBS,
        )
        score_mean = scores.mean()
    except Exception:
        score_mean = np.nan
    cv_results.append({"n_features": n_features, "cv_r2": score_mean})

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(os.path.join(VIS_DIR, "cv_jumlah_fitur.csv"), index=False)

best_n_fs = (
    int(cv_results_df.loc[cv_results_df["cv_r2"].idxmax(), "n_features"])
    if cv_results_df["cv_r2"].notna().any()
    else 2
)

FEATURES_SEL_AUTO  = feat_imp.head(best_n_fs)["Fitur"].tolist()
FEATURES_THRESHOLD = feat_imp[feat_imp["Importance"] >= THRESHOLD]["Fitur"].tolist()
FITUR_WAJIB        = [f for f in ["Curah hujan", "Kelembapan", "Suhu rata-rata"] if f in ALL_FEATURES]

FEATURES_SEL = list(dict.fromkeys(FITUR_WAJIB + FEATURES_THRESHOLD + FEATURES_SEL_AUTO))

if len(FEATURES_SEL) < MIN_FEATURES:
    FEATURES_SEL = list(dict.fromkeys(FEATURES_SEL + feat_imp.head(MIN_FEATURES)["Fitur"].tolist()))

FEATURES_DROP = [f for f in ALL_FEATURES if f not in FEATURES_SEL]

print(f"\n  Jumlah fitur optimal dari CV : {best_n_fs}")
print(f"  Threshold Importance        : {THRESHOLD} ({THRESHOLD*100:.3f}%)")
print(f"  Minimal fitur digunakan     : {MIN_FEATURES}")
print(f"  Jumlah fitur sebelum        : {len(ALL_FEATURES)}")
print(f"  Jumlah fitur sesudah        : {len(FEATURES_SEL)}")

print(f"\n  Fitur otomatis dari CV ({len(FEATURES_SEL_AUTO)}):")
for i, f in enumerate(FEATURES_SEL_AUTO, 1):
    print(f"    {i}. {f}")

print(f"\n  Fitur lolos threshold ({len(FEATURES_THRESHOLD)}):")
for i, f in enumerate(FEATURES_THRESHOLD, 1):
    imp_val = feat_imp.loc[feat_imp["Fitur"] == f, "Importance"].values[0]
    print(f"    {i}. {f:<34} importance={imp_val:.6f} ({imp_val*100:.4f}%)")

print(f"\n  Fitur wajib penelitian ({len(FITUR_WAJIB)}):")
for i, f in enumerate(FITUR_WAJIB, 1):
    print(f"    {i}. {f}")

print(f"\n  Fitur final terpilih ({len(FEATURES_SEL)}):")
for i, f in enumerate(FEATURES_SEL, 1):
    imp_arr = feat_imp.loc[feat_imp["Fitur"] == f, "Importance"].values
    imp_val = imp_arr[0] if len(imp_arr) else 0
    print(f"    {i}. {f:<34} importance={imp_val:.6f} ({imp_val*100:.4f}%)")

print(f"\n  Fitur dibuang ({len(FEATURES_DROP)}):")
for f in FEATURES_DROP:
    print(f"    - {f}")

plt.figure(figsize=(10, 5))
plt.plot(cv_results_df["n_features"], cv_results_df["cv_r2"], marker="o", color="steelblue")
if cv_results_df["cv_r2"].notna().any():
    best_cv_r2_fs = cv_results_df.loc[cv_results_df["cv_r2"].idxmax(), "cv_r2"]
    plt.axvline(
        x=best_n_fs, color="red", linestyle="--",
        label=f"Optimal: {best_n_fs} fitur (CV R²={best_cv_r2_fs:.4f})",
    )
plt.xlabel("Jumlah Fitur")
plt.ylabel("CV R² Produktivitas")
plt.title("Pemilihan Jumlah Fitur Optimal (CV-Based)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "1b_optimal_n_features.png"), dpi=200, bbox_inches="tight")
plt.close()
print("\n  Gambar disimpan: 1b_optimal_n_features.png")

plt.figure(figsize=(10, max(5, len(FEATURES_SEL) * 0.35)))
fi_sel = feat_imp[feat_imp["Fitur"].isin(FEATURES_SEL)].sort_values("Importance")
plt.barh(fi_sel["Fitur"], fi_sel["Importance"] * 100, color="steelblue")
plt.xlabel("Importance (%)")
plt.title("Feature Importance — Fitur Terpilih")
plt.grid(True, axis="x", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "feature_importance_terpilih.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Gambar disimpan: feature_importance_terpilih.png")

# ==============================================================================
# Kode Program Standardisasi Data
# ==============================================================================
X_train        = X_train_all[FEATURES_SEL].copy()
X_test         = X_test_all[FEATURES_SEL].copy()
X_selected_all = X_all[FEATURES_SEL].copy()

scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=FEATURES_SEL,
    index=X_train.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=FEATURES_SEL,
    index=X_test.index
)
X_selected_all = pd.DataFrame(
    scaler.transform(X_selected_all),
    columns=FEATURES_SEL,
    index=X_selected_all.index
)

print(f"\n  [StandardScaler] Fitur dinormalisasi — mean=0, std=1")
print(f"  X_train setelah seleksi dan scaling: {X_train.shape}")
print(f"  X_test setelah seleksi dan scaling  : {X_test.shape}")
# ==============================================================================
# Kode Program Baseline Linear Regression
# ==============================================================================
section("5. BASELINE MODEL — LINEAR REGRESSION")

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr_test = lr.predict(X_test)

prod_pred_lr_test = y_pred_lr_test * df_test["Luas Panen"].values / 10
m_lr_prod = regression_metrics(
    df_test["Produksi"].values, prod_pred_lr_test,
    "Linear Regression - Produksi Test"
)

print("\n  Evaluasi baseline berdasarkan Produksi Test:")
print(f"  R²      : {m_lr_prod['R2']*100:.2f}%")
print(f"  RMSE    : {m_lr_prod['RMSE']:,.2f} ton")
print(f"  MAE     : {m_lr_prod['MAE']:,.2f} ton")
print(f"  MAPE    : {m_lr_prod['MAPE']:.2f}%")
print(f"  SMAPE   : {m_lr_prod['SMAPE']:.2f}%")
# ==============================================================================
# Kode Program Optimasi Hyperparameter GridSearchCV
# ==============================================================================
section("6. GRIDSEARCHCV — OPTIMASI HYPERPARAMETER")

param_grid = {
    "n_estimators":          [100, 200, 300],
    "max_depth":             [4, 6, 8, None],
    "min_samples_split":     [2, 5, 10],
    "min_samples_leaf":      [1, 2, 4],
    "max_features":          ["sqrt", 0.3, 0.5],
    "min_impurity_decrease": [0.0, 0.0001],
    "bootstrap":             [True],
}

total_comb = 1
for values in param_grid.values():
    total_comb *= len(values)

print(f"\n  Total kombinasi : {total_comb} × {cv_splits} folds = {total_comb * cv_splits} iterasi")
print(f"  Folder temp joblib: {TEMP_DIR}")
print(f"  n_jobs yang dipakai: {SKLEARN_N_JOBS}")
print("  [INFO] GridSearchCV dilakukan pada target Produktivitas.")
print("  [INFO] Produksi prediksi dihitung setelah produktivitas diprediksi.")
print("\n  Proses GridSearchCV berjalan.\n")

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=SKLEARN_N_JOBS),
    param_grid,
    cv=cv_ts,
    n_jobs=SKLEARN_N_JOBS,
    scoring="neg_mean_absolute_error",
    verbose=1,
    return_train_score=True,
    pre_dispatch=1,
)
grid_search.fit(X_train, y_train)

best_model       = grid_search.best_estimator_
best_params      = grid_search.best_params_
best_cv_mae_prod = -grid_search.best_score_
best_idx         = grid_search.best_index_
train_score      = grid_search.cv_results_["mean_train_score"][best_idx]
test_score       = grid_search.cv_results_["mean_test_score"][best_idx]

print(f"\n  {'Parameter':<24} {'Nilai':>12}")
print(f"  {'-' * 40}")
for p, v in best_params.items():
    print(f"  {p:<24} {str(v):>12}")
print(f"\n  Best CV MAE Produktivitas   : {best_cv_mae_prod:,.4f} ku/ha")
print(f"  Mean Train MAE Produktivitas: {-train_score:,.4f} ku/ha")
print(f"  Mean CV MAE Produktivitas   : {-test_score:,.4f} ku/ha")

pd.DataFrame(grid_search.cv_results_).to_csv(os.path.join(VIS_DIR, "gridsearchcv_results.csv"), index=False)
print("  CSV disimpan: gridsearchcv_results.csv")

# ==============================================================================
# Kode Program Evaluasi Model Regresi
# ==============================================================================
section("7. EVALUASI MODEL REGRESI — VALIDASI PRODUKSI")
best_model.fit(X_train, y_train)
y_pred_prod_train = best_model.predict(X_train)
y_pred_prod_test  = best_model.predict(X_test)

produksi_pred_train = y_pred_prod_train * df_train["Luas Panen"].values / 10
produksi_pred_test  = y_pred_prod_test  * df_test["Luas Panen"].values / 10

m_prod_train    = regression_metrics(y_train.values, y_pred_prod_train, "Produktivitas Train")
m_prod_test     = regression_metrics(y_test.values, y_pred_prod_test, "Produktivitas Test")
m_produksi_train = regression_metrics(df_train["Produksi"].values, produksi_pred_train, "Produksi Train")
m_produksi_test  = regression_metrics(df_test["Produksi"].values, produksi_pred_test, "Produksi Test")

pd.DataFrame([m_prod_train, m_prod_test, m_produksi_train, m_produksi_test]).to_csv(
    os.path.join(VIS_DIR, "ringkasan_metrik_validasi.csv"), index=False
)

print("\n  METRIK PRODUKTIVITAS")
print("  " + "-" * 60)
print(f"  Train R²   : {m_prod_train['R2']*100:.2f}%")
print(f"  Test R²    : {m_prod_test['R2']*100:.2f}%")
print(f"  Train RMSE : {m_prod_train['RMSE']:.4f} ku/ha")
print(f"  Test RMSE  : {m_prod_test['RMSE']:.4f} ku/ha")
print(f"  Test MAE   : {m_prod_test['MAE']:.4f} ku/ha")
print(f"  Test MAPE  : {m_prod_test['MAPE']:.2f}%")
print(f"  Test SMAPE : {m_prod_test['SMAPE']:.2f}%")

print("\n  METRIK VALIDASI PRODUKSI")
print("  " + "-" * 60)
print(f"  Train R²   : {m_produksi_train['R2']*100:.2f}%  → {status_r2_text(m_produksi_train['R2'])}")
print(f"  Test R²    : {m_produksi_test['R2']*100:.2f}%  → {status_r2_text(m_produksi_test['R2'])}")
print(f"  Test RMSE  : {m_produksi_test['RMSE']:,.2f} ton")
print(f"  Test MAE   : {m_produksi_test['MAE']:,.2f} ton")
print(f"  Test MAPE  : {m_produksi_test['MAPE']:.2f}%")
print(f"  Test SMAPE : {m_produksi_test['SMAPE']:.2f}%")

r2          = m_produksi_test["R2"]
rmse        = m_produksi_test["RMSE"]
mae         = m_produksi_test["MAE"]
mse         = m_produksi_test["MSE"]
mape        = m_produksi_test["MAPE"]
smape_score = m_produksi_test["SMAPE"]
adj_r2      = adjusted_r2(r2, len(df_test), len(FEATURES_SEL))
status_r2   = status_r2_text(r2)
status_mape = (
    "Sangat Akurat" if mape < 10 else
    "Akurat" if mape < 20 else
    "Cukup" if mape < 50 else "Perlu Perhatian"
)

cv_r2_prod   = cross_val_score(best_model, X_train, y_train, cv=cv_ts, scoring="r2", n_jobs=SKLEARN_N_JOBS)
cv_rmse_prod = np.sqrt(-cross_val_score(best_model, X_train, y_train, cv=cv_ts, scoring="neg_mean_squared_error", n_jobs=SKLEARN_N_JOBS))

gap_cv = abs(m_prod_train["R2"] - cv_r2_prod.mean()) if not np.isnan(m_prod_train["R2"]) else np.nan
status_gap = (
    "Konsisten — model stabil" if gap_cv < 0.10 else
    "Sedikit gap — masih dapat diterima" if gap_cv < 0.25 else
    "Gap besar — potensi overfitting"
)

kf_cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_kf = cross_val_score(
    best_model, X_train, y_train,
    cv=kf_cv, scoring="r2", n_jobs=SKLEARN_N_JOBS
)
cv_mae_kf = -cross_val_score(
    best_model, X_train, y_train,
    cv=kf_cv, scoring="neg_mean_absolute_error", n_jobs=SKLEARN_N_JOBS
)
gap_cv_kf = abs(m_prod_train["R2"] - cv_r2_kf.mean()) if not np.isnan(m_prod_train["R2"]) else np.nan
status_gap_kf = (
    "Konsisten — model stabil"           if gap_cv_kf < 0.10 else
    "Sedikit gap — masih dapat diterima" if gap_cv_kf < 0.25 else
    "Gap besar — potensi overfitting"
)

line = "-" * 50
print(f"""
  RINGKASAN VALIDASI PRODUKSI TEST TAHUN {TEST_YEAR_START}–{TEST_YEAR_END}
  {line}
  R²                      : {r2*100:.2f}%  → {status_r2}
  Adjusted R²             : {adj_r2*100 if not np.isnan(adj_r2) else np.nan:.2f}%
  RMSE                    : {rmse:,.2f} ton
  MAE                     : {mae:,.2f} ton
  MSE                     : {mse:,.2f}
  MAPE                    : {mape:.2f}%  → {status_mape}
  SMAPE                   : {smape_score:.2f}%

  CROSS-VALIDATION PRODUKTIVITAS — PERBANDINGAN 2 METODE
  {line}
  [TimeSeriesSplit {cv_splits}-Fold — mempertahankan urutan waktu]
  CV R² Mean ± Std   : {cv_r2_prod.mean()*100:.2f}% ± {cv_r2_prod.std()*100:.2f}%
  CV R² Range        : {cv_r2_prod.min()*100:.2f}% – {cv_r2_prod.max()*100:.2f}%
  CV RMSE Mean       : {cv_rmse_prod.mean():,.4f} ku/ha
  Gap Train vs CV R² : {gap_cv*100:.2f}%  → {status_gap}

  [KFold 5-Fold — acak, tidak mempertahankan urutan waktu]
  CV R² Mean ± Std   : {cv_r2_kf.mean()*100:.2f}% ± {cv_r2_kf.std()*100:.2f}%
  CV R² Range        : {cv_r2_kf.min()*100:.2f}% – {cv_r2_kf.max()*100:.2f}%
  CV MAE Mean        : {cv_mae_kf.mean():,.4f} ku/ha
  Gap Train vs CV R² : {gap_cv_kf*100:.2f}%  → {status_gap_kf}

  INTERPRETASI:
  TimeSeriesSplit mensimulasikan prediksi ke depan (out-of-sample temporal).
  KFold digunakan sebagai pembanding untuk melihat kestabilan pola antarprovinsi.
  Jika TimeSeriesSplit lebih rendah daripada KFold, model lebih kuat menangkap
  pola spasial antarprovinsi dibandingkan perubahan lintas waktu.
  {line}
""")

print("\n  Perbandingan dengan baseline Linear Regression berdasarkan Produksi Test:")
print(f"  {'Metrik':<20} {'Lin.Reg':>15} {'Rand.Forest':>15}")
print(f"  {'-' * 50}")
print(f"  {'R²':<20} {m_lr_prod['R2']*100:>14.2f}% {r2*100:>14.2f}%")
print(f"  {'RMSE':<20} {m_lr_prod['RMSE']:>15,.0f} {rmse:>15,.0f}")
print(f"  {'MAE':<20} {m_lr_prod['MAE']:>15,.0f} {mae:>15,.0f}")
print(f"  {'MAPE':<20} {m_lr_prod['MAPE']:>14.2f}% {mape:>14.2f}%")
print(f"  {'SMAPE':<20} {m_lr_prod['SMAPE']:>14.2f}% {smape_score:>14.2f}%")
# ==============================================================================
# Kode Program Evaluasi Model Klasifikasi
# ==============================================================================
section("8. EVALUASI KLASIFIKASI — 4 MODEL BERTAHAP")

y_cls_train = df_train["Kategori_Label"]
y_cls_test  = df_test["Kategori_Label"]

skf_clf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def get_prob_positive(model, X_data):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_data)
        if proba.shape[1] > 1:
            return proba[:, 1]
        return proba[:, 0]
    return model.predict(X_data)

def evaluasi_klasifikasi(
    nama_model,
    model,
    X_tr,
    X_te,
    y_tr,
    y_te,
    cmap,
    nama_file_cm,
    cv_obj
):
    y_pred_test  = model.predict(X_te)
    y_pred_train = model.predict(X_tr)
    y_prob_test  = get_prob_positive(model, X_te)

    acc = accuracy_score(y_te, y_pred_test)
    bal = balanced_accuracy_score(y_te, y_pred_test)
    kap = cohen_kappa_score(y_te, y_pred_test)
    mcc_score = matthews_corrcoef(y_te, y_pred_test)

    train_acc = accuracy_score(y_tr, y_pred_train)
    test_acc  = acc
    gap       = train_acc - test_acc

    cv_acc_scores = cross_val_score(
        model,
        X_tr,
        y_tr,
        cv=cv_obj,
        scoring="accuracy",
        n_jobs=SKLEARN_N_JOBS
    )

    cv_bal_scores = cross_val_score(
        model,
        X_tr,
        y_tr,
        cv=cv_obj,
        scoring="balanced_accuracy",
        n_jobs=SKLEARN_N_JOBS
    )

    try:
        auc_score = roc_auc_score(y_te, y_prob_test)
    except Exception:
        auc_score = np.nan

    print(f"\n  === {nama_model} ===")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Balanced Accuracy : {bal*100:.2f}%")
    print(f"  Cohen Kappa       : {kap:.4f}")
    print(f"  MCC               : {mcc_score:.4f}")
    print(f"  ROC-AUC           : {auc_score:.4f}")

    print(f"\n  Train Accuracy : {train_acc*100:.2f}%")
    print(f"  Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  Gap Train-Test : {gap*100:.2f}%")
    print(f"  CV Accuracy    : {cv_acc_scores.mean()*100:.2f}% ± {cv_acc_scores.std()*100:.2f}%")
    print(f"  CV Balanced    : {cv_bal_scores.mean()*100:.2f}% ± {cv_bal_scores.std()*100:.2f}%")

    print(f"\n  Classification Report — {nama_model}:")
    print_classification_report_persen(y_te, y_pred_test, le_target.classes_)

    cm = confusion_matrix(
        y_te,
        y_pred_test,
        labels=list(range(len(le_target.classes_)))
    )

    print(f"\n  Confusion Matrix {nama_model}:")
    print(f"  {cm}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le_target.classes_
    ).plot(
        cmap=cmap,
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"Confusion Matrix {nama_model}\nAccuracy={acc:.2%}",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, nama_file_cm), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Gambar disimpan: {nama_file_cm}")

    return {
        "Nama Model": nama_model,
        "Model": model,
        "y_pred": y_pred_test,
        "y_prob": y_prob_test,
        "Accuracy": acc,
        "Balanced Accuracy": bal,
        "Kappa": kap,
        "MCC": mcc_score,
        "ROC-AUC": auc_score,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Gap": gap,
        "CV Accuracy Scores": cv_acc_scores,
        "CV Balanced Scores": cv_bal_scores,
        "Confusion Matrix": cm
    }

m1_baseline = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=SKLEARN_N_JOBS
)

m1_baseline.fit(X_train, y_cls_train)

hasil_rm1 = evaluasi_klasifikasi(
    nama_model="RM1 BASELINE RANDOM FOREST",
    model=m1_baseline,
    X_tr=X_train,
    X_te=X_test,
    y_tr=y_cls_train,
    y_te=y_cls_test,
    cmap="Blues",
    nama_file_cm="cm_rm1_baseline.png",
    cv_obj=skf_clf
)

fi_clf = pd.DataFrame({
    "Fitur": FEATURES_SEL,
    "Importance": m1_baseline.feature_importances_,
}).sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

cv_fs_scores = []

for n in range(5, len(FEATURES_SEL) + 1):
    top_f = fi_clf.head(n)["Fitur"].tolist()

    score = cross_val_score(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=SKLEARN_N_JOBS
        ),
        X_train[top_f],
        y_cls_train,
        cv=skf_clf,
        scoring="accuracy",
        n_jobs=SKLEARN_N_JOBS
    ).mean()

    cv_fs_scores.append({
        "n_features": n,
        "cv_accuracy": score
    })

cv_fs_df = pd.DataFrame(cv_fs_scores)
cv_fs_df.to_csv(
    os.path.join(VIS_DIR, "cv_jumlah_fitur_klasifikasi.csv"),
    index=False
)

best_n_clf_cv = int(
    cv_fs_df.loc[
        cv_fs_df["cv_accuracy"].idxmax(),
        "n_features"
    ]
)

best_cv_acc_clf = float(
    cv_fs_df.loc[
        cv_fs_df["cv_accuracy"].idxmax(),
        "cv_accuracy"
    ]
)

MIN_FEATURES_CLF = 8
best_n_clf_pakai = max(best_n_clf_cv, MIN_FEATURES_CLF)

FEATURES_CLF = fi_clf.head(best_n_clf_pakai)["Fitur"].tolist()

FITUR_WAJIB_KLASIFIKASI = [
    "Curah hujan",
    "Kelembapan",
    "Suhu rata-rata"
]

for fw in FITUR_WAJIB_KLASIFIKASI:
    if fw in FEATURES_SEL and fw not in FEATURES_CLF:
        FEATURES_CLF.append(fw)

FEATURES_CLF = list(dict.fromkeys(FEATURES_CLF))
best_n_clf = len(FEATURES_CLF)

print(f"\n  Jumlah fitur terbaik menurut CV klasifikasi : {best_n_clf_cv}")
print(f"  CV Accuracy terbaik klasifikasi             : {best_cv_acc_clf*100:.2f}%")
print(f"  Minimal fitur klasifikasi yang dipakai      : {MIN_FEATURES_CLF}")
print(f"  Jumlah fitur berdasarkan batas minimal      : {best_n_clf_pakai}")
print(f"  Jumlah fitur sebelum seleksi klasifikasi    : {len(FEATURES_SEL)}")
print(f"  Jumlah fitur setelah seleksi klasifikasi    : {best_n_clf}")
print(f"  Fitur terpilih                              : {FEATURES_CLF}")

print(f"\n  {'Rank':<5} {'Fitur':<34} {'Importance':>12} {'Persen':>10}")
print(f"  {'-' * 65}")

for i, row in fi_clf.iterrows():
    print(
        f"  {i+1:<5} {row['Fitur']:<34} "
        f"{row['Importance']:>12.6f} {row['Importance']*100:>9.4f}%"
    )

plt.figure(figsize=(10, 5))
plt.plot(
    cv_fs_df["n_features"],
    cv_fs_df["cv_accuracy"],
    marker="o",
    color="steelblue"
)

plt.axvline(
    x=best_n_clf_cv,
    color="red",
    linestyle="--",
    label=f"Optimal CV: {best_n_clf_cv} fitur (Acc={best_cv_acc_clf:.4f})"
)

plt.xlabel("Jumlah Fitur")
plt.ylabel("CV Accuracy")
plt.title("Pemilihan Jumlah Fitur Optimal — Klasifikasi")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(VIS_DIR, "optimal_n_features_clf.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()
print("  Gambar disimpan: optimal_n_features_clf.png")

X_train_clf = X_train[FEATURES_CLF].copy()
X_test_clf  = X_test[FEATURES_CLF].copy()

print(f"\n  Shape X_train_clf : {X_train_clf.shape}")
print(f"  Shape X_test_clf  : {X_test_clf.shape}")

m2_fs = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=SKLEARN_N_JOBS
)

m2_fs.fit(X_train_clf, y_cls_train)
print("\n  Model RM2 RF + Feature Selection berhasil dibuat")

hasil_rm2 = evaluasi_klasifikasi(
    nama_model="RM2 RANDOM FOREST + FEATURE SELECTION",
    model=m2_fs,
    X_tr=X_train_clf,
    X_te=X_test_clf,
    y_tr=y_cls_train,
    y_te=y_cls_test,
    cmap="Greens",
    nama_file_cm="cm_rm2_feature_selection.png",
    cv_obj=skf_clf
)

param_grid_clf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "criterion": ["gini"],
    "class_weight": ["balanced"]
}

gs3 = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=SKLEARN_N_JOBS
    ),
    param_grid=param_grid_clf,
    cv=skf_clf,
    scoring="accuracy",
    n_jobs=SKLEARN_N_JOBS,
    verbose=1,
    return_train_score=True
)

gs3.fit(X_train, y_cls_train)

pd.DataFrame(gs3.cv_results_).to_csv(
    os.path.join(VIS_DIR, "gridsearchcv_classification_rm3.csv"),
    index=False
)

print(f"\n  Best Parameters RM3 : {gs3.best_params_}")
print(f"  Best CV Score RM3   : {gs3.best_score_*100:.2f}%")

m3_grid = gs3.best_estimator_
print("\n  Model RM3 RF + GridSearchCV berhasil dibuat")

hasil_rm3 = evaluasi_klasifikasi(
    nama_model="RM3 RANDOM FOREST + GRIDSEARCHCV",
    model=m3_grid,
    X_tr=X_train,
    X_te=X_test,
    y_tr=y_cls_train,
    y_te=y_cls_test,
    cmap="Oranges",
    nama_file_cm="cm_rm3_gridsearchcv.png",
    cv_obj=skf_clf
)

print(f"\n  Shape X_train_clf untuk RM4 : {X_train_clf.shape}")
print(f"  Shape X_test_clf untuk RM4  : {X_test_clf.shape}")
print(f"  Fitur RM4                  : {FEATURES_CLF}")

gs4 = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=SKLEARN_N_JOBS
    ),
    param_grid=param_grid_clf,
    cv=skf_clf,
    scoring="accuracy",
    n_jobs=SKLEARN_N_JOBS,
    verbose=1,
    return_train_score=True
)

gs4.fit(X_train_clf, y_cls_train)

pd.DataFrame(gs4.cv_results_).to_csv(
    os.path.join(VIS_DIR, "gridsearchcv_classification_rm4.csv"),
    index=False
)

print(f"\n  Best Parameters RM4 : {gs4.best_params_}")
print(f"  Best CV Score RM4   : {gs4.best_score_*100:.2f}%")

m4_final = gs4.best_estimator_
print("\n  Model RM4 RF + Feature Selection + GridSearchCV berhasil dibuat")

hasil_rm4 = evaluasi_klasifikasi(
    nama_model="RM4 RF + FEATURE SELECTION + GRIDSEARCHCV",
    model=m4_final,
    X_tr=X_train_clf,
    X_te=X_test_clf,
    y_tr=y_cls_train,
    y_te=y_cls_test,
    cmap="Purples",
    nama_file_cm="confusion_matrix_test.png",
    cv_obj=skf_clf
)

report_df = pd.DataFrame(
    classification_report(
        y_cls_test,
        hasil_rm4["y_pred"],
        labels=list(range(len(le_target.classes_))),
        target_names=le_target.classes_,
        output_dict=True,
        zero_division=0
    )
).transpose()

for col in ["precision", "recall", "f1-score"]:
    if col in report_df.columns:
        report_df[col] = (report_df[col] * 100).round(2)

report_df.to_csv(
    os.path.join(VIS_DIR, "classification_report_test.csv")
)
print("  CSV disimpan: classification_report_test.csv")

# ==============================================================================
# Kode Program Perbandingan Empat Model
# ==============================================================================
comparison_df = pd.DataFrame({
    "Model": [
        "RM1 Baseline RF",
        "RM2 RF + Feature Selection",
        "RM3 RF + GridSearchCV",
        "RM4 RF + FS + GridSearchCV"
    ],
    "Jumlah Fitur": [
        len(FEATURES_SEL),
        len(FEATURES_CLF),
        len(FEATURES_SEL),
        len(FEATURES_CLF)
    ],
    "Accuracy": [
        round(hasil_rm1["Accuracy"], 4),
        round(hasil_rm2["Accuracy"], 4),
        round(hasil_rm3["Accuracy"], 4),
        round(hasil_rm4["Accuracy"], 4)
    ],
    "Balanced Acc": [
        round(hasil_rm1["Balanced Accuracy"], 4),
        round(hasil_rm2["Balanced Accuracy"], 4),
        round(hasil_rm3["Balanced Accuracy"], 4),
        round(hasil_rm4["Balanced Accuracy"], 4)
    ],
    "Kappa": [
        round(hasil_rm1["Kappa"], 4),
        round(hasil_rm2["Kappa"], 4),
        round(hasil_rm3["Kappa"], 4),
        round(hasil_rm4["Kappa"], 4)
    ],
    "MCC": [
        round(hasil_rm1["MCC"], 4),
        round(hasil_rm2["MCC"], 4),
        round(hasil_rm3["MCC"], 4),
        round(hasil_rm4["MCC"], 4)
    ],
    "ROC-AUC": [
        round(hasil_rm1["ROC-AUC"], 4),
        round(hasil_rm2["ROC-AUC"], 4),
        round(hasil_rm3["ROC-AUC"], 4),
        round(hasil_rm4["ROC-AUC"], 4)
    ],
    "CV Accuracy": [
        round(hasil_rm1["CV Accuracy Scores"].mean(), 4),
        round(hasil_rm2["CV Accuracy Scores"].mean(), 4),
        round(hasil_rm3["CV Accuracy Scores"].mean(), 4),
        round(hasil_rm4["CV Accuracy Scores"].mean(), 4)
    ],
})

comparison_df.to_csv(
    os.path.join(VIS_DIR, "perbandingan_model_klasifikasi.csv"),
    index=False
)

print(f"\n  TABEL PERBANDINGAN 4 MODEL:")
print(f"  {'─'*100}")
print(comparison_df.to_string(index=False))
print(f"  {'─'*100}")
print("  CSV disimpan: perbandingan_model_klasifikasi.csv")

overfitting_df = pd.DataFrame({
    "Model": [
        "RM1 Baseline RF",
        "RM2 RF + Feature Selection",
        "RM3 RF + GridSearchCV",
        "RM4 RF + FS + GridSearchCV"
    ],
    "Train Accuracy": [
        round(hasil_rm1["Train Accuracy"], 4),
        round(hasil_rm2["Train Accuracy"], 4),
        round(hasil_rm3["Train Accuracy"], 4),
        round(hasil_rm4["Train Accuracy"], 4)
    ],
    "Test Accuracy": [
        round(hasil_rm1["Test Accuracy"], 4),
        round(hasil_rm2["Test Accuracy"], 4),
        round(hasil_rm3["Test Accuracy"], 4),
        round(hasil_rm4["Test Accuracy"], 4)
    ],
    "Gap": [
        round(hasil_rm1["Gap"], 4),
        round(hasil_rm2["Gap"], 4),
        round(hasil_rm3["Gap"], 4),
        round(hasil_rm4["Gap"], 4)
    ],
})

overfitting_df.to_csv(
    os.path.join(VIS_DIR, "gap_train_test_klasifikasi.csv"),
    index=False
)

print(f"\n  REKAP GAP TRAIN-TEST:")
print(f"  {'─'*70}")
print(overfitting_df.to_string(index=False))
print(f"  {'─'*70}")

labels_short = [
    "RM1\nBaseline",
    "RM2\nFS",
    "RM3\nGridCV",
    "RM4\nFS+GridCV"
]

colors_bar = [
    "steelblue",
    "darkorange",
    "seagreen",
    "crimson"
]


def plot_bar_model(vals, ylabel, title, filename, ylim_bottom=None, ylim_top=1.02):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        labels_short,
        vals,
        color=colors_bar,
        edgecolor="white",
        width=0.5
    )

    for bar, v in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    if ylim_bottom is None:
        ylim_bottom = max(0, min(vals) - 0.05)

    plt.ylim(ylim_bottom, ylim_top)
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Gambar disimpan: {filename}")


plot_bar_model(
    [
        hasil_rm1["Accuracy"],
        hasil_rm2["Accuracy"],
        hasil_rm3["Accuracy"],
        hasil_rm4["Accuracy"]
    ],
    "Accuracy",
    "Perbandingan Accuracy Antar Model Klasifikasi",
    "perbandingan_accuracy_model.png"
)

plot_bar_model(
    [
        hasil_rm1["Balanced Accuracy"],
        hasil_rm2["Balanced Accuracy"],
        hasil_rm3["Balanced Accuracy"],
        hasil_rm4["Balanced Accuracy"]
    ],
    "Balanced Accuracy",
    "Perbandingan Balanced Accuracy Antar Model",
    "perbandingan_balanced_acc_model.png"
)

plot_bar_model(
    [
        hasil_rm1["ROC-AUC"],
        hasil_rm2["ROC-AUC"],
        hasil_rm3["ROC-AUC"],
        hasil_rm4["ROC-AUC"]
    ],
    "ROC-AUC",
    "Perbandingan ROC-AUC Antar Model",
    "perbandingan_roc_auc_model.png"
)

fitur_vals = [
    len(FEATURES_SEL),
    len(FEATURES_CLF),
    len(FEATURES_SEL),
    len(FEATURES_CLF)
]

plt.figure(figsize=(8, 5))
bars = plt.bar(
    labels_short,
    fitur_vals,
    color=colors_bar,
    edgecolor="white",
    width=0.5
)

for bar, v in zip(bars, fitur_vals):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        str(v),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

plt.ylabel("Jumlah Fitur")
plt.title("Jumlah Fitur yang Digunakan Tiap Model", fontweight="bold")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(VIS_DIR, "perbandingan_jumlah_fitur_model.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()
print("  Gambar disimpan: perbandingan_jumlah_fitur_model.png")

gap_vals = [
    hasil_rm1["Gap"],
    hasil_rm2["Gap"],
    hasil_rm3["Gap"],
    hasil_rm4["Gap"]
]

plt.figure(figsize=(8, 5))
bars = plt.bar(
    labels_short,
    gap_vals,
    color=colors_bar,
    edgecolor="white",
    width=0.5
)

for bar, v in zip(bars, gap_vals):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{v:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.ylabel("Gap Train-Test Accuracy")
plt.title("Gap Train-Test Accuracy Antar Model", fontweight="bold")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(VIS_DIR, "gap_train_test_model.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()
print("  Gambar disimpan: gap_train_test_model.png")

fpr_b,  tpr_b,  _ = roc_curve(y_cls_test, hasil_rm1["y_prob"])
fpr_fs, tpr_fs, _ = roc_curve(y_cls_test, hasil_rm2["y_prob"])
fpr_g,  tpr_g,  _ = roc_curve(y_cls_test, hasil_rm3["y_prob"])
fpr_f,  tpr_f,  _ = roc_curve(y_cls_test, hasil_rm4["y_prob"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

curves = [
    (
        fpr_b,
        tpr_b,
        "steelblue",
        2.0,
        f"RM1 Baseline RF (AUC={hasil_rm1['ROC-AUC']:.4f})"
    ),
    (
        fpr_fs,
        tpr_fs,
        "darkorange",
        2.0,
        f"RM2 RF + FS (AUC={hasil_rm2['ROC-AUC']:.4f})"
    ),
    (
        fpr_g,
        tpr_g,
        "seagreen",
        2.0,
        f"RM3 RF + GridSearchCV (AUC={hasil_rm3['ROC-AUC']:.4f})"
    ),
    (
        fpr_f,
        tpr_f,
        "crimson",
        2.5,
        f"RM4 RF + FS + GridSearchCV (AUC={hasil_rm4['ROC-AUC']:.4f})"
    ),
]

for fpr, tpr, color, lw, label in curves:
    ax1.plot(fpr, tpr, color=color, lw=lw, label=label)

ax1.plot(
    [0, 1],
    [0, 1],
    color="gray",
    linestyle="--",
    lw=1,
    label="Random Classifier"
)

ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve — Full", fontweight="bold")
ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)

for fpr, tpr, color, lw, label in curves:
    ax2.plot(fpr, tpr, color=color, lw=lw, label=label)

ax2.set_xlim(-0.005, 0.15)
ax2.set_ylim(0.85, 1.005)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve — Zoom Area Kritis", fontweight="bold")
ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    "Perbandingan ROC Curve Seluruh Model Klasifikasi",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    os.path.join(VIS_DIR, "roc_curve_4model.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()
print("  Gambar disimpan: roc_curve_4model.png")

x = np.arange(4)
width = 0.35

train_vals_plot = [
    hasil_rm1["Train Accuracy"],
    hasil_rm2["Train Accuracy"],
    hasil_rm3["Train Accuracy"],
    hasil_rm4["Train Accuracy"]
]

test_vals_plot = [
    hasil_rm1["Test Accuracy"],
    hasil_rm2["Test Accuracy"],
    hasil_rm3["Test Accuracy"],
    hasil_rm4["Test Accuracy"]
]

plt.figure(figsize=(10, 6))

bars1 = plt.bar(
    x - width / 2,
    train_vals_plot,
    width,
    label="Train Accuracy",
    color="steelblue",
    edgecolor="white"
)

bars2 = plt.bar(
    x + width / 2,
    test_vals_plot,
    width,
    label="Test Accuracy",
    color="darkorange",
    edgecolor="white"
)

for bar in list(bars1) + list(bars2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xticks(x, labels_short)
plt.ylabel("Accuracy")
plt.ylim(
    max(0, min(train_vals_plot + test_vals_plot) - 0.05),
    1.05
)
plt.title(
    "Perbandingan Train vs Test Accuracy — 4 Model",
    fontweight="bold"
)
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(VIS_DIR, "gap_train_test_4model.png"),
    dpi=200,
    bbox_inches="tight"
)
plt.close()
print("  Gambar disimpan: gap_train_test_4model.png")

accuracy        = hasil_rm4["Accuracy"]
bal_acc         = hasil_rm4["Balanced Accuracy"]
kappa           = hasil_rm4["Kappa"]
mcc             = hasil_rm4["MCC"]
cv_acc          = hasil_rm4["CV Accuracy Scores"]
cv_bal_acc      = hasil_rm4["CV Balanced Scores"]
y_pred_cls_test = hasil_rm4["y_pred"]
rf_clf          = m4_final
# ==============================================================================
# Kode Program Visualisasi Hasil Evaluasi
# ==============================================================================
section("9. VISUALISASI EVALUASI REGRESI")

df_test_pred = df_test.copy()
df_test_pred["Produktivitas_Prediksi"] = y_pred_prod_test.round(4)
df_test_pred["Produksi_Aktual"]   = df_test_pred["Produksi"]
df_test_pred["Produksi_Prediksi"] = produksi_pred_test.round(0)
df_test_pred["Selisih"]   = (df_test_pred["Produksi_Aktual"] - df_test_pred["Produksi_Prediksi"]).round(0)
df_test_pred["Error_%"]   = (
    np.abs(df_test_pred["Selisih"]) /
    np.where(df_test_pred["Produksi_Aktual"] == 0, np.nan, df_test_pred["Produksi_Aktual"]) * 100
).round(2)
df_test_pred["SMAPE_%"] = (
    np.abs(df_test_pred["Produksi_Aktual"] - df_test_pred["Produksi_Prediksi"]) /
    ((np.abs(df_test_pred["Produksi_Aktual"]) + np.abs(df_test_pred["Produksi_Prediksi"])) / 2) * 100
).round(2)
df_test_pred["Status"] = df_test_pred["Error_%"].apply(status_error)
df_test_pred.to_csv(os.path.join(VIS_DIR, f"validasi_produksi_test_{TEST_YEAR}.csv"), index=False)
print(f"  CSV disimpan: validasi_produksi_test_{TEST_YEAR}.csv")

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(
    df_test_pred["Produksi_Aktual"] / 1e6,
    df_test_pred["Produksi_Prediksi"] / 1e6,
    alpha=0.85, c=df_test_pred["Provinsi_Encoded"], cmap="tab20",
    s=90, edgecolors="white",
)
lims = [
    min(df_test_pred["Produksi_Aktual"].min(), df_test_pred["Produksi_Prediksi"].min()) / 1e6 - 0.1,
    max(df_test_pred["Produksi_Aktual"].max(), df_test_pred["Produksi_Prediksi"].max()) / 1e6 + 0.1,
]
ax.plot(lims, lims, "r--", lw=2, label="Prediksi sempurna (y=x)")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Produksi aktual (juta ton)")
ax.set_ylabel("Produksi prediksi (juta ton)")
ax.set_title(
    f"Validasi Produksi Aktual vs Prediksi — Test {TEST_YEAR}\n"
    f"R² = {r2*100:.2f}%  |  RMSE = {rmse:,.0f} ton  |  SMAPE = {smape_score:.2f}%",
    fontweight="bold",
)
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, f"validasi_scatter_test_{TEST_YEAR}.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Gambar disimpan: validasi_scatter_test_{TEST_YEAR}.png")

# --- Gambar 4b: Scatterplot Produktivitas Aktual vs Prediksi (ku/ha) ---
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(
    y_test, y_pred_prod_test,
    alpha=0.85, c=df_test_pred["Provinsi_Encoded"], cmap="tab20",
    s=90, edgecolors="white",
)
lims_prod = [
    min(y_test.min(), y_pred_prod_test.min()) - 0.5,
    max(y_test.max(), y_pred_prod_test.max()) + 0.5,
]
ax.plot(lims_prod, lims_prod, "r--", lw=2, label="Prediksi sempurna (y=x)")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Produktivitas aktual (ku/ha)")
ax.set_ylabel("Produktivitas prediksi (ku/ha)")
ax.set_title(
    f"Validasi Produktivitas Aktual vs Prediksi — Test {TEST_YEAR}\n"
    f"R² = {m_prod_test['R2']*100:.2f}%  |  RMSE = {m_prod_test['RMSE']:.2f} ku/ha  |  MAPE = {m_prod_test['MAPE']:.2f}%",
    fontweight="bold",
)
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, f"4b_scatter_produktivitas_test_{TEST_YEAR}.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Gambar disimpan: 4b_scatter_produktivitas_test_{TEST_YEAR}.png")

# --- Gambar 4c: Learning Curve (Expanding Window, dievaluasi terhadap Test Asli 2023-2024) ---
# Catatan teknis: learning_curve() bawaan sklearn dengan cv=TimeSeriesSplit akan membatasi
# ukuran data latih maksimum berdasarkan fold TimeSeriesSplit TERKECIL, sehingga sumbu-X
# tidak representatif terhadap jumlah data latih yang sesungguhnya (165 baris).
# Pendekatan expanding window di bawah ini melatih model pada porsi data latih kronologis
# yang terus bertambah (20%, 40%, ..., 100%), lalu setiap tahap dievaluasi langsung
# terhadap data uji asli (2023-2024) — bukan validasi CV internal — sehingga kurva
# mencerminkan skala data latih sebenarnya dan langsung relevan dengan pertanyaan
# "apakah menambah tahun data historis membantu prediksi ke periode uji nyata?".

n_train_total = len(X_train)
fractions_lc  = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
train_sizes_lc = np.maximum(2, np.round(fractions_lc * n_train_total)).astype(int)
train_sizes_lc = np.unique(train_sizes_lc)

train_r2_lc = []
test_r2_lc  = []

for n_sub in train_sizes_lc:
    X_sub = X_train.iloc[:n_sub]
    y_sub = y_train.iloc[:n_sub]

    model_lc = RandomForestRegressor(random_state=42, n_jobs=SKLEARN_N_JOBS, **best_params)
    model_lc.fit(X_sub, y_sub)

    r2_train_sub = r2_score(y_sub, model_lc.predict(X_sub))
    r2_test_sub  = r2_score(y_test, model_lc.predict(X_test))

    train_r2_lc.append(r2_train_sub)
    test_r2_lc.append(r2_test_sub)

    print(f"  Data latih: {n_sub:>4} baris  |  Train R²: {r2_train_sub*100:6.2f}%  |  Test R² (2023-2024): {r2_test_sub*100:6.2f}%")

train_r2_lc = np.array(train_r2_lc)
test_r2_lc  = np.array(test_r2_lc)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_sizes_lc, train_r2_lc, "o-", color="steelblue", lw=2, markersize=7, label="Training R²")
ax.plot(train_sizes_lc, test_r2_lc, "s--", color="darkorange", lw=2, markersize=7, label="Test R² (Data Uji 2023-2024)")
ax.axhline(0, color="gray", linestyle=":", lw=1, alpha=0.7)
for x_val, y_val in zip(train_sizes_lc, test_r2_lc):
    ax.annotate(f"{y_val*100:.1f}%", (x_val, y_val), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=9, color="darkorange")
for x_val, y_val in zip(train_sizes_lc, train_r2_lc):
    ax.annotate(f"{y_val*100:.1f}%", (x_val, y_val), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9, color="steelblue")
ax.set_xlabel("Jumlah Data Latih (baris, kronologis 2018→2022)")
ax.set_ylabel("R² Score")
ax.set_title(
    f"Learning Curve — Random Forest Produktivitas\n(Expanding Window, dievaluasi terhadap Test {TEST_YEAR_START}-{TEST_YEAR_END})",
    fontweight="bold"
)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "4c_learning_curve.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Gambar disimpan: 4c_learning_curve.png")

pd.DataFrame({
    "jumlah_data_latih": train_sizes_lc,
    "train_r2": train_r2_lc,
    "test_r2_2023_2024": test_r2_lc,
}).to_csv(os.path.join(VIS_DIR, "learning_curve_data.csv"), index=False)
print("  CSV disimpan: learning_curve_data.csv")

residuals_test = df_test_pred["Produksi_Aktual"].values - df_test_pred["Produksi_Prediksi"].values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(residuals_test / 1e3, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(0, color="red", linestyle="--", lw=2, label="Nol (ideal)")
axes[0].axvline(residuals_test.mean() / 1e3, color="orange", linestyle="--", lw=2, label=f"Mean = {residuals_test.mean()/1e3:.1f}k ton")
axes[0].set_title("Distribusi Residual Test", fontweight="bold")
axes[0].set_xlabel("Residual (ribu ton)"); axes[0].set_ylabel("Frekuensi")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[1].scatter(
    df_test_pred["Produksi_Prediksi"] / 1e6, residuals_test / 1e3,
    alpha=0.75, c=df_test_pred["Provinsi_Encoded"], cmap="tab20", s=70, edgecolors="white",
)
axes[1].axhline(0, color="red", linestyle="--", lw=2, label="Nol (ideal)")
axes[1].set_title("Residual vs Fitted Test", fontweight="bold")
axes[1].set_xlabel("Prediksi (juta ton)"); axes[1].set_ylabel("Residual (ribu ton)")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.suptitle("Analisis Residual — Validasi Produksi Test", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, f"analisis_residual_test_{TEST_YEAR}.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Gambar disimpan: analisis_residual_test_{TEST_YEAR}.png")

fig, ax = plt.subplots(figsize=(12, 10))
top_feat_heatmap = FEATURES_SEL[:15] if len(FEATURES_SEL) > 15 else FEATURES_SEL
sns.heatmap(
    df_clean[top_feat_heatmap + ["Produktivitas", "Produksi"]].corr(),
    annot=True, cmap="coolwarm", fmt=".3f",
    linewidths=0.5, square=True, ax=ax, annot_kws={"size": 7},
)
ax.set_title("Heatmap Korelasi — Fitur Terpilih, Produktivitas, Produksi", fontsize=12, fontweight="bold", pad=15)
plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "4_heatmap_korelasi.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Gambar disimpan: 4_heatmap_korelasi.png")

fig, ax = plt.subplots(figsize=(8, 4))
fold_labels = [f"Fold {i+1}" for i in range(len(cv_r2_prod))]
colors_cv   = ["#2E8B57" if v >= 0.50 else "#CD5C5C" for v in cv_r2_prod]
bars = ax.bar(fold_labels, cv_r2_prod * 100, color=colors_cv, edgecolor="white", width=0.5)
for bar, val in zip(bars, cv_r2_prod):
    ax.text(bar.get_x() + bar.get_width()/2, max(val*100, 0) + 0.5,
            f"{val*100:.2f}%", ha="center", va="bottom", fontsize=10)
ax.axhline(cv_r2_prod.mean() * 100, color="red", linestyle="--", lw=2, label=f"Mean R² = {cv_r2_prod.mean()*100:.2f}%")
ax.axhline(50, color="orange", linestyle=":", lw=1.5, label="Batas 50%")
ax.set_title(f"Cross-Validation R² Produktivitas per Fold\n(TimeSeriesSplit {cv_splits}-Fold pada data train)", fontsize=12, fontweight="bold")
ax.set_ylabel("R² Score (%)"); ax.set_ylim(min(0, cv_r2_prod.min()*100 - 10), 110)
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "5_cv_r2_per_fold.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Gambar disimpan: 5_cv_r2_per_fold.png")

fig, ax = plt.subplots(figsize=(10, max(5, len(df_test_pred) * 0.22)))
err_prov_test = df_test_pred.groupby("Provinsi")["Error_%"].mean().sort_values()
colors_err    = ["#2E8B57" if v < 5 else ("#FFA500" if v < 10 else "#CD5C5C") for v in err_prov_test.values]
bars = ax.barh(err_prov_test.index, err_prov_test.values, color=colors_err, edgecolor="white", height=0.55)
for bar, val in zip(bars, err_prov_test.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=9)
ax.axvline(5,  color="#2E8B57", linestyle="--", lw=1.5, label="< 5%  → Sangat Akurat")
ax.axvline(10, color="#FFA500", linestyle="--", lw=1.5, label="< 10% → Akurat")
ax.set_title(f"Error Prediksi Produksi per Provinsi — Test {TEST_YEAR}", fontsize=13, fontweight="bold")
ax.set_xlabel("Error (%)"); ax.legend(); ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, f"error_per_provinsi_test_{TEST_YEAR}.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"  Gambar disimpan: error_per_provinsi_test_{TEST_YEAR}.png")
# ==============================================================================
# Kode Program Penyimpanan Model dan Hasil
# ==============================================================================
section("10. SIMPAN MODEL")

final_model = RandomForestRegressor(random_state=42, n_jobs=SKLEARN_N_JOBS, **best_params)
final_model.fit(X_selected_all, y_prod_all)

final_clf = RandomForestClassifier(random_state=42, n_jobs=SKLEARN_N_JOBS, **gs4.best_params_)
X_all_clf = X_selected_all[FEATURES_CLF].copy()
final_clf.fit(X_all_clf, y_cls_all)

joblib.dump(final_model,  os.path.join(MODEL_DIR, "model_rf_regresi_produktivitas.pkl"))
joblib.dump(final_clf,    os.path.join(MODEL_DIR, "model_rf_klasifikasi_produksi.pkl"))
joblib.dump(le,           os.path.join(MODEL_DIR, "label_encoder_provinsi.pkl"))
joblib.dump(le_target,    os.path.join(MODEL_DIR, "label_encoder_target.pkl"))
joblib.dump(le_icon,      os.path.join(MODEL_DIR, "label_encoder_icon.pkl"))
joblib.dump(FEATURES_SEL, os.path.join(MODEL_DIR, "selected_features.pkl"))
joblib.dump(FEATURES_CLF, os.path.join(MODEL_DIR, "selected_features_clf.pkl"))
joblib.dump(best_params,  os.path.join(MODEL_DIR, "best_hyperparameters.pkl"))
joblib.dump(scaler,       os.path.join(MODEL_DIR, "scaler_fitur.pkl"))

print(f"\n  Model disimpan di: {MODEL_DIR}")
for f in [
    "model_rf_regresi_produktivitas.pkl", "model_rf_klasifikasi_produksi.pkl",
    "label_encoder_provinsi.pkl",          "label_encoder_target.pkl",
    "label_encoder_icon.pkl",              "selected_features.pkl",
    "selected_features_clf.pkl",           "best_hyperparameters.pkl",
    "scaler_fitur.pkl",
]:
    print(f"  · {f}")

section("11. VERIFIKASI — PREDIKSI vs DATA ASLI PER PROVINSI")

PROVINSI_LIST = sorted(df_clean["Provinsi"].unique())

df_hist = df_clean.copy()
df_hist_scaled = pd.DataFrame(
    scaler.transform(df_hist[FEATURES_SEL]),
    columns=FEATURES_SEL,
    index=df_hist.index
)
df_hist["Produktivitas_Prediksi"] = final_model.predict(df_hist_scaled).round(4)
df_hist["Produksi_Aktual"]   = df_hist["Produksi"]
df_hist["Produksi_Prediksi"] = (df_hist["Produktivitas_Prediksi"] * df_hist["Luas Panen"] / 10).round(0)
df_hist["Selisih"]   = (df_hist["Produksi_Aktual"] - df_hist["Produksi_Prediksi"]).round(0)
df_hist["Error_%"]   = (
    np.abs(df_hist["Selisih"]) /
    np.where(df_hist["Produksi_Aktual"] == 0, np.nan, df_hist["Produksi_Aktual"]) * 100
).round(2)
df_hist["SMAPE_%"] = (
    np.abs(df_hist["Produksi_Aktual"] - df_hist["Produksi_Prediksi"]) /
    ((np.abs(df_hist["Produksi_Aktual"]) + np.abs(df_hist["Produksi_Prediksi"])) / 2) * 100
).round(2)
df_hist["Status"] = df_hist["Error_%"].apply(status_error)

W = 128
for provinsi in PROVINSI_LIST:
    sub = df_hist[df_hist["Provinsi"] == provinsi].sort_values("Tahun")
    if sub.empty:
        continue

    print(f"\n  Provinsi: {provinsi}")
    print(f"  {'-' * W}")
    print(f"  {'Tahun':<6} {'Aktual (ton)':>15} {'Prediksi (ton)':>16} {'Selisih':>13} {'Error%':>8} {'SMAPE%':>8}  Status")
    print(f"  {'-' * W}")
    for _, row in sub.iterrows():
        sel_str = f"{row['Selisih']:+,.0f}"
        print(
            f"  {int(row['Tahun']):<6} {row['Produksi_Aktual']:>15,.0f} "
            f"{row['Produksi_Prediksi']:>16,.0f} {sel_str:>13} "
            f"{row['Error_%']:>8.2f}% {row['SMAPE_%']:>8.2f}%  {row['Status']}"
        )

    sa = (sub["Status"] == "Sangat Akurat").sum()
    ak = (sub["Status"] == "Akurat").sum()
    pp = (sub["Status"] == "Perlu Perhatian").sum()
    print(f"  {'-' * W}")
    print(
        f"  Avg Error: {sub['Error_%'].mean():.2f}%  |  Avg SMAPE: {sub['SMAPE_%'].mean():.2f}%  |  "
        f"Sangat Akurat (<5%): {sa} thn  |  Akurat (<10%): {ak} thn  |  Perlu Perhatian: {pp} thn"
    )

df_hist.to_csv(os.path.join(VIS_DIR, "perbandingan_aktual_vs_prediksi_semua_data.csv"), index=False)
print("\n  CSV disimpan: perbandingan_aktual_vs_prediksi_semua_data.csv")
section("12. VISUALISASI PERBANDINGAN PER PROVINSI")

n_prov = len(PROVINSI_LIST)
n_cols = 3
n_rows = int(np.ceil(n_prov / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
axes = np.array(axes).reshape(-1)

for i, provinsi in enumerate(PROVINSI_LIST):
    sub = df_hist[df_hist["Provinsi"] == provinsi].sort_values("Tahun")
    ax  = axes[i]
    ax.plot(sub["Tahun"], sub["Produksi_Aktual"] / 1e6, "o-", color="steelblue",  lw=2, markersize=6, label="Aktual")
    ax.plot(sub["Tahun"], sub["Produksi_Prediksi"] / 1e6, "s--", color="darkorange", lw=2, markersize=6, label="Prediksi")
    ax.fill_between(sub["Tahun"], sub["Produksi_Aktual"] / 1e6, sub["Produksi_Prediksi"] / 1e6, alpha=0.12, color="red")
    ax.set_title(f"{provinsi}\n(Avg Error: {sub['Error_%'].mean():.1f}%)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Tahun", fontsize=9); ax.set_ylabel("Produksi (juta ton)", fontsize=9)
    ax.set_xticks(sub["Tahun"]); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

for j in range(n_prov, len(axes)):
    axes[j].axis("off")

plt.suptitle("Perbandingan Produksi Aktual vs Prediksi per Provinsi", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(os.path.join(VIS_DIR, "6_perbandingan_per_provinsi.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Gambar disimpan: 6_perbandingan_per_provinsi.png")

fig, ax = plt.subplots(figsize=(10, max(6, n_prov * 0.28)))
err_prov    = df_hist.groupby("Provinsi")["Error_%"].mean().sort_values()
colors_err  = ["#2E8B57" if v < 5 else ("#FFA500" if v < 10 else "#CD5C5C") for v in err_prov.values]
bars = ax.barh(err_prov.index, err_prov.values, color=colors_err, edgecolor="white", height=0.55)
for bar, val in zip(bars, err_prov.values):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=9)
ax.axvline(5,  color="#2E8B57", linestyle="--", lw=1.5, label="< 5%  → Sangat Akurat")
ax.axvline(10, color="#FFA500", linestyle="--", lw=1.5, label="< 10% → Akurat")
ax.set_title("Rata-rata Error Prediksi per Provinsi", fontsize=13, fontweight="bold")
ax.set_xlabel("Rata-rata Error (%)"); ax.legend(); ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "7_error_per_provinsi.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  Gambar disimpan: 7_error_per_provinsi.png")
print(f"\n{SEP}\n  RINGKASAN AKHIR\n{SEP}")

line60 = "─" * 60

ringkasan = {
    "Jumlah data awal hasil merge": df.shape[0],
    "Jumlah provinsi": df["Provinsi"].nunique(),
    "Rentang tahun": f"{df['Tahun'].min()}–{df['Tahun'].max()}",
    "Dihapus IQR": n_iqr_out, "Dihapus anomali": n_anomali,
    "Data bersih": df_clean.shape[0], "Tahun test": f"{TEST_YEAR_START}–{TEST_YEAR_END}",
    "Jumlah fitur awal": len(ALL_FEATURES), "Jumlah fitur terpilih": len(FEATURES_SEL),
    "Fitur terpilih": ", ".join(FEATURES_SEL),
    "Test R2 Produksi": r2, "Test RMSE Produksi": rmse,
    "Test MAE Produksi": mae, "Test MAPE Produksi": mape, "Test SMAPE Produksi": smape_score,
    "Accuracy Klasifikasi Test": accuracy, "Balanced Accuracy Test": bal_acc,
    "Cohen Kappa Test": kappa, "MCC Test": mcc,
}
pd.DataFrame([ringkasan]).to_csv(os.path.join(VIS_DIR, "ringkasan_akhir_validasi_produksi.csv"), index=False)

print(f"""
  DATASET
  {line60}
  Jumlah data awal   : {df.shape[0]} baris  |  {df['Provinsi'].nunique()} provinsi
  Rentang tahun      : {df['Tahun'].min()} – {df['Tahun'].max()}
  Dihapus (IQR)      : {n_iqr_out} baris
  Dihapus (anomali)  : {n_anomali} baris
  Data bersih        : {df_clean.shape[0]} baris
  Semua data dipakai : Ya, termasuk provinsi Papua baru yang match dengan data cuaca
  Validasi test      : Tahun {TEST_YEAR_START}–{TEST_YEAR_END}

  KONSEP MODEL
  {line60}
  Target model       : Produktivitas (ku/ha)
  Validasi akhir     : Produksi Prediksi vs Produksi Aktual
  Rumus produksi     : Produktivitas_Prediksi × Luas Panen / 10

  SELEKSI FITUR
  {line60}
  Jumlah fitur awal  : {len(ALL_FEATURES)} fitur
  Fitur dipilih      : {len(FEATURES_SEL)} fitur (threshold + fitur wajib + minimal fitur)
  Fitur dibuang      : {len(FEATURES_DROP)} fitur
  Fitur terpilih     : {FEATURES_SEL}
""")

print(f"\n  HYPERPARAMETER TERBAIK (GridSearchCV TimeSeriesSplit {cv_splits}-Fold)")
print(f"  {line60}")
for p, v in best_params.items():
    print(f"  {p:<24}: {v}")
print(f"  {'Best CV MAE Produktivitas':<24}: {best_cv_mae_prod:,.4f} ku/ha")

print(f"""
  EVALUASI REGRESI PRODUKSI TEST {TEST_YEAR_START}–{TEST_YEAR_END}
  {line60}
  R²            : {r2*100:.2f}%  → {status_r2}
  Adjusted R²   : {adj_r2*100 if not np.isnan(adj_r2) else np.nan:.2f}%
  RMSE / MAE    : {rmse:,.0f} / {mae:,.0f} ton
  MAPE          : {mape:.2f}%  → {status_mape}
  SMAPE         : {smape_score:.2f}%

  EVALUASI KLASIFIKASI TEST {TEST_YEAR_START}–{TEST_YEAR_END}
  {line60}
  Accuracy        : {accuracy*100:.2f}%
  Balanced Acc    : {bal_acc*100:.2f}%
  Cohen Kappa     : {kappa:.4f}
  MCC             : {mcc:.4f}
  CV Accuracy     : {np.nanmean(cv_acc)*100:.2f}% ± {np.nanstd(cv_acc)*100:.2f}%
  CV Balanced Acc : {np.nanmean(cv_bal_acc)*100:.2f}% ± {np.nanstd(cv_bal_acc)*100:.2f}%

  AKURASI PER PROVINSI — SEMUA DATA
  {line60}
  {'Provinsi':<28} {'Avg Error':>10}  {'Avg SMAPE':>10}  {'Sangat Akurat':>14}  {'Akurat':>8}  {'Perlu Perhatian':>16}""")

for provinsi in PROVINSI_LIST:
    s  = df_hist[df_hist["Provinsi"] == provinsi]
    sa = (s["Status"] == "Sangat Akurat").sum()
    ak = (s["Status"] == "Akurat").sum()
    pp = (s["Status"] == "Perlu Perhatian").sum()
    print(
        f"  {provinsi:<28} {s['Error_%'].mean():>9.2f}%"
        f"  {s['SMAPE_%'].mean():>9.2f}%  {sa:>14} thn  {ak:>7} thn  {pp:>15} thn"
    )

print(f"""
  FILE OUTPUT
  {line60}
  visualizations/
    dataset_hasil_merge_padi_cuaca.csv       | data_padi_tidak_punya_cuaca.csv
    cek_kelengkapan_tahun_per_provinsi.csv   | feature_importance_awal.csv
    cv_jumlah_fitur.csv                      | gridsearchcv_results.csv
    ringkasan_metrik_validasi.csv            | ringkasan_akhir_validasi_produksi.csv
    validasi_produksi_test_{TEST_YEAR}.csv         | perbandingan_aktual_vs_prediksi_semua_data.csv
    classification_report_test.csv           | perbandingan_model_klasifikasi.csv
    learning_curve_data.csv

  visualizations/ (gambar)
    1b_optimal_n_features.png   feature_importance_terpilih.png
    validasi_scatter_test_{TEST_YEAR}.png    analisis_residual_test_{TEST_YEAR}.png
    4b_scatter_produktivitas_test_{TEST_YEAR}.png   4c_learning_curve.png
    4_heatmap_korelasi.png      5_cv_r2_per_fold.png
    error_per_provinsi_test_{TEST_YEAR}.png  6_perbandingan_per_provinsi.png
    7_error_per_provinsi.png    cm_rm1_baseline.png
    cm_rm2_feature_selection.png             cm_rm3_gridsearchcv.png
    confusion_matrix_test.png   optimal_n_features_clf.png
    perbandingan_accuracy_model.png          perbandingan_balanced_acc_model.png
    perbandingan_jumlah_fitur_model.png      roc_curve_4model.png
    gap_train_test_4model.png

  models/
    model_rf_regresi_produktivitas.pkl       model_rf_klasifikasi_produksi.pkl
    label_encoder_provinsi.pkl               label_encoder_target.pkl
    label_encoder_icon.pkl                   selected_features.pkl
    selected_features_clf.pkl                best_hyperparameters.pkl
""")
print(f"{SEP}\n  SELESAI.\n{SEP}")
progress.finish()