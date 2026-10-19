
# ============================================
# REGRESI LINEAR BERGANDA KOTA BANDAR LAMPUNG
# Y  : KEMISKINAN
# X1 : DBH_PAJAK
# X2 : bm_pendidikan
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go

# 1. Membaca dataset
url = "https://raw.githubusercontent.com/Alam-A99/DATAAKSI/refs/heads/main/dataset_fiskal_97_hipoFull.csv"

df = pd.read_csv(url)

print("Ukuran dataset:", df.shape)
print("Daftar kolom:", df.columns.tolist())

# 2. Memfilter data Kota Bandar Lampung
data = df[
    df["Kota"].astype(str).str.strip().str.lower()
    == "kota bandar lampung"
].copy()

# 3. Memilih variabel penelitian
kolom = ["Tahun", "DBH_PAJAK", "bm_pendidikan", "KEMISKINAN"]
data = data[kolom].copy()

# Memastikan variabel berbentuk numerik
for col in kolom:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Menghapus baris yang memiliki nilai kosong
data = data.dropna().sort_values("Tahun")

print("\nDATA KOTA BANDAR LAMPUNG:")
display(data)

print("\nJumlah observasi:", len(data))

# 4. Menentukan X dan Y
X = data[["DBH_PAJAK", "bm_pendidikan"]]
Y = data["KEMISKINAN"]

# Menambahkan konstanta
X_model = sm.add_constant(X)

# 5. Membuat model regresi
model = sm.OLS(Y, X_model).fit()

# 6. Menampilkan hasil regresi
print("\nHASIL REGRESI LINEAR BERGANDA")
print(model.summary())

# 7. Menampilkan persamaan regresi
b0 = model.params["const"]
b1 = model.params["DBH_PAJAK"]
b2 = model.params["bm_pendidikan"]

print("\nPERSAMAAN REGRESI:")
print(
    f"KEMISKINAN = {b0:.4f} "
    f"+ ({b1:.4f} * DBH_PAJAK) "
    f"+ ({b2:.4f} * bm_pendidikan)"
)

print("\nR-squared:", round(model.rsquared, 4))
print("Adjusted R-squared:", round(model.rsquared_adj, 4))
print("P-value DBH_PAJAK:", round(model.pvalues["DBH_PAJAK"], 4))
print("P-value bm_pendidikan:", round(model.pvalues["bm_pendidikan"], 4))

# 8. Membuat prediksi
data["Prediksi_Kemiskinan"] = model.predict(X_model)

print("\nPERBANDINGAN NILAI AKTUAL DAN PREDIKSI:")
display(data)

# 9. Visualisasi regresi 3D
# Sumbu X = DBH_PAJAK
# Sumbu Y = bm_pendidikan
# Sumbu Z = KEMISKINAN

x_range = np.linspace(
    data["DBH_PAJAK"].min(),
    data["DBH_PAJAK"].max(),
    30
)

y_range = np.linspace(
    data["bm_pendidikan"].min(),
    data["bm_pendidikan"].max(),
    30
)

xx, yy = np.meshgrid(x_range, y_range)

zz = (
    b0
    + b1 * xx
    + b2 * yy
)

fig = go.Figure()

# Titik data aktual
fig.add_trace(go.Scatter3d(
    x=data["DBH_PAJAK"],
    y=data["bm_pendidikan"],
    z=data["KEMISKINAN"],
    mode="markers+text",
    text=data["Tahun"].astype(int).astype(str),
    textposition="top center",
    marker=dict(size=5),
    name="Data aktual"
))

# Bidang regresi
fig.add_trace(go.Surface(
    x=xx,
    y=yy,
    z=zz,
    opacity=0.65,
    colorscale="Viridis",
    showscale=True,
    name="Bidang regresi"
))

fig.update_layout(
    title="Regresi Linear Berganda 3D - Kota Bandar Lampung",
    scene=dict(
        xaxis_title="DBH_PAJAK (X1)",
        yaxis_title="Belanja Modal Pendidikan (X2)",
        zaxis_title="KEMISKINAN (Y)"
    ),
    template="plotly_white",
    width=1000,
    height=700
)

fig.show()

# 10. Menyimpan hasil data ke CSV
data.to_csv("hasil_regresi_bandar_lampung.csv", index=False)

print("\nSelesai! Hasil data tersimpan dalam file CSV.")
