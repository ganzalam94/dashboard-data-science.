import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Muat dataset
df = pd.read_csv('https://raw.githubusercontent.com/Alam-A99/Pandas_Latih/10ebb3af000e930a44989075733f495d5c2f4be5/penjualan.csv')
df
# 2. Menambahkan kolom total untuk mempermudah beberapa grafik
df['total'] = df[['jan', 'feb', 'mar', 'apr']].sum(axis=1)
sns.set_theme(style="whitegrid")

# Bar Chart
plt.figure(figsize=(8, 5))  # Ukuran kompak (lebar 8, tinggi 5 inci)

# Urutkan data dari total tertinggi
df_sorted = df.sort_values("total", ascending=False)

sns.barplot(data=df_sorted, x="nama", y="total", palette="viridis")

plt.title("Total Penjualan per Orang", fontsize=12)
plt.xlabel("Nama Karyawan")
plt.ylabel("Total Penjualan")
plt.xticks(rotation=45)  # Miringkan label agar tidak tumpang tindih
plt.tight_layout()
plt.show()

#Pie Chart
plt.figure(figsize=(5, 5)) #Ukuran persegi agar pie chart tidak gepeng

#Hitung total penjualan per bulan
total_per_bulan = df[['jan', 'feb', 'mar', 'apr']].sum()

plt.pie(total_per_bulan, labels=total_per_bulan.index, autopct='%1.1f%%', startangle=140, colors = sns.color_palette('muted'))
plt.title('proporsi penjualan per bulan', fontsize=12)
plt.tight_layout()
plt.show()
