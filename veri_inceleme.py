import pandas as pd

dosya_yolu = "Liver_disease_data.csv"

# Veriyi oku
df = pd.read_csv(dosya_yolu)

# İlk 5 satırı göster
print("--- Verinin İlk 5 Satırı ---")
print(df.head())

# Sütun bilgileri ve veri tipleri
print("\n--- Veri Seti Bilgileri ---")
print(df.info())

# İstatistiksel özet (Ortalama, standart sapma vb.)
print("\n--- İstatistiksel Özet ---")
print(df.describe())

# Eksik değer kontrolü
print("\n--- Eksik Değer Sayıları ---")
print(df.isnull().sum())

# Hedef değişken dağılımı (Hasta / Hasta Değil)
print("\n--- Hedef Değişken Dağılımı (Diagnosis) ---")
print(df['Diagnosis'].value_counts())