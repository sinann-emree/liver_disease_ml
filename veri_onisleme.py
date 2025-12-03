import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Veriyi Yükle
# Dosya yolunu senin sistemine göre aynı klasörde varsayıyorum
df = pd.read_csv('Liver_disease_data.csv')

# 2. Ölçeklenecek Sütunları Belirle
# Sadece sayısal aralığı geniş olanları seçiyoruz.
# 0-1 olan (Gender, Smoking vb.) sütunlara dokunmuyoruz.
cols_to_scale = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'LiverFunctionTest']

print("Ölçekleme öncesi ilk 3 satır (Age ve BMI'a dikkat):")
print(df[cols_to_scale].head(3))

# 3. Standartlaştırma İşlemi (StandardScaler)
# Bu işlem verilerin ortalamasını 0, standart sapmasını 1 yapar.
scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("\n------------------------------------------------\n")
print("Ölçekleme sonrası ilk 3 satır (Değerler küçüldü ve standartlaştı):")
print(df[cols_to_scale].head(3))

# 4. Veriyi Kaydet
# İşlenmiş veriyi yeni bir dosya olarak kaydediyoruz ki orijinali bozulmasın.
output_filename = 'Liver_disease_data_processed.csv'
df.to_csv(output_filename, index=False)

print(f"\nİşlem tamamlandı! Yeni veri seti '{output_filename}' olarak kaydedildi.")