import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Orjinal veriyi oku
print("Veri okunuyor...")
df = pd.read_csv('Liver_disease_data.csv')

# 2. Sadece ölçeklenmesi gereken sütunları seç (Ön işlemedekiyle aynı olmalı)
cols_to_scale = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'LiverFunctionTest']

# 3. Scaler'ı eğit (Veri setindeki ortalama ve sapmayı öğrensin)
scaler = StandardScaler()
scaler.fit(df[cols_to_scale])

# 4. Scaler'ı kaydet
joblib.dump(scaler, 'scaler.joblib')
print("Başarılı! 'scaler.joblib' dosyası oluşturuldu.")
print("Artık arayüz bu dosyayı kullanarak girilen değerleri doğru formata çevirebilecek.")