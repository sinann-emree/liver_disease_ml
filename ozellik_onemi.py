import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Modeli Yükle
model = joblib.load('en_iyi_model.joblib')

# 2. Sütun İsimleri (Eğitim sırasıyla aynı olmalı)
feature_names = ['Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 
                 'GeneticRisk', 'PhysicalActivity', 'Diabetes', 
                 'Hypertension', 'LiverFunctionTest']

# 3. Önem Değerlerini Al
importances = model.feature_importances_

# 4. Görselleştirme
# Veriyi DataFrame yapıp sıralayalım
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

print("\n--- Modelin En Çok Dikkat Ettiği Özellikler ---")
print(feature_imp_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Hangi Özellik Sonucu Daha Çok Etkiliyor?')
plt.xlabel('Önem Düzeyi (Puan)')
plt.ylabel('Özellikler')
plt.tight_layout()
plt.savefig('ozellik_onemi.png')
plt.show()

print("\nGrafik 'ozellik_onemi.png' olarak kaydedildi.")