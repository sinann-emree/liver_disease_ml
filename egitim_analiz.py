import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn kütüphaneleri
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score, recall_score, precision_score

# Tensorflow / Keras (Yapay Sinir Ağları için)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Uyarıları kapatmak için
import warnings
warnings.filterwarnings('ignore')

# 1. Veriyi Yükleme
print("Veri yükleniyor...")
df = pd.read_csv('Liver_disease_data_processed.csv')
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# 2. Modellerin Tanımlanması
random_state = 42

models = {
    "Lojistik Regresyon": LogisticRegression(random_state=random_state),
    "SVM": SVC(probability=True, random_state=random_state), # probability=True ROC için gerekli
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Rastgele Orman": RandomForestClassifier(n_estimators=100, random_state=random_state)
}

# 3. YSA (Yapay Sinir Ağı) Modelini Oluşturma Fonksiyonu
def create_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu')) # Gizli katman 1
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu')) # Gizli katman 2
    model.add(Dense(1, activation='sigmoid')) # Çıkış katmanı (0 veya 1)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- A) DIŞARIDA TUTMA (HOLD-OUT) YÖNTEMİ (%70 Eğitim - %30 Test) ---
print("\n--- A) Dışarıda Tutma (Hold-out) Eğitimi Başlıyor ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

results_holdout = []
predictions = {} # McNemar testi için tahminleri saklayacağız
trained_models = {} # Modelleri saklayacağız

# Klasik Modellerin Eğitimi
for name, model in models.items():
    print(f"{name} eğitiliyor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    predictions[name] = y_pred
    trained_models[name] = model
    
    # Metrikler
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    sens = recall_score(y_test, y_pred) # Duyarlılık
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp) # Özgüllük
    
    results_holdout.append({
        "Model": name,
        "Yöntem": "Hold-out",
        "Doğruluk (Accuracy)": acc,
        "Duyarlılık (Sensitivity)": sens,
        "Özgüllük (Specificity)": spec,
        "F1-Skoru": f1
    })

# YSA Eğitimi (Özel Grafik İstendiği İçin Ayrı)
print("Yapay Sinir Ağları (ANN) eğitiliyor...")
ann_model = create_ann_model(X_train.shape[1])
history = ann_model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=100, # Hoca >= 100 istedi
                        batch_size=32, 
                        verbose=0) # Çıktı kirliliği olmasın diye 0

# YSA Tahminleri
y_prob_ann = ann_model.predict(X_test).ravel()
y_pred_ann = (y_prob_ann > 0.5).astype(int)
predictions["YSA"] = y_pred_ann
trained_models["YSA"] = ann_model

# YSA Metrikleri
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ann).ravel()
results_holdout.append({
    "Model": "YSA",
    "Yöntem": "Hold-out",
    "Doğruluk (Accuracy)": accuracy_score(y_test, y_pred_ann),
    "Duyarlılık (Sensitivity)": recall_score(y_test, y_pred_ann),
    "Özgüllük (Specificity)": tn / (tn + fp),
    "F1-Skoru": f1_score(y_test, y_pred_ann)
})

# --- GRAFİKLER ---

# 1. YSA Eğitim ve Kayıp Grafiği
plt.figure(figsize=(12, 5))

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('YSA Model Kaybı (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

# Başarı (Accuracy) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.title('YSA Model Başarısı (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Başarı')
plt.legend()

plt.savefig('ysa_egitim_grafigi.png')
print("YSA Grafiği kaydedildi: ysa_egitim_grafigi.png")

# 2. Karışıklık Matrisleri (Confusion Matrices)
plt.figure(figsize=(15, 10))
for i, (name, pred) in enumerate(predictions.items()):
    plt.subplot(2, 3, i+1)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} Karışıklık Matrisi")
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
plt.tight_layout()
plt.savefig('karisiklik_matrisleri.png')
print("Karışıklık matrisleri kaydedildi: karisiklik_matrisleri.png")

# 3. ROC Eğrileri
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# YSA ROC Eğrisi
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
roc_auc_ann = auc(fpr_ann, tpr_ann)
plt.plot(fpr_ann, tpr_ann, label=f'YSA (AUC = {roc_auc_ann:.2f})', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı (FPR)')
plt.ylabel('Doğru Pozitif Oranı (TPR)')
plt.title('Alıcı İşletim Karakteristiği (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_egrileri.png')
print("ROC eğrileri kaydedildi: roc_egrileri.png")


# --- B) K-KAT ÇAPRAZ DOĞRULAMA (K-FOLD CV) ---
print("\n--- B) K-Kat Çapraz Doğrulama (10-Kat) Başlıyor ---")
# Not: YSA için K-Fold maliyetlidir, burada diğer 4 algoritma için yapıyoruz.
# Ancak tam ödev için YSA'yı döngüye sokmak gerekebilir ama sklearn wrapper gerekir.
# Şimdilik 4 temel algoritma üzerinden CV sonuçlarını verelim.

results_cv = []
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

for name, model in models.items():
    cv_acc = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    cv_f1 = cross_val_score(model, X, y, cv=kfold, scoring='f1').mean()
    # Not: cross_val_score tek seferde çoklu metrik döndürmez, tekrar çalışır.
    
    results_cv.append({
        "Model": name,
        "Yöntem": "10-Kat CV",
        "Ortalama Doğruluk": cv_acc,
        "Ortalama F1": cv_f1
    })

# --- SONUÇLARIN TABLO HALİNDE SUNUMU ---
df_holdout = pd.DataFrame(results_holdout)
df_cv = pd.DataFrame(results_cv)

print("\n--- A) Hold-out Sonuçları ---")
print(df_holdout)

print("\n--- B) K-Fold Cross Validation Sonuçları ---")
print(df_cv)

# Tüm sonuçları Excel/CSV olarak kaydet
df_holdout.to_csv('sonuclar_holdout.csv', index=False)
df_cv.to_csv('sonuclar_cv.csv', index=False)
print("\nTablolar CSV dosyası olarak kaydedildi.")


# --- 2. McNemar Testi ---
print("\n--- McNemar Testi (Model Karşılaştırması) ---")
# McNemar testi için bir fonksiyon: Model A ve Model B arasındaki istatistiksel farka bakar.
def mcnemar_test(y_true, pred_a, pred_b):
    # Tabloyu oluştur:
    #          Model B Doğru   Model B Yanlış
    # Model A Doğru      a               b
    # Model A Yanlış     c               d
    
    # Bizim odaklandığımız b ve c: Modellerin anlaşamadığı durumlar.
    correct_a = (y_true == pred_a)
    correct_b = (y_true == pred_b)
    
    b = np.sum(correct_a & ~correct_b) # A bildi, B bilemedi
    c = np.sum(~correct_a & correct_b) # A bilemedi, B bildi
    
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    return chi2

# En iyi modeli seçip diğerleriyle karşılaştıralım
# Hold-out sonuçlarına göre doğruluğu en yüksek olanı bulalım
best_model_name = df_holdout.loc[df_holdout['Doğruluk (Accuracy)'].idxmax()]['Model']
print(f"En başarılı model (Hold-out setinde): {best_model_name}")

# En iyi modeli kaydedelim (Arayüzde kullanmak için)
best_model = trained_models[best_model_name]
joblib.dump(best_model, 'en_iyi_model.joblib')
print(f"En iyi model '{best_model_name}' dosyaya kaydedildi: en_iyi_model.joblib")

print(f"\n{best_model_name} ile Diğer Modellerin Karşılaştırması (McNemar):")
print(f"{'Karşılaştırılan':<20} | {'Chi2 Değeri':<10} | {'Sonuç'}")
print("-" * 50)

for name, pred in predictions.items():
    if name != best_model_name:
        chi2_val = mcnemar_test(y_test.values, predictions[best_model_name], pred)
        # 3.841 kritik değerdir (p<0.05 için 1 serbestlik derecesinde)
        sonuc = "Farklı (p<0.05)" if chi2_val > 3.841 else "Fark Yok"
        print(f"{name:<20} | {chi2_val:.4f}     | {sonuc}")