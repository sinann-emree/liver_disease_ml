import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import numpy as np

# --- 1. MODEL VE SCALER YÜKLEME ---
try:
    model = joblib.load('en_iyi_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model ve Scaler başarıyla yüklendi.")
except FileNotFoundError as e:
    messagebox.showerror("Hata", f"Gerekli dosyalar bulunamadı!\n{e}\nLütfen 'en_iyi_model.joblib' ve 'scaler.joblib' dosyalarının aynı klasörde olduğundan emin olun.")
    exit()

# --- 2. TAHMİN FONKSİYONU ---
def tahmin_et():
    try:
        # Kullanıcıdan verileri al
        # Not: Arayüzdeki giriş sırası, modelin eğitim sırasıyla AYNI olmalı
        # Eğitimdeki Sütun Sırası: Age, Gender, BMI, AlcoholConsumption, Smoking, GeneticRisk, PhysicalActivity, Diabetes, Hypertension, LiverFunctionTest
        
        age = float(entry_age.get())
        gender = int(combo_gender.get().split('-')[0]) # "1-Kadın" -> 1 alır
        bmi = float(entry_bmi.get())
        alcohol = float(entry_alcohol.get())
        smoking = int(combo_smoking.get().split('-')[0])
        genetic = int(combo_genetic.get())
        physical = float(entry_physical.get())
        diabetes = int(combo_diabetes.get().split('-')[0])
        hypertension = int(combo_hypertension.get().split('-')[0])
        liver_test = float(entry_liver.get())

        # Veriyi DataFrame'e çevir (İsimler eğitimdekiyle aynı olmalı)
        input_data = pd.DataFrame([[age, gender, bmi, alcohol, smoking, genetic, physical, diabetes, hypertension, liver_test]],
                                  columns=['Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest'])

        # Ölçekleme İşlemi (Sadece belirli sütunlar)
        cols_to_scale = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'LiverFunctionTest']
        input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

        # Tahmin Yap
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # Sonucu Göster
        if prediction == 1:
            sonuc_text = "HASTA (Pozitif)"
            renk = "red"
            olasilik = probability[1] * 100
        else:
            sonuc_text = "HASTA DEĞİL (Negatif)"
            renk = "green"
            olasilik = probability[0] * 100

        lbl_sonuc_text.config(text=f"Sonuç: {sonuc_text}", fg=renk)
        lbl_sonuc_prob.config(text=f"Olasılık: %{olasilik:.2f}")

    except ValueError:
        messagebox.showwarning("Uyarı", "Lütfen tüm alanları sayısal olarak doldurunuz.")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")

# --- 3. ARAYÜZ TASARIMI (Tkinter) ---
root = tk.Tk()
root.title("Karaciğer Hastalığı Tahmin Sistemi")
root.geometry("450x650")
root.configure(bg="#f0f0f0")

# Başlık
lbl_baslik = tk.Label(root, text="Karaciğer Hastalığı Risk Analizi", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
lbl_baslik.pack(pady=15)

# Form Alanı
frame_form = tk.Frame(root, bg="#f0f0f0")
frame_form.pack(pady=10)

def create_row(parent, label_text, row, widget_type="entry", options=None):
    tk.Label(parent, text=label_text, bg="#f0f0f0", font=("Arial", 10)).grid(row=row, column=0, sticky="e", padx=10, pady=5)
    
    if widget_type == "entry":
        widget = tk.Entry(parent, width=20)
    elif widget_type == "combo":
        widget = ttk.Combobox(parent, values=options, width=17, state="readonly")
        widget.current(0)
    
    widget.grid(row=row, column=1, sticky="w", padx=10, pady=5)
    return widget

# Giriş Alanları
entry_age = create_row(frame_form, "Yaş:", 0)
combo_gender = create_row(frame_form, "Cinsiyet:", 1, "combo", ["0-Erkek", "1-Kadın"]) # Dataset genelde 0:Erkek 1:Kadın varsayılır
entry_bmi = create_row(frame_form, "Vücut Kitle İndeksi (BMI):", 2)
entry_alcohol = create_row(frame_form, "Alkol Tüketimi (Birim):", 3)
combo_smoking = create_row(frame_form, "Sigara Kullanımı:", 4, "combo", ["0-Hayır", "1-Evet"])
combo_genetic = create_row(frame_form, "Genetik Risk Seviyesi:", 5, "combo", ["0", "1", "2"])
entry_physical = create_row(frame_form, "Fiziksel Aktivite (Saat):", 6)
combo_diabetes = create_row(frame_form, "Diyabet:", 7, "combo", ["0-Yok", "1-Var"])
combo_hypertension = create_row(frame_form, "Hipertansiyon:", 8, "combo", ["0-Yok", "1-Var"])
entry_liver = create_row(frame_form, "Karaciğer Fonksiyon Testi:", 9)

# Hesapla Butonu
btn_hesapla = tk.Button(root, text="ANALİZ ET", command=tahmin_et, bg="#007bff", fg="white", font=("Arial", 12, "bold"), width=20, height=2)
btn_hesapla.pack(pady=20)

# Sonuç Ekranı
frame_sonuc = tk.Frame(root, bg="white", bd=2, relief="groove")
frame_sonuc.pack(fill="x", padx=20, pady=10)

lbl_sonuc_text = tk.Label(frame_sonuc, text="Sonuç Bekleniyor...", font=("Arial", 14, "bold"), bg="white", fg="#555")
lbl_sonuc_text.pack(pady=5)

lbl_sonuc_prob = tk.Label(frame_sonuc, text="Olasılık: -", font=("Arial", 12), bg="white", fg="#555")
lbl_sonuc_prob.pack(pady=5)

# Örnek Değer Doldur Butonu (Test için)
def ornek_doldur():
    entry_age.delete(0, tk.END); entry_age.insert(0, "58")
    combo_gender.current(0)
    entry_bmi.delete(0, tk.END); entry_bmi.insert(0, "35.85")
    entry_alcohol.delete(0, tk.END); entry_alcohol.insert(0, "17.2")
    combo_smoking.current(0)
    combo_genetic.current(1)
    entry_physical.delete(0, tk.END); entry_physical.insert(0, "0.65")
    combo_diabetes.current(0)
    combo_hypertension.current(0)
    entry_liver.delete(0, tk.END); entry_liver.insert(0, "42.7")

btn_ornek = tk.Button(root, text="Örnek Veri Doldur", command=ornek_doldur, bg="#6c757d", fg="white", font=("Arial", 9))
btn_ornek.pack(pady=5)

root.mainloop()