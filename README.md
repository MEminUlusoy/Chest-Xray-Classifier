# 🩺 Chest X-Ray Pneumonia Detection (AI-Powered)

Bu proje, göğüs röntgeni (X-Ray) görüntülerinden **Zatürre (Pneumonia)** teşhisini otomatize eden yüksek doğruluklu bir Derin Öğrenme modelidir.

## 🚀 Proje Özeti
Model, tıbbi görüntüleme verilerindeki dengesiz dağılımı ve düşük kontrast sorunlarını aşarak **%91 F1-Score (Dengeli Başarı)** oranına ulaşmıştır. Gerçek dünya senaryolarına uygun, "False Negative" (hastayı kaçırma) oranını minimize eden bir yapı kurulmuştur.

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python 3.11
* **Kütüphaneler:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib
* **Model:** MobileNetV2 (Transfer Learning)
* **Veri Yönetimi:** Google Colab (GPU T4), Google Drive Integration

## 📊 Model Performansı ve Görseller

### 1. Confusion Matrix (Hata Matrisi)
![Confusion Matrix](model/reports/confusion_matrix.png)

**Açıklama:**
* **Dengeli Tahmin:** Modelimiz "Normal" sınıfta 27, "Zatürre" sınıfında 29 hata yaparak son derece dengeli bir performans sergilemiştir.
* **Güvenilirlik:** Sınıflar arasındaki hata dağılımının birbirine yakın olması, modelin herhangi bir sınıfa yanlılık (bias) göstermediğini ve doku bozukluklarını gerçekten ayırt edebildiğini kanıtlar.

### 2. Model Success Samples (Tahmin Örnekleri)
![Model Success Samples](model/reports/model_success_samples.png)

**Açıklama:**
* Bu görselde modelin test setindeki rastgele seçilmiş resimler üzerindeki tahminlerini görebilirsiniz.
* Model, farklı kontrast ve açıdaki röntgenlerde bile yüksek güven (Confidence) skorları ile doğru teşhis koyabilmektedir.

### 3. Classification Report (Sınıflandırma Raporu)
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Normal** | 0.89 | 0.89 | 0.89 |
| **Pneumonia** | 0.93 | 0.92 | 0.93 |
| **Average / Total** | **0.91** | **0.91** | **0.91** |

**Analiz:**
* **Recall (Duyarlılık):** Zatürre sınıfındaki yüksek Recall değeri, hasta olan bireylerin doğru tespit edilme oranının çok yüksek olduğunu gösterir (Tıbbi projelerde en kritik metrik).
* **F1-Score:** %91'lik F1 skoru, modelin hem kesinlik hem de duyarlılık arasında kusursuz bir denge kurduğunu belgeler.

## 💡 Uygulanan Teknik Stratejiler
* **Data Augmentation:** `RandomFlip` ve `Rescaling` ile modelin geometrik varyasyonlara karşı dayanıklılığı artırıldı.
* **Class Weighting:** Veri setindeki dengesizliği (Imbalance) gidermek için sınıflara özel ağırlıklar verildi.
* **Label Smoothing:** Gerçek dünyadaki etiket gürültüsünü (Label Noise) kompanse etmek için etiket yumuşatma uygulandı.
* **Learning Rate Scheduler:** Eğitim sırasında performansın durakladığı anlarda hız otomatik düşürülerek en iyi sonuca odaklanıldı.
