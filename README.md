🩺 Chest X-Ray Pneumonia Detection (AI-Powered)
Bu proje, göğüs röntgeni (X-Ray) görüntülerinden Zatürre (Pneumonia) teşhisini otomatize eden yüksek doğruluklu bir Derin Öğrenme modelidir.

🚀 Proje Özeti
Model, tıbbi görüntüleme verilerindeki dengesiz dağılımı ve düşük kontrast sorunlarını aşarak %91 doğruluk (Accuracy) oranına ulaşmıştır. Gerçek dünya senaryolarına uygun, "False Negative" (hastayı kaçırma) oranını minimize eden bir yapı kurulmuştur.

🛠️ Kullanılan Teknolojiler
Dil: Python

Kütüphaneler: TensorFlow, Keras, OpenCV, NumPy, Matplotlib

Model: MobileNetV2 (Transfer Learning)

Veri Yönetimi: Google Colab (GPU T4), Google Drive Integration

📊 Model Performansı ve Görseller
1. Confusion Matrix (Hata Matrisi)
Açıklama:

Dengeli Tahmin: Modelimiz "Normal" sınıfta 27, "Zatürre" sınıfında 29 hata yaparak son derece dengeli bir performans sergilemiştir.

Güvenilirlik: Sınıflar arasındaki hata dağılımının birbirine yakın olması, modelin herhangi bir sınıfa yanlılık (bias) göstermediğini ve doku bozukluklarını gerçekten ayırt edebildiğini kanıtlar.

2. Model Success Samples (Tahmin Örnekleri)
Açıklama:

Bu görselde modelin test setindeki rastgele seçilmiş resimler üzerindeki tahminlerini görebilirsiniz.

Model, farklı kontrast ve açıdaki röntgenlerde bile yüksek güven (Confidence) skorları ile doğru teşhis koyabilmektedir.