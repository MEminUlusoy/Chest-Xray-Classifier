import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  #? Model yüklerken kullanıcaz bu import'u aşağıda anlıcaksın.

# 1. PARAMETRELER (Eğitim dosyasıyla aynı olmalı)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
test_dir = "../data/test"                     #? Burada tekrar, test data setiyle genel doğruluğu ölçmek istediğim için bu url'yi aldım ve aşağıda da  32'şerli halde resimleri alıyorum ve model.evaluate()  ile test accuracy ve loss değerlerini alıyorum.  Ve bu şekilde yazarak  Normal ve Zatüre label isimlerini alıyorum =>   class_names = test_ds.class_names     .Bu class_names 'i sonra kullanıcam aşağıda
model_path = "../model/chest_xray_model_7.keras"  #? train_model.py 'de eğittiğimiz modeli çağırıyoruz.
os.makedirs("../model/reports", exist_ok=True) #? raporların kaydedileceği dosya olurda oluşturulamazsa diye burada tekrar kontrol ettim. Eğer oluşturulamamışsa , reports dosyası oluşturacak.

# 2. MODELİ YÜKLE
print("Model yükleniyor...")
model = tf.keras.models.load_model(model_path, custom_objects={'preprocess_input': preprocess_input})   #? train_model'de eğttiğimiz modeli yükledik ve model değişkenine attık.  Fakat böyle yazdığımda "serileştirme" (serialization) hatası aldım  =>    model = tf.keras.models.load_model(model_path)     Nedeni ise,  Modeli train_model.py 'de  kaydederken içine bir Lambda katmanı ve bu katmanın içinde preprocess_input fonksiyonunu koydun. Ancak modeli geri yüklerken (load_model), Keras bu "preprocess_input" isminin ne anlama geldiğini ve hangi kütüphaneden geldiğini hatırlayamıyor. Bu sorunu çözmek için Keras'a bu fonksiyonu custom_objects (özel nesneler) parametresi ile tanıtman gerekiyor.  Bunun için yukarıda bu importu yaptık. Yani işini özeti:   Keras modeli .keras dosyasına kaydederken "burada bir fonksiyon var ve adı preprocess_input" diye not alır. Ancak dosyayı açarken (load_model), Keras bu ismin hangi kütüphaneye (MobileNetV2 mi, ResNet mi?) ait olduğunu bilmez. custom_objects sözlüğü ile biz Keras'a şunu diyoruz: "Bak, dosyada gördüğün o 'preprocess_input' ismi aslında benim sana verdiğim bu fonksiyondur." 



# 3. TEST VERİSİNİ HAZIRLA
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,                 #? test_dir  içindeki fotoğrafları ise test dataseti kullanmak için yapıcam. Dikkat ettiysen:  validation_split, subset, seed   kullanmadık. Çünkü veri setini bölmüyorum. Yukarıda train klasörü içindeki verileri böldüğüm için hem validation hemde train için subset,seed gibi parametreler kullanıyorduk. FAkat soldaki bütün dataseti test dataseti olarak kullanacağım için bölmeye gerek yok bu yüzden de bu parametreleri kullanmaya gerek yok.
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)

class_names = test_ds.class_names


# 4. GENEL BAŞARIYI ÖLÇ (Title için lazım)
print("Test seti değerlendiriliyor...")
test_loss, test_acc = model.evaluate(test_ds)  #? Aşağıda for döngüsü içinde kullanıcam. test_loss ve  test_acc değerlerini.  test_acc modelin genel doğruluğunu vericek.




# 5. GÖRSELLEŞTİRME

# 1. Test setinden 1 batch (32 resim) alalım
for images, labels in test_ds.take(1):      #? Burada amacımız test dataseti içindeki resimleri, train_model.py içinde eğittiğimiz modelle tahminde bulundurmak. Yani eğittiğimiz model hiç görmediği test datasetindeki fotoğrafları tahmin edicek. Ve bu fotoğrafları  zatüre mi yoksa sağlıklı mı diye tahmin edip yazıcak ve bu tahminini yüzde kaç doğrulukla bu label tahminlerini yaptı onu da yazıcak. Ve ayrıca gerçek label adınıda yazıp. Bize grafikte gösterecek. 
    predictions = model.predict(images)     #? test_ds 'nin (yani test datasetindeki) ilk 32 resmi alıyor. Çünkü ilk batchi alıyor.  test_ds.take(1) dediği zaman ilk 32 resim gelir test datasetindeki. Burayı anlamadıysan tensorflow ders anlatımındaki en sondaki yapraklarla ilgili yaptığımız projeye bak oradan anlarsın.  
                                            #? images bu test_ds içindeki 32 resmin her birinin piksellerini tutuyor tensor olarak. Labels ise bu 32 resmin her birinin gerçekte hangi label'a ait olduğunun bilgisini tutuyor. Yani sıfır ve birlerden oluşan bir array. ve bu 32 tane çünkü 32 resmin hangi class'a ait olduğunu tutuyor.  Eğer sıfır yazıyorsa ilk indexde, bu 32 resim içindeki ilk resmin Normal sağlıklı olduğunu ifade eder. Eğer 1 yazıyorsa, Bu da 32 resim içindeki ilk resmin zatüre olduğunu ifade eder.    
                                            #?  model.predict() çıktısı şöyledir: [[0.92], [0.12], [0.85], ...]   .   predictions[i] dediğinde, i'inci resmin sonucuna ulaşırsın. Bu sonuç şöyledir: [0.92]      .Bu hâlâ bir listedir. İçindeki yalın sayıya (0.92) ulaşmak için listenin ilk elemanını, yani [0]'ı çağırmamız gerekir.  Neden np.argmax() kullanmadık?   np.argmax(), "en büyük sayının olduğu indeksi (sırayı) ver" demektir.  Softmax (Çok Sınıflı) Olsaydı: Çıktı her resim için [0.1, 0.9] gibi iki nöronlu olurdu. argmax burada 1. indeksteki sayının büyük olduğunu söylerdi.  Sigmoid (İkili) Olduğu İçin: Bizim modelimizde sadece tek bir çıkış nöronu var. Çıktı sadece tek bir sayı: [0.92].  Eğer tek bir sayıya argmax uygularsan, o sayı kaç olursa olsun her zaman "0. indeksteki sayı en büyüktür" diyerek sana 0 döndürür. Bu da modelin her şeye "Normal" demesine sebep olurdu.
                                            #?     
    # Grafik alanını oluşturuyoruz
    plt.figure(figsize=(16, 12))          #? Bir figür oluşturacaz. 8 tane resimi gösterip, eğittiğimiz model ne tahmin ediyor, yüzde kaç doğrulukla tahmin ediyor ve gerçekte asıl label adı ne bunun bilgisini tutan bir grafik oluşturacaz. Bunun için önce figsize yazdık.
                                          #?   plt.figure(): "Şimdi bir grafik alanı oluştur" komutudur. Eğer bunu yazmazsan Python varsayılan, küçük bir pencere açar.   figsize=(16, 12): Kağıdın boyutudur. İlk rakam (16) genişliği, ikinci rakam (12) yüksekliği temsil eder (birim inçtir).  Neden bu değerleri verdik?: Biz ekranda 8 tane (2 satır, 4 sütun) resim göstereceğiz. Eğer küçük bir değer verseydik resimler iç içe girer, yazılar okunmazdı. 16x12, 8 resmi ferah ferah göstermek için ideal bir "sinema ekranı" formatıdır.         

    # --- BURASI YENİ: GENEL BAŞARIYI ANA BAŞLIK OLARAK EKLE ---
    # test_acc zaten model.evaluate() kısmından elimizde var
    plt.suptitle(f"Model Genel Test Doğruluğu: %{test_acc*100:.2f}\n(Örnek Tahminler)",     #? Figürün en başında başlık olarak. test_ds 'nin genel doğruluğunu başta yazdıracaz. Sonra tek tek fotoğrafların tahminlerini gösterecez. Yani bu soldaki kod, evaluate() ile yaptığımız doğruluğu yazdırmak içindi. Aşağıdaki kodlar ise predict() ile yaptırdığımız kodlar için olacak.  Ne İşe Yarar?: Eğer bir pencerede (figure) birden fazla alt grafik (subplot) varsa, her birinin kendi başlığı (title) olur. suptitle() ise tüm bu küçük grafiklerin en tepesine, tüm sayfayı kapsayan genel bir başlık atar.  Farkı Nedir?: plt.title() sadece o anki küçük kutunun başlığıdır; plt.suptitle() ise tüm tablonun ismidir.
                 fontsize=20, fontweight='bold', color='darkblue')
    # --------------------------------------------------------



    for i in range(8):      #? İlk 8 resmi göstermek için for içinde 0'dan 8'e kadar(8 dahil değil) giden sayılar oluşturdum. Sırayla  "i" değeri bu değerleri alarak, bize resimleri gösterecek.
        ax = plt.subplot(2, 4, i + 1)   #? 2 satır 4 sütun şeklinde gösterilecek resimler ve bu subplot  1'den başlamak zorunda olduğu için ve bizim "i" değerimiz başta 1 olduğu için   i+1  şeklinde yazdık
        
        # Resmi göster
        img = images[i].numpy().astype("uint8")   #? images değişkeni 32 resmin her birinin piksellerini tutuyor demiştik.  images[i]  diyerek her döngüde sırayla bu 32 resimden birini gösterecek.  "i" değerleri  =>  0,1,2,3,4,5,6,7    şeklinde olacağı için,  32 resim içindeki bu indexlerdeki resimleri tek tek gösterecek.  plt.imshow() ile.  Fakat göstermeden önce bu resimler tensor olarak tutulduğu için önce numpy()  array'e çevirmek lazım bu yüzden numpy() yazdık.  Ayrıca  , astype("uint8") yazmazsak resimler abuk subuk gösteriyordu.
        plt.imshow(img)                       #? Burada ise img değişkenini gösteriyoruz grafikte.
        
        # Etiketler ve Olasılıklar
        true_label = class_names[int(labels[i])]   #? labels değişkeni, test datasetindeki  gerçek label değerlerini tutuyordu. 0 ve 1 şeklinde. Biz bu ilk 8 verinin sıfır ve 1 değerlerini nolur nolmaz önce integer'a çevirdik.  Ve 32 resmin içindeki ilk 8 verinin label'ını yani sıfır ve 1 değerini aldıktan sonra,  class_names  içine verdik çünkü class_names = ["NORMAL","PNEUMONIA"] şeklinde array'di, Biz sıfırsa zaten NORMAL,  1 ise  PNEUMONIA diyorduk. Bu yüzden sıfır ve 1 değerlerini class_names içine vererek  "NORMAL" ve "PNEUMONIA"  isimlerini almış oluyoruz. Böylece sıfır ve 1 tutmak yerine daha anlamlı olan => "NORMAL"  ve "PNEUMONIA"  label isimlerini tutuyoruz.
        prob = predictions[i][0]                    #? Nedenini model.predic() yanında yazdım. Buradan 0.92 gibi bir değer alıyoruz örneğin veya 0.30 gibi. Her döngüde , Sadece tek bir değer alıyoruz dikkat et.  Yani 8 tane resim için sırayla 8 tane olasılık alıcaz. Ve bu olasılıklar yüzde kaç doğruluk oranıyla her resmi tahmin ediyor onu ifade ediyordu. Ve bunu prob değişkenine atıyoruz
        pred_label = "PNEUMONIA" if prob > 0.8 else "NORMAL"   #? Burada diyoruz ki eğer prob değişkeni 0.5 'den büyükse  tahmin ettiğimiz sınıf  PNEUMONIA(Zatüre) olsun , eğer küçüksede tahmin ettiğimiz label NORMAL(yani sağlıklı) olsun diyoruz. Çünkü 1 ise label değerimiz PNEUMONIA(Zatüre) oluyordu  bu yüzde 0.5'den büyük yani 1'e yakın değerler için bu label etiketini veriyoruz, sıfır label'ı ise NORMAL(sağlıklı) görseller için kullanıldığı içinde 0.5'den küçük değer olursada label değerimiz NORMAL yazsın dedik. Amaç bu.   Neden prob > 0.5 gibi bir şart yazdık?  Sigmoid fonksiyonu sana bir "kimlik" değil, bir olasılık verir.  0.0: Kesinlikle Sağlıklı (Normal)  ,  1.0: Kesinlikle Hasta (Zatürre),  0.5: Tam sınır (Yazı-tura gibi).   Matematiksel bir karar vermemiz gerekiyor. Biz de diyoruz ki: "Eğer bu olasılık %50'den büyükse (> 0.5), artık bu 1'e (Pneumonia) yakındır. Eğer küçükse 0'a (Normal) yakındır."
        
        # Renk ve Güven Skoru
        color = "green" if pred_label == true_label else "red"  #? Burada yaptığımız şey sadece renk belirlemek için yazdık. Ya yeşil ya da kırmızı yazısını tutacak color değişkeni. Bu değeri aşağıda color parametresi içine vererek görsel eğer doğru tahmin etmişse label'ı yeşil olarka gözükecek, eğer yanlış tahmin etmişse kırmızı olarak gözükecek tabloda.

        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100  #? Aslında model, 0 ile 1 arasındaki tüm aralığı Zatürre için kullanır.   Prob değerleri eğer böyleyse =>    0.99: "Neredeyse eminim, bu %99 Zatürre."    ,  0.01: "Zatürre olma ihtimali neredeyse %1."       Zatürre olma ihtimali %1 ise, o zaman sağlıklı olma ihtimali %99'dur.  Yani zatüre olma ihtimli bu görselin yüzde 1  demek yerine sağlıklı olma ihtimali yüzde 99 diyoruz
                                                                     #? Neden 1'den çıkarıyoruz?  Bir tarafta Sağlıklı (0), diğer tarafta Zatürre (1) var. Tam ortası ise 0.5 noktası.    Eğer model bir resim için 0.80 değerini veriyorsa: "Bu resmin %80 ihtimalle Zatürre olduğunu düşünüyorum" diyordur.   Eğer model 0.20 değerini veriyorsa: "Bu resmin %20 ihtimalle Zatürre olduğunu düşünüyorum" diyordur.  %20 ihtimalle Zatürre olan bir resim için biz ne deriz? "%80 ihtimalle Sağlıklı" deriz. İşte o 80 rakamına ulaşmak için   1 - 0.20 = 0.80   işlemini yaparız.  Örneğin:  Değer: 0.80 (> 0.5)  , Karar: PNEUMONIA (Zatürre), Confidence Hesabı:  0.80 * 100 = %80  ,   Yorum: Model %80 güvenle "Bu hasta zatürre" diyor.    Başka örnek =>  Değer: 0.47 (< 0.5),  Karar: NORMAL (Sağlıklı) ,   Confidence Hesabı: (1 - 0.47) * 100  =>  0.53 * 100 = %53  ,  Yorum: Model %53 güvenle "Bu çocuk sağlıklı" diyor.   Neden direkt 47 ile çarpmadık? Eğer ekrana "Tahmin: NORMAL (%47.00)" yazdırsaydık, bu kafa karıştırıcı olurdu. Çünkü %47 düşük bir oran gibi duruyor. Halbuki model "Normal" olduğuna %53 ihtimal veriyor (merkeze yani 0'a daha yakın). Biz her zaman kazanan tarafın yüzdesini göstermek istiyoruz.
                                                                     #? Soldaki kod şunu söylüyor =>   Eğer sayı 0.5'ten büyükse; zaten 1'e (Zatürreye) yakındır, sayıyı direkt al ve yüzdeye çevir.   Değilse (yani 0.5'ten küçükse); bu sayı 0'a (Sağlıklıya) daha yakındır. 0'a olan uzaklığını (yani gerçek güven oranını) bulmak için onu 1'den çıkar ve sonra yüzdeye çevir.  Peki neden sigmoid kullandık ?  Eğer senin 2 tane ayrı nöronun (Softmax) olsaydı çıktı şöyle olurdu:  [Sağlıklı: 0.80, Zatürre: 0.20]     Ama biz tek nöron (Sigmoid) kullanıyoruz ve o bize sadece Zatürre tarafını söylüyor: 0.20.   Biz de diyoruz ki: "Eee, Zatürre %20 ise Sağlıklı zaten %80'dir. Boşuna ikinci bir nöronla sistemi yormayalım, biz matematiksel olarak 1 - 0.20 yaparak Sağlıklı'yı buluruz."          Softmax kullandığında (yani 2 nöronun olduğunda), model.predict() sana her resim için tek bir sayı değil, iki sayılık bir liste verirdi:    Örnek çıktı: [0.20, 0.80] (Birincisi "Normal" ihtimali, ikincisi "Zatürre" ihtimali)     Bu durumda predictions[i] dediğinde eline şu liste geçerdi: [0.20, 0.80].        İşte burada [0] yazmak yerine np.argmax() kullanırdın çünkü listenin içindeki en büyük sayının konumunu bulman gerekirdi.
                                                                     #? İster Sigmoid kullan ister Softmax, TensorFlow her zaman sonuçları Batch (Paket) halinde döndürür. Yani sen modele 32 resim verdiysen, model sana her zaman 32 satırlık bir matris verir.   Sigmoid'de (Bizimki): [[0.92], [0.12], [0.85]] -> Her satırda 1 sayı var.   Softmax'ta: [[0.08, 0.92], [0.88, 0.12], [0.15, 0.85]] -> Her satırda 2 sayı var.     Yani her iki durumda da o "dıştaki" liste (batch) hep var. predictions[i] diyerek o satıra girdikten sonra;   Sigmoid'de: [0] diyerek o tek sayıyı paketten çıkarıyoruz    Softmax'ta: np.argmax() diyerek hangi kutunun daha dolu olduğuna bakıyoruz.


        
        plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label} (%{confidence:.2f})", color=color)   #? Bütün verileri resimlerin üstüne title olarak yazar ve color parametresinede yukardaki color değişkenini koyar.
        plt.axis("off")       #? bunu yazmazsam resmin altında ve üstünde x ve y eksenleri gözükür. Bu o eksenleri göstermiyor.


    plt.tight_layout(rect=[0, 0, 1, 0.95]) #? Bu fonksiyon, grafiğin içindeki elemanların (resimler, başlıklar, eksenler) birbirine çarpmasını engeller, araları otomatik açar.  rect=[sol, alt, sağ, üst]: Bu, grafiğin "çerçeve sınırlarını" belirler. Değerler 0 ile 1 arasındadır.   (Döngü bitince, 8 resim de kağıda dizilince, şimdi araları açmak için bunu kullandık.)  En sondaki 0.95, grafiğin en üstünden %5'lik bir boşluk bırak demektir.  Neye göre verdik?: Biz plt.suptitle ile en tepeye dev bir başlık yazdık ya; eğer bunu 1.0 yapsaydık, o dev başlık en üstteki resimlerin üzerine binerdi. Biz resimlere "Siz biraz aşağıda durun (0.95'te bitin), en üstteki %5'lik alan başlığa kalsın" demiş olduk.   sol = 0: Grafiğin en sol sınırını 0 noktasına (kağıdın en soluna) yasla demektir. Burada 0 vererek soldan ekstra bir boşluk bırakmadık, kağıdı tam kullandık.   alt = 0.03: Grafiğin alt sınırını %3 yukarıdan başlat demektir. Yani kağıdın en altında %3'lük çok ince bir boşluk bıraktık. Bu, genellikle alttaki eksen yazıları (etiketler) kağıdın dışına taşmasın diye yapılır.  sağ = 1: Grafiğin sağ sınırını 1 noktasına (kağıdın en sağına) yasla demektir. Sağından hiç boşluk bırakmadık, sonuna kadar kullandık.
    plt.savefig("../model/reports/model_success_samples.png")  #? plt.savefig(): Hazırladığın bu şık tabloyu bir dosya (PNG, JPG vb.) olarak bilgisayara kaydeder. GitHub'da paylaşmak için bu dosya şarttır.
    plt.show()  #? plt.show() Sadece savefig'i mi Gösterir?  Hayır, aslında ikisi birbirinden tamamen bağımsızdır.  plt.show(): Bu, Python'ın bilgisayarının ekranında canlı bir pencere açmasını sağlar. O pencere açıldığında grafiği inceleyebilir, yakınlaştırabilir veya manuel olarak kaydedebilirsin. Eğer bu kodu yazmazsan, program arka planda her şeyi çizer ama sana hiçbir şey göstermeden biter.   plt.savefig(): Bu, ekranda bir şey gösterip göstermediğine bakmaksızın, o anki çizimi sessizce bir dosyaya (.png, .jpg) yazar.  Özetle: savefig dosyaya yazar, show ise senin gözüne gösterir.

#? Neden figure, tight_layout ve show dışarıda?:  Çünkü biz 8 farklı resim için 8 ayrı pencere istemiyoruz. Bir tane büyük pencere açıyoruz (figure), içine 8 resmi diziyoruz, işimiz bitince de o tek pencereyi düzenleyip kaydediyoruz.
#? Neden take(1) olan döngünün içindeler?: Aslında teknik olarak take(1) sadece bir kez çalışacağı için içerde veya dışarda olması sonucu değiştirmez. Ancak images ve labels değişkenleri bu döngünün içinde doğduğu için, resimlere ulaşıp onları subplot ile kağıda basan kodlar mecburen bu döngünün içinde olmalı.
#?  plt.figure() İçinde Resimleri Gösteren Şey Ne?  plt.subplot(2, 4, i + 1): Bu kod, o boş çerçeveyi hayali bir ızgaraya böler. "Benim 2 satırlık ve 4 sütunluk bir masam var, şimdi sıradaki resmi şu numaralı kutuya koy" der.   plt.imshow(img): İşte resmi o kutunun içine fırçayla çizen asıl komut budur. "Bu sayısal veriyi (pikselleri) al ve onu bir resim olarak o kutuya bas" der.
#?  Mekanizmanın Tam Akışı (Adım Adım):
#?  plt.figure(): Boş bir sergi salonu kiralar.,  plt.subplot(): Salondaki duvarlara resim asılacak yerleri (çerçeveleri) işaretler.  plt.imshow(): Her bir çerçeveye gidip resmi içine yerleştirir.  plt.title(): Resmin altına/üstüne etiketini yapıştırır.  plt.savefig(): Salonun bir fotoğrafını çeker ve albüme (klasöre) koyar. plt.show(): Salonun kapılarını açar ve seni içeri davet eder.


# ---  CONFUSION MATRIX (HATA MATRİSİ) ---
print("\nConfusion Matrix için tüm test seti taranıyor...")

y_true = []  #? test_ds içindeki gerçek label'ları (etiketleri) tutacak
y_pred = []  #? test_ds içindeki resimleri model.predict() ile tahmin edip, tahmine ettiği etiketleri tutacak

for images, labels in test_ds:  #? Yukarıda zaten test_ds yapmışken şimdi niye burada tekrar test_ds çağırdık ? Yukarıda yazdığın test_ds.take(1) kodu, test setinden sadece ilk 32 resmi (1 batch) çekip bırakmıştı.  Görselleştirme için: Sadece 8 veya 32 resim yeterlidir.   Confusion Matrix için: Test setindeki bütün resimlerin (örneğin 624 resim) tahmin edilmesi gerekir. Sadece 32 resimle matris yaparsan, modelin başarısını tüm veri setinde ölçemezsin. Bu yüzden   for images, labels in test_ds:   diyerek listenin en başından en sonuna kadar her batch'i (her 32'lik grubu) tek tek geziyoruz.

    y_true.extend(labels.numpy().astype(int).flatten())  #? Neden append kullanmadık ?   append(): İçine ne verirsen onu tek bir paket olarak listenin sonuna ekler.  extend(): İçindeki paketi açar, elemanları tek tek listenin içine saçar. Aşağıda daha detaylı anlıcaksın.    labels değeri labels.numpy() yazamadan önce batch halinde etiketler alındığı için tensör şeklinde veri tutuyordu yani bu şekilde => <tf.Tensor: shape=(32,), dtype=float32, numpy=array([0., 1., 0., ...])>     Fakat biz Bu objeyi doğrudan Python listesine atarsak, scikit-learn kütüphanesi bunu anlamaz. labels.numpy() diyerek onu saf bir NumPy Array ([0, 1, 0...]) haline getiriyoruz.  Bu işlem sonrasında y_true şöyle gözükecek => Batchlerimiz 32'şerli olduğu için 32'li şekilde veriler y_true içine eklenecekti [0,1,....,1] gibi Böyle her bir 32 'şerli grup eklenirse ve bunu append ile yapmaya kalkarsak böyle olurdu => Eğer append kullansaydık: [[0, 1,...., 0], [1, 1,...., 0]] (Liste içinde listeler olurdu - Yanlış)      .Ama extend kullandığımız için =>  [0, 1, 0, 1, 1, 0] (Dümdüz bir liste olur - Doğru) . Ayrıca Yukarıda label_mode="binary" kullandığın için labels verisi [[0.], [1.], [1.]] şeklinde 2 boyutlu (matris) olarak gelir. Sklearn confusion_matrix bunu genelde tolere edip kendi düzeltse de, bazen versiyon farklılıklarından dolayı "Boyutlar uyuşmuyor" hatası fırlatabilir.  Böyle sorun olmaması için bu ondalık sayıları integer'a ve sonrada flatten ile düzleştirmelisin. flatten ne iş yapıyor aşağıda daha iyi anlıcaksın. Şuanlık böyle bil

    preds = model.predict(images, verbose=0) #? Neden verbose=0 Yazdık?  model.predict() normalde çalışırken ekrana bir ilerleme çubuğu (progress bar) basar: 1/1 [==========].   Biz zaten bir for döngüsünün içindeyiz ve her batch için bu fonksiyonu çağırıyoruz. Eğer verbose=0 (sessiz mod) demezsek, ekran yüzlerce "1/1" satırıyla dolar ve görmek istediğin çıktıları bir arada görmen zorlaşır.    verbose=0 sadece "sessizce arka planda tahmini yap ve sonucu ver" demektir.
    
    y_pred.extend((preds > 0.8).astype(int).flatten())  #? yukarıdaki   test_ds.take(1)  için olan for döngüsünde biz   predictions    değişkenine zaten predict ile atmıştık şimdi neden bidaha böyle prediction hesaplıyoruz dersen ?  Yukarıdaki predictions değişkeni sadece test_ds.take(1) ile aldığın ilk 32 resmin tahminlerini tutuyordu.   Confusion matrix için bize tüm test setinin tahminleri lazım. Bu yüzden döngü içerisinde her gelen yeni images batch'i için taze bir tahmin alıyoruz ki bütün veri setini kapsamış olalım. 
                                                         #? preds değişkeni modelden çıktığında şöyledir: [[0.98], [0.12], [0.85]]. Bizim bunu [1, 0, 1] haline getirmemiz lazım. Adım adım bakalım:   preds > 0.5: Olasılıkları doğru/yanlış yapar. Sonuç: [[True], [False], [True]]     .astype(int): Boolean değerleri sayıya çevirir. Sonuç: [[1], [0], [1]]    .flatten(): İç içe geçmiş listeyi dümdüz eder. Sonuç: [1, 0, 1] şeklinde olur her batch için.   y_pred.extend(...): Her batchden gelen düz verileri yukardaki olay gibi bide bu düzleştirir. Bu temizlenmiş sayıları ana listeye ekler. Sonuçta y_pred şuna benzer: [1, 0, 1, 1, 0, 0, 1 ...] (Yüzlerce tahminin arka arkaya dizildiği düz bir liste).  

# Matrisi hesapla
cm = confusion_matrix(y_true, y_pred)  #? Yani yukardaki olayların hepsini confision_matrix içine verebilmek için yaptık. Çünkü confision_matrix içine  test verisetindeki bütün gerçek label'lar ve  test verisetindeki resimleri predict eden tahmin edilmiş label'lar lazım. Eğer elimizde bu ikisi varsa confision matrix yapabiliriz. Peki neden   tf.math.confusion_matrix(labels=y_test,predictions=y_pred)   şeklinde yazmadık ?  Neden tf.math yerine sklearn kullandık?  Aslında her ikisi de aynı matematiksel işlemi yapar, ancak kullanım amaçları farklıdır:   tf.math.confusion_matrix: Genelde eğitim (training) sırasında, modelin içinde bir metrik hesaplamak istediğinde kullanılır. Çıktısı bir "Tensor"dur ve görselleştirmesi daha zahmetlidir. sklearn.metrics.confusion_matrix: Veri bilimi dünyasının standartıdır. Python listeleri ve NumPy dizileriyle doğrudan çalışır. Ayrıca classification_report gibi yan araçlarla tam uyumludur.  Hangisi daha iyi? Analiz ve raporlama aşamasındaysan sklearn çok daha pratiktir. Modelin içine özel bir "loss" fonksiyonu yazıyorsan tf.math kullanılır. 
                                       #?  Neden labels ve predictions parametre isimlerini yazmadık içine ?    confusion_matrix(y_true, y_pred) -> Sklearn kural olarak ilk sıradakini gerçek, ikinciyi tahmin kabul eder. Sırası önemli yani yazarken. Daha kısa olsun diye kod yazmadık.  


# Matrisi Görselleştir (Heatmap)
plt.figure(figsize=(8,6))
sns.heatmap(cm,           #? yukardaki confision_matrixden aldığımız değeri heatmap içine koyuyoruz görselleştirmek için.
            annot=True,   #? annot=True  =>	Kutuların içine sayıları yazar. Sayıları (örneğin 150 tane doğru tahmin) gözle görmek için.
            fmt="d",      #? fmt='d'     => Sayı formatını belirler ('d' = decimal/tam sayı). Eğer bunu yazmazsan sayılar bilimsel notasyonla (1.5e+02 gibi) çirkin görünür.
            cmap="Blues", #? Renk paletini belirler. Klasik, profesyonel bir mavi tonu verir. Değer arttıkça renk koyulaşır.
            xticklabels= class_names, #? X ekseni (Tahmin) isimleri. Heatmap'de grafikte x ve y ekseninde,  0 ve 1 yerine "NORMAL" ve "PNEUMONIA" yazması için class_names verdik.
            yticklabels= class_names) #? Y ekseni (Gerçek) isimleri. Sol tarafta neyin ne olduğunu anlamak için yine isimleri verdik. Aynı şey üsttekiyle

plt.title(f"Confusion Matrix\nGenel Doğruluk: %{test_acc*100:.2f}")
plt.xlabel("Tahmin Edilen (Predicted)")  #? Her zaman xlabel predicted için midir heatmap'de ?  Evet , X Ekseni (Horizontal): Her zaman Tahmin Edilen (Predicted) değerleri temsil eder 
plt.ylabel("Gerçek Değer (True)")        #? Y Ekseni (Vertical): Her zaman Gerçek (True) değerleri temsil eder.

# Grafiği Kaydet
plt.savefig("../model/reports/confusion_matrix.png")  #? Yine reports içine kaydedicez her çalıştırdığımızda eski olanın üzerine yeni dosya yazılacağı için hep en güncel olan rapor gözükecek.
plt.show()

# Metinsel Rapor (Precision, Recall, F1-Score)
print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred, target_names=class_names))  #? classification report içinede önce y_true sonra y_pred yaz.  target_names parametresi ise, bu parametreyi vermezsen raporunda precision, recall yanında 0 ve 1 gözükür. Ama sen 0 ve 1'in hangi class olduğunu hatırlamayabilirsin. Burada 0 ve 1 yazmak yerine class isimleri yazsın daha anlaşılır olsun diye bu parametreyi kullanıyoruz. target_names=class_names yazdığında ise rapor otomatik olarak 0 yerine "NORMAL", 1 yerine "PNEUMONIA" yazar. Bu, GitHub'da projenin README dosyasını okuyan birinin hiçbir teknik bilgiye sahip olmasa bile sonuçları anlamasını sağlar.   




















