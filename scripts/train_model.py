import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt

#! Kodu çalıştırmak için   C:\Users\Ulusoy\Desktop\YAPAY ZEKA PROJELER\Chest_Xray_Project>   terminal soldaki gibiyken => cd scripts   yaz ve sonra böyle olacak =>   C:\Users\Ulusoy\Desktop\YAPAY ZEKA PROJELER\Chest_Xray_Project\scripts>     Sonrasında ise => python train_model.py    şeklinde yaz ve sonra train_model.py çalışacak. Sonra ise visualize_results çalıştırmak için yine scripts klasörü içinde olduğundan emin ol ve bu sefer  python visaulize_results.py  demen gerekecek.

data_dir = "../data"   # Biz   FaceID_Project   projesinde, data klasörü içinde fotoğraflarım vardı hatırlarsan emin ve ahmet klasörleri içinde. Fakat bu projede Biz kaggle sitesindeki datasets bölümüne gidip seçtiğim projedeki download kısmına bastıktan sonra  Download as Zip dedim ve zip olarak dataseti indirdim. Ve içinde fotoğrafları tutan test,train ve val adında klasörler vardı ve bu klasörlerin hepsinin içinde NORMAL(normal, sağlıklı) ve PNEUMONIA (zatüre)  adında klasörler vardı ve bunların içinde ise NORMAL klasörü içinde sağlıklı 0-5 yaş arası çocukların akciğer röntgenleri,  PNEUMONIA klasörleri içinde ise bakterili veya virüslü şekilde zatüre olan çocukların akciğerlerinin röntgenleri vardı. Biz train, test ve val klasörlerini direk data diye klasör oluşturup onun içine attım. 
                       # Şimdi şuraya dikkat et. FaceID_Project'de direk data klasörü içinde emin ve ahmet adında klasörler vardı ve içinde resimler vardı. FAkat şimdi data klasörü içinde train,test ve val klasörleri var ve onların içinde NORMAL ve PNEUMONIA  adında klasörler var. Yani aslında emin ve ahmet klasörleri sanki train,test ve val klasörleri içindeymiş gibi düşün. Bu yüzden biz aşağıda  data_dir  değişkenin tuttuğu  ../data  yolunun içini   ../data/train  diyerek train dosyası içine girmeyi sağlıcaz ve train içindeki NORMAL, PNEUMONIA dosyalarına yani resimleri tutan dosyalara ulaşabilecez    ve ../data/test   diyerekte   test dosyasının içine girmiş olacaz ve yine bu içindeki resimleri tutan NORMAL ve PNEUMONIA dosyalarına erişebilicez. Dikkat edersen ../data/val  klasörü içine girmiyorum çünkü içindeki NORMAL ve PNEUMONIA klasörleri içindeki resimler çok az sayıda bu yüzden validation için gerekli olan resimleri ben train içinden çekicem. Aşağıda anlıcaksın.



train_dir = os.path.join(data_dir, "train")  #? ../data/train   şeklinde yolu birleştirmiş oldu bu kod.  Bu kod sayesinde ben train dosyası içindeki NORMAL ve PNEUMONIA  klasörlerine erişebilicem.  FaceID_Project 'de böyle bir şey yapmıyordum çünkü zaten ../data   adresi zaten emin ve ahmet klasörlerine erişebilmeme yetiyordu. Fakat soldaki durumda train içine girmem lazım çünkü resimleri tutan klasörlerin adresine kadar girmen lazım etiketleme yapabilmek için resimleri. Yani,  aşağıda biz   image_dataset_from_directory()   fonksiyonu ile train_dir değişkeni vererek, bu NORMAL klasörü içindeki her resime sıfır etiketini yapıştırıyoruz,  PNEUMIONIA klasörü içindeki her resime ise, 1 etiketini yapıştırıyoruz. Böyle yapabiliyoruz çünkü train_dir değişkenin tuttuğu adresde resimleri tutan klasörler var.   Eğer direk   data_dir  demiş olsaydık alttaki fonksiyonda bu sefer  test,train ve val klasörlerine sıfır ve 1 verirdi bu klasörlerde içinde resim bulundurmadığı için saçma bi durum olurdu. Yani etiketleme problemi yaşanırdı. Önceki projeden farkı bu yani. Biz önceki projede train,test,val diye klasörlerimiz yoktu fakat şimdi var ve bu klasörler içine girip bu train, test, val için ayrılmış fotoğrafları alıp eğiticem. Bunları almak içinde fotoğrafları tutan klasörlerin bir üst adresine kadar girmem lazım. Eğer train içinden Normal içine girmeye çalışsaydık. Normal klasörü içinde 100 tane sağlıklı röntgen resmi var diyelim bunların her birine bir class etiketi atamya çalışırdı 0,1,2 gibi yine saçma olurdu. 
test_dir = os.path.join(data_dir, "test")    #? Dikkat et validation set için fotoğraflar az olduğu için onun adresini almadım. Validation için gerekli olan yeteri sayıdaki fotoğrafları train dosyası içindeki fotoğrafları bölerek alıcam. Yani train klasörü içindeki NORMAL ve PNEUMONIA içindeki fotoğrafların yüzde 80'nini train için ayırıcam yüzde 20'sini ise validation için ayırıcam. Bu yüzden aşağıda hem train_ds  , hemde val_ds  için   train_dir  yolunu kullandım dikkat et. Yani train klasörü içindeki NORMAL ve PENUMONIA klasörlerine etiket vermiş oluyorum 0 ve 1 diye ve bu  train klasörü içindeki fotoğrafların yüzde 80'inini  train_ds içine atıyorum  ve yüzde 20'ini val_ds içine atıyorum



IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,                   #? train_dir  içinden %80 'ini training için verdik
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"         #?  label_mode = "binary"   Nedir?   Klasördeki verileri 0 ve 1 olarak (sayısal) etiketle demektir. (Örn: Normal = 0, Zatürre = 1).  Ne zaman kullanılır: Sadece iki sınıfın olduğu (Evet/Hayır, Kedi/Köpek) durumlarda kullanılır. Neden kullanılır: Bellekten tasarruf sağlar. Eğer categorical seçseydin verileri [1, 0] veya [0, 1] gibi vektörler olarak tutardı, bu da gereksiz yer kaplardı.   Yazmazsak Ne Olur?   image_dataset_from_directory() fonksiyonunda label_mode varsayılan (default) olarak   int   gelir. Yani bir şey yazmazsan   int  gelir. Önceki projede bunu yazmamıştık ve klasörde sadece 2 klasör olduğu için bunları 0 ve 1 diye numaralandırmıştı. Yani  int  olarak etiketlenmişti.    Neden şimdi binary yazdık? binary modunda etiketler float32 tipinde ve [batch_size, 1] şeklinde bir matris olarak döner. Bu, binary_crossentropy kaybı (loss) hesaplanırken en optimize ve sorunsuz haldir. Yazmazsan hata almazsın ama açıkça belirtmek "temiz kod" prensibidir. Yani  model.compile() fonksiyonunda binary_crossentropy kullanacaksan bunu yaz.  
)                                 #?  Başka ne gibi label_mode'lar var ?     int => 0, 1, 2, 3 gibi tam sayılar verir. Çok sınıflı ama basit etiketleme istediğinde kullan.   categorical =>  One-hot encoding yapar ([1, 0, 0] gibi).  Çok sınıflı (Kedi, Köpek, Kuş) sınıflarda kullan. 


val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,                         #? Dikkat et yine,  train_dir  içinden %20 'sini validation için verdik. Yani val klasörünü kullanmadık üstteki durumda da bu soldaki durumda da  train klasörü içini kullandık.
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)



test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,                 #? test_dir  içindeki fotoğrafları ise test dataseti kullanmak için yapıcam. Dikkat ettiysen:  validation_split, subset, seed   kullanmadık. Çünkü veri setini bölmüyorum. Yukarıda train klasörü içindeki verileri böldüğüm için hem validation hemde train için subset,seed gibi parametreler kullanıyorduk. FAkat soldaki bütün dataseti test dataseti olarak kullanacağım için bölmeye gerek yok bu yüzden de bu parametreleri kullanmaya gerek yok.
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)


print("\nYol Tanımlama Başarılı! Veriler  çekildi.")


#? HATIRLATMA:  
#? Validation (Ara Sınav): Eğitim sırasında her epoch sonunda kullanılır. Modelin nasıl gittiğini kontrol edip, gerekirse ayarları (hiperparametreleri) değiştirmene yardım eder.
#? Test (Final Sınavı): Eğitim tamamen bittikten sonra bir kez kullanılır. Modelin daha önce hiç görmediği bu veriler, senin "gerçek dünya" başarı skorundur.

#? Neden Hem Validation Hem Test Var? (Saçma mı?)  Aslında hiç saçma değil, aksine çok profesyonelce. Şöyle düşün:  Validation (Yol Gösterici): Sen modeli eğitirken her epoch sonunda "Nasıl gidiyorum?" diye bakarsın. Eğer validation başarısı düşüyorsa eğitimi durdurursun. Yani aslında sen farkında olmadan eğitim stratejini validation setine göre belirliyorsun. Bu da modelin validation setine hafifçe "alışmasına" neden olur.   Test (Acı Gerçek): Test seti, modelin eğitim boyunca hiç görmediği, hiçbir ayar için kullanılmadığı tertemiz bir veridir. Test setindeki skorun, modelin gerçek hayatta hiç görmediği bir hastanın röntgenine baktığında vereceği skordur.  Hangisi daha iyi? Her ikisinin de olması en iyisidir. Sadece Val kullanırsan, modelin başarısı hakkında biraz "iyimser" (overfit riskli) olabilirsin. Test seti seni dürüst tutar. 


class_names = train_ds.class_names
print(f"Class Names: {class_names}")






# TRANSFER LEARNING
 
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input   #? Burayı yeni yazdık önceki projede yoktu.  Neden Rescaling(1./255) yerine preprocess_input Kullandık?  Her dev modelin (MobileNet, ResNet, VGG) "beslenme alışkanlığı" farklıdır. Bazısı pikselleri 0 ile 1 arasında ister, MobileNetV2 ise -1 ile 1 arasında ister. 0 ile 1 arasında istemez. Fark: Eğer biz 1./255 yapsaydık, MobileNet'e yanlış formatta veri vermiş olurduk ve modelin başarısı düşerdi. preprocess_input fonksiyonu, MobileNetV2'nin tam olarak beklediği matematiksel dönüşümü (scaling + centering) otomatik yapar.   Rescaling(1./255) vs preprocess_input: Hangisi Doğru?  Önceki projede mobilenet kullanıp  0 ve 1 arasında ölçekleyebilmemiz modelin esnekliğinden kaynaklanıyor.  eden 1./255 bazen yeterli olur? Eğer modeli sıfırdan eğitiyorsan veya "fine-tuning" (ince ayar) yapıyorsan, model [0, 1] arasındaki verilere uyum sağlar.   Neden preprocess_input kullanmalısın? MobileNetV2, ImageNet veri setinde eğitilirken pikseller [-1, 1] arasına çekilerek eğitildi. Eğer sen 1./255 yaparsan veri $[0, 1]$ arasında olur. Modelin içindeki o devasa bilgi birikimi (ağırlıklar), veriyi $[-1, 1]$ arasında görmeye ayarlı olduğu için kafası karışabilir. Sonuç: Hazır bir model (MobileNet, ResNet vb.) kullanıyorsan, o modelin kendi preprocess_input fonksiyonunu kullanmak altın kuraldır. Hata payını minimize eder.  MobileNetV2 piksellerin [-1, 1] arasında olmasını ister.

base_model = tf.keras.applications.MobileNetV2(input_shape = (224,224,3),
                                               include_top = False,
                                               weights = "imagenet")

base_model.trainable = False


# DATA AUGMENTATION

data_augmentation = keras.Sequential([    #? data_augmentation temel olarak overfitting (ezberleme) sorununu çözmek için kullanılır, ancak tek amacı bu değildir.  Değişmezlik (Invariance) Kazandırmak: Modelin bir nesneyi her açıda, her ışıkta ve her boyutta tanımasını sağlar. Örneğin, bir akciğer röntgeni biraz yamuk çekilse bile modelin "Bu hala bir akciğer" demesini öğretir.   Gürültüye Karşı Dayanıklılık (Robustness): Gerçek dünyadaki veriler (tozlu lensler, düşük çözünürlük) eğitim verileri kadar temiz olmayabilir. Augmentation, modeli bu tür bozulmalara karşı eğitir.  Özellik Odaklaması: Gereksiz detayları (arka plan rengi gibi) değiştirerek modelin asıl önemli olan yapıya (ciğerdeki dokuya) odaklanmasını sağlar.
    layers.RandomRotation(0.01), #? 360 derecenin yüzdesi değil, radyan cinsinden bir oran veya tam tur (360°) üzerinden bir katsayıdır.  0.02 (360 * 0.02  => yaklaşık => 7 derece) . Hasta hafif eğik durarak röntgende durmuş olabilir yani röntgeni etkilemez bu küçük 7 derecelik dönme.  Overfitting olduğu için data augmentation yapıyorum.
    layers.RandomTranslation(height_factor=0.01, width_factor=0.01), #? Bu katman, resmi olduğu yerde sağa-sola veya yukarı-aşağı "ittirir".  height_factor (Yükseklik): Resmin dikey eksende (yukarı/aşağı) ne kadar kaydırılacağını belirler. 0.1 demek, resmin boyunun %10'u kadar yukarı veya aşağı kayabilir demektir.  width_factor (Genişlik): Resmin yatay eksende (sağ/sol) ne kadar kaydırılacağını belirler.
    #layers.RandomContrast(0.01) #? parlaklığı yüzde 1 arttırır veya azaltır.
])




# SEQUENTIAL

model = models.Sequential([

    layers.Input(shape=(224,224,3)),

    data_augmentation,

    layers.Lambda(preprocess_input),  #? layers.Lambda Nedir?  Lambda ne demek: Kendi yazdığın veya hazır olan bir fonksiyonu, sanki bir "model katmanıymış" gibi modelin içine gömmeye yarar. Neden kullandık: preprocess_input aslında bir Python fonksiyonudur. Onu layers.Lambda içine alarak modelin bir parçası haline getirdik. Böylece modeli kaydedip başka bir yerde açtığında, dışarıdan tekrar işlem yapmana gerek kalmaz; model içine giren ham resmi kendi kendine işler.  Her zaman preprocess_input kullanmak daha mantıklıdır.   Neden? Fine-tuning yapsan bile, modelin temelindeki (dondurduğun katmanlardaki) ağırlıklar belirli bir veri dağılımıyla eğitildi. MobileNetV2 için bu [-1, 1] arasıdır. Eğer sen 1./255 yapıp [0, 1] arası veri verirsen, modelin o devasa bilgi birikimi "beklediği sayılar gelmediği için" tam verimle çalışmaz. Ne zaman 1./255 yapılır? Modeli tamamen sıfırdan (ağırlıkları rastgele başlatarak) eğitiyorsan veya kullandığın modelin dokümantasyonu "ben sadece 0-1 arası istiyorum" diyorsa (bazı basit modeller gibi) yapılır.  Böyle yazmıştık önceden =>  layers.Lambda(preprocess_input)   Eğer bundan hata alırsan böyle yaz => layers.Lambda(lambda x: preprocess_input(x))  .Aynı şeyi yapıyor ikiside.

    base_model,

    layers.GlobalAveragePooling2D(),

    #layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),    #? GlobalAveragePooling2D katmanından sonra direkt Dropout ve Output koymak aslında çok modern bir yaklaşımdır.  Neden Dense eklemedik? MobileNetV2 zaten resmi 1280 tane çok güçlü özelliğe (feature) indirger. Arada devasa bir Dense layer (mesela 512 nöronlu) koyarsan, model bu 5200 tane röntgen resmini çok hızlı ezberler (overfitting). * Hangisi daha iyi? Eğer verin azsa (5-10 bin resim az sayılır), az katman daha iyidir. Eğer test başarın düşük kalırsa araya bir Dense(256, activation='relu') eklemek bir seçenektir ama genelde direkt geçiş daha "genellenebilir" sonuçlar verir.

    layers.Dense(1, activation="sigmoid")  #? Neden 1 Output: İkili (binary) sınıflandırmada tek bir nöron yeterlidir. Çıkan sonuç 0'a yakınsa "Normal", 1'e yakınsa "Zatürre" deriz. İkinci bir nöron gereksizdir çünkü birinin ihtimali %70 ise diğerininki zaten %30'dur.  Sınıf meselesi: Kaggle veri setinde "Bakteriyel" ve "Viral" diye iki tip zatürre olsa da, klasör yapısında bunlar tek bir PNEUMONIA klasörü altındadır. Eğer onları da ayırmak isteseydin 3 output ve softmax kullanman gerekirdi.  Neden Sigmoid: Sigmoid fonksiyonu her sayıyı 0 ile 1 arasına sıkıştırır. Bu da bize bir "olasılık" verir (Örn: %85 Zatürre).  Sadece iki sınıf olduğu için (zature veya sağlıklı) sigmoid kullandık yani.

])


# COMPILE 
model.compile(optimizer="adam",
              loss = "binary_crossentropy",   #? binary_crossentropy ve Sigmoid Bağlantısı =>  Sigmoid ile bağlantısı: Bunlar ayrılmaz bir ikilidir. Sigmoid bir olasılık üretir, binary_crossentropy ise bu olasılığın gerçek etiketle (0 veya 1) arasındaki hatayı (loss) hesaplar.  Ne zaman kullanılır: Sadece iki sınıflı (Binary) problemlerde kullanılır.  Matematiksel mantığı: Tahmin ile gerçek arasındaki mesafeyi ölçer.  Sigmoid'in görevi: Sadece matematiği bükmektir. Gelen karmaşık sayıları $0$ ile $1$ arasına hapseder. O bir "olasılık" üreticisidir.   Binary Crossentropy'nin görevi: Eğitim sırasında Sigmoid'den çıkan o 0.85 değerini alıp, "Gerçek değer 1'di, aradaki fark çok, seni cezalandırıyorum!" diyerek modelin hatasını hesaplamaktır.  0.5 Kararını Kim Verir? Aslında ne Sigmoid ne de Loss fonksiyonu buna karar verir. TensorFlow'un accuracy metriği, arka planda otomatik olarak "Eğer sonuç $>0.5$ ise ben bunu 1 (Zatürre) sayarım" diye kabul eder. Sen tahmin yaparken bu eşiği istersen $0.7$ bile yapabilirsin.  1'i zatüre sayme sebebi başta label verirken  NORMAL için sıfır, PNEUMONIA için 1 vermesinden kaynaklı oluyor. 
              metrics=["accuracy"])




# CHANGING VERSION 

model_dir = "../model"          #? Burada yaptığımız olay biz modelleri kaydederken veya callbacks'de csv logger içine koyarken şöyle her seferinde dosya ismi oluşturuyorduk =>  chest_xray_model_1.keras    şeklinde model oluşturuyorduk. Sonra tekrar eğittiğimizde elimle değiştiriyordum dosya numarasını 1 arttırıyordum ve  chest_xray_model_2.keras   yapıyordum.  Veya csv dosyası oluştururken =>  chest_xray_model_1_report.csv   gibi sürekli yine dosya adını elimle değiştiriyordum. Şimdi böyle yapmamak için aşağıdaki kodda kendi kendine bu sayıları değiştirmesini sağlıcak kodu yazdık.
log_dir = "../model/reports"    #? İlk başta soldaki url'leri bu değişkenlere atadık

os.makedirs(model_dir, exist_ok=True)   #? Eğer model_dir adresinde gerçekten model diye klasör varsa bir şey yapma ama yoksa model_dir adresi oluşacak şekilde klasör oluştur. Yani scripts klasörünün dışında model klasörü yoksa soldaki kodda model klasörü oluşturur. Biz eskiden bunu şöyle yazıyorduk =>  if not os.path.exists(log_dir):  os.makedirs(log_dir)       Fakat, bu yazım uzun yazım oluyo. Ama exist_ok = True daha kısa bi yazım.  Eğer varsa öyle dosya bir şey yapma ama yoksa o dosyayı oluşturacak makedir() fonksiyonu.
os.makedirs(log_dir, exist_ok=True)    #? Bu soldaki kodu yukardakinin bir altına yazdık çünkü eğer model klasörü yoksa o oluşsun, sonra ise model klasörü içine  reports diye klasör oluşsun. Eğer varsa yine bir şey yapmasın diye exist_ok = True dedik.

version = 1
while True:
    current_model = os.path.join(model_dir, f"chest_xray_model_{version}.keras")   #? İlk başta version = 1 olduğu için ve model_dir 'deki adresle birleştiği için buradaki url şöyle olur =>   ../model/chest_xray_model_1.keras     Yani soldaki adresi current_model  değişkeni tutsun dedim.
    current_log = os.path.join(log_dir, f"chest_xray_model_{version}_report.csv")  #? Yine başta version = 1 olduğu için ve log_dir adresiyle birleştiğinde soldaki  current_log değişkeni şu adresi tutar =>  ../model/reports/chest_xray_model_1_report.csv   
    current_graph = os.path.join(log_dir, f"training_history_{version}.png")

    if not os.path.exists(current_log) and not os.path.exists(current_model):   #? Eğer yukarıdaki yazdığım adresler gibi dosyalar yoksa o zaman version arttırmaya gerek olmadığı için while 'dan çıkabiliriz.  While 'dan çıkınca aşağıda  model.save()  ve  CSVLogger()  fonksiyonlarına bu adresler yazılabilir ve modeli bu adrese kaydeder =>  ../model/chest_xray_model_1.keras     ve csv report dosyasını ise CSVLogger içine koyduğumuz içinse bu adrese kaydeder =>  ../model/reports/chest_xray_model_1_report.csv     
        break                                                                 #? Eğer ama bizim  version = 1 iken dosyalarımız varsa yani yukardaki gibi dosyalarımız varsa , soldaki if koşulunda böyle dosyalar olduğu için while break edilmez ve aşağıda version bir arttırılır.  Eğer version= 2  iken dosyalarım yoksa break yapılır, ve ikinci dosyalar kaydedilir yoksa tekrar version arttırılır ve üçüncü dosyalar kaydedilir. Bu böyle gider.  Bu yüzden artık model.save() demeden önce  ../model klasörü var mı yok mu diye kontrol etmeye gerek yok çünkü while dışında yapıyoruz. Aynı şekilde  ../model/reports  klasörü var mı yok mu diye CSVLogger() fonksiyonu öncesinde de kontrol etmeye gerek yok.   

    version += 1

print(f"\nBu eğitim için kullanılacak versiyon: {version}")
print(f"Model Kayıt Yolu: {current_model}")
print(f"Rapor Kayıt Yolu: {current_log}\n")





# CALLBACKS
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True  #? Senaryo: En iyi başarısını 18. epoch'ta yakaladı ama 25. epoch'a kadar eğitim devam etti. Sonuç: Eğitim bittiğinde Keras hafızasındaki ağırlıkları kontrol eder. restore_best_weights=True olduğu için, 25. epoch'taki ağırlıkları siler ve bellekte tuttuğu 18. epoch'taki en iyi ağırlıkları modele geri yükler. Yani model erkenden durdurulmasa bile sonuna kadar eğitilse bile yine en iyi ağırlıkları alır en sondaki ağırlıkları almaz   restore_best_weights=True  sayesinde
)

csv_logger = CSVLogger(
    current_log,
    append=True,
    separator=";"
)



# FIT
print("\nMobileNetV2 ile Eğitim Başlıyor...")

class_weights = {0: 4.0, 1: 1.0}  #? Normal(sağlıklı) resim sayısı az olduğu için Zatüre olana göre, bu yüzden model sağlıklı olan röntgenlerede zatüre diyor. Yani ezber yapıyor. Bunun önüne geçmek için oversampling'de yapılabilir veya böyle class weights 'de yapılabilir. Soldaki değişkeni model.fit()  içine class_weights parametresine vererek şunu demiş oluyorsun =>  "Bir Normal resmi yanlış tahmin edersen, seni bir Zatürre hatasından 2.8 kat daha fazla cezalandırırım."  class_weight kullandığında Loss (Kayıp) fonksiyonunun çalışma prensibi şuna dönüşür:  Normal (Sınıf 0) hatası: Hesaplanan hata değeri 2.8 ile çarpılır.  Zatürre (Sınıf 1) hatası: Hesaplanan hata değeri 1.0 ile çarpılır.    Modelin amacı bu Toplam Loss değerini minimuma indirmektir. Eğer model bir "Normal" resmi yanlış tahmin ederse, toplam loss değeri 2.8 kat daha fazla artar. * Bu yüksek artış, modelin parametrelerini (ağırlıklarını) güncelleyen Gradient Descent algoritmasına şu sinyali gönderir: "Bu sınıftaki hata çok büyük, ağırlıklarını bu hatayı azaltacak yönde daha sert güncelle."  Model, toplam loss değerini düşürebilmek için hata payı yüksek olan (2.8 ile çarpılan) "Normal" sınıfını doğru tahmin etmeye öncelik verir. Yani loss fonksiyonunun düşmesi için model, "Normal" sınıfındaki başarısını artırmak zorunda kalır.  Özet: class_weight kullanmak, loss fonksiyonunun değerini yapay olarak yükseltir; model de bu yüksek değeri düşürmek için azınlıkta kalan sınıfa daha fazla odaklanır.

history = model.fit(train_ds,    #? Grafik çizdirmek için history değişkenine atadık.
          validation_data=val_ds, 
          epochs=25, 
          class_weight = class_weights,  #? Dikkat et class_weight parametresine yukardaki class_weightS değişkenini atadık
          callbacks = [early_stopping, csv_logger]
        )




# TEST
print("\nModel Değerlendiriliyor (Test)...")
test_loss, test_acc = model.evaluate(test_ds)       #? Biz normalde  model.evaluate(X_test, y_test)  şeklinde verirdik fakat yine model.fit() 'deki olay gibi bu şekilde verincede model kendisi ayırabiliyor y(label etiketi) ve x verisini.  Yani model.evaluate() sonucunda bana test verisetindeki fotoğrafları modelim tahmin etmeye çalışacak ve genel doğruluk ve loss oranını verecek.  Böyle bi sonuç gelecek yani =>  accuracy: 0.9157 - loss: 0.3024    .Buradaki sonuç test verisetindeki fotoğrafları modelim atıyorum yüzde 91 oranında doğru tahmin edecek demek. Yani toplam bir doğruluk oranı. Verisetindeki bütün resimleri bu kadar oranda doğruluğunu buluyor. 
                                                    #? evaluate() vs predict() => İkisi de tahmin yapar ama amaçları farklıdır:   evaluate() (Ölçmek): Elinde hem resimler hem de o resimlerin gerçek etiketleri (0 veya 1 oldukları) olduğunda kullanılır. Sana "Modelin %92 doğrulukla çalışıyor" gibi istatistiksel bir özet verir. Yani "Sınavı oku ve bana notu söyle" demektir.     predict() (Tahmin Etmek): Sadece resimleri verirsin (etiketleri vermezsin). Model sana her resim için [0, 1] arası bir olasılık verir (Örn: 0.98). Sen bu değere bakıp "Hah, bu hasta zatürre" dersin. Yani "Sınav kağıdındaki soruları çöz" demektir.
                                                    #?  model.evaluate() ne zaman kullanılır?   Elinde hem resimler hem de o resimlerin ne olduğu bilgisi (etiketi) varsa kullanılır.  Amacı: Modelin başarısını (doğruluk oranını) ölçmektir.  Örnek: Kaggle'dan indirdiğin test klasörü. Orada hangi resmin normal, hangisinin zatürre olduğunu biliyoruz (çünkü klasör isimleri belli). Bu yüzden "Hadi model, bu sınavı çöz ve kaç puan aldığını söyle" diyoruz.
                                                    #?  model.predict() ne zaman kullanılır?    Elinde sadece resimler varsa ve modelin ne diyeceğini merak ediyorsan kullanılır.  Amacı: Bilinmeyen bir veri hakkında tahmin yürütmektir.  Örnek: İnternetten yeni bir röntgen buldun veya bir hastanın röntgenini sisteme yükledin. Bunun cevabı sende yok. Model sana bir sayı üretir (Örn: 0.92). Sen de dersin ki "Tamam, model %92 ihtimalle bu zatürre diyor."      
                                                    #? Küçük bir not: Yeni bir test veri setin olsa bile, eğer o veri setinin içinde etiketler (cevaplar) varsa yine evaluate ile genel başarıyı ölçebilirsin. Ama amacın sadece "Bu resim ne?" sorusuna cevap almaksa predict tek çaredir.


print(f"\nFinal Test Doğruluğu: %{test_acc*100:.2f}")   #? . Neden test_acc * 100 yaptık? TensorFlow, doğruluk (accuracy) oranını her zaman 0 ile 1 arasında ondalık bir sayı olarak verir (Örneğin: 0.9157).  Sayıyı 100 ile çarparak onu standart yüzdelik formata çevirmiş oluyoruz (0.9157 * 100 = 91.57).  Neden : (İki nokta) koyduk?  Python'daki f-string (f"...") yapısının kuralıdır. İki nokta üst üste (:), koda şunu söyler:  "Sol taraftaki değeri (test_acc*100) al, ama ekrana basmadan önce sağ taraftaki kurala göre şekillendir (formatla)."  demektir.  Neden .2f yazdık?  Bu, sağ taraftaki o "şekillendirme kuralı"nın kendisidir. İki parçadan oluşur:  .2 : Noktadan (virgülden) sonra sadece 2 basamak göster ve gerisini yuvarla demektir.  f : Float (ondalıklı sayı) tipinde yazdırılacağını belirtir.  Özetle Ne İşe Yarar?  Eğer sen o kuralı yazmasaydın, Python sayıyı ham haliyle şöyle upuzun ve çirkin yazdırabilirdi: Final Test Doğruluğu: %91.578941345    .2f sayesinde bunu tertemiz kesip yuvarlıyor ve şöyle gösteriyor:  Final Test Doğruluğu: %91.58
                                                        #? Neden her zaman yüzde ile çarpıyoruz dersen yüzdelik değer bulmaya çalışıyoruz. Örneğin şöyle verimiz olsun =>  0.0091578   Biz bunu yüz ile çarparsak sıfırlar sadeleşir ve  matematiksel kural gereği nokta 2 basamak sağa kayar.  ve böyle sonuç olur =>  100 ile çarpılmış hali: 0.91578 (Artık bu bir "yüzde" değeridir).   Eğer buna da .2f kuralını uygularsak =>  Elimizdeki sayı: 0.91|5... (3. basamak 5 olduğu için yukarı yuvarlar).  Ekranda görünecek sonuç: %0.92.   Peki %9 gibi değer için nasıl bir ondalık sayı olmalıydı buna bakalım => Ham sayı: 0.09     İşlem: 0.09 * 100 = 9.00    Görünüm (.2f): %9.00  şeklinde gözükür.  





# SAVE 
model.save(current_model)   #? odel kaydetmenin birkaç yolu var ve her biri farklı bir dönemi temsil ediyor:    .h5 (Eski Standart): Uzun yıllar kullanılan klasik formattır. Ancak bazen modelin içindeki "özel katmanları" (custom layers) kaydederken sorun çıkarabilir.    .keras (Yeni ve Güncel Standart): TensorFlow ve Keras'ın şu anki favorisi. Daha hafif, daha güvenli ve modelin her şeyini (mimarisi, ağırlıkları, optimizer ayarları) tek bir dosyada kusursuz tutar. Bundan sonra bunu kullanmanı öneririm.   model.export() (SavedModel): Bu sadece bir dosya değil, bir klasör oluşturur. Genelde modeli "üretim aşamasına" (Production/Deployment) taşırken, yani bir web sitesine veya mobil uygulamaya gömerken (TensorFlow Serving ile) kullanılır.
print(f"\nİşlem Tamamlandı! Model başarıyla kaydedildi: {current_model}")



# HISTORY GRAPHIC  
acc = history.history["accuracy"]   #? Kodun bu kısımlarını anlamak için tensorflow ders anlatımındaki hastalıklı çiçekli projeye bak
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(12,5))  

#? 1. Başarı (Accuracy) Grafiği
plt.subplot(1,2,1)
plt.plot(acc, label="Training Accuracy", color="blue", linewidth=2)
plt.plot(val_acc, label="Validation Accuracy", color="orange", linewidth=2)
plt.title("Accuracy Graph of Model")
plt.xlabel('Epoch')   #? xlabel(), ylabel() sadece string yazı yazar. Bu x ve y eksenine değerleri zaten otamatikman matplotlib modelden çeker ve grafiğe yukarıda plt.plot() dediğin için çizer. Bu xlabel(), ylabel() fonksiyonlarında sadece isim veriyorsun x ve y grafiğine.
plt.ylabel('Başarı Oranı')
plt.legend()
plt.grid(True)  #? Arka plana kareli harita metot defteri gibi ince çizgiler (ızgara) ekler. Okuma kolaylığı sağlar grafiği.

#? 2. Hata (Loss) Grafiği
plt.subplot(1,2,2)
plt.plot(loss, label="Training Loss", color="blue", linewidth=2)
plt.plot(val_loss, label="Validation Loss", color="orange", linewidth=2)
plt.title("Loss Graph of Model")
plt.xlabel('Epoch')  
plt.ylabel('Hata Skoru')
plt.legend()
plt.grid(True)

plt.tight_layout()  #? tight_layout içi boş farkındaysan böyle olunca =>  Sayfadaki hiçbir yazı, başlık veya sayı birbirine çarpmasın, aralarında ideal minimum bir boşluk olsun" der ve otomatik bir hesaplama yapar. Fakat visualize_results.py içinde farklı değerler olmasının sebebi başına başlık koymamızdı bundan dolayı düzenlememiz gerekmişti.
plt.savefig(current_graph)  #? visualize_results.py  içinde açıkladım bunu ve plt.show() fonksiyonunu
plt.show()

print(f"Eğitim geçmişi grafiği kaydedildi: {current_graph}")


























#? FaceID_Project 'de aşağıdaki gibi veri eğitimi yapıyorduk
""" 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os

data_dir = "../data"

train_dir = os.path.join(data_dir, "train")

test_dir = os.path.join(data_dir, "test")


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

print("\nYol Tanımlama Başarılı! Veriler  çekildi.")


class_names = train_ds.class_names
print(f"Class Names: {class_names}")



# TRANSFER LEARNING

base_model = tf.keras.applications.MobileNetV2(input_shape = (224,224,3),
                                               include_top = False,
                                               weights = "imagenet")

base_model.trainable = False


# SEQUENTIAL

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224,224,3)),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")

])


# COMPILE 
model.compile(optimizer="adam",
              loss = "sparse_categorical_crossentropy",
              accuracy=["metrics"])


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)



# FIT
model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks = [early_stopping])











"""
































