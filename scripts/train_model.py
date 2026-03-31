import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt

#! Kodu çalıştırmak için   C:\Users\Ulusoy\Desktop\YAPAY ZEKA PROJELER\Chest_Xray_Project>   terminal soldaki gibiyken => cd scripts   yaz ve sonra böyle olacak =>   C:\Users\Ulusoy\Desktop\YAPAY ZEKA PROJELER\Chest_Xray_Project\scripts>     Sonrasında ise => python train_model.py    şeklinde yaz ve sonra train_model.py çalışacak. Sonra ise visualize_results çalıştırmak için yine scripts klasörü içinde olduğundan emin ol ve bu sefer  python visaulize_results.py  demen gerekecek. Fakat biz train_model.py'yi çalıştırmadık bunun yerine google colab'de en aşağıda alttaki kodu çalıştırdık. Yani Google Colab kodunu kırmızı başlıkla yazıcam. Hemen alttaki google colab kodu değil ve projenin son nihai sonucunu üreten kod değil. Öğrenmek için alttakileri oku ama.

data_dir = "../data"   # Biz   FaceID_Project   projesinde, data klasörü içinde fotoğraflarım vardı hatırlarsan emin ve ahmet klasörleri içinde. Fakat bu projede Biz kaggle sitesindeki datasets bölümüne gidip seçtiğim projedeki download kısmına bastıktan sonra  Download as Zip dedim ve zip olarak dataseti indirdim. Ve içinde fotoğrafları tutan test,train ve val adında klasörler vardı ve bu klasörlerin hepsinin içinde NORMAL(normal, sağlıklı) ve PNEUMONIA (zatüre)  adında klasörler vardı ve bunların içinde ise NORMAL klasörü içinde sağlıklı 0-5 yaş arası çocukların akciğer röntgenleri,  PNEUMONIA klasörleri içinde ise bakterili veya virüslü şekilde zatüre olan çocukların akciğerlerinin röntgenleri vardı. Biz train, test ve val klasörlerini direk data diye klasör oluşturup onun içine attım. 
                       # Şimdi şuraya dikkat et. FaceID_Project'de direk data klasörü içinde emin ve ahmet adında klasörler vardı ve içinde resimler vardı. FAkat şimdi data klasörü içinde train,test ve val klasörleri var ve onların içinde NORMAL ve PNEUMONIA  adında klasörler var. Yani aslında emin ve ahmet klasörleri sanki train,test ve val klasörleri içindeymiş gibi düşün. Bu yüzden biz aşağıda  data_dir  değişkenin tuttuğu  ../data  yolunun içini   ../data/train  diyerek train dosyası içine girmeyi sağlıcaz ve train içindeki NORMAL, PNEUMONIA dosyalarına yani resimleri tutan dosyalara ulaşabilecez    ve ../data/test   diyerekte   test dosyasının içine girmiş olacaz ve yine bu içindeki resimleri tutan NORMAL ve PNEUMONIA dosyalarına erişebilicez. Dikkat edersen ../data/val  klasörü içine girmiyorum çünkü içindeki NORMAL ve PNEUMONIA klasörleri içindeki resimler çok az sayıda bu yüzden validation için gerekli olan resimleri ben train içinden çekicem. Aşağıda anlıcaksın.


print("Gördüğüm GPU sayısı: ", len(tf.config.list_physical_devices('GPU'))) # Eğer Gördüğüm GPU sayısı: 1  yazarsa gpu vardır sıfır yazarsa gpu yoktur eğitim için.

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

class_names = train_ds.class_names  #? train_ds unbatch edilmeden önce soldaki kodu yazdırman lazım yoksa hata verir.
print(f"Class Names: {class_names}")


# OVERSAMPLING 
train_ds = train_ds.unbatch()  #? Şimdi, bizim train_ds içindeki röntgen fotoğrafları eşit sayıda değil. Yani Normal ve Zatüre röntgen fiyatları eşit sayıda olmadığı için modelim tahmin ederken hep Zatüre röntgenleri görmeye alıştığı için, Normal röntgenlerde zorlanıyor ve düşük olasılıkla bunları tahmin ediyor. Bunun önüne geçmek için yani Normal röntgenleri iyi tahmin etmesi için oversampling yapıcaz. Yani Normal röntgen sayısını, Zatüre röntgen sayısına eşitlicez. Böylece daha doğru karar vericek modelim. Yani bizim Normal röntgen sayımız 1300 civarı, Zatüre röntgen sayısıda 3500 civarı. Normal röntgen sayılarını arttırarak(oversampling) Zatüre röntgen sayısını yakalamaya çalışıcam. Ama bu bir sanal oversampling aşağıda yapacağım şey. Yani biz Normal klasörün içine girip fiziksel olarak işte hepsini kopyala diyip 3 kere aynı şeyi yapıştırmıyoruz. Böyle yapmış olsaydık, veri setimizdeki Normal röntgen görüntüsü 1 GB yer kaplamıcaktıda örneğin 3 GB yer kaplıcaktı. Ayrıyetten, fiziksel oversampling yapsaydık normal fotoğraflara normal fotoğraf sayısı 3000 olacaktı ve zatüre fotoğraf sayısıda 3000 olduğu için toplam 6000 fotoğraf olacak. Ve bu önceden 1000 normal ,3000 zatüre fotoğrafları cpu ile okumak için 1 saniye vakit harcanıyorsa, artık 3000 normal, 3000 zatüre fotoğrafını cpu ile okumak için örneğin artık 3 saniye harcıcak. Bundan dolayı bir yavaşlık oluşurken ayrıca modeli eğitirken cpu grup grup bu fotoğrafları yollarken fiziksel oversampling'de daha uzun sürede fotoğrafları okuduğu için, gpu modeli eğitirken cpu'nun fotoğraf yollamasını daha uzun süre bekliceği için model eğitm süreside çok artacaktı. 
                               #? Bütün bu sebeplerden dolayı,  fiziksel röntgen sayısı arttırmak yerine, sanal oversampling yaparak sadece eğitim sırasında sırası gelince resimlerim RAM'e çağrılırlar. Yani fiziksel bir normal fotoğraf artışı olmuyor. Yani hala 1000 fotoğraf normal, 3000 fotoğraf Zatüre fotoğrafı var. Fakat ram'e çağırırken 1000 normal fotoğrafı daha çok çağıracaz gibi düşün. Bu yüzden de sanki 3000 normal fotoğraf olmuş gibi olacak. Yani, daha küçük dosya boyutumuzla daha kısa model eğitmimizle model eğitmiş olacaz. Sanal oversampling bu yüzden yapılıyor.Yani fiziksel bir artış söz konusu değil sanal oversampling'de.  Ve ayrıca sanal oversampling'de fotoğrafları çağırdığımızda yeni oluşacak görüntüleri tekrar aşağıda görüceksin train_ds içine atıyoruz ve sonra train_ds 'ye data augmentation uygulanıyor. Yani bu röntgenleri çeviriyo, değiştiriyor böylece aynı görüntü olmamış oluyor. Yani aynı fotoğrafları vermemiş oluyoruz. Aynı resim , data augmentationa girip örneğin 3 farklı şekilde çıkıyor. Birinde sağa daha fazla dönmüş, diğerinde sola kaymış şekilde falan. Böylece farklı röntgen görüntüsü görmüş gibi oluyor model. 
                               #? Unbatch Ne Demek?    Bir palet dolusu kutu (batch) düşün. unbatch() yapmak, o paleti dağıtıp içindeki kutuları tek tek yere dizmektir.   Önce: Veri yapısı (Batch_Size, 224, 224, 3) şeklindedir. Yani "32 tane 224x224'lük renkli resim bir arada" demektir.   Sonra: Veri yapısı (224, 224, 3) şekline döner. Artık her bir eleman tek bir resimdir.  Neden unbatch yapıyoruz?   Filtreleme (filter) ve örnekleme (sample_from_datasets) işlemleri, veri setinin içindeki her bir elemana (her bir resme) grupca değil, tek tek bakmak ister.    Eğer paketleri açmazsan (unbatch yapmazsan), TensorFlow filtreleme yaparken "Bu 32'li paketin içinde normal resim var mı?" diye sormaz; "Bu paketin kendisi 0 mı 1 mi?" diye sorar ve hata verir.    Bizim amacımız her bir resmi tek tek kontrol edip "Sen normalsin, sen zatürresin" diye ayırmak olduğu için paketleri açmak zorundayız.  Yani unbatch() sayesinde önce batch'leri (32'lik paketleri) bozup resimleri tek tek akışa alıyoruz.
                               #? unbatch() öncesi train_ds böyle gözüküyordu =>  [ (32 resim + 32 etiket), (32 resim + 32 etiket), (32 resim + 32 etiket) ... ]    Unbatch Sonrası (Tekil) =>  [ (Resim1, Etiket1), (Resim2, Etiket2), (Resim3, Etiket3), (Resim4, Etiket4) ... ]      Artık train_ds içinde tam olarak diskteki resim sayısı kadar eleman vardır. Her eleman tek bir tuple (resim matrisi, etiket) halindedir.
                               #? Neden Önce image_dataset_from_directory Yaptık? Yani, Madem unbatch edicektik neden başta yukardan batch halinde aldık dersen ?  Önce klasörü tanımlaman lazım. Yani önce   image_dataset_from_directory()  sayesinde klasörden train fotoğrafları çekiyorum için yüzde 80% olarak. Yani önce fotoğrafları elime bi almam lazım, çünkü elimde fotoğraf olmadan bu zatüre bu normal diyemem. Önce elime fotoğrafı alıcam. Sonra  tek tek bunları çözüp bu normal, bu zatüre dicem. Yani önce fotoğraf albümünü elime alıyorum, sonra bu resimde ben varım veya bu resimde başkası var diye ayırabiliyorum.


normal_ds = train_ds.filter(lambda img, label: tf.squeeze(label) == 0).cache().repeat() #? Resimleri filtreliyoruz. Önce lambda sayesinde tek tek train_ds içindeki unbatch() edilmiş [ (Resim1, Etiket1), (Resim2, Etiket2), ... ]  röntgenlerini tek tek alıyoruz. Ve bunları resim(img) ve etiketlerine(label) göre ayırıyoruz çünkü iki değeri var resim(img) ve etiket(label) diye. Sonra bu resimlerin etiket değerleri sıfıra eşit ise  normal_ds içine atıyoruz. Eğer 1'e eşitse bu sefer  pneumonia_ds  değişkenine atıyoruz. Yani normal_ds içinde sadece normal röntgenlerin fotoğrafları oluyor bu sayede, pneumonia_ds içinde ise sadece zatüre fotoğrafları oluyor.  Dikkat et bu görüntüleri train_ds'den aldık.    Ve sonra bu normal görüntüleri zatüre görüntüleri ayırdığımız için aşağıda eşit oranda dağıtabilirim.    label_mode="binary" seçtiğin için etiketlerin şekli bir liste gibi geliyor (Örn: [0] veya [1]).  Yani label [0], [1] gibi veriler tutuyor.  Fakat benim bu arrayden çıkarmam lazım ki sıfıra eşit mi değil mi karşılaştırıyım yoksa arrayi sıfıra eşit mi diye kontrol ettiğim için hata alırım. Bunun için  tf.squeeze onu o gereksiz parantezlerden kurtarıp doğrudan 0 haline getiriyor. Böylece 0 == 0 karşılaştırması bir liste değil, doğrudan bir Boolean (True/False) döndürüyor. Ve böylece az önceki dediğim dağıtma işlemi başarılı oldu normal_ds ve pneumonia_ds ayrılmış oldu başarılı bi şekilde.
pneumonia_ds = train_ds.filter(lambda img, label: tf.squeeze(label) == 1).cache()  #? DİKKAT: Normal sınıfı az olduğu için sonuna .repeat() ekledik. Bu sayede torba hiç boşalmayacak.   repeat() ne demek ?  normal_ds.repeat() dediğimizde,  normal_ds içindeki resimler bitmiyor. Yani , normal_ds içindeki 1300 resim bittiği an, TensorFlow otomatik olarak başa dönüyor.  RAM'de veya diskte "Resim_1_kopya.jpg" gibi bir dosya oluşmuyor. Bu olay şu şekilde oluyor => CPU, diskteki o resmin adresini tekrar okuyor ve modele "yeni bir veriymiş gibi" gönderiyor. Böylece train_ds, büyük olan sınıf (pneumonia_ds) bitene kadar sürekli başa dönerek yeni görüntüler üretmeye eder. Yani sanal oversampling yapmış olur. Yani .repeat() sayesinde  Çıktıda her iki sınıftan da eşit sayıda (yaklaşık 3800 + 3800 = 7600) görsel geçmiş olur.  
                                                                #? Tabiki bu eşit sayıda görsel üretmeyei aşağıdaki sample_from_datasets() fonksiyonu yapıyor.  Fakat repeat() soldaki kodda aslında bir not düşüyor diyor ki => "Eğer senin sonuna gelinirse, durma; en başa dön."      . repeat() fonksiyonu asıl işini sample_from_datasets() içinde veri çekilmeye başlandığında yapar.   sample_from_datasets   elini normal_ds torbasına atar ve bir resim çeker.   normal_ds içindeki 1341. (son) resmi de verdikten sonra normalde "ben bittim" sinyali göndermesi gerekir.    Ancak üzerinde .repeat() notu olduğu için, "ben bittim" demek yerine gizlice başa döner ve 1. resmi tekrar verir.    Bu sırada pneumonia_ds (zatürre torbası) hala kendi resimlerini vermeye devam ediyordur.   Eğer sample_from_datasets içinde özel bir durma şartı (örneğin stop_on_empty_dataset=True) belirtmediysen ve diğer torba (pneumonia_ds) hala doluysa, repeat() sayesinde normal_ds,  pneumonia_ds bitene kadar tekrar tekrar veri besleyebilir.  İşlem, en büyük küme (Pneumonia) bitene kadar devam eder. Pneumonia musluğu kesildiği an, sample_from_datasets işlemi bitirir ve eğitim o epoch (devir) için tamamlanır.


train_ds = tf.data.Dataset.sample_from_datasets([normal_ds, pneumonia_ds], 
            weights=[0.5, 0.5]).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  #? sample_from_datasets NE YAPAR?  İki farklı veri kaynağından (torbadan) belirli bir kurala göre veri çekme işlemini yönetir.   Neden [normal_ds, pneumonia_ds] ?  Çünkü kaynaklarımızı filter komutuyla ayırdık. Bir torbada sadece sağlıklı, diğerinde sadece hasta röntgenleri var.   Neden weights=[0.5, 0.5]?   Bu bir olasılık ayarıdır. "Her adımda %50 ihtimalle soldaki torbadan, %50 ihtimalle sağdaki torbadan çek" demiş oluyoruz. Bu da sınıfları otomatik olarak eşitler.   Eğer [0.7, 0.3] deseydin, verilerin %70'i Normal, %30'u Pneumonia olurdu. Biz tam denge istediğimiz için 0.5 verdik ikisinede.  
                                                                              #?  batch(32): Resimleri tek tek değil, 32'li gruplar halinde tekrar train_ds'ye atıyoruz. Yani yukarıda bu 32'li grubu açmıştık açtıktan sonra yukardaki işlemleri yaptık, şimdi ise işlemlerimiz bittiği için tekrar 32'şerli gruplar haline getiriyoruz batch()  ile.   prefetch(): "Önceden getir" demektir. GPU mevcut 32 resmi eğitirken, CPU boş durmaz ve diskten bir sonraki 32 resmi hazırlayıp RAM'e koyar. GPU işini bitirince veri beklemez, hemen bir sonraki gruba geçer.  AUTOTUNE: TensorFlow'un bilgisayarının hızına göre "kaç tane paketi önceden hazırlayacağına" (buffer size) kendisinin karar vermesini sağlar. Manuel sayı vermekten daha profesyoneldir.                 
                                                                              #? Şu an train_ds, diskteki resimleri fiziksel olarak kopyalamadan, sanal oversampling sayesinde, eşit sayıda (%50-%50 oranında) normal ve zatüre fotoğrafları var. Ve eğitim sırasında 32'li paketler halinde hazırlayan ve GPU'yu hiç bekletmeyen profesyonel bir veri boru hattı (pipeline) haline geldi.  
                                                                              #? Bu yukardaki olayı sadece train_ds için yaptık dersen ?  yani val_ds Neden Değişmedi dersen?   Validation (Doğrulama) setine asla dokunulmaz, dengelenmez veya artırılmaz (augmentation yapılmaz).  Çünkü validation seti "gerçek dünyayı" temsil etmelidir. Hastaneye gelen 100 kişinin 75'i hastaysa, modelin bu gerçek dağılımda ne kadar başarılı olduğunu görmen gerekir. Sınıfları eşitlersen, modelin başarısı hakkında kendini kandırmış olursun.  Bu arada bu işlemlerden sonra hala train_ds (%80), val_ds(%20) oranı korunur. Yani val_ds hiç bi zaman bu olaya dahil olmadı train_ds içinde yaptık bu değişiklikleri.            
                                                                              #? Biz [0.5, 0.5] dediğimz için



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

# class_weights = {0: 4.0, 1: 1.0}  #? DİKKAT YA sample_from_datasets() kullanılır yani oversampling ya da soldaki teknik. Eğer sample_from_datasets() kullanıyorsan ASLA SOLDAKİ KODU KULLANMA. EĞER BUNU KULLANICAKSANDA YUKARDAKİ sanal oversampling kodunu kullanma !!!   Normal(sağlıklı) resim sayısı az olduğu için Zatüre olana göre, bu yüzden model sağlıklı olan röntgenlerede zatüre diyor. Yani ezber yapıyor. Bunun önüne geçmek için oversampling'de yapılabilir veya böyle class weights 'de yapılabilir. Soldaki değişkeni model.fit()  içine class_weights parametresine vererek şunu demiş oluyorsun =>  "Bir Normal resmi yanlış tahmin edersen, seni bir Zatürre hatasından 2.8 kat daha fazla cezalandırırım."  class_weight kullandığında Loss (Kayıp) fonksiyonunun çalışma prensibi şuna dönüşür:  Normal (Sınıf 0) hatası: Hesaplanan hata değeri 2.8 ile çarpılır.  Zatürre (Sınıf 1) hatası: Hesaplanan hata değeri 1.0 ile çarpılır.    Modelin amacı bu Toplam Loss değerini minimuma indirmektir. Eğer model bir "Normal" resmi yanlış tahmin ederse, toplam loss değeri 2.8 kat daha fazla artar. * Bu yüksek artış, modelin parametrelerini (ağırlıklarını) güncelleyen Gradient Descent algoritmasına şu sinyali gönderir: "Bu sınıftaki hata çok büyük, ağırlıklarını bu hatayı azaltacak yönde daha sert güncelle."  Model, toplam loss değerini düşürebilmek için hata payı yüksek olan (2.8 ile çarpılan) "Normal" sınıfını doğru tahmin etmeye öncelik verir. Yani loss fonksiyonunun düşmesi için model, "Normal" sınıfındaki başarısını artırmak zorunda kalır.  Özet: class_weight kullanmak, loss fonksiyonunun değerini yapay olarak yükseltir; model de bu yüksek değeri düşürmek için azınlıkta kalan sınıfa daha fazla odaklanır.

history = model.fit(train_ds,    #? Grafik çizdirmek için history değişkenine atadık.
          validation_data=val_ds, 
          epochs=25, 
          #class_weight = class_weights,  #? SANAL OVERSAMPLING YAPTIĞIMIZ İÇİN YUKARIDA ARTIK BU SOLDAKİ CLASS_WEIGHTS KISMINI KULLANMIYORUM !!!!. ÇÜNKÜ,  Biz yukarıdaki boru hattı (train_ds.unbatch()...) ile veriyi zaten %50 Normal - %50 Zatürre olacak şekilde mükemmel bir dengeye oturttuk. Modelin önüne zaten eşit sayıda fotoğraf geliyor.  Eğer sen dengelenmiş bu verinin üzerine bir de class_weights = {0: 4.0, 1: 1.0} eklersen modele şunu emretmiş olursun:  "Önüne eşit sayıda resim geliyor biliyorum ama sen Normal (0) etiketli resimlerde hata yaparsan seni 4 kat daha fazla cezalandıracağım!"  Sonuç: Model cezadan korktuğu için Normal sınıfına aşırı takıntılı (biased) hale gelir. Kararsız kaldığı, hatta hafif zatürre olan her röntgene sırf ceza yememek için "Normal" demeye başlar. Şimdi bu soldaki konudan bağımsız diğer notum => Dikkat et class_weight parametresine yukardaki class_weightS değişkenini atamışız.
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





#! ---------------- GOOGLE COLAB 'DE YAZDIĞIM NİHAİ KOD AŞAĞIDA -----------------------------------
#! ------------------------------------------------------------------------------------------------

""" 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt



#data_dir = "../data"   #? ==> ÖNEMLİ:  Eskiden soldaki gibi yazmıştık ama google colab kullanmak için verileri drive atmamız lazım.
                        #? Önce chest_xray_project içindeki data klasörünü (içinde train ve test olan) kopyalayıp masaüstüne aldın. Bu data klasörünü sağ tıklayıp .zip haline getirdin çünkü böylece fotoğrafları drive'a daha hızlı atmış olucaz. Sonra bu zip dosyasını google drive içine sürükleyip attın. Ve bu data.zip hiç bir klasörün altında olmadan drive'ım başlığı altında gözüküyordu.
                        #? Sonra google colab açtın ve sonra  Sol taraftaki klasör simgesine basıp "Drive" ikonuna tıkladın. Ondan sonra  Terminal yerine, başına ünlem işareti ! koyarak bir kabuk komutu (shell command) çalıştırdın. Bu komut, Drive'daki data.zip dosyasını Colab'ın hızlı diskine saniyeler içinde açtı bu kod sayesinde =>  !unzip -q /content/drive/MyDrive/data.zip -d /content/dataset     Sonra bu soldaki kodu küçük hücre içinde çalıştırmak için solundaki devam işareti gibi olan üçgene bastın ve çalıştırdın.
                        #?  !(Ünlem işareti): Python hücresine "Bu bir Python kodu değil, bir sistem komutudur, bunu terminaldeymişsin gibi çalıştır" demektir.   unzip: Dosyaları zipten çıkaran temel komut.       -q (Quiet): "Sessiz ol" anlamına gelir. Eğer bunu yazmasaydın, zipten çıkan 5000 tane dosyanın ismini ekrana tek tek yazdırır ve sayfayı inanılmaz uzatırdı. -q sayesinde her şeyi arka planda hızlıca halletti.    /content/drive/MyDrive/data.zip: Drive'daki kaynak dosyanın yolu.        -d /content/dataset: "Directory" (Dizin) demek. Dosyaları nereye boşaltacağını söyler. /content zaten google colab'in çalışma dizini özel isim biz uydurmadık.  Biz /content/dataset  diyerek  content dizinine  dataset diye klasör oluştur ve içine bu data.zip 'i  koy demiş oluyoruz yani.
                        #? Bunları yaptıktan sonra   GOOGLE COLAB 'DE YAZDIĞIM NİHAİ KOD AŞAĞIDA  başlığının altındaki kodları çalıştırmadan önce  google colab sitesinde  üst menüden "Düzenle" (Edit) -> "Not Defteri Ayarları" (Notebook Settings) yolunu izleyerek Donanım Hızlandırıcı (Hardware Accelerator) kısmından T4 GPU'yu seçmiştik. Bu sayede veri eğitirken google'ın gpu 'ları kullanılıyor. Ve daha hızlı veri eğitimi yapıyor.
                        #? Daha sonra,  işte yeni bir kod hücre bloğu oluşturduk ve içine    GOOGLE COLAB 'DE YAZDIĞIM NİHAİ KOD AŞAĞIDA   başlığının altındaki kodu buraya yazdık ve sonrada çalıştırdık o kod bloğunu. 


data_dir = "/content/dataset/data"   #?   Zip dosyasını /content/dataset içine açtığımız için yollar artık soldaki gibi oldu yukardaki gibi olmadı. Çünkü google colab üzerinden data.zip 'e ulaşıyoruz. Bu google colab üzerinden data.zip içindeki fotoğraflara bakarak modelimizi eğiticez ve bunun sonucunda biz her veri eğitimizdeki gibi  .keras uzantılı dosya elde etmiş olucaz sonra bunu indirip masaüstündeki projendeki .keras uzantılı dosyalar içine atıp kullanıcaksın.


print("Gördüğüm GPU sayısı: ", len(tf.config.list_physical_devices('GPU')))  #? Eğer 1 yazarsa gpu var demektir eğer sıfır yazarsa model eğitirken gpu kullanmıyo demektir.

train_dir = os.path.join(data_dir, "train")  #? /content/dataset/data/train   yolunu alıcak artık bu alttakide aynı şekilde  test'i alıcak
test_dir = os.path.join(data_dir, "test")



IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,                   #? Buralar aynı hiç değişiklik yok.
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)

class_names = train_ds.class_names
print(f"Class Names: {class_names}")


# OVERSAMPLING
train_ds = train_ds.unbatch()


normal_ds = train_ds.filter(lambda img, label: tf.squeeze(label) == 0).repeat()       #?  Neden İkisine de .repeat() Koyduk?   Çünkü kullandığımız sample_from_datasets fonksiyonu bir "sonsuz karıştırıcı" gibi çalışır.   Kural: Bu fonksiyon, verdiğin veri setlerinden sürekli örnek çekmek ister. Eğer sınıflardan biri (çok olan bile olsa) biterse, karıştırıcı çekiliş yapamaz hale gelir ve hata verir veya eğitimi durdurur.  Mantık: "Az olanı çoğaltmak" için değil, "çekiliş havuzunu asla boş bırakmamak" için ikisini de sonsuz yapıyoruz. Biz steps_per_epoch ile eğitime "DUR" diyene kadar o havuzdan veri çekilmeye devam eder.  
pneumonia_ds = train_ds.filter(lambda img, label: tf.squeeze(label) == 1).repeat()   #? Yukarıda .cache() verdik ama neden şimdi tercih etmedik ?   .cache(), veriyi ilk okuduğunda RAM'e (belleğe) veya yerel diske kopyalar. Böylece sonraki epoch'larda dosyayı tekrar diskten okumakla vakit kaybetmez, direkt RAM'den şimşek hızıyla çeker.  Şimdi Neden Tercih Etmedik?  RAM Kapasitesi: Colab'ın RAM'i kısıtlıdır. 5000+ yüksek çözünürlüklü röntgen resmini RAM'e sığdırmaya çalışırsan Colab "RAM aşıldı" diyerek kendini kapatır (çöker).     Hız Dengesi: Zaten prefetch(tf.data.AUTOTUNE) kullandığımız için, model bir resmi işlerken bir sonrakini arka planda hazırlıyor. Bu yüzden .cache() riskine girmeden de yeterli hıza ulaştık.


train_ds = tf.data.Dataset.sample_from_datasets([normal_ds, pneumonia_ds],
            weights=[0.6, 0.4]).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) #? Neden weights=[0.6, 0.4] yaptık ?  Elimizde çok fazla "Zatürre" (Pneumonia) ve az "Normal" resim vardı. Eğer müdahale etmeseydik, model sürekli zatürre görüp "Her şey zatürredir" diye ezberleyecekti.  Verileri havuzdan çekerken şöyle demiş olduk:  "Batch oluştururken resimlerin %60'ını Normal klasöründen, %40'ını Zatürre klasöründen çek."   Az olan sınıfı (Normal) daha sık modele göstererek dengeyi kurdun. Model artık her iki sınıfı da eşit ağırlıkta öğreniyor.



val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)



test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)


print("\nYol Tanımlama Başarılı! Veriler  çekildi.")



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
    layers.RandomZoom(0.1),
    layers.RandomFlip("horizontal")  #? Bu, "Ayna Görüntüsü" oluşturma katmanıdır. Görüntüyü yatayda (sağ-sol) rastgele çevirir.  Bir akciğer röntgeninde lekelerin hep sağda veya hep solda olması önemli değildir. Modelin "Leke soldaysa zatürredir" gibi yanlış bir kural ezberlemesini istemiyoruz.  Modeli "geometrik" olarak şaşırtır. Modelin sadece konuma değil, dokudaki bozulmaya odaklanmasını sağlar. Böylece veri setini yapay olarak iki katına çıkarmış olursun.
])




# SEQUENTIAL

model = models.Sequential([

    layers.Input(shape=(224,224,3)),

    data_augmentation,

    layers.Lambda(preprocess_input),  #? layers.Lambda Nedir?  Lambda ne demek: Kendi yazdığın veya hazır olan bir fonksiyonu, sanki bir "model katmanıymış" gibi modelin içine gömmeye yarar. Neden kullandık: preprocess_input aslında bir Python fonksiyonudur. Onu layers.Lambda içine alarak modelin bir parçası haline getirdik. Böylece modeli kaydedip başka bir yerde açtığında, dışarıdan tekrar işlem yapmana gerek kalmaz; model içine giren ham resmi kendi kendine işler.  Her zaman preprocess_input kullanmak daha mantıklıdır.   Neden? Fine-tuning yapsan bile, modelin temelindeki (dondurduğun katmanlardaki) ağırlıklar belirli bir veri dağılımıyla eğitildi. MobileNetV2 için bu [-1, 1] arasıdır. Eğer sen 1./255 yapıp [0, 1] arası veri verirsen, modelin o devasa bilgi birikimi "beklediği sayılar gelmediği için" tam verimle çalışmaz. Ne zaman 1./255 yapılır? Modeli tamamen sıfırdan (ağırlıkları rastgele başlatarak) eğitiyorsan veya kullandığın modelin dokümantasyonu "ben sadece 0-1 arası istiyorum" diyorsa (bazı basit modeller gibi) yapılır.  Böyle yazmıştık önceden =>  layers.Lambda(preprocess_input)   Eğer bundan hata alırsan böyle yaz => layers.Lambda(lambda x: preprocess_input(x))  .Aynı şeyi yapıyor ikiside.

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation="relu"), # 1280 özelliği 256'ya indirerek rafine eder.  Veri Miktarı Faktörü: 5000 resim derin öğrenme için çok büyük bir rakam değil. Eğer araya 256-128-64 gibi çok fazla katman koyarsan, modelin parametre sayısı (öğrenmesi gereken değişken sayısı) çok artar. Tehlike: Model, röntgenin mantığını öğrenmek yerine, o 5000 resmi "ezberlemeye" (Overfitting) başlar. Katman sayısı arttıkça ezber riski artar.  MobileNetV2 Zaten Güçlü: MobileNetV2 zaten resmi 1280 tane çok kaliteli "özelliğe" (feature) indirgemiş durumda. Bizim amacımız bu 1280 özelliği "sınıflandırmak", yeni özellikler icat etmek değil. Tek bir 256'lık katman, bu 1280 veriyi süzüp rafine etmek için fazlasıyla yeterli.  Sıkıştırma (Dense 256): 1280 tane farklı bilgi (ciğerin köşesi, kalbin sınırı, lekenin yoğunluğu vb.) 256 tane "akıllı" nörona aktarılır. Bu, gereksiz gürültülerin elendiği bir "özetleme" aşamasıdır.
    layers.BatchNormalization(),  #  Dengeleme (BatchNormalization): Bu 256 nöronun ürettiği sayılar BN'ye girer. BN der ki: "Senin sayın çok büyük çıkmış, senin ki çok küçük; hepinizi standart bir aralığa getiriyorum." Bu sayede model bir sonraki adımda (Sigmoid) daha kararlı kararlar verir. Düşün ki elinde 32 tane röntgen var (bir batch). MobileNetV2 bu resimleri inceledi ve her biri için bize sayılar çıkardı. Bir resim için çıkan sayı 0.001 (çok küçük). Diğeri için 500.0 (çok büyük). Modelin bir sonraki katmanı (Dense), bu kadar uçurum olan sayılarla karşılaşınca "Hangi sayıya güveneceğim?" diye şaşırır. İşte BatchNormalization burada devreye girer: Hizaya Sokar:  32 resimlik grubun ortalamasını alır. Her sayıyı, birbirine yakın ve standart bir aralığa (genelde 0 ile 1 arasına) çeker. Eğitimi Hızlandırır: Sayılar "hizaya" girince, modelin ağırlıkları (weights) çok daha dengeli güncellenir. Ezberi Engeller: Her seferinde o anki 32 resme göre bir "ayarlama" yaptığı için, modele hafif bir gürültü ekler. Bu gürültü, modelin pikselleri ezberlemesini değil, genel mantığı anlamasını sağlar.
    layers.Dropout(0.5),    #? GlobalAveragePooling2D katmanından sonra direkt Dropout ve Output koymak aslında çok modern bir yaklaşımdır.  Neden Dense eklemedik? MobileNetV2 zaten resmi 1280 tane çok güçlü özelliğe (feature) indirger. Arada devasa bir Dense layer (mesela 512 nöronlu) koyarsan, model bu 5200 tane röntgen resmini çok hızlı ezberler (overfitting). * Hangisi daha iyi? Eğer verin azsa (5-10 bin resim az sayılır), az katman daha iyidir. Eğer test başarın düşük kalırsa araya bir Dense(256, activation='relu') eklemek bir seçenektir ama genelde direkt geçiş daha "genellenebilir" sonuçlar verir. Bir tane %50 Dropout: Nöronların yarısını kapatır ve modeli zorlar. İki tane Dropout: Eğer peş peşe çok fazla dropout eklersen, modelin "öğrenecek nöronu" kalmaz. Bilgi akışı kesilir (Gradient Vanishing).

    layers.Dense(1, activation="sigmoid")  #? Neden 1 Output: İkili (binary) sınıflandırmada tek bir nöron yeterlidir. Çıkan sonuç 0'a yakınsa "Normal", 1'e yakınsa "Zatürre" deriz. İkinci bir nöron gereksizdir çünkü birinin ihtimali %70 ise diğerininki zaten %30'dur.  Sınıf meselesi: Kaggle veri setinde "Bakteriyel" ve "Viral" diye iki tip zatürre olsa da, klasör yapısında bunlar tek bir PNEUMONIA klasörü altındadır. Eğer onları da ayırmak isteseydin 3 output ve softmax kullanman gerekirdi.  Neden Sigmoid: Sigmoid fonksiyonu her sayıyı 0 ile 1 arasına sıkıştırır. Bu da bize bir "olasılık" verir (Örn: %85 Zatürre).  Sadece iki sınıf olduğu için (zature veya sağlıklı) sigmoid kullandık yani.

])


# COMPILE
model.compile(optimizer="adam",
              #loss = "binary_crossentropy",   #? binary_crossentropy ve Sigmoid Bağlantısı =>  Sigmoid ile bağlantısı: Bunlar ayrılmaz bir ikilidir. Sigmoid bir olasılık üretir, binary_crossentropy ise bu olasılığın gerçek etiketle (0 veya 1) arasındaki hatayı (loss) hesaplar.  Ne zaman kullanılır: Sadece iki sınıflı (Binary) problemlerde kullanılır.  Matematiksel mantığı: Tahmin ile gerçek arasındaki mesafeyi ölçer.  Sigmoid'in görevi: Sadece matematiği bükmektir. Gelen karmaşık sayıları 0 ile 1 arasına hapseder. O bir "olasılık" üreticisidir.   Binary Crossentropy'nin görevi: Eğitim sırasında Sigmoid'den çıkan o 0.85 değerini alıp, "Gerçek değer 1'di, aradaki fark çok, seni cezalandırıyorum!" diyerek modelin hatasını hesaplamaktır.  0.5 Kararını Kim Verir? Aslında ne Sigmoid ne de Loss fonksiyonu buna karar verir. TensorFlow'un accuracy metriği, arka planda otomatik olarak "Eğer sonuç > 0.5 ise ben bunu 1 (Zatürre) sayarım" diye kabul eder. Sen tahmin yaparken bu eşiği istersen 0.7 bile yapabilirsin.  1'i zatüre sayme sebebi başta label verirken  NORMAL için sıfır, PNEUMONIA için 1 vermesinden kaynaklı oluyor.
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),  #? Label Smoothing (Etiket Yumuşatma), modele "asla %100 emin olma, her zaman küçük bir hata payı bırak" demektir.   Bir doktorun "Normal" dediğine diğeri "Başlangıç seviyesinde zatürre" diyebilir. Yani veri setindeki bazı 0'lar aslında 1, bazı 1'ler aslında 0 olabilir.   Eğer modele "Bu kesinlikle 1 (Zatürre)" dersen (Hard Label) ve o resim aslında yanlış etiketlenmiş bir "Normal" ise; model o hatayı düzeltmek için kendi içindeki tüm mantıklı ağırlıkları bozar.   label_smoothing=0.05 kullandığımızda modele şunu demiş oluyoruz: "Bu resmin etiketi 1 ama sen yine de %5'lik bir hata payı bırak. Eğer görsel özellikler aksini söylüyorsa, etikete körü körüne inanıp kendini parçalama." Model "inatçı" bir ezberci olmak yerine, verilere karşı daha şüpheci ve esnek bir dedektif gibi davranır. Bu da senin test setindeki başarını (accuracy) artıran en kritik ince ayarlardan biriydi.  Fine tuning işleminde de aynısını yapıyoruz.
              metrics=["accuracy"])




# CHANGING VERSION

#model_dir = "../model"          #? Burada yaptığımız olay biz modelleri kaydederken veya callbacks'de csv logger içine koyarken şöyle her seferinde dosya ismi oluşturuyorduk =>  chest_xray_model_1.keras    şeklinde model oluşturuyorduk. Sonra tekrar eğittiğimizde elimle değiştiriyordum dosya numarasını 1 arttırıyordum ve  chest_xray_model_2.keras   yapıyordum.  Veya csv dosyası oluştururken =>  chest_xray_model_1_report.csv   gibi sürekli yine dosya adını elimle değiştiriyordum. Şimdi böyle yapmamak için aşağıdaki kodda kendi kendine bu sayıları değiştirmesini sağlıcak kodu yazdık.
#log_dir = "../model/reports"    #? İlk başta soldaki url'leri bu değişkenlere atadık

model_dir = "/content/drive/MyDrive/Xray_Modelleri"      # BURA DEĞİŞTİ !! Model ve raporların Drive'ına kaydedilmesi için yolu Drive içine veriyoruz. Eğer Xray_modelleri adında klasör yoksa aşağıda exist_ok=True dediği için böyle klasör açılacak sonra ise içine reports klasörü açılacak. Drive'ımın içine bu dosyalar oluşacak ve buradan raporları alabileceğim.
log_dir = "/content/drive/MyDrive/Xray_Modelleri/reports" # Böylece Colab kapansa bile modellerin Drive'ında kalır.

os.makedirs(model_dir, exist_ok=True)   #? Eğer model_dir adresinde gerçekten model diye klasör varsa bir şey yapma ama yoksa model_dir adresi oluşacak şekilde klasör oluştur. Yani scripts klasörünün dışında model klasörü yoksa soldaki kodda model klasörü oluşturur. Biz eskiden bunu şöyle yazıyorduk =>  if not os.path.exists(log_dir):  os.makedirs(log_dir)       Fakat, bu yazım uzun yazım oluyo. Ama exist_ok = True daha kısa bi yazım.  Eğer varsa öyle dosya bir şey yapma ama yoksa o dosyayı oluşturacak makedir() fonksiyonu.
os.makedirs(log_dir, exist_ok=True)    #? Bu soldaki kodu yukardakinin bir altına yazdık çünkü eğer model klasörü yoksa o oluşsun, sonra ise model klasörü içine  reports diye klasör oluşsun. Eğer varsa yine bir şey yapmasın diye exist_ok = True dedik.

version = 9
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

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(  #? Eğitim sırasında bazen hata skoru (val_loss) bir noktada takılır ve daha fazla düşmez. Model sanki bir çıkmaza girmiş gibi davranır. Bunun sebebi genellikle öğrenme hızının (Learning Rate) o aşama için çok yüksek olmasıdır. Model, çözümün etrafında büyük adımlarla zıplıyor ama tam merkeze (en düşük hataya) odaklanamıyordur.
    monitor='val_loss',
    factor=0.2,     # Hızı 5 kat azalt.   Eğer model 3 epoch boyunca gelişme gösteremezse, mevcut hızını 5 kat azaltırız ($0.001 \rightarrow 0.0002$). Bu, modelin "koşmayı bırakıp dikkatlice yürümesini" sağlar.
    patience=3,      # 3 epoch gelişme olmazsa devreye gir
    min_lr=1e-7      # En son inebileceği hız böylece Hızın çok fazla düşüp tamamen durmasını engeller.
)


# FIT
print("\nMobileNetV2 ile Eğitim Başlıyor...")

# class_weights = {0: 4.0, 1: 1.0}  #? DİKKAT YA sample_from_datasets() kullanılır yani oversampling ya da soldaki teknik. Eğer sample_from_datasets() kullanıyorsan ASLA SOLDAKİ KODU KULLANMA. EĞER BUNU KULLANICAKSANDA YUKARDAKİ sanal oversampling kodunu kullanma !!!   Normal(sağlıklı) resim sayısı az olduğu için Zatüre olana göre, bu yüzden model sağlıklı olan röntgenlerede zatüre diyor. Yani ezber yapıyor. Bunun önüne geçmek için oversampling'de yapılabilir veya böyle class weights 'de yapılabilir. Soldaki değişkeni model.fit()  içine class_weights parametresine vererek şunu demiş oluyorsun =>  "Bir Normal resmi yanlış tahmin edersen, seni bir Zatürre hatasından 2.8 kat daha fazla cezalandırırım."  class_weight kullandığında Loss (Kayıp) fonksiyonunun çalışma prensibi şuna dönüşür:  Normal (Sınıf 0) hatası: Hesaplanan hata değeri 2.8 ile çarpılır.  Zatürre (Sınıf 1) hatası: Hesaplanan hata değeri 1.0 ile çarpılır.    Modelin amacı bu Toplam Loss değerini minimuma indirmektir. Eğer model bir "Normal" resmi yanlış tahmin ederse, toplam loss değeri 2.8 kat daha fazla artar. * Bu yüksek artış, modelin parametrelerini (ağırlıklarını) güncelleyen Gradient Descent algoritmasına şu sinyali gönderir: "Bu sınıftaki hata çok büyük, ağırlıklarını bu hatayı azaltacak yönde daha sert güncelle."  Model, toplam loss değerini düşürebilmek için hata payı yüksek olan (2.8 ile çarpılan) "Normal" sınıfını doğru tahmin etmeye öncelik verir. Yani loss fonksiyonunun düşmesi için model, "Normal" sınıfındaki başarısını artırmak zorunda kalır.  Özet: class_weight kullanmak, loss fonksiyonunun değerini yapay olarak yükseltir; model de bu yüksek değeri düşürmek için azınlıkta kalan sınıfa daha fazla odaklanır.

history = model.fit(train_ds,    #? Grafik çizdirmek için history değişkenine atadık.
          validation_data=val_ds,
          epochs=25,
          steps_per_epoch=163,  # Oversampling/repeat() kullandığın için bu şart!  Yukarıda .repeat() fonksiyonunu kullandın. Bu, veri setini sonsuz bir döngüye sokar.  Eğer steps_per_epoch değerini belirtmeseydin, Keras veri setinin bittiğini asla anlayamazdı.  Model, birinci epoch'u bitirip ikinciye geçmek yerine sonsuza kadar veri çekmeye devam ederdi. Bu yüzden ona "Şu kadar adım sonra dur ve bir sonraki epoch'a geç" dedik.  Toplam Resim Sayısı: Veri setinde (Oversampling öncesi veya orijinal eğitim setinde) yaklaşık 5216 civarı resim olduğunu varsayalım.  Batch Size: Senin kodunda bu değer 32.   Hesaplama: 5216 / 32 = 163  diyerek 163 sayısını bulduk.  Yani modele şunu demiş oldun: "Her epoch'ta 32'lik paketlerden 163 tane çek ($163 \times 32 = 5216$ resim eder). Bu kadar resim gördüğünde bir turu tamamlamış say"    .  Oversampling (Örnekleme) yaptığın için aslında elinde "sanal" olarak daha fazla veri var. 163 diyerek modelin her epoch'ta veri setinin tamamını en az bir kez görmesini garanti altına aldın. Eğer bu sayıyı çok küçük (mesela 50) verseydin, model her epoch'ta verinin sadece bir kısmını öğrenip geçerdi; çok büyük verseydin de bir epoch bitmek bilmezdi.
          #class_weight = class_weights,  #? SANAL OVERSAMPLING YAPTIĞIMIZ İÇİN YUKARIDA ARTIK BU SOLDAKİ CLASS_WEIGHTS KISMINI KULLANMIYORUM !!!!. ÇÜNKÜ,  Biz yukarıdaki boru hattı (train_ds.unbatch()...) ile veriyi zaten %50 Normal - %50 Zatürre olacak şekilde mükemmel bir dengeye oturttuk. Modelin önüne zaten eşit sayıda fotoğraf geliyor.  Eğer sen dengelenmiş bu verinin üzerine bir de class_weights = {0: 4.0, 1: 1.0} eklersen modele şunu emretmiş olursun:  "Önüne eşit sayıda resim geliyor biliyorum ama sen Normal (0) etiketli resimlerde hata yaparsan seni 4 kat daha fazla cezalandıracağım!"  Sonuç: Model cezadan korktuğu için Normal sınıfına aşırı takıntılı (biased) hale gelir. Kararsız kaldığı, hatta hafif zatürre olan her röntgene sırf ceza yememek için "Normal" demeye başlar. Şimdi bu soldaki konudan bağımsız diğer notum => Dikkat et class_weight parametresine yukardaki class_weightS değişkenini atamışız.
          callbacks = [early_stopping, csv_logger, lr_reducer]
        )




# FINE-TUNING
print("İnce ayar (Fine-Tuning) başlatılıyor...")
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              #loss="binary_crossentropy",
              loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
              metrics=["accuracy"])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=20,
          steps_per_epoch=163,
          callbacks=[early_stopping, csv_logger, lr_reducer])


# TEST
print("\nModel Değerlendiriliyor (Test)...")
test_loss, test_acc = model.evaluate(test_ds)


print(f"\nFinal Test Doğruluğu: %{test_acc*100:.2f}")




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
"""




#! -------------------------------- GOOGLE COLAB KOD SONU ----------------------------------------------------
#! -----------------------------------------------------------------------------------------------------------









""" 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt



#data_dir = "../data"

# Zip dosyasını /content/dataset içine açtığımız için yollar artık böyle:
data_dir = "/content/dataset/data"


print("Gördüğüm GPU sayısı: ", len(tf.config.list_physical_devices('GPU')))

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")



IMG_SIZE = (224, 224)
BATCH_SIZE = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,                   #? train_dir  içinden %80 'ini training için verdik
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)

class_names = train_ds.class_names
print(f"Class Names: {class_names}")


# OVERSAMPLING
train_ds = train_ds.unbatch()


normal_ds = train_ds.filter(lambda img, label: tf.squeeze(label) == 0).repeat()
pneumonia_ds = train_ds.filter(lambda img, label: tf.squeeze(label) == 1).repeat()


train_ds = tf.data.Dataset.sample_from_datasets([normal_ds, pneumonia_ds],
            weights=[0.6, 0.4]).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)



test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "binary"
)


print("\nYol Tanımlama Başarılı! Veriler  çekildi.")



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
    layers.RandomZoom(0.1),
    layers.RandomFlip("horizontal")
])




# SEQUENTIAL

model = models.Sequential([

    layers.Input(shape=(224,224,3)),

    data_augmentation,

    layers.Lambda(preprocess_input),  #? layers.Lambda Nedir?  Lambda ne demek: Kendi yazdığın veya hazır olan bir fonksiyonu, sanki bir "model katmanıymış" gibi modelin içine gömmeye yarar. Neden kullandık: preprocess_input aslında bir Python fonksiyonudur. Onu layers.Lambda içine alarak modelin bir parçası haline getirdik. Böylece modeli kaydedip başka bir yerde açtığında, dışarıdan tekrar işlem yapmana gerek kalmaz; model içine giren ham resmi kendi kendine işler.  Her zaman preprocess_input kullanmak daha mantıklıdır.   Neden? Fine-tuning yapsan bile, modelin temelindeki (dondurduğun katmanlardaki) ağırlıklar belirli bir veri dağılımıyla eğitildi. MobileNetV2 için bu [-1, 1] arasıdır. Eğer sen 1./255 yapıp [0, 1] arası veri verirsen, modelin o devasa bilgi birikimi "beklediği sayılar gelmediği için" tam verimle çalışmaz. Ne zaman 1./255 yapılır? Modeli tamamen sıfırdan (ağırlıkları rastgele başlatarak) eğitiyorsan veya kullandığın modelin dokümantasyonu "ben sadece 0-1 arası istiyorum" diyorsa (bazı basit modeller gibi) yapılır.  Böyle yazmıştık önceden =>  layers.Lambda(preprocess_input)   Eğer bundan hata alırsan böyle yaz => layers.Lambda(lambda x: preprocess_input(x))  .Aynı şeyi yapıyor ikiside.

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),    #? GlobalAveragePooling2D katmanından sonra direkt Dropout ve Output koymak aslında çok modern bir yaklaşımdır.  Neden Dense eklemedik? MobileNetV2 zaten resmi 1280 tane çok güçlü özelliğe (feature) indirger. Arada devasa bir Dense layer (mesela 512 nöronlu) koyarsan, model bu 5200 tane röntgen resmini çok hızlı ezberler (overfitting). * Hangisi daha iyi? Eğer verin azsa (5-10 bin resim az sayılır), az katman daha iyidir. Eğer test başarın düşük kalırsa araya bir Dense(256, activation='relu') eklemek bir seçenektir ama genelde direkt geçiş daha "genellenebilir" sonuçlar verir.

    layers.Dense(1, activation="sigmoid")  #? Neden 1 Output: İkili (binary) sınıflandırmada tek bir nöron yeterlidir. Çıkan sonuç 0'a yakınsa "Normal", 1'e yakınsa "Zatürre" deriz. İkinci bir nöron gereksizdir çünkü birinin ihtimali %70 ise diğerininki zaten %30'dur.  Sınıf meselesi: Kaggle veri setinde "Bakteriyel" ve "Viral" diye iki tip zatürre olsa da, klasör yapısında bunlar tek bir PNEUMONIA klasörü altındadır. Eğer onları da ayırmak isteseydin 3 output ve softmax kullanman gerekirdi.  Neden Sigmoid: Sigmoid fonksiyonu her sayıyı 0 ile 1 arasına sıkıştırır. Bu da bize bir "olasılık" verir (Örn: %85 Zatürre).  Sadece iki sınıf olduğu için (zature veya sağlıklı) sigmoid kullandık yani.

])


# COMPILE
model.compile(optimizer="adam",
              loss = "binary_crossentropy",   #? binary_crossentropy ve Sigmoid Bağlantısı =>  Sigmoid ile bağlantısı: Bunlar ayrılmaz bir ikilidir. Sigmoid bir olasılık üretir, binary_crossentropy ise bu olasılığın gerçek etiketle (0 veya 1) arasındaki hatayı (loss) hesaplar.  Ne zaman kullanılır: Sadece iki sınıflı (Binary) problemlerde kullanılır.  Matematiksel mantığı: Tahmin ile gerçek arasındaki mesafeyi ölçer.  Sigmoid'in görevi: Sadece matematiği bükmektir. Gelen karmaşık sayıları $0$ ile $1$ arasına hapseder. O bir "olasılık" üreticisidir.   Binary Crossentropy'nin görevi: Eğitim sırasında Sigmoid'den çıkan o 0.85 değerini alıp, "Gerçek değer 1'di, aradaki fark çok, seni cezalandırıyorum!" diyerek modelin hatasını hesaplamaktır.  0.5 Kararını Kim Verir? Aslında ne Sigmoid ne de Loss fonksiyonu buna karar verir. TensorFlow'un accuracy metriği, arka planda otomatik olarak "Eğer sonuç $>0.5$ ise ben bunu 1 (Zatürre) sayarım" diye kabul eder. Sen tahmin yaparken bu eşiği istersen $0.7$ bile yapabilirsin.  1'i zatüre sayme sebebi başta label verirken  NORMAL için sıfır, PNEUMONIA için 1 vermesinden kaynaklı oluyor.
              metrics=["accuracy"])




# CHANGING VERSION

#model_dir = "../model"          #? Burada yaptığımız olay biz modelleri kaydederken veya callbacks'de csv logger içine koyarken şöyle her seferinde dosya ismi oluşturuyorduk =>  chest_xray_model_1.keras    şeklinde model oluşturuyorduk. Sonra tekrar eğittiğimizde elimle değiştiriyordum dosya numarasını 1 arttırıyordum ve  chest_xray_model_2.keras   yapıyordum.  Veya csv dosyası oluştururken =>  chest_xray_model_1_report.csv   gibi sürekli yine dosya adını elimle değiştiriyordum. Şimdi böyle yapmamak için aşağıdaki kodda kendi kendine bu sayıları değiştirmesini sağlıcak kodu yazdık.
#log_dir = "../model/reports"    #? İlk başta soldaki url'leri bu değişkenlere atadık

model_dir = "/content/drive/MyDrive/Xray_Modelleri"      #  Model ve raporların Drive'ına kaydedilmesi için yolu Drive içine veriyoruz
log_dir = "/content/drive/MyDrive/Xray_Modelleri/reports" # Böylece Colab kapansa bile modellerin Drive'ında kalır.

os.makedirs(model_dir, exist_ok=True)   #? Eğer model_dir adresinde gerçekten model diye klasör varsa bir şey yapma ama yoksa model_dir adresi oluşacak şekilde klasör oluştur. Yani scripts klasörünün dışında model klasörü yoksa soldaki kodda model klasörü oluşturur. Biz eskiden bunu şöyle yazıyorduk =>  if not os.path.exists(log_dir):  os.makedirs(log_dir)       Fakat, bu yazım uzun yazım oluyo. Ama exist_ok = True daha kısa bi yazım.  Eğer varsa öyle dosya bir şey yapma ama yoksa o dosyayı oluşturacak makedir() fonksiyonu.
os.makedirs(log_dir, exist_ok=True)    #? Bu soldaki kodu yukardakinin bir altına yazdık çünkü eğer model klasörü yoksa o oluşsun, sonra ise model klasörü içine  reports diye klasör oluşsun. Eğer varsa yine bir şey yapmasın diye exist_ok = True dedik.

version = 9
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

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,     # Hızı 5 kat azalt
    patience=3,      # 3 epoch gelişme olmazsa devreye gir
    min_lr=1e-7      # En son inebileceği hız
)


# FIT
print("\nMobileNetV2 ile Eğitim Başlıyor...")

# class_weights = {0: 4.0, 1: 1.0}  #? DİKKAT YA sample_from_datasets() kullanılır yani oversampling ya da soldaki teknik. Eğer sample_from_datasets() kullanıyorsan ASLA SOLDAKİ KODU KULLANMA. EĞER BUNU KULLANICAKSANDA YUKARDAKİ sanal oversampling kodunu kullanma !!!   Normal(sağlıklı) resim sayısı az olduğu için Zatüre olana göre, bu yüzden model sağlıklı olan röntgenlerede zatüre diyor. Yani ezber yapıyor. Bunun önüne geçmek için oversampling'de yapılabilir veya böyle class weights 'de yapılabilir. Soldaki değişkeni model.fit()  içine class_weights parametresine vererek şunu demiş oluyorsun =>  "Bir Normal resmi yanlış tahmin edersen, seni bir Zatürre hatasından 2.8 kat daha fazla cezalandırırım."  class_weight kullandığında Loss (Kayıp) fonksiyonunun çalışma prensibi şuna dönüşür:  Normal (Sınıf 0) hatası: Hesaplanan hata değeri 2.8 ile çarpılır.  Zatürre (Sınıf 1) hatası: Hesaplanan hata değeri 1.0 ile çarpılır.    Modelin amacı bu Toplam Loss değerini minimuma indirmektir. Eğer model bir "Normal" resmi yanlış tahmin ederse, toplam loss değeri 2.8 kat daha fazla artar. * Bu yüksek artış, modelin parametrelerini (ağırlıklarını) güncelleyen Gradient Descent algoritmasına şu sinyali gönderir: "Bu sınıftaki hata çok büyük, ağırlıklarını bu hatayı azaltacak yönde daha sert güncelle."  Model, toplam loss değerini düşürebilmek için hata payı yüksek olan (2.8 ile çarpılan) "Normal" sınıfını doğru tahmin etmeye öncelik verir. Yani loss fonksiyonunun düşmesi için model, "Normal" sınıfındaki başarısını artırmak zorunda kalır.  Özet: class_weight kullanmak, loss fonksiyonunun değerini yapay olarak yükseltir; model de bu yüksek değeri düşürmek için azınlıkta kalan sınıfa daha fazla odaklanır.

history = model.fit(train_ds,    #? Grafik çizdirmek için history değişkenine atadık.
          validation_data=val_ds,
          epochs=25,
          steps_per_epoch=163,  # Oversampling/repeat() kullandığın için bu şart!
          #class_weight = class_weights,  #? SANAL OVERSAMPLING YAPTIĞIMIZ İÇİN YUKARIDA ARTIK BU SOLDAKİ CLASS_WEIGHTS KISMINI KULLANMIYORUM !!!!. ÇÜNKÜ,  Biz yukarıdaki boru hattı (train_ds.unbatch()...) ile veriyi zaten %50 Normal - %50 Zatürre olacak şekilde mükemmel bir dengeye oturttuk. Modelin önüne zaten eşit sayıda fotoğraf geliyor.  Eğer sen dengelenmiş bu verinin üzerine bir de class_weights = {0: 4.0, 1: 1.0} eklersen modele şunu emretmiş olursun:  "Önüne eşit sayıda resim geliyor biliyorum ama sen Normal (0) etiketli resimlerde hata yaparsan seni 4 kat daha fazla cezalandıracağım!"  Sonuç: Model cezadan korktuğu için Normal sınıfına aşırı takıntılı (biased) hale gelir. Kararsız kaldığı, hatta hafif zatürre olan her röntgene sırf ceza yememek için "Normal" demeye başlar. Şimdi bu soldaki konudan bağımsız diğer notum => Dikkat et class_weight parametresine yukardaki class_weightS değişkenini atamışız.
          callbacks = [early_stopping, csv_logger, lr_reducer]
        )




# FINE-TUNING 
print("İnce ayar (Fine-Tuning) başlatılıyor...")
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, 
          validation_data=val_ds,
          epochs=10,
          steps_per_epoch=163,
          callbacks=[early_stopping, csv_logger, lr_reducer])


# TEST
print("\nModel Değerlendiriliyor (Test)...")
test_loss, test_acc = model.evaluate(test_ds)


print(f"\nFinal Test Doğruluğu: %{test_acc*100:.2f}")




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

 """




















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
































