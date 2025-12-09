from ultralytics import YOLO

# 1. Önceden eğitilmiş bir YOLOv8 modeli yükleyin (Hız/Boyut/Doğruluk tercihinize göre 'n', 's', 'm', 'l' veya 'x' seçebilirsiniz)
# Örn: YOLOv8s (small) hızlı bir başlangıç için iyidir.
#model = YOLO('yolov8s.pt')  

# 2. Eğitimi başlatın
# 'data.yaml' dosyanızın yolu ve eğitim parametreleri (epochs, image size)
"""results = model.train(
   data='yolo_train/data.yaml',       # Oluşturduğunuz veri yapılandırma dosyası
   epochs=50,              # Eğitim döngüsü sayısı (model performansına göre artırılabilir)
   imgsz=640,              # Görüntü boyutu (640 yaygın bir başlangıç değeridir)
   batch=16,               # Aynı anda işlenecek görüntü sayısı (GPU belleğine göre ayarlanır)
   name='id_card_train' # Eğitim çalışmasına bir isim verin
)"""

# 3. Eğitim tamamlandıktan sonra modeli test verisi üzerinde değerlendirin (isteğe bağlı)
# model.val() 

# 4. Modeli kullanıma hazır hale getirin (ONNX, OpenVINO vb.)
# model.export(format='onnx')
from ultralytics import YOLOv8

model = YOLOv8.from_pretrained("tech4humans/yolov8s-signature-detector")
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)