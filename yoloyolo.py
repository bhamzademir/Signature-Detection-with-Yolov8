from ultralytics import YOLO

# Modeli yükle (yolov8s.pt dosyasını aynı dizinde olduğundan emin olun)
model = YOLO('yolov8s.pt')

# Modelin mimarisini görüntüleme
print(model.yaml) # Modelin yapılandırma YAML dosyasını gösterir (katmanlar, parametreler vb.)
print(model.names) # Modelin algıladığı sınıfların isimlerini gösterir (örneğin, {0: 'signature'})
print(model) # Modelin kısa bir özetini ve katmanlarını gösterir

# Modeli doğrudan Python kodu içinden çağırma
# Bu, komut satırındaki 'yolo predict' ile aynı işlevi görür ancak sonuçları doğrudan Python nesneleri olarak verir.
results = model('halfSigned.jpg', conf=0.12, iou=0.5, imgsz=640)

# Sonuçlara programatik olarak erişme
for r in results:
    boxes = r.boxes  # Bounding box'lar
    masks = r.masks  # Segmentasyon maskeleri (eğer model destekliyorsa)
    probs = r.probs  # Sınıflandırma olasılıkları (eğer sınıflandırma modeli ise)

    for box in boxes:
        # normalize edilmiş [x_center, y_center, width, height] formatında
        xywhn = box.xywhn.cpu().numpy()
        conf = box.conf.cpu().numpy()  # Güven skoru
        cls = box.cls.cpu().numpy()    # Sınıf ID'si

        print(f"Detected: Class {int(cls)}, Confidence {conf:.2f}, Box {xywhn}")

# Sonuçları kaydetme veya gösterme
r.save(filename='toulouse.jpg') # Tespitli görüntüyü kaydet
r.show() # Tespitli görüntüyü pop-up pencerede göster