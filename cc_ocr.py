from ultralytics import YOLO
import cv2
import os
from datetime import datetime

from paddleocr import PaddleOCR

# === Genel Ayarlar ===
ONNX_MODEL_PATH  = "model.onnx"                       # Kullanılacak ONNX model yolu
PT_MODEL_PATH    = "runs/detect/train9/weights/best.pt"  # Sınıf isimleri için .pt model yolu
TEST_IMAGE_PATH  = "bank_cc.jpg"                      # Test edilecek resim

OUT_DIR   = "onnx_test_results"
CROP_DIR  = os.path.join(OUT_DIR, "crops")
VIS_DIR   = os.path.join(OUT_DIR, "frames")

CONF_THR        = 0.35
TARGET_CLASSES  = {"card", "id card", "credit card"}


# ==============================
# Yardımcı Fonksiyonlar
# ==============================

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_dirs():
    """Gerekli klasörleri oluştur."""
    os.makedirs(CROP_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)


def load_class_names(pt_model_path: str):
    """
    Sınıf isimlerini .pt modelinden okumaya çalışır.
    Olmazsa manuel fallback döner.
    """
    try:
        temp_model = YOLO(pt_model_path)
        class_names = temp_model.names
        print("Sınıf Adları (.pt modelinden alındı):", class_names)
        return class_names
    except Exception as e:
        print(f"UYARI: Orijinal .pt model yüklenemedi, manuel sınıf adları kullanılacak: {e}")
        # BURAYI kendi modelinin index-sınıf ismi eşleşmesine göre düzenle
        return {0: "card", 1: "id card", 2: "credit card"}


def load_onnx_model(onnx_model_path: str):
    """ONNX YOLO modelini yükler."""
    try:
        model = YOLO(onnx_model_path)
        print(f"ONNX Model '{onnx_model_path}' yüklendi.")
        return model
    except Exception as e:
        raise RuntimeError(f"ONNX Model yüklenirken hata: {e}")


def detect_cards_on_frame(model, frame, class_names):
    """
    Verilen frame üzerinde YOLO ONNX modeli ile inference yapar.
    Kart sınıflarına ait kutu ve crop'ları kaydeder.
    Annotated frame ve crop kaydı yapılıp yapılmadığını döner.
    """
    res = model.predict(source=frame, imgsz=640, conf=0.25, verbose=True)[0]

    h, w = frame.shape[:2]
    saved_any = False

    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls.item())
            conf   = float(box.conf.item())
            cls_nm = class_names.get(cls_id, str(cls_id)).lower()

            # Yalnızca hedef sınıflar
            if cls_nm not in TARGET_CLASSES:
                continue
            if conf < CONF_THR:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            pad = 8
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad)
            y2p = min(h, y2 + pad)

            crop = frame[y1p:y2p, x1p:x2p]

            ts = now_str()
            crop_name = os.path.join(CROP_DIR, f"card_{ts}_c{int(conf*100)}.jpg")
            cv2.imwrite(crop_name, crop)
            print(f"[SAVE] CROP  -> {crop_name}  cls={cls_nm} conf={conf:.2f}")
            saved_any = True

            # Kutu ve label çiz
            cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls_nm} {conf:.2f}",
                (x1p, max(0, y1p - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return frame, saved_any


def read_from_frame(frame, model, class_names):
    """
    Verilen frame üzerinde kart tespiti yapar ve crop'ları kaydeder.
    Annotated frame ve crop kaydı yapılıp yapılmadığını döner.
    """
    annotated_frame, saved_any = detect_cards_on_frame(model, frame, class_names)
    return annotated_frame, saved_any


def process_image(image_path: str, model, class_names):
    """
    Tek bir resmi okur, kart tespiti yapar,
    crop'ları kaydeder, kutulu resmi kaydedip ekranda gösterir.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Resim dosyası bulunamadı veya açılamadı: {image_path}")
    print(f"Resim '{image_path}' yüklendi. Boyut: {frame.shape}")

    annotated_frame, saved_any = detect_cards_on_frame(model, frame, class_names)

    if saved_any:
        vis_name = os.path.join(VIS_DIR, f"result_frame_{now_str()}.jpg")
        cv2.imwrite(vis_name, annotated_frame)
        print(f"[SAVE] FRAME -> {vis_name}")
        print("\n--- Test Bitti ---")
        print(f"Sonuçlar '{OUT_DIR}' klasörüne kaydedildi.")

        cv2.imshow("ONNX Test Sonucu", annotated_frame)
        print("Resmi kapatmak için herhangi bir tuşa basın.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n--- Test Bitti ---")
        print(f"Belirtilen resimde (CONF_THR={CONF_THR}) eşleşen kart bulunamadı.")


# ==============================
# main
# ==============================

def main():
    ensure_dirs()
    class_names = load_class_names(PT_MODEL_PATH)
    onnx_model = load_onnx_model(ONNX_MODEL_PATH)

    # Tek bir görüntü işleme
    process_image(TEST_IMAGE_PATH, onnx_model, class_names)

    # İstersen klasördeki tüm .jpg/.png'leri gezmek için:
    # import glob
    # for img_path in glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.jpeg"):
    #     process_image(img_path, onnx_model, class_names)


if __name__ == "__main__":
    main()