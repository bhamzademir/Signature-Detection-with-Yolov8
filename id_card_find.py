from ultralytics import YOLO


model = "runs/detect/id_card_train7/weights/best.pt"

yolo_model = YOLO(model)
def detect_id_card(image_path: str):
    results = yolo_model.predict(source=image_path, conf=0.25, save=True, save_txt=True)
    return results

if __name__ == "__main__":
    image_path = "dataset/images/train_deskewed2/file_279652_page2.jpg"
    detect_id_card(image_path)