import cv2
import supervision as sv

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(
  repo_id="tech4humans/yolov8s-signature-detector", 
  filename="yolov8s.pt"
)

model = YOLO(model_path)

image_path = "yamuk.jpg"  # Resim dosyanızın yolu
image = cv2.imread(image_path)

results = model(image_path)

detections = sv.Detections.from_ultralytics(results[0])

box_annotator = sv.BoxAnnotator()
annotated_image = box_annotator.annotate(scene=image, detections=detections)

cv2.imshow("Detections", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
