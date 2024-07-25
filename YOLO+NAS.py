# try out this way: https://medium.com/@Mert.A/how-to-use-yolov8-and-yolo-nas-for-object-detection-8c5893939480


from super_gradients.training import models
import cv2
import torch

device = 'cuda' if torch.cuda.is_available() else "cpu"

model = models.get("yolo_nas_m", pretrained_weights="coco")

img_path = "assets/semantic_segmentation/in.jpg"
model.predict(img_path, conf=0.25).show()