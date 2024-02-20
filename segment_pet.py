import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")

results = model.predict("C:/Users/synml/Desktop/images", device="cuda", retina_masks=True, classes=[15, 16, 77], stream=True, conf=0.7)
for result in results:
    mask = result.masks.data.cpu().numpy()
    appended_mask = np.zeros_like(mask[0])
    for m in mask:
        appended_mask += m
    processed_img = cv2.bitwise_and(result.orig_img, result.orig_img, mask=appended_mask.astype(np.uint8))
    cv2.imwrite(result.path.replace("images", "syshin"), processed_img)
