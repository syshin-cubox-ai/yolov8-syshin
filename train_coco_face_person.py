from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8c-pose.yaml").load("yolov8n-pose.pt")

    model.train(data="coco-face-person.yaml", epochs=550, imgsz=640, batch=128, optimizer="SGD",
                device=[0, 1, 2, 3, 4, 5, 6, 7], workers=8, plots=True)