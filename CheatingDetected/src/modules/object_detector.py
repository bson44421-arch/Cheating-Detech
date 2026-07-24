from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="models/yolov8n-detect.pt", phone_class_id=67, conf_thresh=0.5):
        """Khởi tạo mô hình YOLO Object Detection"""
        self.model = YOLO(model_path)
        self.phone_class_id = phone_class_id
        self.conf_thresh = conf_thresh

    def detect_phone(self, image):
        """
        Kiểm tra xem có phát hiện điện thoại trong ảnh không.
        Return: (has_phone: bool, phone_boxes: list)
        """
        results = self.model(image, verbose=False)[0]
        has_phone = False
        phone_boxes = []

        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                if int(cls) == self.phone_class_id and conf >= self.conf_thresh:
                    has_phone = True
                    phone_boxes.append(box)

        return has_phone, phone_boxes