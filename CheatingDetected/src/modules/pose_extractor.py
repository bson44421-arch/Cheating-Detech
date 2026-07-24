import numpy as np
from ultralytics import YOLO
from src.utils.normalization import normalize_bbox_keypoints

class PoseExtractor:
    def __init__(self, model_path="models/yolov8n-pose.pt"):
        """Khởi tạo mô hình YOLO Pose"""
        self.model = YOLO(model_path)

    def extract_keypoints(self, image):
        """
        Trích xuất và chuẩn hóa keypoints của người trong ảnh/frame.
        Return: (norm_kpts_34d, bbox, raw_kpts)
        """
        results = self.model(image, verbose=False)[0]

        if len(results.boxes) == 0:
            return None, None, None

        # Lấy người đầu tiên (có độ tin cậy cao nhất)
        boxes = results.boxes.xyxy.cpu().numpy()
        keypoints = results.keypoints.xy.cpu().numpy()

        bbox = boxes[0]           # [xmin, ymin, xmax, ymax]
        raw_kpts = keypoints[0]   # Shape (17, 2)

        # Chuẩn hóa về vector 34D tương đối
        norm_kpts_34d = normalize_bbox_keypoints(raw_kpts, bbox)

        return norm_kpts_34d, bbox, raw_kpts