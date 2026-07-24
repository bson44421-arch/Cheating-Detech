import torch

# Thiết bị tính toán
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tham số Kỹ thuật
NUM_KEYPOINTS = 17
INPUT_DIM = NUM_KEYPOINTS * 2  # 34 giá trị (x, y tương đối)
NUM_CLASSES = 2                # 0: Normal, 1: Cheating

# Ngưỡng (Thresholds)
PHONE_CLASS_ID = 67            # Class ID của Cell Phone trong COCO Dataset
SMOOTHING_WINDOW = 5           # Độ dài cửa sổ trượt để trung bình cộng xác suất (Lọc nhiễu)
CHEAT_THRESHOLD = 0.70         # Xác suất >= 0.70 thì cảnh báo Gian lận

# Đường dẫn File
YOLO_POSE_PATH = "models/yolov8n-pose.pt"
YOLO_DETECT_PATH = "models/yolov8n-detect.pt"
MLP_MODEL_PATH = "models/mlp_pose_classifier.pth"