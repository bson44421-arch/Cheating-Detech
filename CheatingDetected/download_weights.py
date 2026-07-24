from ultralytics import YOLO

# Ultralytics sẽ tự tải yolov8n-pose.pt về nếu máy chưa có
pose_model = YOLO('yolov8n-pose.pt')

# Tương tự cho phát hiện vật thể/điện thoại
detect_model = YOLO('yolov8n.pt')