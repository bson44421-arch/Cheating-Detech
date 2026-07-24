import cv2
import torch

from src.config import DEVICE, MLP_MODEL_PATH, SMOOTHING_WINDOW, CHEAT_THRESHOLD
from src.modules.pose_extractor import PoseExtractor
from src.modules.object_detector import ObjectDetector
from src.modules.pose_mlp import PoseMLP
from src.tracker.real_time_tracker import RealTimeTracker


def main():
    print("=== Đang khởi tạo hệ thống nhận diện gian lận ===")

    # 1. Khởi tạo các Module trích xuất dữ liệu
    print("[*] Loading YOLOv8 Modules...")
    pose_extractor = PoseExtractor("models/yolov8n-pose.pt")
    object_detector = ObjectDetector("models/yolov8n-detect.pt")

    # 2. Load mô hình MLP Phân loại Hành vi
    print(f"[*] Loading PoseMLP từ {MLP_MODEL_PATH}...")
    mlp_model = PoseMLP(input_dim=34, hidden_dim=64, num_classes=2).to(DEVICE)

    try:
        mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))
        mlp_model.eval()
    except FileNotFoundError:
        print(f"[X LỖI] Không tìm thấy file {MLP_MODEL_PATH}. Hãy chạy 'train_mlp.py' trước!")
        return

    # 3. Khởi tạo Real-time Tracker
    tracker = RealTimeTracker(
        model=mlp_model,
        window_size=SMOOTHING_WINDOW,
        cheat_thresh=CHEAT_THRESHOLD,
        device=DEVICE
    )

    # 4. Mở Webcam (ID 0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[X LỖI] Không thể kết nối với Webcam!")
        return

    print("=== Hệ thống sẵn sàng! Nhấn 'q' để thoát ===")

    # 5. Vòng lặp xử lý từng khung hình (Frame)
    while cap.isOpened():
        ret, frame = cap.read()  # <--- Biến 'frame' được tạo ra tại đây
        if not ret:
            print("[!] Không đọc được khung hình từ webcam.")
            break

        # Lật ảnh giống soi gương
        frame = cv2.flip(frame, 1)

        # A. Kiểm tra phát hiện điện thoại
        has_phone, phone_boxes = object_detector.detect_phone(frame)

        # B. Trích xuất Pose & Chuẩn hóa keypoints
        norm_kpts, bbox, raw_kpts = pose_extractor.extract_keypoints(frame)

        status = "NO PERSON"
        prob = 0.0
        color = (128, 128, 128)  # Xám

        # C. Dự đoán hành vi nếu tìm thấy người
        if norm_kpts is not None:
            status, prob = tracker.process_frame(norm_kpts, has_phone=has_phone)

            color = (0, 0, 255) if "CHEATING" in status else (0, 255, 0)  # Đỏ hoặc Xanh

            # Vẽ Bounding Box xung quanh sinh viên
            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Vẽ Khung xương (17 điểm keypoints)
            for x, y in raw_kpts:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1)

        # D. Vẽ Bounding Box đỏ nếu phát hiện Điện thoại
        if has_phone:
            for p_box in phone_boxes:
                px1, py1, px2, py2 = map(int, p_box)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # E. Hiển thị bảng thông số ở góc trái
        cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Prob: {prob:.2f}", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show kết quả lên màn hình
        cv2.imshow("Exam Monitoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()