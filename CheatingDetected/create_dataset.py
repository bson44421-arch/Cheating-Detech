import os
import glob
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Import các tham số cấu hình và hàm chuẩn hóa
from src.config import YOLO_POSE_PATH
from src.utils.normalization import normalize_bbox_keypoints


def build_dataset():
    print("=== BẮT ĐẦU QUY TRÌNH TRÍCH XUẤT DATASET TỪ ẢNH ===")

    # 1. Đường dẫn thư mục ảnh gốc và đường dẫn lưu CSV
    raw_images_dir = "data/raw_images"
    output_csv_path = "data/processed/train_kpts.csv"

    # Định nghĩa nhãn: 0 là Normal (Bình thường), 1 là Cheating (Gian lận)
    label_map = {
        "normal": 0,
        "cheating": 1
    }

    # 2. Kiểm tra thư mục ảnh có tồn tại không
    if not os.path.exists(raw_images_dir):
        print(f"[X LỖI] Không tìm thấy thư mục {raw_images_dir}!")
        print("[!] Hãy tạo thư mục 'data/raw_images/normal' và 'data/raw_images/cheating' rồi bỏ ảnh vào đó.")
        return

    # 3. Load mô hình YOLOv8-Pose
    print(f"[*] Loading YOLOv8-Pose từ {YOLO_POSE_PATH}...")
    pose_model = YOLO(YOLO_POSE_PATH)

    dataset_rows = []
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']

    # 4. Duyệt qua từng thư mục nhãn (normal, cheating)
    for class_name, label_id in label_map.items():
        folder_path = os.path.join(raw_images_dir, class_name)
        if not os.path.exists(folder_path):
            print(f"[!] Bỏ qua thư mục '{folder_path}' do chưa được tạo.")
            continue

        # Gom tất cả file ảnh trong thư mục
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        print(f"[*] Đang xử lý nhãn '{class_name}' (ID: {label_id}) - Tìm thấy {len(image_files)} ảnh...")

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Run YOLOv8-Pose trên ảnh
            results = pose_model(img, verbose=False)[0]

            # Nếu phát hiện thấy người trong ảnh
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                keypoints = results.keypoints.xy.cpu().numpy()

                # Lấy người đầu tiên (người có score cao nhất)
                bbox = boxes[0]  # [xmin, ymin, xmax, ymax]
                kpts_17x2 = keypoints[0]  # Shape (17, 2)

                # Chuẩn hóa keypoints thành vector 34 chiều
                norm_kpts = normalize_bbox_keypoints(kpts_17x2, bbox)

                # Ghép vector 34D với cột label_id ở cuối
                row_data = list(norm_kpts) + [label_id]
                dataset_rows.append(row_data)

    # 5. Lưu ra file CSV
    if len(dataset_rows) == 0:
        print("[X LỖI] Không trích xuất được mẫu dữ liệu nào! Kiểm tra lại thư mục ảnh của bạn.")
        return

    # Tạo tiêu đề cột: kpt_0_x, kpt_0_y, ..., kpt_16_y, label
    columns = []
    for i in range(17):
        columns.extend([f"kpt_{i}_x", f"kpt_{i}_y"])
    columns.append("label")

    df = pd.DataFrame(dataset_rows, columns=columns)

    # Tạo thư mục chứa file CSV đầu ra nếu chưa có
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print("-" * 50)
    print(f"[✔ SUCCESS] Đã tạo thành công tập dữ liệu tại: {output_csv_path}")
    print(f"[✔ SUCCESS] Tổng số mẫu thu thập được: {len(df)} mẫu")
    print(f"  - Normal: {len(df[df['label'] == 0])} mẫu")
    print(f"  - Cheating: {len(df[df['label'] == 1])} mẫu")


if __name__ == "__main__":
    build_dataset()