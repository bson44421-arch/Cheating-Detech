import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import cấu hình và kiến trúc mô hình
from src.config import DEVICE, INPUT_DIM, NUM_CLASSES, MLP_MODEL_PATH
from src.modules.pose_mlp import PoseMLP


# 1. Custom Dataset cho PyTorch
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train():
    print("=== Đang khởi tạo quá trình huấn luyện PoseMLP ===")

    # 2. Kiểm tra file dữ liệu CSV
    csv_path = "data/processed/train_kpts.csv"
    if not os.path.exists(csv_path):
        print(f"[X LỖI] Không tìm thấy file {csv_path}!")
        print("[!] Hãy chạy file 'create_dataset.py' trước để tạo dữ liệu huấn luyện từ ảnh.")
        return

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)
    print(f"[*] Tổng số mẫu dữ liệu thu thập được: {len(df)}")

    # Giả định 34 cột đầu là Keypoints (0 -> 33) và cột cuối cùng là 'label'
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 3. Chia tập dữ liệu thành Train (80%) và Validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Khởi tạo DataLoader
    train_dataset = PoseDataset(X_train, y_train)
    val_dataset = PoseDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 4. Khởi tạo Mô hình, Loss Function và Optimizer
    model = PoseMLP(input_dim=INPUT_DIM, hidden_dim=64, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 5. Vòng lặp Huấn luyện (Training Loop)
    epochs = 30
    best_val_acc = 0.0

    print(f"[*] Bắt đầu huấn luyện trên thiết bị: {DEVICE}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        # A. Chế độ Train
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # B. Chế độ Validation (Đánh giá)
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        # In kết quả sau mỗi Epoch
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%")

        # C. Lưu lại Trọng số Mô hình tốt nhất (Best Model Weights)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Tự động tạo thư mục chứa model nếu chưa có
            os.makedirs(os.path.dirname(MLP_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MLP_MODEL_PATH)

    print("-" * 50)
    print(f"[✔ SUCCESS] Huấn luyện hoàn tất!")
    print(f"[✔ SUCCESS] Độ chính xác cao nhất trên tập Val: {best_val_acc * 100:.2f}%")
    print(f"[✔ SUCCESS] Trọng số mô hình đã lưu tại: {MLP_MODEL_PATH}")


if __name__ == "__main__":
    train()