from collections import deque
import numpy as np
import torch


class RealTimeTracker:
    def __init__(self, model, window_size=5, cheat_thresh=0.7, device="cpu"):
        self.model = model
        self.window_size = window_size
        self.cheat_thresh = cheat_thresh
        self.device = device
        self.prob_history = deque(maxlen=window_size)

    def process_frame(self, kpt_vector_34d, has_phone):
        # 1. Luồng ưu tiên cao nhất: Phát hiện điện thoại
        if has_phone:
            return "CHEATING (Phone)", 1.0

        if kpt_vector_34d is None:
            return "NO PERSON", 0.0

        # 2. Dự đoán trên khung hình hiện tại
        self.model.eval()
        with torch.no_grad():
            inp = torch.tensor([kpt_vector_34d], dtype=torch.float32).to(self.device)
            out = self.model(inp)
            probs = torch.softmax(out, dim=1)
            cheat_prob = probs[0][1].item()  # Xác suất của class 1 (Cheating)

        # 3. Làm mịn bằng Moving Average
        self.prob_history.append(cheat_prob)
        smoothed_prob = float(np.mean(self.prob_history))

        # 4. Quyết định trạng thái
        status = "CHEATING" if smoothed_prob >= self.cheat_thresh else "NORMAL"
        return status, smoothed_prob