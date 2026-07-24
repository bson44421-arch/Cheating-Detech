import numpy as np

def normalize_bbox_keypoints(kpts_17x2, bbox):
    """
    kpts_17x2: np.array shape (17, 2) - Tọa độ pixel (x, y)
    bbox: list/array [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)

    norm_kpts = np.zeros_like(kpts_17x2, dtype=np.float32)
    norm_kpts[:, 0] = (kpts_17x2[:, 0] - xmin) / w
    norm_kpts[:, 1] = (kpts_17x2[:, 1] - ymin) / h

    # Phẳng hóa thành vector 34 chiều
    return norm_kpts.flatten()