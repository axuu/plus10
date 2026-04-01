import os
import cv2
import numpy as np


def _crop_center(img: np.ndarray, ratio: float) -> np.ndarray:
    """裁取图片中心区域，只保留数字部分，忽略外围包子轮廓。

    Args:
        img: 输入图片
        ratio: 保留比例 (0-1)，如 0.5 表示裁取中心 50% 区域
    """
    h, w = img.shape[:2]
    margin_x = int(w * (1 - ratio) / 2)
    margin_y = int(h * (1 - ratio) / 2)
    return img[margin_y:h - margin_y, margin_x:w - margin_x]


class GridRecognizer:
    """基于模板匹配的数字网格识别器"""

    def __init__(
        self,
        template_dir: str,
        confidence_threshold: float = 0.7,
        empty_variance_threshold: float = 500,
        crop_ratio: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.empty_variance_threshold = empty_variance_threshold
        self.crop_ratio = crop_ratio
        self.templates: dict[int, np.ndarray] = {}

        for d in range(1, 10):
            path = os.path.join(template_dir, f"{d}.png")
            if os.path.exists(path):
                tpl = cv2.imread(path)
                if tpl is not None:
                    # 模板也只保留中心区域
                    self.templates[d] = _crop_center(tpl, crop_ratio)

    def recognize_cell(self, cell_img: np.ndarray) -> int:
        """识别单个格子中的数字。

        Args:
            cell_img: 格子图片 (BGR)

        Returns:
            1-9 表示数字，0 表示空格
        """
        # 裁取中心区域
        center = _crop_center(cell_img, self.crop_ratio)

        # 快速空格检测
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        if np.var(gray) < self.empty_variance_threshold:
            return 0

        best_digit = 0
        best_score = -1.0

        for digit, tpl in self.templates.items():
            tpl_resized = cv2.resize(tpl, (center.shape[1], center.shape[0]))
            result = cv2.matchTemplate(center, tpl_resized, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]

            if score > best_score:
                best_score = score
                best_digit = digit

        if best_score < self.confidence_threshold:
            return 0

        return best_digit

    def extract_grid(
        self,
        screenshot: np.ndarray,
        origin_x: int,
        origin_y: int,
        cell_width: int,
        cell_height: int,
        cols: int = 10,
        rows: int = 16,
    ) -> np.ndarray:
        """从截图中提取数字矩阵。"""
        grid = np.zeros((rows, cols), dtype=int)

        for r in range(rows):
            for c in range(cols):
                x = origin_x + c * cell_width
                y = origin_y + r * cell_height
                cell = screenshot[y:y + cell_height, x:x + cell_width]

                if cell.shape[0] == 0 or cell.shape[1] == 0:
                    continue

                grid[r][c] = self.recognize_cell(cell)

        return grid
