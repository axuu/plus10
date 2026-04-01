import os
import cv2
import numpy as np


def _extract_dark_pixels(img: np.ndarray, threshold: int = 80) -> np.ndarray:
    """只保留深色像素（数字），其余变白。

    Args:
        img: BGR 图片
        threshold: 灰度阈值，低于此值视为深色（数字）

    Returns:
        二值化灰度图，数字为黑(0)，背景为白(255)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


class GridRecognizer:
    """基于模板匹配的数字网格识别器"""

    def __init__(
        self,
        template_dir: str,
        confidence_threshold: float = 0.7,
        empty_variance_threshold: float = 500,
        dark_threshold: int = 80,
    ):
        self.confidence_threshold = confidence_threshold
        self.empty_variance_threshold = empty_variance_threshold
        self.dark_threshold = dark_threshold
        self.templates: dict[int, np.ndarray] = {}

        for d in range(1, 10):
            path = os.path.join(template_dir, f"{d}.png")
            if os.path.exists(path):
                tpl = cv2.imread(path)
                if tpl is not None:
                    self.templates[d] = _extract_dark_pixels(tpl, dark_threshold)

    def recognize_cell(self, cell_img: np.ndarray) -> int:
        """识别单个格子中的数字。

        Args:
            cell_img: 格子图片 (BGR)

        Returns:
            1-9 表示数字，0 表示空格
        """
        binary = _extract_dark_pixels(cell_img, self.dark_threshold)

        # 快速空格检测：深色像素太少说明无数字
        dark_ratio = np.count_nonzero(binary) / binary.size
        if dark_ratio < 0.02:
            return 0

        best_digit = 0
        best_score = -1.0

        for digit, tpl in self.templates.items():
            tpl_resized = cv2.resize(tpl, (binary.shape[1], binary.shape[0]))
            result = cv2.matchTemplate(binary, tpl_resized, cv2.TM_CCOEFF_NORMED)
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
