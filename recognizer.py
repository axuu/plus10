import os
import cv2
import numpy as np


class GridRecognizer:
    """基于模板匹配的数字网格识别器"""

    def __init__(
        self,
        template_dir: str,
        confidence_threshold: float = 0.7,
        empty_variance_threshold: float = 500,
    ):
        self.confidence_threshold = confidence_threshold
        self.empty_variance_threshold = empty_variance_threshold
        self.templates: dict[int, np.ndarray] = {}

        for d in range(1, 10):
            path = os.path.join(template_dir, f"{d}.png")
            if os.path.exists(path):
                tpl = cv2.imread(path)
                if tpl is not None:
                    self.templates[d] = tpl

    def recognize_cell(self, cell_img: np.ndarray) -> int:
        """识别单个格子中的数字。

        Args:
            cell_img: 格子图片 (BGR)

        Returns:
            1-9 表示数字，0 表示空格
        """
        # 快速空格检测：方差低说明无内容
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        if np.var(gray) < self.empty_variance_threshold:
            return 0

        best_digit = 0
        best_score = -1.0

        for digit, tpl in self.templates.items():
            # 调整模板大小匹配格子
            tpl_resized = cv2.resize(tpl, (cell_img.shape[1], cell_img.shape[0]))
            result = cv2.matchTemplate(cell_img, tpl_resized, cv2.TM_CCOEFF_NORMED)
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
        """从截图中提取数字矩阵。

        Args:
            screenshot: 完整截图 (BGR)
            origin_x, origin_y: 网格左上角像素坐标
            cell_width, cell_height: 格子尺寸
            cols, rows: 网格列数和行数

        Returns:
            rows x cols 的 numpy 数组，0=空，1-9=数字
        """
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
