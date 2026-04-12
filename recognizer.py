import os
import logging
import cv2
import numpy as np

log = logging.getLogger("recognizer")


def _extract_dark_pixels(img: np.ndarray, threshold: int = 80) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


class GridRecognizer:
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
        # 存储原始彩色模板，不预先二值化
        self.templates_raw: dict[int, np.ndarray] = {}

        log.info(f"加载模板目录: {template_dir}, dark_threshold={dark_threshold}")
        for d in range(1, 10):
            path = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                candidate = os.path.join(template_dir, f"{d}{ext}")
                if os.path.exists(candidate):
                    path = candidate
                    break

            if path is None:
                log.warning(f"  模板 {d}: 文件不存在 ({template_dir}/{d}.*)")
                continue

            tpl = cv2.imread(path)
            if tpl is not None:
                self.templates_raw[d] = tpl
                log.info(f"  模板 {d}: shape={tpl.shape}, 文件={path}")
            else:
                log.warning(f"  模板 {d}: 读取失败 ({path})")

        log.info(f"共加载 {len(self.templates_raw)} 个模板")

        # 兼容旧代码引用
        self.templates = self.templates_raw

    def recognize_cell(self, cell_img: np.ndarray, row: int = -1, col: int = -1) -> int:
        cell_h, cell_w = cell_img.shape[:2]

        # 对格子做二值化
        binary = _extract_dark_pixels(cell_img, self.dark_threshold)

        dark_ratio = np.count_nonzero(binary) / binary.size
        if dark_ratio < 0.02:
            log.debug(f"  cell({row},{col}): 空格 (dark_ratio={dark_ratio:.4f})")
            return 0

        best_digit = 0
        best_score = -1.0
        scores = {}

        for digit, tpl_raw in self.templates_raw.items():
            # 先缩放原图到格子尺寸，再二值化（关键改动）
            tpl_resized = cv2.resize(tpl_raw, (cell_w, cell_h))
            tpl_binary = _extract_dark_pixels(tpl_resized, self.dark_threshold)

            result = cv2.matchTemplate(binary, tpl_binary, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]
            scores[digit] = score

            if score > best_score:
                best_score = score
                best_digit = digit

        if best_score < self.confidence_threshold:
            log.debug(f"  cell({row},{col}): 低置信度 best={best_digit}({best_score:.3f}) scores={scores}")
            return 0

        log.debug(f"  cell({row},{col}): 识别为 {best_digit} (score={best_score:.3f})")
        return best_digit

    def extract_grid(
        self,
        screenshot: np.ndarray,
        origin_x: float,
        origin_y: float,
        cell_width: float,
        cell_height: float,
        cols: int = 10,
        rows: int = 16,
    ) -> np.ndarray:
        h, w = screenshot.shape[:2]

        # 如果值 <= 1.0，视为比例，乘以截图尺寸转换为像素
        if origin_x <= 1.0:
            origin_x = int(origin_x * w)
        else:
            origin_x = int(origin_x)
        if origin_y <= 1.0:
            origin_y = int(origin_y * h)
        else:
            origin_y = int(origin_y)
        if cell_width <= 1.0:
            cell_width = int(cell_width * w)
        else:
            cell_width = int(cell_width)
        if cell_height <= 1.0:
            cell_height = int(cell_height * h)
        else:
            cell_height = int(cell_height)

        log.debug(f"提取网格: origin=({origin_x},{origin_y}), cell=({cell_width}x{cell_height}), grid=({cols}x{rows}), 截图=({w}x{h})")
        grid = np.zeros((rows, cols), dtype=int)

        for r in range(rows):
            for c in range(cols):
                x = origin_x + c * cell_width
                y = origin_y + r * cell_height
                cell = screenshot[y:y + cell_height, x:x + cell_width]

                if cell.shape[0] == 0 or cell.shape[1] == 0:
                    log.warning(f"  cell({r},{c}): 空图片 shape={cell.shape}")
                    continue

                grid[r][c] = self.recognize_cell(cell, row=r, col=c)

        return grid
