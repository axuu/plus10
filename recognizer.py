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
            # 支持多模板: d.png, d_2.png, d_3.png ...
            variants = [str(d)] + [f"{d}_{i}" for i in range(2, 10)]
            loaded = []
            for name in variants:
                for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                    candidate = os.path.join(template_dir, f"{name}{ext}")
                    if os.path.exists(candidate):
                        tpl = cv2.imread(candidate)
                        if tpl is not None:
                            loaded.append(tpl)
                            log.info(f"  模板 {d}: shape={tpl.shape}, 文件={candidate}")
                        break

            if loaded:
                self.templates_raw[d] = loaded
            else:
                log.warning(f"  模板 {d}: 文件不存在 ({template_dir}/{d}.*)")

        total = sum(len(v) for v in self.templates_raw.values())
        log.info(f"共加载 {total} 个模板 ({len(self.templates_raw)} 个数字)")

        # 兼容旧代码引用
        self.templates = self.templates_raw

    def recognize_cell(self, cell_img: np.ndarray, row: int = -1, col: int = -1) -> int:
        cell_h, cell_w = cell_img.shape[:2]

        # 用二值化判断是否为空
        binary = _extract_dark_pixels(cell_img, self.dark_threshold)
        dark_ratio = np.count_nonzero(binary) / binary.size
        if dark_ratio < 0.005:
            log.debug(f"  cell({row},{col}): 空格 (dark_ratio={dark_ratio:.4f})")
            return 0

        # 中心裁剪：只保留内部 60% 区域（去掉圆形边框，聚焦数字）
        crop = 0.2  # 每边裁掉 20%
        cx1 = int(cell_w * crop)
        cy1 = int(cell_h * crop)
        cx2 = cell_w - cx1
        cy2 = cell_h - cy1

        cell_center = cell_img[cy1:cy2, cx1:cx2]
        cell_bin = _extract_dark_pixels(cell_center, self.dark_threshold)

        best_digit = 0
        best_score = -1.0
        scores = {}

        for digit, tpl_list in self.templates_raw.items():
            digit_best = -1.0
            for tpl_raw in tpl_list:
                tpl_resized = cv2.resize(tpl_raw, (cell_w, cell_h))
                tpl_center = tpl_resized[cy1:cy2, cx1:cx2]
                tpl_bin = _extract_dark_pixels(tpl_center, self.dark_threshold)

                result = cv2.matchTemplate(cell_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
                score = result[0][0]
                if score > digit_best:
                    digit_best = score

            scores[digit] = digit_best
            if digit_best > best_score:
                best_score = digit_best
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

        # 如果值 <= 1.0，视为比例，乘以截图尺寸（保持浮点精度）
        ox = origin_x * w if origin_x <= 1.0 else float(origin_x)
        oy = origin_y * h if origin_y <= 1.0 else float(origin_y)
        cw = cell_width * w if cell_width <= 1.0 else float(cell_width)
        ch = cell_height * h if cell_height <= 1.0 else float(cell_height)

        log.debug(f"提取网格: origin=({ox:.1f},{oy:.1f}), cell=({cw:.1f}x{ch:.1f}), grid=({cols}x{rows}), 截图=({w}x{h})")
        grid = np.zeros((rows, cols), dtype=int)

        for r in range(rows):
            for c in range(cols):
                # 每个格子独立从浮点计算像素位置，避免误差累积
                x = int(round(ox + c * cw))
                y = int(round(oy + r * ch))
                x2 = int(round(ox + (c + 1) * cw))
                y2 = int(round(oy + (r + 1) * ch))
                cell = screenshot[y:y2, x:x2]

                if cell.shape[0] == 0 or cell.shape[1] == 0:
                    log.warning(f"  cell({r},{c}): 空图片 shape={cell.shape}")
                    continue

                grid[r][c] = self.recognize_cell(cell, row=r, col=c)

        return grid
