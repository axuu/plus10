import os
import logging
import cv2
import numpy as np

log = logging.getLogger("recognizer")

NORM_SIZE = 40  # 归一化数字尺寸


def _extract_dark_pixels(img: np.ndarray, threshold: int = 80) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def _normalize_digit(binary_img: np.ndarray) -> np.ndarray | None:
    """从二值图中提取数字区域，归一化到 NORM_SIZE x NORM_SIZE。"""
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 2 or h < 2:
        return None
    digit = binary_img[y:y + h, x:x + w]
    # 保持宽高比缩放，留 2px 边距
    scale = min((NORM_SIZE - 4) / h, (NORM_SIZE - 4) / w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 居中放到画布上
    canvas = np.zeros((NORM_SIZE, NORM_SIZE), dtype=np.uint8)
    y_off = (NORM_SIZE - new_h) // 2
    x_off = (NORM_SIZE - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


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
        self.templates_raw: dict[int, list[np.ndarray]] = {}
        # 预计算归一化后的模板数字
        self.template_norms: dict[int, list[np.ndarray]] = {}

        log.info(f"加载模板目录: {template_dir}, dark_threshold={dark_threshold}")
        for d in range(1, 10):
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

        # 预计算：对每个模板做中心裁剪 → 二值化 → 归一化
        for d, tpl_list in self.templates_raw.items():
            norms = []
            for tpl in tpl_list:
                th, tw = tpl.shape[:2]
                crop = 0.15
                cx1 = int(tw * crop)
                cy1 = int(th * crop)
                tpl_center = tpl[cy1:th - cy1, cx1:tw - cx1]
                tpl_bin = _extract_dark_pixels(tpl_center, dark_threshold)
                norm = _normalize_digit(tpl_bin)
                if norm is not None:
                    norms.append(norm)
                    log.debug(f"  模板 {d}: 归一化成功")
                else:
                    log.warning(f"  模板 {d}: 归一化失败（无深色像素）")
            self.template_norms[d] = norms

    def recognize_cell(self, cell_img: np.ndarray, row: int = -1, col: int = -1) -> int:
        cell_h, cell_w = cell_img.shape[:2]

        # 用二值化判断是否为空
        binary = _extract_dark_pixels(cell_img, self.dark_threshold)
        dark_ratio = np.count_nonzero(binary) / binary.size
        if dark_ratio < 0.005:
            log.debug(f"  cell({row},{col}): 空格 (dark_ratio={dark_ratio:.4f})")
            return 0

        # 中心裁剪：去掉圆形边框
        crop = 0.15
        cx1 = int(cell_w * crop)
        cy1 = int(cell_h * crop)
        cell_center = cell_img[cy1:cell_h - cy1, cx1:cell_w - cx1]

        # 二值化 → 提取数字 → 归一化
        cell_bin = _extract_dark_pixels(cell_center, self.dark_threshold)
        cell_norm = _normalize_digit(cell_bin)
        if cell_norm is None:
            log.debug(f"  cell({row},{col}): 归一化失败")
            return 0

        best_digit = 0
        best_score = -1.0
        scores = {}

        for digit, norms in self.template_norms.items():
            digit_best = -1.0
            for tpl_norm in norms:
                result = cv2.matchTemplate(cell_norm, tpl_norm, cv2.TM_CCOEFF_NORMED)
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

        ox = origin_x * w if origin_x <= 1.0 else float(origin_x)
        oy = origin_y * h if origin_y <= 1.0 else float(origin_y)
        cw = cell_width * w if cell_width <= 1.0 else float(cell_width)
        ch = cell_height * h if cell_height <= 1.0 else float(cell_height)

        log.debug(f"提取网格: origin=({ox:.1f},{oy:.1f}), cell=({cw:.1f}x{ch:.1f}), grid=({cols}x{rows}), 截图=({w}x{h})")
        grid = np.zeros((rows, cols), dtype=int)

        for r in range(rows):
            for c in range(cols):
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
