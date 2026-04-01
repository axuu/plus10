import numpy as np
import cv2
import os
import tempfile
from recognizer import GridRecognizer


def _make_digit_image(digit: int, cell_w: int = 50, cell_h: int = 50) -> np.ndarray:
    """生成一个带数字的测试格子图片"""
    img = np.full((cell_h, cell_w, 3), 200, dtype=np.uint8)  # 灰色背景
    cv2.putText(
        img, str(digit), (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2,
    )
    return img


def _make_empty_image(cell_w: int = 50, cell_h: int = 50) -> np.ndarray:
    """生成空白格子图片"""
    return np.full((cell_h, cell_w, 3), 200, dtype=np.uint8)


def test_recognize_single_digit():
    """识别单个数字"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 为每个数字生成模板
        for d in range(1, 10):
            tpl = _make_digit_image(d)
            cv2.imwrite(os.path.join(tmpdir, f"{d}.png"), tpl)

        recognizer = GridRecognizer(
            template_dir=tmpdir,
            confidence_threshold=0.7,
            empty_variance_threshold=500,
        )

        # 测试识别
        test_img = _make_digit_image(5)
        result = recognizer.recognize_cell(test_img)
        assert result == 5


def test_recognize_empty_cell():
    """空格子应返回0"""
    with tempfile.TemporaryDirectory() as tmpdir:
        for d in range(1, 10):
            tpl = _make_digit_image(d)
            cv2.imwrite(os.path.join(tmpdir, f"{d}.png"), tpl)

        recognizer = GridRecognizer(
            template_dir=tmpdir,
            confidence_threshold=0.7,
            empty_variance_threshold=500,
        )

        empty_img = _make_empty_image()
        result = recognizer.recognize_cell(empty_img)
        assert result == 0


def test_extract_grid():
    """从完整截图提取矩阵"""
    cell_w, cell_h = 50, 50
    cols, rows = 10, 16

    with tempfile.TemporaryDirectory() as tmpdir:
        for d in range(1, 10):
            tpl = _make_digit_image(d, cell_w, cell_h)
            cv2.imwrite(os.path.join(tmpdir, f"{d}.png"), tpl)

        recognizer = GridRecognizer(
            template_dir=tmpdir,
            confidence_threshold=0.7,
            empty_variance_threshold=500,
        )

        # 构造一个完整网格图片
        full_img = np.full((rows * cell_h, cols * cell_w, 3), 200, dtype=np.uint8)
        expected = np.zeros((rows, cols), dtype=int)

        # 在 (0,0) 放数字 3
        digit_img = _make_digit_image(3, cell_w, cell_h)
        full_img[0:cell_h, 0:cell_w] = digit_img
        expected[0][0] = 3

        grid = recognizer.extract_grid(
            full_img,
            origin_x=0, origin_y=0,
            cell_width=cell_w, cell_height=cell_h,
            cols=cols, rows=rows,
        )

        assert grid[0][0] == expected[0][0]
        assert grid.shape == (rows, cols)
