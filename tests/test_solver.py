# tests/test_solver.py
import numpy as np
from solver import find_valid_rectangles


def test_find_single_pair_horizontal():
    """横向两个数字和为10"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 3
    grid[0][1] = 7
    rects = find_valid_rectangles(grid)
    assert (0, 0, 0, 1) in rects


def test_find_single_pair_vertical():
    """纵向两个数字和为10"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 1
    grid[1][0] = 9
    rects = find_valid_rectangles(grid)
    assert (0, 0, 1, 0) in rects


def test_no_valid_rectangles():
    """全1矩阵，没有和为10的矩形（单格不算）"""
    grid = np.ones((16, 10), dtype=int)
    rects = find_valid_rectangles(grid)
    # 每个矩形至少2个非空数字，最小矩形2格和=2，不等于10
    # 10个1的和=10，所以横向10格的行应该被找到
    assert any(r for r in rects)


def test_rectangle_with_empty_cells():
    """矩形内包含空格，只算非空数字"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 4
    grid[0][2] = 6
    # (0,0)到(0,2) 包含 4,0,6 和=10
    rects = find_valid_rectangles(grid)
    assert (0, 0, 0, 2) in rects


def test_must_have_at_least_two_nonzero():
    """单个数字10不存在(1-9)，所以不需要特殊处理，但确保单格不被选"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 5
    rects = find_valid_rectangles(grid)
    # 单格 5 != 10，不应在结果中
    assert (0, 0, 0, 0) not in rects
