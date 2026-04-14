# tests/test_solver.py
import numpy as np
from solver import find_valid_rectangles, solve


def test_find_single_pair_horizontal():
    """横向两个数字和为10"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 3
    grid[0][1] = 7
    rects = find_valid_rectangles(grid)
    assert any(r[:4] == (0, 0, 0, 1) for r in rects)


def test_find_single_pair_vertical():
    """纵向两个数字和为10"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 1
    grid[1][0] = 9
    rects = find_valid_rectangles(grid)
    assert any(r[:4] == (0, 0, 1, 0) for r in rects)


def test_no_valid_rectangles():
    """全1矩阵，横向10格的行和=10应被找到"""
    grid = np.ones((16, 10), dtype=int)
    rects = find_valid_rectangles(grid)
    assert len(rects) > 0


def test_rectangle_with_empty_cells():
    """矩形内包含空格，只算非空数字"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 4
    grid[0][2] = 6
    rects = find_valid_rectangles(grid)
    assert any(r[:4] == (0, 0, 0, 2) for r in rects)


def test_must_have_at_least_two_nonzero():
    """单格不被选"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 5
    rects = find_valid_rectangles(grid)
    assert not any(r[:4] == (0, 0, 0, 0) for r in rects)


def test_solve_simple_pair():
    """最简单情况：只有一对可消"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 2
    grid[0][1] = 8
    moves, score = solve(grid, time_budget=2.0)
    assert len(moves) > 0
    assert moves[0] == (0, 0, 0, 1)
    assert score == 2


def test_solve_prefers_more_digits():
    """应该能清除所有数字"""
    grid = np.zeros((16, 10), dtype=int)
    # 选项1: 2+8=10, 消2个
    grid[0][0] = 2
    grid[0][1] = 8
    # 选项2: 1+2+3+4=10, 消4个
    grid[2][0] = 1
    grid[2][1] = 2
    grid[2][2] = 3
    grid[2][3] = 4
    moves, score = solve(grid, time_budget=2.0)
    assert score == 6  # 全部6个数字都能被清除


def test_solve_no_moves():
    """没有可消的情况"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 1
    moves, score = solve(grid, time_budget=1.0)
    assert moves == []
    assert score == 0


def test_solve_empty_grid():
    """空网格"""
    grid = np.zeros((16, 10), dtype=int)
    moves, score = solve(grid, time_budget=1.0)
    assert moves == []
    assert score == 0


def test_solve_with_target_score():
    """目标分数达标时提前停止"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 2
    grid[0][1] = 8
    grid[1][0] = 3
    grid[1][1] = 7
    moves, score = solve(grid, time_budget=2.0, target_score=4)
    assert score >= 4


def test_solve_warm_start():
    """热启动不会返回更差的结果"""
    grid = np.zeros((16, 10), dtype=int)
    grid[0][0] = 2
    grid[0][1] = 8
    warm_moves = [(0, 0, 0, 1)]
    warm_score = 2
    moves, score = solve(grid, time_budget=1.0, warm_start=(warm_moves, warm_score))
    assert score >= warm_score
