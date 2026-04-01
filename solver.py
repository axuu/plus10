# solver.py
import numpy as np


def find_valid_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形。

    Args:
        grid: 16x10 数组，0=空格，1-9=数字

    Returns:
        list of (r1, c1, r2, c2) 矩形坐标
    """
    rows, cols = grid.shape

    # 前缀和：sum_prefix[i][j] = grid[0..i-1][0..j-1] 的和
    sum_prefix = np.zeros((rows + 1, cols + 1), dtype=int)
    cnt_prefix = np.zeros((rows + 1, cols + 1), dtype=int)  # 非空格子计数

    for i in range(rows):
        for j in range(cols):
            sum_prefix[i + 1][j + 1] = (
                grid[i][j]
                + sum_prefix[i][j + 1]
                + sum_prefix[i + 1][j]
                - sum_prefix[i][j]
            )
            cnt_prefix[i + 1][j + 1] = (
                (1 if grid[i][j] > 0 else 0)
                + cnt_prefix[i][j + 1]
                + cnt_prefix[i + 1][j]
                - cnt_prefix[i][j]
            )

    result = []
    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    rect_sum = (
                        sum_prefix[r2 + 1][c2 + 1]
                        - sum_prefix[r1][c2 + 1]
                        - sum_prefix[r2 + 1][c1]
                        + sum_prefix[r1][c1]
                    )
                    rect_cnt = (
                        cnt_prefix[r2 + 1][c2 + 1]
                        - cnt_prefix[r1][c2 + 1]
                        - cnt_prefix[r2 + 1][c1]
                        + cnt_prefix[r1][c1]
                    )
                    if rect_sum == 10 and rect_cnt >= 2:
                        result.append((r1, c1, r2, c2))

    return result


def _count_nonzero_in_rect(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    """计算矩形内非零元素数量"""
    return int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))


def _apply_move(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    """执行消除，返回新的 grid（不修改原数组）"""
    new_grid = grid.copy()
    new_grid[r1:r2 + 1, c1:c2 + 1] = 0
    return new_grid


def _dfs(grid: np.ndarray, depth: int, current_score: int) -> tuple[int, tuple | None]:
    """DFS 搜索最优消除序列的第一步。

    Returns:
        (best_total_score, best_first_move) — best_first_move 是第一步的矩形坐标
    """
    if depth == 0:
        return current_score, None

    candidates = find_valid_rectangles(grid)
    if not candidates:
        return current_score, None

    # 按消除数字数降序排列；同分时优先选面积更小的矩形
    candidates.sort(
        key=lambda r: (
            _count_nonzero_in_rect(grid, r[0], r[1], r[2], r[3]),
            -((r[2] - r[0] + 1) * (r[3] - r[1] + 1)),
        ),
        reverse=True,
    )

    best_score = current_score
    best_move = None

    for rect in candidates:
        r1, c1, r2, c2 = rect
        eliminated = _count_nonzero_in_rect(grid, r1, c1, r2, c2)
        new_grid = _apply_move(grid, r1, c1, r2, c2)
        sub_score, _ = _dfs(new_grid, depth - 1, current_score + eliminated)

        if sub_score > best_score:
            best_score = sub_score
            best_move = rect

    return best_score, best_move


def solve(grid: np.ndarray, depth: int = 3) -> tuple[int, int, int, int] | None:
    """求解当前局面的最优第一步消除。

    Args:
        grid: 16x10 数组
        depth: 前瞻搜索深度

    Returns:
        (r1, c1, r2, c2) 最优矩形，或 None 表示无合法消除
    """
    _, best_move = _dfs(grid, depth, 0)
    return best_move
