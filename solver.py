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
