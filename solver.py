# solver.py
import numpy as np


def find_valid_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形（向量化实现）。

    Args:
        grid: 16x10 数组，0=空格，1-9=数字

    Returns:
        list of (r1, c1, r2, c2) 矩形坐标
    """
    rows, cols = grid.shape

    # 前缀和
    sum_prefix = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    cnt_prefix = np.zeros((rows + 1, cols + 1), dtype=np.int32)

    sum_prefix[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    cnt_prefix[1:, 1:] = np.cumsum(np.cumsum(grid > 0, axis=0), axis=1)

    # 生成所有 (r1, c1, r2, c2) 组合的索引
    r1_vals = np.arange(rows)
    c1_vals = np.arange(cols)
    r2_vals = np.arange(rows)
    c2_vals = np.arange(cols)

    r1, c1, r2, c2 = np.meshgrid(r1_vals, c1_vals, r2_vals, c2_vals, indexing="ij")
    r1 = r1.ravel()
    c1 = c1.ravel()
    r2 = r2.ravel()
    c2 = c2.ravel()

    # 只保留 r2 >= r1 且 c2 >= c1
    mask = (r2 >= r1) & (c2 >= c1)
    r1, c1, r2, c2 = r1[mask], c1[mask], r2[mask], c2[mask]

    # 向量化计算矩形和与非零计数
    rect_sum = (
        sum_prefix[r2 + 1, c2 + 1]
        - sum_prefix[r1, c2 + 1]
        - sum_prefix[r2 + 1, c1]
        + sum_prefix[r1, c1]
    )
    rect_cnt = (
        cnt_prefix[r2 + 1, c2 + 1]
        - cnt_prefix[r1, c2 + 1]
        - cnt_prefix[r2 + 1, c1]
        + cnt_prefix[r1, c1]
    )

    valid = (rect_sum == 10) & (rect_cnt >= 2)
    indices = np.nonzero(valid)[0]

    return [
        (int(r1[i]), int(c1[i]), int(r2[i]), int(c2[i]))
        for i in indices
    ]


def _count_nonzero_in_rect(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    """计算矩形内非零元素数量"""
    return int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))


def _apply_move(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    """执行消除，返回新的 grid（不修改原数组）"""
    new_grid = grid.copy()
    new_grid[r1:r2 + 1, c1:c2 + 1] = 0
    return new_grid


def _dfs(
    grid: np.ndarray, depth: int, current_score: int, beam_width: int,
) -> tuple[int, tuple | None]:
    """DFS + Beam Search 搜索最优消除序列的第一步。

    Args:
        grid: 当前棋盘
        depth: 剩余搜索深度
        current_score: 当前累计分数
        beam_width: 每层最多探索的候选数

    Returns:
        (best_total_score, best_first_move)
    """
    if depth == 0:
        return current_score, None

    candidates = find_valid_rectangles(grid)
    if not candidates:
        return current_score, None

    # 按消除数字数降序排列，同分时优先小面积矩形；只取前 beam_width 个
    candidates.sort(
        key=lambda r: (
            _count_nonzero_in_rect(grid, r[0], r[1], r[2], r[3]),
            -((r[2] - r[0] + 1) * (r[3] - r[1] + 1)),
        ),
        reverse=True,
    )
    candidates = candidates[:beam_width]

    best_score = current_score
    best_move = None

    for rect in candidates:
        r1, c1, r2, c2 = rect
        eliminated = _count_nonzero_in_rect(grid, r1, c1, r2, c2)
        new_grid = _apply_move(grid, r1, c1, r2, c2)
        sub_score, _ = _dfs(new_grid, depth - 1, current_score + eliminated, beam_width)

        if sub_score > best_score:
            best_score = sub_score
            best_move = rect

    return best_score, best_move


def solve(
    grid: np.ndarray, depth: int = 3, beam_width: int = 15,
) -> tuple[int, int, int, int] | None:
    """求解当前局面的最优第一步消除。

    Args:
        grid: 16x10 数组
        depth: 前瞻搜索深度
        beam_width: 每层最多探索的候选数量

    Returns:
        (r1, c1, r2, c2) 最优矩形，或 None 表示无合法消除
    """
    _, best_move = _dfs(grid, depth, 0, beam_width)
    return best_move
