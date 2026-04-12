# solver.py
import numpy as np
import logging

log = logging.getLogger("solver")


def find_valid_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形（向量化实现）。"""
    rows, cols = grid.shape

    sum_prefix = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    cnt_prefix = np.zeros((rows + 1, cols + 1), dtype=np.int32)

    sum_prefix[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    cnt_prefix[1:, 1:] = np.cumsum(np.cumsum(grid > 0, axis=0), axis=1)

    r1_vals = np.arange(rows)
    c1_vals = np.arange(cols)
    r2_vals = np.arange(rows)
    c2_vals = np.arange(cols)

    r1, c1, r2, c2 = np.meshgrid(r1_vals, c1_vals, r2_vals, c2_vals, indexing="ij")
    r1 = r1.ravel()
    c1 = c1.ravel()
    r2 = r2.ravel()
    c2 = c2.ravel()

    mask = (r2 >= r1) & (c2 >= c1)
    r1, c1, r2, c2 = r1[mask], c1[mask], r2[mask], c2[mask]

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
    return int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))


def _apply_move(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    new_grid = grid.copy()
    new_grid[r1:r2 + 1, c1:c2 + 1] = 0
    return new_grid


def _estimate_potential(grid: np.ndarray) -> float:
    """估算剩余局面的消除潜力（快速启发式）。"""
    candidates = find_valid_rectangles(grid)
    if not candidates:
        return 0.0

    # 取最大的几个不重叠消除作为潜力估算
    used = np.zeros_like(grid, dtype=bool)
    potential = 0
    # 按消除数排序
    scored = []
    for rect in candidates:
        cnt = _count_nonzero_in_rect(grid, *rect)
        scored.append((cnt, rect))
    scored.sort(reverse=True)

    for cnt, (r1, c1, r2, c2) in scored[:20]:
        # 检查是否与已选的重叠
        region = used[r1:r2+1, c1:c2+1]
        if np.any(region):
            continue
        used[r1:r2+1, c1:c2+1] = True
        potential += cnt

    return potential


def _dfs(
    grid: np.ndarray, depth: int, current_score: int, beam_width: int,
) -> tuple[float, tuple | None]:
    """DFS + Beam Search 搜索最优消除序列的第一步。"""
    if depth == 0:
        # 叶子节点：当前分 + 未来潜力估算（折扣）
        potential = _estimate_potential(grid)
        return current_score + potential * 0.3, None

    candidates = find_valid_rectangles(grid)
    if not candidates:
        return current_score, None

    # 排序：按消除数降序；同消除数优先密度高（小面积）的
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
    grid: np.ndarray, depth: int = 4, beam_width: int = 20,
) -> tuple[int, int, int, int] | None:
    """求解当前局面的最优第一步消除。

    Args:
        grid: 16x10 数组
        depth: 前瞻搜索深度（默认4）
        beam_width: 每层最多探索的候选数量（默认20）

    Returns:
        (r1, c1, r2, c2) 最优矩形，或 None 表示无合法消除
    """
    _, best_move = _dfs(grid, depth, 0, beam_width)
    return best_move
