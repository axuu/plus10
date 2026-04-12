# solver.py
import numpy as np
import logging

log = logging.getLogger("solver")


def find_valid_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形。

    Returns:
        list of (r1, c1, r2, c2, count) 按 count 降序排列
    """
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

    # 面积
    areas = (r2 - r1 + 1) * (c2 - c1 + 1)

    valid = (rect_sum == 10) & (rect_cnt >= 2)
    indices = np.nonzero(valid)[0]

    counts = rect_cnt[indices]
    rect_areas = areas[indices]

    # 密度 = 消除数 / 面积，密度越高越好（少留空洞）
    densities = counts.astype(float) / rect_areas

    # 按 (消除数, 密度) 联合排序
    # 主排序: 消除数降序; 副排序: 密度降序
    order = np.lexsort((-densities, -counts))
    indices = indices[order]

    return [
        (int(r1[i]), int(c1[i]), int(r2[i]), int(c2[i]), int(rect_cnt[i]))
        for i in indices
    ]


def _apply_move(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    new_grid = grid.copy()
    new_grid[r1:r2 + 1, c1:c2 + 1] = 0
    return new_grid


def solve(
    grid: np.ndarray, depth: int = 5, beam_width: int = 15,
    branch_limit: int = 10,
) -> list[tuple[int, int, int, int]]:
    """层级 beam search 求解最优操作序列。

    返回完整操作序列而非单步，调用方可以一次性执行所有步骤。

    Args:
        grid: 16x10 数组
        depth: 前瞻步数（也是返回的最大序列长度）
        beam_width: 每层保留的最优路径数
        branch_limit: 每个状态最多展开的候选操作数

    Returns:
        [(r1, c1, r2, c2), ...] 操作序列，空列表表示无合法消除
    """
    # (累计消除数, 操作序列, 当前棋盘)
    beam = [(0, (), grid)]

    for d in range(depth):
        next_candidates = []

        for score, moves, g in beam:
            rectangles = find_valid_rectangles(g)[:branch_limit]
            for r1, c1, r2, c2, cnt in rectangles:
                new_grid = _apply_move(g, r1, c1, r2, c2)
                next_candidates.append((score + cnt, moves + ((r1, c1, r2, c2),), new_grid))

        if not next_candidates:
            break

        next_candidates.sort(key=lambda x: x[0], reverse=True)
        beam = next_candidates[:beam_width]

    if not beam or not beam[0][1]:
        return []

    best_score, best_moves, _ = beam[0]
    log.info(f"solve: depth={depth}, best_score={best_score}, steps={len(best_moves)}")
    return list(best_moves)
