# solver.py
import numpy as np
import logging

log = logging.getLogger("solver")


def find_valid_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形。

    Returns:
        list of (r1, c1, r2, c2, count) — count 为矩形内非零元素数量
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

    valid = (rect_sum == 10) & (rect_cnt >= 2)
    indices = np.nonzero(valid)[0]

    return [
        (int(r1[i]), int(c1[i]), int(r2[i]), int(c2[i]), int(rect_cnt[i]))
        for i in indices
    ]


def _apply_move(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    new_grid = grid.copy()
    new_grid[r1:r2 + 1, c1:c2 + 1] = 0
    return new_grid


def solve(
    grid: np.ndarray, depth: int = 4, beam_width: int = 50,
) -> tuple[int, int, int, int] | None:
    """层级 beam search 求解最优第一步。

    每层展开所有 beam 状态的合法操作，按累计消除数排序，
    只保留 beam_width 条最优路径。复杂度 O(depth × beam_width)。

    Args:
        grid: 16x10 数组
        depth: 前瞻步数
        beam_width: 每层保留的最优路径数

    Returns:
        (r1, c1, r2, c2) 最优矩形，或 None
    """
    # 每条路径: (累计消除数, 第一步操作, 当前棋盘)
    beam = [(0, None, grid)]

    for d in range(depth):
        next_candidates = []

        for score, first_move, g in beam:
            rectangles = find_valid_rectangles(g)
            for r1, c1, r2, c2, cnt in rectangles:
                new_grid = _apply_move(g, r1, c1, r2, c2)
                fm = first_move if first_move is not None else (r1, c1, r2, c2)
                next_candidates.append((score + cnt, fm, new_grid))

        if not next_candidates:
            break

        # 按累计消除数降序，保留 beam_width 条
        next_candidates.sort(key=lambda x: x[0], reverse=True)
        beam = next_candidates[:beam_width]

    if not beam or beam[0][1] is None:
        return None

    best_score, best_move, _ = beam[0]
    log.info(f"beam search: depth={depth}, beam={beam_width}, "
             f"best_score={best_score}, move={best_move}")
    return best_move
