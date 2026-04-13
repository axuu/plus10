# solver.py
import numpy as np
import logging
import time

log = logging.getLogger("solver")


def find_valid_rectangles(grid: np.ndarray, top_n: int = 0) -> list[tuple[int, int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形。

    Args:
        grid: 棋盘
        top_n: 只返回消除数最多的前 N 个（0=全部）

    Returns:
        list of (r1, c1, r2, c2, count) 按 count 降序
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

    if len(indices) == 0:
        return []

    counts = rect_cnt[indices]
    order = np.argsort(-counts)

    if top_n > 0:
        order = order[:top_n]

    indices = indices[order]

    return [
        (int(r1[i]), int(c1[i]), int(r2[i]), int(c2[i]), int(rect_cnt[i]))
        for i in indices
    ]


def _simulate_game(grid: np.ndarray, rng: np.random.Generator) -> tuple[list, int]:
    """模拟一局完整游戏（随机加权贪心）。

    从 top 5 中按消除数的平方为权重随机选择，兼顾贪心和探索。

    Returns:
        (操作序列, 总消除数)
    """
    g = grid.copy()
    moves = []
    total = 0

    while True:
        rects = find_valid_rectangles(g, top_n=8)
        if not rects:
            break

        # 加权随机选择：消除数的平方作为权重
        weights = np.array([cnt ** 2 for _, _, _, _, cnt in rects], dtype=float)
        weights /= weights.sum()

        idx = rng.choice(len(rects), p=weights)
        r1, c1, r2, c2, cnt = rects[idx]

        eliminated = int(np.count_nonzero(g[r1:r2+1, c1:c2+1]))
        moves.append((r1, c1, r2, c2))
        total += eliminated
        g[r1:r2+1, c1:c2+1] = 0

    return moves, total


def solve(
    grid: np.ndarray, n_simulations: int = 200, time_budget: float = 8.0,
    **kwargs,
) -> list[tuple[int, int, int, int]]:
    """蒙特卡洛规划：模拟多局完整游戏，选最优序列。

    Args:
        grid: 16x10 数组
        n_simulations: 最大模拟次数
        time_budget: 最大规划时间（秒）

    Returns:
        [(r1, c1, r2, c2), ...] 完整操作序列
    """
    if int(np.count_nonzero(grid)) == 0:
        return []

    rng = np.random.default_rng()
    best_moves = []
    best_score = 0
    t0 = time.perf_counter()

    # 先跑一次纯贪心（不随机）作为基线
    g = grid.copy()
    greedy_moves = []
    greedy_score = 0
    while True:
        rects = find_valid_rectangles(g, top_n=1)
        if not rects:
            break
        r1, c1, r2, c2, cnt = rects[0]
        eliminated = int(np.count_nonzero(g[r1:r2+1, c1:c2+1]))
        greedy_moves.append((r1, c1, r2, c2))
        greedy_score += eliminated
        g[r1:r2+1, c1:c2+1] = 0

    best_moves = greedy_moves
    best_score = greedy_score
    log.info(f"贪心基线: {greedy_score} 分, {len(greedy_moves)} 步")

    # 蒙特卡洛模拟
    for i in range(n_simulations):
        if time.perf_counter() - t0 > time_budget:
            log.info(f"时间预算用完，完成 {i} 次模拟")
            break

        moves, score = _simulate_game(grid, rng)
        if score > best_score:
            best_score = score
            best_moves = moves
            log.info(f"模拟 {i+1}: 新最优 {score} 分, {len(moves)} 步")

    elapsed = time.perf_counter() - t0
    log.info(f"规划完成: {best_score} 分, {len(best_moves)} 步, 耗时 {elapsed:.1f}s")
    return best_moves
