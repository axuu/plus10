# solver.py
import numpy as np
import logging
import time

log = logging.getLogger("solver")


def find_valid_rectangles(grid: np.ndarray, top_n: int = 0) -> list[tuple[int, int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形。

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


def _greedy_complete(grid: np.ndarray) -> tuple[list, int]:
    """纯贪心：每步选消除数最多的矩形，直到无法继续。"""
    g = grid.copy()
    moves = []
    total = 0
    while True:
        rects = find_valid_rectangles(g, top_n=1)
        if not rects:
            break
        r1, c1, r2, c2, cnt = rects[0]
        eliminated = int(np.count_nonzero(g[r1:r2 + 1, c1:c2 + 1]))
        moves.append((r1, c1, r2, c2))
        total += eliminated
        g[r1:r2 + 1, c1:c2 + 1] = 0
    return moves, total


def _simulate_game(grid: np.ndarray, rng: np.random.Generator) -> tuple[list, int]:
    """随机加权贪心模拟。"""
    g = grid.copy()
    moves = []
    total = 0

    while True:
        rects = find_valid_rectangles(g, top_n=10)
        if not rects:
            break

        weights = np.array([cnt ** 2 for _, _, _, _, cnt in rects], dtype=float)
        weights /= weights.sum()

        idx = rng.choice(len(rects), p=weights)
        r1, c1, r2, c2, cnt = rects[idx]

        eliminated = int(np.count_nonzero(g[r1:r2 + 1, c1:c2 + 1]))
        moves.append((r1, c1, r2, c2))
        total += eliminated
        g[r1:r2 + 1, c1:c2 + 1] = 0

    return moves, total


def solve(
    grid: np.ndarray,
    depth: int = 5,
    beam_width: int = 15,
    n_simulations: int = 50,
    time_budget: float = 3.0,
    **kwargs,
) -> list[tuple[int, int, int, int]]:
    """Beam search + 贪心补全 + Monte Carlo 探索。

    Phase 1: Beam search 前 depth 步，保留 beam_width 个最优分支
    Phase 2: 每个分支用贪心补全到底
    Phase 3: Monte Carlo 随机模拟，在剩余时间内探索更多可能
    """
    if int(np.count_nonzero(grid)) == 0:
        return []

    t0 = time.perf_counter()
    rng = np.random.default_rng()

    # === Phase 1: Beam search ===
    # 每个 beam: (grid, moves, score)
    beams = [(grid.copy(), [], 0)]
    expand_n = max(beam_width, 20)  # 每个状态尝试的候选数

    for d in range(depth):
        candidates = []
        for g, moves, score in beams:
            rects = find_valid_rectangles(g, top_n=expand_n)
            if not rects:
                # 无路可走，保留当前状态
                candidates.append((g, moves, score))
                continue
            for r1, c1, r2, c2, cnt in rects:
                new_g = g.copy()
                eliminated = int(np.count_nonzero(new_g[r1:r2 + 1, c1:c2 + 1]))
                new_g[r1:r2 + 1, c1:c2 + 1] = 0
                candidates.append((new_g, moves + [(r1, c1, r2, c2)], score + eliminated))

        # 按分数降序，保留 beam_width 个
        candidates.sort(key=lambda x: -x[2])
        beams = candidates[:beam_width]

    # === Phase 2: 贪心补全每个 beam ===
    best_moves = []
    best_score = 0

    for g, moves, score in beams:
        extra_moves, extra_score = _greedy_complete(g)
        total_score = score + extra_score
        total_moves = moves + extra_moves
        if total_score > best_score:
            best_score = total_score
            best_moves = total_moves

    elapsed = time.perf_counter() - t0
    log.info(f"Beam search: {best_score} 分, {len(best_moves)} 步, 耗时 {elapsed:.1f}s")

    # === Phase 3: Monte Carlo 探索 ===
    mc_improved = 0
    for i in range(n_simulations):
        if time.perf_counter() - t0 > time_budget:
            log.info(f"时间预算用完，完成 {i} 次 MC 模拟")
            break

        mc_moves, mc_score = _simulate_game(grid, rng)
        if mc_score > best_score:
            best_score = mc_score
            best_moves = mc_moves
            mc_improved += 1
            log.info(f"MC 模拟 {i + 1}: 新最优 {mc_score} 分, {len(mc_moves)} 步")

    elapsed = time.perf_counter() - t0
    log.info(f"规划完成: {best_score} 分, {len(best_moves)} 步, "
             f"MC 改进 {mc_improved} 次, 总耗时 {elapsed:.1f}s")
    return best_moves
