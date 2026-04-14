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


def _simulate_game(
    grid: np.ndarray, rng: np.random.Generator, temp: float = 2.0
) -> tuple[list, int]:
    """随机加权贪心模拟。temp 控制贪心程度：越高越贪心。"""
    g = grid.copy()
    moves = []
    total = 0

    while True:
        rects = find_valid_rectangles(g, top_n=10)
        if not rects:
            break

        if temp <= 0:
            idx = rng.integers(len(rects))
        else:
            weights = np.array([cnt ** temp for _, _, _, _, cnt in rects], dtype=float)
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
    time_budget: float = 120.0,
    target_score: int = 0,
    beam_width: int = 30,
    warm_start: tuple[list, int] | None = None,
    **kwargs,
) -> tuple[list[tuple[int, int, int, int]], int]:
    """两阶段求解器，在时间预算内最大化分数。

    Phase 1: Beam search + 快照收集（快速，< 1s）
    Phase 2: 从多源起点（初始状态 + 中间快照）统一随机探索

    Args:
        grid: 数字矩阵
        time_budget: 时间预算（秒）
        target_score: 目标分数，达标提前停止（0=不设目标）
        beam_width: beam 宽度
        warm_start: (moves, score) 热启动，跳过已知下界以下的方案

    Returns:
        (best_moves, best_score)
    """
    total_cells = int(np.count_nonzero(grid))
    if total_cells == 0:
        return [], 0

    t0 = time.perf_counter()
    rng = np.random.default_rng()

    best_moves: list[tuple[int, int, int, int]] = []
    best_score = 0

    if warm_start is not None:
        best_moves, best_score = warm_start

    def elapsed():
        return time.perf_counter() - t0

    def hit_target():
        return target_score > 0 and best_score >= target_score

    beam_deadline = time_budget * 0.10

    # === Phase 1: Beam Search + 快照收集 ===
    beams: list[tuple[np.ndarray, list, int]] = [(grid.copy(), [], 0)]
    expand_n = max(beam_width, 20)
    depth = 0
    snapshot_pool: list[tuple[np.ndarray, list, int, int]] = []
    SNAPSHOT_INTERVAL = 5
    SNAPSHOT_TOP_N = 10

    while True:
        depth += 1
        candidates = []
        any_expanded = False

        for g, moves, score in beams:
            remaining = int(np.count_nonzero(g))
            if remaining == 0:
                candidates.append((g, moves, score))
                continue
            if score + remaining <= best_score:
                continue

            rects = find_valid_rectangles(g, top_n=expand_n)
            if not rects:
                candidates.append((g, moves, score))
                continue

            any_expanded = True
            for r1, c1, r2, c2, cnt in rects:
                new_g = g.copy()
                eliminated = int(np.count_nonzero(new_g[r1:r2 + 1, c1:c2 + 1]))
                new_g[r1:r2 + 1, c1:c2 + 1] = 0
                candidates.append(
                    (new_g, moves + [(r1, c1, r2, c2)], score + eliminated)
                )

        if not any_expanded or not candidates:
            break

        # 按分数降序 + grid 去重，保留 beam_width 个
        candidates.sort(key=lambda x: -x[2])
        seen: set[bytes] = set()
        unique: list[tuple[np.ndarray, list, int]] = []
        for g, moves, score in candidates:
            h = g.tobytes()
            if h not in seen:
                seen.add(h)
                unique.append((g, moves, score))
            if len(unique) >= beam_width:
                break
        beams = unique

        # 定期保存中间快照用于 Phase 2 rollout
        if depth % SNAPSHOT_INTERVAL == 0:
            for g, moves, score in beams[:SNAPSHOT_TOP_N]:
                remaining = int(np.count_nonzero(g))
                if remaining > 0:
                    snapshot_pool.append((g.copy(), list(moves), score, remaining))

        if elapsed() >= beam_deadline or hit_target():
            break

    # 贪心补全所有 beam，建立基线
    for g, moves, score in beams:
        extra_moves, extra_score = _greedy_complete(g)
        total = score + extra_score
        if total > best_score:
            best_score = total
            best_moves = moves + extra_moves

    # 也贪心补全快照状态（不同的中间路径可能有不同结局）
    for g, prefix_moves, prefix_score, _ in snapshot_pool:
        extra_moves, extra_score = _greedy_complete(g)
        total = prefix_score + extra_score
        if total > best_score:
            best_score = total
            best_moves = prefix_moves + extra_moves

    log.info(
        f"Phase1 beam: depth={depth}, {best_score}分/{total_cells}, "
        f"{len(best_moves)}步, 快照{len(snapshot_pool)}个, {elapsed():.1f}s"
    )

    if hit_target():
        log.info(f"达标 {best_score}>={target_score}, {elapsed():.1f}s")
        return best_moves, best_score

    # === Phase 2: 快照 Rollout（2% 剩余时间，快速试探）===
    remaining_budget = time_budget - elapsed()
    snapshot_deadline = elapsed() + remaining_budget * 0.02

    # 把末端仍有有效矩形的 beam 也加入快照池
    for g, moves, score in beams:
        remaining = int(np.count_nonzero(g))
        if remaining > 0 and find_valid_rectangles(g, top_n=1):
            snapshot_pool.append((g, moves, score, remaining))

    snap_count = 0
    snap_improved = 0

    while snapshot_pool and elapsed() < snapshot_deadline and not hit_target():
        idx = snap_count % len(snapshot_pool)
        g, prefix_moves, prefix_score, remaining = snapshot_pool[idx]
        snap_count += 1

        if prefix_score + remaining <= best_score:
            continue

        sim_moves, sim_score = _simulate_game(g, rng)
        total = prefix_score + sim_score

        if total > best_score:
            best_score = total
            best_moves = prefix_moves + sim_moves
            snap_improved += 1
            log.info(
                f"Snapshot #{snap_count}: {best_score}分/{total_cells}, "
                f"{len(best_moves)}步 ({elapsed():.1f}s)"
            )

    log.info(
        f"Phase2 snapshot: {snap_count}次, 改进{snap_improved}次, "
        f"{best_score}分, {elapsed():.1f}s"
    )

    if hit_target():
        log.info(f"达标 {best_score}>={target_score}, {elapsed():.1f}s")
        return best_moves, best_score

    # === Phase 3: Root MC（多温度策略，深度探索不同路径）===
    mc_count = 0
    mc_improved = 0
    # 混合温度：贪心(3.0) + 标准(2.0) + 平衡(1.0) + 探索(0.5)
    mc_temps = [3.0, 2.0, 2.0, 1.0, 0.5]

    while elapsed() < time_budget and not hit_target():
        temp = mc_temps[mc_count % len(mc_temps)]
        mc_moves, mc_score = _simulate_game(grid, rng, temp)
        mc_count += 1
        if mc_score > best_score:
            best_score = mc_score
            best_moves = mc_moves
            mc_improved += 1
            log.info(
                f"MC #{mc_count}: {best_score}分/{total_cells}, "
                f"{len(best_moves)}步, temp={temp} ({elapsed():.1f}s)"
            )
        if mc_count % 5000 == 0:
            log.debug(f"MC进度: {mc_count}次, 最优{best_score}分 ({elapsed():.1f}s)")

    log.info(f"Phase3 MC: {mc_count}次, 改进{mc_improved}次")

    pct = best_score / total_cells * 100 if total_cells > 0 else 0
    log.info(
        f"求解完成: {best_score}分/{total_cells} ({pct:.0f}%清除率), "
        f"{len(best_moves)}步, 耗时{elapsed():.1f}s"
    )

    return best_moves, best_score
