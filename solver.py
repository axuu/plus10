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


def _build_prefix(grid: np.ndarray):
    """构建前缀和，用于 O(1) 矩形 sum/cnt 查询。"""
    rows, cols = grid.shape
    sp = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    cp = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    sp[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    cp[1:, 1:] = np.cumsum(np.cumsum(grid > 0, axis=0), axis=1)
    return sp, cp


def _eval_candidates(candidates, all_rects_arr, sp, cp, weight: float):
    """向量化评估：清除候选矩形后还剩多少有效矩形。

    Returns: 每个候选的 eval 分数 (np.ndarray)
    """
    a_r1 = all_rects_arr[:, 0]
    a_c1 = all_rects_arr[:, 1]
    a_r2 = all_rects_arr[:, 2]
    a_c2 = all_rects_arr[:, 3]
    a_cnt = all_rects_arr[:, 4]
    a_sum = (sp[a_r2 + 1, a_c2 + 1] - sp[a_r1, a_c2 + 1]
             - sp[a_r2 + 1, a_c1] + sp[a_r1, a_c1])

    nc = len(candidates)
    evals = np.empty(nc, dtype=float)

    for ci in range(nc):
        r1a, c1a, r2a, c2a, cnta = candidates[ci]

        # 向量化交集
        ri1 = np.maximum(r1a, a_r1)
        ri2 = np.minimum(r2a, a_r2)
        ci1 = np.maximum(c1a, a_c1)
        ci2 = np.minimum(c2a, a_c2)
        has_overlap = (ri1 <= ri2) & (ci1 <= ci2)

        isect_sum = np.zeros(len(a_r1), dtype=np.int32)
        isect_cnt = np.zeros(len(a_r1), dtype=np.int32)
        if has_overlap.any():
            m = has_overlap
            isect_sum[m] = (sp[ri2[m] + 1, ci2[m] + 1] - sp[ri1[m], ci2[m] + 1]
                            - sp[ri2[m] + 1, ci1[m]] + sp[ri1[m], ci1[m]])
            isect_cnt[m] = (cp[ri2[m] + 1, ci2[m] + 1] - cp[ri1[m], ci2[m] + 1]
                            - cp[ri2[m] + 1, ci1[m]] + cp[ri1[m], ci1[m]])

        new_sum = a_sum - isect_sum
        new_cnt = a_cnt - isect_cnt
        surviving = int(np.sum((new_sum == 10) & (new_cnt >= 2)))
        evals[ci] = cnta + surviving * weight

    return evals


def _greedy_complete(grid: np.ndarray, lookahead: bool = False,
                     weight: float = 0.3) -> tuple[list, int]:
    """贪心补全。lookahead=True 时每步评估对后续的影响。"""
    g = grid.copy()
    moves = []
    total = 0
    while True:
        if lookahead:
            all_rects = find_valid_rectangles(g, top_n=0)
        else:
            all_rects = find_valid_rectangles(g, top_n=1)
        if not all_rects:
            break

        if not lookahead or len(all_rects) == 1:
            r1, c1, r2, c2, cnt = all_rects[0]
        else:
            candidates = all_rects[:15]
            sp, cp = _build_prefix(g)
            arr = np.array(all_rects, dtype=np.int32)
            evals = _eval_candidates(candidates, arr, sp, cp, weight)
            r1, c1, r2, c2, cnt = candidates[int(np.argmax(evals))]

        eliminated = int(np.count_nonzero(g[r1:r2 + 1, c1:c2 + 1]))
        moves.append((r1, c1, r2, c2))
        total += eliminated
        g[r1:r2 + 1, c1:c2 + 1] = 0
    return moves, total


def _simulate_game(
    grid: np.ndarray, rng: np.random.Generator, temp: float = 2.0
) -> tuple[list, int]:
    """随机加权贪心模拟（无前瞻，快速）。"""
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


def _simulate_lookahead(
    grid: np.ndarray, rng: np.random.Generator, weight: float = 0.3
) -> tuple[list, int]:
    """前瞻模拟：评估每步对后续可行矩形数量的影响。"""
    g = grid.copy()
    moves = []
    total = 0

    while True:
        all_rects = find_valid_rectangles(g, top_n=0)
        if not all_rects:
            break

        if len(all_rects) == 1:
            r1, c1, r2, c2, cnt = all_rects[0]
        else:
            candidates = all_rects[:10]
            sp, cp = _build_prefix(g)
            arr = np.array(all_rects, dtype=np.int32)
            evals = _eval_candidates(candidates, arr, sp, cp, weight)

            # 加权随机选择
            evals = evals ** 2
            evals /= evals.sum()
            idx = rng.choice(len(candidates), p=evals)
            r1, c1, r2, c2, cnt = candidates[idx]

        eliminated = int(np.count_nonzero(g[r1:r2 + 1, c1:c2 + 1]))
        moves.append((r1, c1, r2, c2))
        total += eliminated
        g[r1:r2 + 1, c1:c2 + 1] = 0

    return moves, total


def _perturb_solution(
    grid: np.ndarray, moves: list, score: int,
    rng: np.random.Generator, weight: float = 0.3,
    use_random: bool = False,
) -> tuple[list, int]:
    """扰动搜索：从已有方案中删除若干步，重放有效步骤后重新补全。

    use_random=True 时用随机前瞻补全（增加多样性），否则用确定性前瞻贪心。
    """
    if len(moves) <= 2:
        return moves, score

    # 随机删除 1~6 步
    n_remove = int(rng.integers(1, min(7, len(moves))))
    remove_set = set(rng.choice(len(moves), size=n_remove, replace=False).tolist())

    # 重放剩余步骤，跳过因删除导致失效的
    g = grid.copy()
    new_moves = []
    new_score = 0
    for i, (r1, c1, r2, c2) in enumerate(moves):
        if i in remove_set:
            continue
        sub = g[r1:r2 + 1, c1:c2 + 1]
        rect_sum = int(sub.sum())
        rect_cnt = int(np.count_nonzero(sub))
        if rect_sum == 10 and rect_cnt >= 2:
            new_score += rect_cnt
            g[r1:r2 + 1, c1:c2 + 1] = 0
            new_moves.append((r1, c1, r2, c2))

    # 补全：随机前瞻 or 确定性前瞻贪心
    if use_random:
        extra_moves, extra_score = _simulate_lookahead(g, rng, weight=weight)
    else:
        extra_moves, extra_score = _greedy_complete(g, lookahead=True, weight=weight)
    return new_moves + extra_moves, new_score + extra_score


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

    # 前瞻贪心补全所有 beam + 快照，建立基线
    all_states = [(g, moves, score) for g, moves, score in beams]
    all_states += [(g, m, s) for g, m, s, _ in snapshot_pool]

    for g, moves, score in all_states:
        for la in (False, True):
            extra_moves, extra_score = _greedy_complete(g, lookahead=la)
            total = score + extra_score
            if total > best_score:
                best_score = total
                best_moves = moves + extra_moves

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

    # === Phase 3: 交替搜索（MC 探索 + 扰动调优 + 解池）===
    POOL_SIZE = 5
    pool: list[tuple[list, int]] = []
    if best_moves:
        pool.append((best_moves, best_score))

    mc_count = 0
    mc_improved = 0
    perturb_count = 0
    perturb_improved = 0
    explore_weights = [0.05, 0.2, 0.3, 0.6, 1.0]
    total_iter = 0
    MC_WARMUP = 30  # 先跑一批 MC 填充解池

    while elapsed() < time_budget and not hit_target():
        total_iter += 1
        w = explore_weights[total_iter % len(explore_weights)]

        # MC 探索 or 扰动调优（交替，初期偏 MC）
        do_mc = total_iter <= MC_WARMUP or total_iter % 3 == 0 or not pool

        if do_mc:
            mc_moves, mc_score = _simulate_lookahead(grid, rng, weight=w)
            mc_count += 1
            if mc_score > best_score:
                best_score = mc_score
                best_moves = mc_moves
                mc_improved += 1
                log.info(
                    f"MC #{mc_count}: {best_score}分/{total_cells}, "
                    f"{len(best_moves)}步, w={w} ({elapsed():.1f}s)"
                )
            # 加入解池
            pool.append((mc_moves, mc_score))
        else:
            # 从解池中随机选一个解进行扰动
            pidx = int(rng.integers(len(pool)))
            p_moves, p_score = pool[pidx]
            use_random = rng.random() < 0.3
            new_moves, new_score = _perturb_solution(
                grid, p_moves, p_score, rng, weight=w, use_random=use_random
            )
            perturb_count += 1
            if new_score > best_score:
                best_score = new_score
                best_moves = new_moves
                perturb_improved += 1
                log.info(
                    f"Perturb #{perturb_count}: {best_score}分/{total_cells}, "
                    f"{len(best_moves)}步, w={w} ({elapsed():.1f}s)"
                )
            # 加入解池
            pool.append((new_moves, new_score))

        # 定期裁剪解池：保留 top POOL_SIZE
        if len(pool) > POOL_SIZE * 2:
            pool.sort(key=lambda x: -x[1])
            del pool[POOL_SIZE:]

        if total_iter % 2000 == 0:
            log.debug(
                f"进度: MC {mc_count}次 扰动 {perturb_count}次, "
                f"池{len(pool)}个, 最优{best_score}分 ({elapsed():.1f}s)"
            )

    log.info(
        f"Phase3: MC {mc_count}次/改进{mc_improved}, "
        f"扰动 {perturb_count}次/改进{perturb_improved}"
    )

    pct = best_score / total_cells * 100 if total_cells > 0 else 0
    log.info(
        f"求解完成: {best_score}分/{total_cells} ({pct:.0f}%清除率), "
        f"{len(best_moves)}步, 耗时{elapsed():.1f}s"
    )

    return best_moves, best_score
