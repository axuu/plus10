# solver.py
import numpy as np
import logging
import time

log = logging.getLogger("solver")

# 尝试加载 Cython 加速模块
try:
    from _solver_fast import find_potential_c, eval_candidates_c
    _USE_CYTHON = True
    log.info("Cython 加速模块已加载")
except ImportError:
    _USE_CYTHON = False

# 缓存 meshgrid 索引，同尺寸 grid 只算一次
_INDEX_CACHE: dict[tuple[int, int], tuple] = {}


def _build_prefix(grid: np.ndarray):
    """构建前缀和，用于 O(1) 矩形 sum/cnt 查询。"""
    rows, cols = grid.shape
    sp = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    cp = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    sp[1:, 1:] = np.cumsum(np.cumsum(grid, axis=0), axis=1)
    cp[1:, 1:] = np.cumsum(np.cumsum(grid > 0, axis=0), axis=1)
    return sp, cp


def find_valid_rectangles(
    grid: np.ndarray, top_n: int = 0, prefix=None,
) -> list[tuple[int, int, int, int, int]]:
    """找出所有和为10且至少包含2个非空数字的矩形。

    Args:
        prefix: (sum_prefix, cnt_prefix) 可选，避免重复计算
    Returns:
        list of (r1, c1, r2, c2, count) 按 count 降序
    """
    rows, cols = grid.shape
    key = (rows, cols)

    # 缓存 meshgrid 索引
    if key not in _INDEX_CACHE:
        r1v = np.arange(rows)
        c1v = np.arange(cols)
        r1, c1, r2, c2 = np.meshgrid(r1v, c1v, r1v, c1v, indexing="ij")
        r1, c1, r2, c2 = r1.ravel(), c1.ravel(), r2.ravel(), c2.ravel()
        mask = (r2 >= r1) & (c2 >= c1)
        _INDEX_CACHE[key] = (r1[mask], c1[mask], r2[mask], c2[mask])

    r1, c1, r2, c2 = _INDEX_CACHE[key]

    if prefix is not None:
        sum_prefix, cnt_prefix = prefix
    else:
        sum_prefix, cnt_prefix = _build_prefix(grid)

    rect_sum = (
        sum_prefix[r2 + 1, c2 + 1] - sum_prefix[r1, c2 + 1]
        - sum_prefix[r2 + 1, c1] + sum_prefix[r1, c1]
    )
    rect_cnt = (
        cnt_prefix[r2 + 1, c2 + 1] - cnt_prefix[r1, c2 + 1]
        - cnt_prefix[r2 + 1, c1] + cnt_prefix[r1, c1]
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


def _find_potential(grid: np.ndarray, prefix=None):
    """找 sum∈[10,20] 且 cnt>=2 的全部矩形（含当前有效 + 可解锁的）。

    Returns: (np, 6) int32 数组，列=(r1,c1,r2,c2,cnt,sum)
    """
    rows, cols = grid.shape
    key = (rows, cols)
    if key not in _INDEX_CACHE:
        find_valid_rectangles(grid, top_n=1)  # 触发缓存
    r1, c1, r2, c2 = _INDEX_CACHE[key]
    sp, cp = prefix if prefix else _build_prefix(grid)

    rect_sum = sp[r2 + 1, c2 + 1] - sp[r1, c2 + 1] - sp[r2 + 1, c1] + sp[r1, c1]
    rect_cnt = cp[r2 + 1, c2 + 1] - cp[r1, c2 + 1] - cp[r2 + 1, c1] + cp[r1, c1]

    mask = (rect_sum >= 10) & (rect_sum <= 20) & (rect_cnt >= 2)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return np.empty((0, 6), dtype=np.int32)

    return np.column_stack([
        r1[idx], c1[idx], r2[idx], c2[idx], rect_cnt[idx], rect_sum[idx]
    ]).astype(np.int32)


def _eval_candidates(cand_arr, potential_arr, sp, cp, weight: float):
    """解锁感知的前瞻评估：考虑清除后哪些 sum>10 矩形被激活。

    Args:
        cand_arr: 候选矩形 (nc, 5) int32, 列=(r1,c1,r2,c2,cnt)
        potential_arr: 潜在矩形 (np, 6) int32, 列=(r1,c1,r2,c2,cnt,sum)
                       包含 sum∈[10,20] 的全部矩形
    Returns:
        每个候选的 eval 分数 (nc,) float
    """
    if len(potential_arr) == 0:
        return cand_arr[:, 4].astype(float)

    # (nc,1) × (1,np) 广播
    ri1 = np.maximum(cand_arr[:, 0:1], potential_arr[np.newaxis, :, 0])
    ri2 = np.minimum(cand_arr[:, 2:3], potential_arr[np.newaxis, :, 2])
    ci1 = np.maximum(cand_arr[:, 1:2], potential_arr[np.newaxis, :, 1])
    ci2 = np.minimum(cand_arr[:, 3:4], potential_arr[np.newaxis, :, 3])
    has_overlap = (ri1 <= ri2) & (ci1 <= ci2)

    nc, np_ = len(cand_arr), len(potential_arr)
    isect_sum = np.zeros((nc, np_), dtype=np.int32)
    isect_cnt = np.zeros((nc, np_), dtype=np.int32)
    if has_overlap.any():
        m = has_overlap
        isect_sum[m] = (sp[ri2[m] + 1, ci2[m] + 1] - sp[ri1[m], ci2[m] + 1]
                        - sp[ri2[m] + 1, ci1[m]] + sp[ri1[m], ci1[m]])
        isect_cnt[m] = (cp[ri2[m] + 1, ci2[m] + 1] - cp[ri1[m], ci2[m] + 1]
                        - cp[ri2[m] + 1, ci1[m]] + cp[ri1[m], ci1[m]])

    # 潜在矩形的原始 sum: (1, np_)
    a_sum = potential_arr[np.newaxis, :, 5]
    new_sum = a_sum - isect_sum
    new_cnt = potential_arr[np.newaxis, :, 4] - isect_cnt

    # 清除候选后有多少矩形变为 sum=10 且 cnt>=2（含存活 + 新解锁）
    future_valid = np.sum((new_sum == 10) & (new_cnt >= 2), axis=1).astype(float)

    return cand_arr[:, 4].astype(float) + future_valid * weight


def _greedy_complete(grid: np.ndarray, lookahead: bool = False,
                     weight: float = 0.3) -> tuple[list, int]:
    """贪心补全。lookahead=True 时用解锁感知评估。"""
    g = grid.copy()
    moves = []
    total = 0
    while True:
        if lookahead:
            if _USE_CYTHON:
                valid_list, potential, sp, cp = find_potential_c(g)
                if not valid_list:
                    break
            else:
                sp, cp = _build_prefix(g)
                potential = _find_potential(g, prefix=(sp, cp))
                if len(potential) == 0:
                    break
                valid_mask = potential[:, 5] == 10
                if not valid_mask.any():
                    break
                va = potential[valid_mask]
                order = np.argsort(-va[:, 4])
                valid_list = [(int(va[i,0]),int(va[i,1]),int(va[i,2]),int(va[i,3]),int(va[i,4])) for i in order]
        else:
            valid_list = find_valid_rectangles(g, top_n=1)
            if not valid_list:
                break

        if not lookahead or len(valid_list) == 1:
            r1, c1, r2, c2, cnt = valid_list[0]
        else:
            nc = min(15, len(valid_list))
            cand_arr = np.array(valid_list[:nc], dtype=np.int32)
            if _USE_CYTHON:
                evals = eval_candidates_c(cand_arr, potential, sp, cp, weight)
            else:
                evals = _eval_candidates(cand_arr, potential, sp, cp, weight)
            best = int(np.argmax(evals))
            r1, c1, r2, c2, cnt = valid_list[best]

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
    """解锁感知的前瞻模拟：评估每步对未来可行矩形（含解锁）的影响。"""
    g = grid.copy()
    moves = []
    total = 0

    while True:
        if _USE_CYTHON:
            valid_list, potential, sp, cp = find_potential_c(g)
            if not valid_list:
                break
        else:
            sp, cp = _build_prefix(g)
            potential = _find_potential(g, prefix=(sp, cp))
            if len(potential) == 0:
                break
            valid_mask = potential[:, 5] == 10
            if not valid_mask.any():
                break
            va = potential[valid_mask]
            order = np.argsort(-va[:, 4])
            valid_list = [(int(va[i,0]),int(va[i,1]),int(va[i,2]),int(va[i,3]),int(va[i,4])) for i in order]

        if len(valid_list) == 1:
            r1, c1, r2, c2, cnt = valid_list[0]
        else:
            nc = min(10, len(valid_list))
            cand_arr = np.array(valid_list[:nc], dtype=np.int32)
            if _USE_CYTHON:
                evals = eval_candidates_c(cand_arr, potential, sp, cp, weight)
            else:
                evals = _eval_candidates(cand_arr, potential, sp, cp, weight)

            evals = evals ** 2
            evals /= evals.sum()
            idx = rng.choice(nc, p=evals)
            r1, c1, r2, c2, cnt = valid_list[idx]

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

    # 50% 散点删除 (1-6步)，50% 段删除 (连续块，最多半数步骤)
    if rng.random() < 0.5 or len(moves) <= 4:
        n_remove = int(rng.integers(1, min(7, len(moves))))
        remove_set = set(rng.choice(len(moves), size=n_remove, replace=False).tolist())
    else:
        max_seg = max(2, len(moves) // 2)
        seg_len = int(rng.integers(2, min(max_seg + 1, len(moves))))
        start = int(rng.integers(0, len(moves) - seg_len + 1))
        remove_set = set(range(start, start + seg_len))

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


def _replay_moves(grid: np.ndarray, moves: list) -> tuple[np.ndarray, list, int]:
    """在 grid 上重放步骤序列，跳过失效的。返回 (结果grid, 有效步骤, 得分)。"""
    g = grid.copy()
    valid_moves = []
    score = 0
    for r1, c1, r2, c2 in moves:
        sub = g[r1:r2 + 1, c1:c2 + 1]
        if int(sub.sum()) == 10 and int(np.count_nonzero(sub)) >= 2:
            score += int(np.count_nonzero(sub))
            g[r1:r2 + 1, c1:c2 + 1] = 0
            valid_moves.append((r1, c1, r2, c2))
    return g, valid_moves, score


def _crossover(
    grid: np.ndarray, moves_a: list, moves_b: list,
    rng: np.random.Generator, weight: float = 0.3,
) -> tuple[list, int]:
    """解交叉：取 A 的前缀 + B 的后缀，重放有效步骤后补全。"""
    min_len = min(len(moves_a), len(moves_b))
    if min_len <= 2:
        return moves_a, 0  # 太短无法交叉

    cut = int(rng.integers(1, min_len))
    combined = list(moves_a[:cut]) + list(moves_b[cut:])
    g, valid_moves, score = _replay_moves(grid, combined)

    extra_moves, extra_score = _greedy_complete(g, lookahead=True, weight=weight)
    return valid_moves + extra_moves, score + extra_score


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

    # === Phase 3: 交替搜索（MC + 扰动 + 交叉 + 解池 + 重启）===
    POOL_SIZE = 8
    pool: list[tuple[list, int]] = []
    if best_moves:
        pool.append((best_moves, best_score))

    mc_count = 0
    mc_improved = 0
    perturb_count = 0
    perturb_improved = 0
    cross_count = 0
    cross_improved = 0
    explore_weights = [0.05, 0.2, 0.3, 0.6, 1.0]
    total_iter = 0
    no_improve = 0
    MC_WARMUP = 20
    RESTART_THRESHOLD = 5000

    while elapsed() < time_budget and not hit_target():
        total_iter += 1
        w = explore_weights[total_iter % len(explore_weights)]

        # 重启：长时间无改进 → 清池 + MC 爆发
        if no_improve >= RESTART_THRESHOLD:
            log.debug(f"重启: {no_improve}次无改进 ({elapsed():.1f}s)")
            no_improve = 0
            pool = [(best_moves, best_score)]

        # 操作选择：初期 MC; 之后 1/7 MC + 2/7 交叉 + 4/7 扰动
        in_warmup = total_iter <= MC_WARMUP or not pool
        force_mc = no_improve >= RESTART_THRESHOLD - MC_WARMUP
        do_mc = in_warmup or total_iter % 7 == 0 or force_mc
        do_cross = not do_mc and total_iter % 7 in (1, 2) and len(pool) >= 2

        improved = False

        if do_mc:
            mc_moves, mc_score = _simulate_lookahead(grid, rng, weight=w)
            mc_count += 1
            if mc_score > best_score:
                best_score, best_moves = mc_score, mc_moves
                mc_improved += 1
                improved = True
                log.info(
                    f"MC #{mc_count}: {best_score}分/{total_cells}, "
                    f"{len(best_moves)}步, w={w} ({elapsed():.1f}s)"
                )
            pool.append((mc_moves, mc_score))

        elif do_cross:
            i = int(rng.integers(len(pool)))
            j = int(rng.integers(len(pool) - 1))
            if j >= i:
                j += 1
            new_moves, new_score = _crossover(
                grid, pool[i][0], pool[j][0], rng, weight=w
            )
            cross_count += 1
            if new_score > best_score:
                best_score, best_moves = new_score, new_moves
                cross_improved += 1
                improved = True
                log.info(
                    f"Cross #{cross_count}: {best_score}分/{total_cells}, "
                    f"{len(best_moves)}步, w={w} ({elapsed():.1f}s)"
                )
            pool.append((new_moves, new_score))

        else:
            pidx = int(rng.integers(len(pool)))
            p_moves, _ = pool[pidx]
            use_random = rng.random() < 0.3
            new_moves, new_score = _perturb_solution(
                grid, p_moves, 0, rng, weight=w, use_random=use_random
            )
            perturb_count += 1
            if new_score > best_score:
                best_score, best_moves = new_score, new_moves
                perturb_improved += 1
                improved = True
                log.info(
                    f"Perturb #{perturb_count}: {best_score}分/{total_cells}, "
                    f"{len(best_moves)}步, w={w} ({elapsed():.1f}s)"
                )
            pool.append((new_moves, new_score))

        no_improve = 0 if improved else no_improve + 1

        if len(pool) > POOL_SIZE * 2:
            pool.sort(key=lambda x: -x[1])
            del pool[POOL_SIZE:]

        if total_iter % 5000 == 0:
            log.debug(
                f"进度: MC{mc_count} 扰动{perturb_count} 交叉{cross_count}, "
                f"池{len(pool)}个, 最优{best_score}分 ({elapsed():.1f}s)"
            )

    log.info(
        f"Phase3: MC {mc_count}/改进{mc_improved}, "
        f"扰动 {perturb_count}/改进{perturb_improved}, "
        f"交叉 {cross_count}/改进{cross_improved}"
    )

    pct = best_score / total_cells * 100 if total_cells > 0 else 0
    log.info(
        f"求解完成: {best_score}分/{total_cells} ({pct:.0f}%清除率), "
        f"{len(best_moves)}步, 耗时{elapsed():.1f}s"
    )

    return best_moves, best_score
