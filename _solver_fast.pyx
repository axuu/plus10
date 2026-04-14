# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython 加速版的 solver 核心函数。"""
import numpy as np
cimport numpy as np

ctypedef np.int32_t INT32
ctypedef np.float64_t FLOAT64


def find_potential_c(grid_input):
    """一次扫描：构建前缀和 + 找出 sum∈[10,20] 且 cnt>=2 的全部矩形。

    Returns:
        valid_list: list of (r1,c1,r2,c2,cnt) sum=10 的有效矩形，按 cnt 降序
        potential:  (N, 6) int32 数组, 列=(r1,c1,r2,c2,cnt,sum)
        sp:         sum 前缀和
        cp:         cnt 前缀和
    """
    cdef np.ndarray[INT32, ndim=2] grid = np.asarray(grid_input, dtype=np.int32)
    cdef int rows = grid.shape[0]
    cdef int cols = grid.shape[1]
    cdef int r1, c1, r2, c2, rsum, rcnt

    # 构建前缀和
    cdef np.ndarray[INT32, ndim=2] sp = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    cdef np.ndarray[INT32, ndim=2] cp = np.zeros((rows + 1, cols + 1), dtype=np.int32)

    cdef int r, c
    for r in range(rows):
        for c in range(cols):
            sp[r + 1, c + 1] = grid[r, c] + sp[r, c + 1] + sp[r + 1, c] - sp[r, c]
            cp[r + 1, c + 1] = (1 if grid[r, c] > 0 else 0) + cp[r, c + 1] + cp[r + 1, c] - cp[r, c]

    # 扫描所有矩形
    cdef list valid_raw = []
    cdef list potential_raw = []

    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1, rows):
                for c2 in range(c1, cols):
                    rsum = sp[r2 + 1, c2 + 1] - sp[r1, c2 + 1] - sp[r2 + 1, c1] + sp[r1, c1]
                    if rsum < 10 or rsum > 20:
                        continue
                    rcnt = cp[r2 + 1, c2 + 1] - cp[r1, c2 + 1] - cp[r2 + 1, c1] + cp[r1, c1]
                    if rcnt < 2:
                        continue
                    potential_raw.append((r1, c1, r2, c2, rcnt, rsum))
                    if rsum == 10:
                        valid_raw.append((r1, c1, r2, c2, rcnt))

    # 有效矩形按 cnt 降序
    valid_raw.sort(key=lambda x: -x[4])

    # 潜在矩形转 numpy
    cdef np.ndarray[INT32, ndim=2] potential
    if len(potential_raw) > 0:
        potential = np.array(potential_raw, dtype=np.int32)
    else:
        potential = np.empty((0, 6), dtype=np.int32)

    return valid_raw, potential, sp, cp


def eval_candidates_c(
    np.ndarray[INT32, ndim=2] cand,
    np.ndarray[INT32, ndim=2] pot,
    np.ndarray[INT32, ndim=2] sp,
    np.ndarray[INT32, ndim=2] cp,
    double weight,
):
    """C 循环版前瞻评估，无临时数组分配。

    Args:
        cand: (nc, 5) 候选矩形, 列=(r1,c1,r2,c2,cnt)
        pot:  (np, 6) 潜在矩形, 列=(r1,c1,r2,c2,cnt,sum)
        sp:   sum 前缀和
        cp:   cnt 前缀和
        weight: 存活矩形权重

    Returns:
        (nc,) float64 评估分数
    """
    cdef int nc = cand.shape[0]
    cdef int np_ = pot.shape[0]
    cdef np.ndarray[FLOAT64, ndim=1] evals = np.empty(nc, dtype=np.float64)

    cdef int ci, pi
    cdef int r1a, c1a, r2a, c2a, cnta
    cdef int r1b, c1b, r2b, c2b, cntb, sumb
    cdef int ri1, ri2, ci1, ci2
    cdef int isect_sum, isect_cnt, new_sum, new_cnt
    cdef int surviving

    for ci in range(nc):
        r1a = cand[ci, 0]
        c1a = cand[ci, 1]
        r2a = cand[ci, 2]
        c2a = cand[ci, 3]
        cnta = cand[ci, 4]
        surviving = 0

        for pi in range(np_):
            r1b = pot[pi, 0]
            c1b = pot[pi, 1]
            r2b = pot[pi, 2]
            c2b = pot[pi, 3]
            cntb = pot[pi, 4]
            sumb = pot[pi, 5]

            # 交集边界
            ri1 = r1a if r1a > r1b else r1b
            ri2 = r2a if r2a < r2b else r2b
            ci1 = c1a if c1a > c1b else c1b
            ci2 = c2a if c2a < c2b else c2b

            if ri1 <= ri2 and ci1 <= ci2:
                isect_sum = sp[ri2 + 1, ci2 + 1] - sp[ri1, ci2 + 1] - sp[ri2 + 1, ci1] + sp[ri1, ci1]
                isect_cnt = cp[ri2 + 1, ci2 + 1] - cp[ri1, ci2 + 1] - cp[ri2 + 1, ci1] + cp[ri1, ci1]
            else:
                isect_sum = 0
                isect_cnt = 0

            new_sum = sumb - isect_sum
            new_cnt = cntb - isect_cnt
            if new_sum == 10 and new_cnt >= 2:
                surviving += 1

        evals[ci] = <double>cnta + <double>surviving * weight

    return evals
