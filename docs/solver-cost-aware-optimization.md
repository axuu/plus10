# Solver Cost-Aware 优化记录

> 把 3 块真实数据上的总分从 **345 → 363-364 (+18~19, +5.5%)**。
> 时间预算 15s/board，每块跑 5 trials 取最佳。
> 日期：2026-05-16。

## 1. 问题诊断

### 1.1 跑残局分析看 solver 失败模式

让 baseline solver 跑完三块真实棋盘，剩余的数字分布：

| Board | 残局总数 | 残局数字分布 (1 → 9) |
|-------|---------|---------------------|
| 1 | 54 | 1:**1**, 2:1, 3:1, 4:2, 5:12, 6:7, 7:7, 8:12, 9:11 |
| 2 | 45 | 2:2, 4:3, 5:2, 6:6, 7:8, 8:11, 9:**13** |
| 3 | 42 | 2:1, 3:1, 4:1, 5:4, 6:6, 7:9, 8:6, 9:**14** |

**典型病灶：小数字（1、2）已经被消光，剩下一大堆 8、9 配不出来。** Board 3 残局 14 个 9 全部孤立就是最极端例子。

### 1.2 原始 board 的供需失衡

每个数字 `d` 主要和 `10-d` 配对。原始供给对比：

| Board | count(1) | count(9) | 1 vs 9 缺口 |
|-------|---------|----------|------------|
| 1 | 14 | 16 | 欠 2 |
| 2 | 15 | 20 | 欠 5 |
| 3 | 17 | 23 | **欠 6** |

不平衡越严重，原算法残局越糟（Board 3 best 仅 120/160）。

### 1.3 根因

原 evaluator 公式：
```python
eval = cnt + future_valid * weight
```

只看"这次能消几个"和"消完后还剩几个 sum=10 矩形"，完全不区分"消的是哪些数字"。在它眼里：
- 一个 (5, 5) 双消（消两个 5，资源充裕）
- 一个 (1, 9) 双消（消一个珍贵的 1）

cnt 都是 2、收益相等。但前者保留稀缺资源，后者透支。

## 2. 优化方案

### 2.1 数字稀缺度度量 — `_compute_digit_cost`

定义每个数字 `d` 的边际消耗代价 `cost[d] ∈ [0, 1]`，含义：消去一个 `d` 后，未来可能少消多少个 cell（启发式估计）。

```python
def _compute_digit_cost(grid):
    c = bincount(grid)        # 当前每个数字剩余量
    cost = zeros(10)
    # d=5 自配对：每两个 5 配一对，消一个折损 0.5
    if c[5] >= 2:  cost[5] = 0.5
    elif c[5] == 1: cost[5] = 1.0
    # d != 5：对面 = 10-d
    for d in (1,2,3,4,6,7,8,9):
        o = 10 - d
        if c[d] <= c[o]:
            cost[d] = 1.0           # 稀缺方：消一个让对面一个变孤儿
        else:
            cost[d] = c[o] / c[d]   # 过剩方：按比例打折
    return cost
```

直觉：
- Board 1 中 `count(1)=14 ≤ count(9)=16`：消一个 1 的代价是 1.0（让一个 9 没人配）
- 反过来消一个 9 的代价 = 14/16 ≈ 0.88（其余 9 还有 1 可配，只略损）
- 消一个 5 的代价 0.5（5+5 自配对，每两个配一对）

### 2.2 O(1) 矩形 cost 查询

仿照原代码的 sum/cnt 前缀和，加一份 cost 前缀和：

```python
def _build_cost_prefix(grid, digit_cost):
    cell_cost = digit_cost[grid]  # 每个 cell 的代价 (R, C)
    cp = zeros((R+1, C+1))
    cp[1:, 1:] = cumsum(cumsum(cell_cost, axis=0), axis=1)
    return cp

def _rect_cost_batch(cand_arr, cost_prefix):
    # 给一批 (r1,c1,r2,c2) 一次性算出每个 rect 的总 cost
    ...
```

每步 grid 变化后重算（开销 O(R·C) ≈ 几百 ops，可忽略）。

### 2.3 新 evaluator

```python
eval = cnt + future_valid * β  -  α * sum(cost[cells in rect])
       └────────────────────┘     └─────────────────────────┘
       原始：当前 + 未来潜力         新增：消耗稀缺资源的折损
```

`α` 是关键超参。

### 2.4 α 调参（关键发现）

15s × 2 trials × 5 个 α 并行扫：

| α | B1 | B2 | B3 | 总和 |
|---|----|----|----|------|
| 0.0 | 107 | 116 | 120 | 343 |
| 0.15 | 109 | 114 | 120 | 343 |
| 0.3 | 110 | 116 | 121 | 347 |
| **0.5** | **114** | 116 | 120 | **350** |
| 0.8 | 112 | 115 | 121 | 348 |

**α=0.5 是甜蜜点。** 太小没用，太大把高 cnt rect 也压下去（贪不到该贪的）。

### 2.5 Phase 1 beam search 同步 cost-aware

原始 beam 只按当前分排序、按 cnt 扩展候选。改：

1. **排序加未来潜力**：按 `score + _pair_upper_bound(grid)` 排序，其中
   ```python
   _pair_upper_bound(g) = 2 * Σ_{d<5} min(c[d], c[10-d]) + 2*(c[5]//2)
   ```
   给"剩余能配对潜力大"的状态优先级。
   > ⚠️ 注意：pair bound 不是严格上界（三元组 1+3+6 比配对消的多），所以**只能用于排序，不能用于剪枝**。剪枝条件仍用 `score + remaining`。

2. **扩展时拉宽并重排**：先取 `top expand_n*2` 候选，再按 `cnt - α·cost` 重排取 top expand_n。促进"虽然 cnt 不顶但资源便宜"的 rect 被探索。

### 2.6 Phase 3 多源种子 + 时间感知 restart

- **多源种子**：Phase 1 完成阶段会对 top-30 状态各做 3 种补全（纯贪心 / 前瞻无cost / 前瞻+cost），共 ~90 个解，全部放进 `phase1_seeds`。Phase 3 初始 pool 优先用这些（按分去重）。
- **时间感知 restart**：原版每 5000 迭代无改进才重启（在 15s 内基本触发不到）。改成"max(2s, 剩余时间·20%) 无改进就重启"，每次重启混入 3 个 phase1 随机种子，立即注入新搜索方向。

### 2.7 MC 候选拉宽 10 → 15

`_simulate_lookahead` 每步从 sum=10 rect 中按 eval² 概率采样。把候选从 10 提到 15，增加每步选择多样性。

### 2.8 经验教训

- ❌ **α 轮换/抖动伤 best-of-N**：试过 [0, 0.2, 0.5, 0.5, 0.7]、[α·0.6, α, α, α·1.2, α]，avg 反而比固定 α=0.5 略高（更稳）但 best 显著下降（少了"凑巧大胆探索到好路径"）。**对求 best 这种 max 目标，方差是朋友。** 最终用固定 α=0.5。
- ❌ **过度激进剪枝**：实验中一度用 `pair_upper_bound` 做 Phase 1 剪枝；表面上变更紧但其实是错的（三元组能消更多），导致截断有效路径。
- ✅ **代价是用 cell 单位、不要做归一化**：`eval = cnt - α·rect_cost` 这里 cnt 和 cost 都是 cell 单位，相减就是"净 cell 收益"。

## 3. 最终结果

3 块真实棋盘，15s × 5 trials：

| Board | 原 best | 新 best | Δ | 原 avg | 新 avg |
|-------|---------|---------|-----|--------|--------|
| 1 | 105 | **115** | **+10** | 102.7 | 114.0 |
| 2 | 120 | **121** | +1 | 117.7 | 117.4 |
| 3 | 120 | **128** | **+8** | 119.0 | 126.0 |
| **合计** | **345** | **364** | **+19 (+5.5%)** | | |

### 收益模式
- **数字越不平衡，cost-aware 增益越大**：Board 3 收益最大（极端不平衡），Board 1 也显著（轻度不平衡），Board 2 几乎没动（瓶颈是几何约束）。
- Board 2 即使新 best 也才 121/160，残局结构受地理排布限制，光优化数字代价不够。如果还想提升，下一步应是几何感知（比如孤立单元识别）。

## 4. 没做但可能继续提分的方向

- **几何感知**：识别"孤立的稀缺数字"（周围没有合适配对邻居），降低其 cost；反向，给"在密集区的稀缺数字"加权（容易被配掉）。
- **Cython 内嵌 cost penalty**：当前 cost 是 Python 端在 Cython 输出后做减法。把它推进 Cython 能省点常数开销，估计每步省几 μs。
- **多进程多起点**：用 multiprocessing 并行跑 4 个不同 seed 的 solver，取 best。对 15s 预算，wallclock 不变但实际计算量 ×4。
- **LP 上界估计代替 pair bound**：用 transportation LP 求更准的剩余可消上界（含 triples / quads），更精准的 Phase 1 排序与（如果是严格上界的话）剪枝。
- **MCTS-light**：把 Phase 3 的随机滚动换成 UCT 搜索，理论上更收敛但实现复杂。

## 5. 复现命令

```bash
# 准备环境（首次）
cd ~/code/plus10
uv venv --python 3.11 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/pip install -r requirements.txt
.venv/bin/python setup_cython.py build_ext --inplace

# 验证测试通过
.venv/bin/python -m pytest tests/test_solver.py

# 跑 3 块真实数据 baseline
.venv/bin/python -u bench_boards.py 15 5 0.5  # 15s/board, 5 trials, α=0.5
.venv/bin/python -u bench_compare.py 5 15      # 同上，输出更易读

# 看 solver 残局（诊断用）
.venv/bin/python analyze_residual.py
```

## 6. 文件清单

主要改动：`solver.py`（约 +80 行）

辅助脚本（这次新建）：
- `bench_boards.py` — 用提供的 3 块真实数据跑 N 个 trial × M 秒，输出每块的 best/avg/all
- `bench_compare.py` — bench_boards 的整理版，输出更结构化
- `analyze_residual.py` — 跑完一遍后打印每块的残局 grid 和数字分布，用于诊断
