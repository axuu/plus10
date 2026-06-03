"""Run solver on each board and show residual structure to find weaknesses."""
import logging
import numpy as np

from bench_boards import BOARDS_RAW, parse_board
from solver import solve, find_valid_rectangles

logging.basicConfig(level=logging.WARNING)


def print_grid(g, title):
    print(title)
    for r in range(g.shape[0]):
        print("  " + " ".join(f"{x:d}" if x else "." for x in g[r]))


def main():
    for bi, s in enumerate(BOARDS_RAW):
        grid = parse_board(s)
        moves, score = solve(grid.copy(), time_budget=15.0)
        # replay to get residual
        g = grid.copy()
        for r1, c1, r2, c2 in moves:
            g[r1:r2 + 1, c1:c2 + 1] = 0
        nz = int(np.count_nonzero(g))
        rsum = int(g.sum())
        # 分数统计
        from collections import Counter
        digit_counts = Counter(int(x) for x in g.ravel() if x > 0)
        # 剩余可消的矩形
        leftover_rects = find_valid_rectangles(g)
        print(f"\n===== Board {bi + 1}: score={score}, residual={nz}, sum_left={rsum} =====")
        print(f"  剩余数字分布: {dict(sorted(digit_counts.items()))}")
        print(f"  剩余可消 sum=10 矩形 (cnt>=2): {len(leftover_rects)}")
        if leftover_rects:
            print(f"  例: {leftover_rects[:5]}")
        print_grid(g, f"  Residual board {bi + 1}:")


if __name__ == "__main__":
    main()
