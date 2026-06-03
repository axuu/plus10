"""Final comprehensive comparison: confirm new vs report deltas."""
import sys
import time
import numpy as np
import logging

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.WARNING)

from bench_boards import BOARDS_RAW, parse_board
from solver import solve


def run_config(label, boards, trials, time_budget, **solve_kw):
    print(f"\n===== {label} =====")
    per_board = []
    for bi, grid in enumerate(boards):
        total_cells = int(np.count_nonzero(grid))
        scores = []
        for t in range(trials):
            t0 = time.perf_counter()
            moves, score = solve(grid, time_budget=time_budget, **solve_kw)
            dt = time.perf_counter() - t0
            scores.append(score)
            print(
                f"  B{bi + 1} t{t + 1}: {score}/{total_cells} "
                f"({score / total_cells:.1%}) {dt:.1f}s"
            )
        best = max(scores)
        avg = sum(scores) / len(scores)
        per_board.append((best, avg, scores))
        print(f"  ==> Board {bi + 1}: best={best}, avg={avg:.1f}, all={scores}")
    grand = sum(b for b, _, _ in per_board)
    print(f"\n  >>> GRAND BEST: {grand}/{160 * len(boards)} "
          f"({grand / (160 * len(boards)):.1%})")
    return per_board, grand


def main():
    boards = [parse_board(s) for s in BOARDS_RAW]
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    time_budget = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    # default config
    print(f"\nConfig: trials={trials}, time_budget={time_budget}s/board")
    run_config("alpha=0.5 (current best)", boards, trials, time_budget, alpha=0.5)


if __name__ == "__main__":
    main()
