"""Benchmark the current solver against the 3 real boards the user provided.

Usage:
    .venv/bin/python bench_boards.py [time_budget] [trials]
"""
import sys
import time
import numpy as np
import logging

sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.WARNING)

from solver import solve


BOARDS_RAW = [
    # board 1
    """5 3 9 6 1 8 6 5 2 3
       7 5 1 8 6 5 9 7 7 6
       7 6 1 5 2 7 5 1 1 8
       4 2 2 9 5 6 7 6 7 6
       3 5 4 7 9 8 7 5 7 4
       1 2 3 7 3 5 9 8 4 2
       9 5 6 6 4 7 5 8 7 5
       6 8 3 5 1 3 1 9 5 9
       4 4 5 4 8 3 8 9 2 9
       1 2 8 2 2 3 6 2 1 5
       6 5 9 2 1 4 1 7 3 9
       8 7 3 8 2 7 8 4 3 9
       8 5 9 5 6 6 3 5 3 8
       3 8 5 9 3 7 2 2 9 4
       1 1 6 4 3 4 4 2 6 8
       8 4 4 6 5 5 8 8 2 6""",
    # board 2
    """5 2 7 4 7 6 2 2 1 5
       2 4 4 8 6 8 4 5 5 6
       5 2 5 5 3 9 6 7 3 7
       9 3 2 3 8 6 9 1 3 1
       3 2 2 3 2 1 9 6 8 8
       4 9 6 4 3 6 3 9 7 7
       6 9 7 1 5 7 7 6 4 8
       6 4 4 5 7 5 6 1 8 7
       9 5 8 2 1 1 3 1 9 7
       1 7 4 5 8 3 6 3 4 1
       6 7 8 6 8 5 4 8 2 2
       8 1 6 4 1 7 8 5 9 8
       2 1 9 3 7 5 3 9 8 2
       6 9 9 2 7 3 5 1 7 9
       9 2 4 9 5 3 5 4 3 9
       2 2 4 9 8 9 5 6 5 7""",
    # board 3
    """7 3 8 2 7 4 4 1 4 8
       3 7 8 6 1 4 4 2 1 7
       5 9 9 9 4 7 3 5 6 5
       8 9 5 8 2 2 2 7 2 6
       6 9 1 2 3 1 1 1 9 2
       7 6 8 5 6 3 4 7 3 6
       9 2 6 5 1 7 4 9 8 5
       8 8 7 3 3 1 1 4 7 5
       2 5 1 7 3 3 8 2 4 3
       2 5 8 9 8 5 3 2 3 3
       4 7 5 1 4 7 1 7 2 7
       7 1 9 9 9 5 1 6 9 9
       3 5 5 1 7 9 9 9 6 5
       9 3 5 3 2 1 7 9 6 3
       7 6 4 6 7 9 7 3 2 6
       7 8 9 3 3 7 4 9 6 9""",
]


def parse_board(s: str) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in s.strip().splitlines()]
    return np.array(rows, dtype=np.int32)


def main():
    time_budget = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    boards = [parse_board(s) for s in BOARDS_RAW]
    print(f"== alpha={alpha}, time_budget={time_budget}s, trials={trials} ==")

    grand_total = 0
    grand_possible = 0
    per_board_scores = []
    for bi, grid in enumerate(boards):
        total_cells = int(np.count_nonzero(grid))
        best_seen = 0
        scores = []
        for t in range(trials):
            t0 = time.perf_counter()
            moves, score = solve(grid, time_budget=time_budget, alpha=alpha)
            dt = time.perf_counter() - t0
            scores.append(score)
            best_seen = max(best_seen, score)
            print(
                f"Board {bi + 1} trial {t + 1}: score={score}/{total_cells} "
                f"({score / total_cells:.1%}), moves={len(moves)}, {dt:.1f}s"
            )
        per_board_scores.append((best_seen, total_cells, scores))
        grand_total += best_seen
        grand_possible += total_cells
        avg = sum(scores) / len(scores)
        print(
            f"  Board {bi + 1}: best={best_seen}/{total_cells}, avg={avg:.1f}"
        )

    print("\n===== Summary =====")
    for bi, (best, total, scores) in enumerate(per_board_scores):
        print(
            f"Board {bi + 1}: best={best}/{total} ({best / total:.1%}), "
            f"all={scores}"
        )
    print(
        f"\nGrand total best: {grand_total}/{grand_possible} "
        f"({grand_total / grand_possible:.1%})"
    )


if __name__ == "__main__":
    main()
