# main.py
import time
import traceback
import yaml
import keyboard
import numpy as np

import win32gui
from capture import set_dpi_aware, find_game_window, capture_window
from recognizer import GridRecognizer
from solver import solve
from executor import Executor

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("miemie.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("main")


def load_config(path: str = "config.yaml") -> dict:
    log.info(f"加载配置: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    log.debug(f"配置内容: {config}")
    return config


def main():
    try:
        log.info("========== 程序启动 ==========")

        set_dpi_aware()
        log.info("DPI 设置完成")

        config = load_config()

        # 查找窗口
        title = config["window_title"]
        log.info(f"正在查找窗口: {title}")
        hwnd = find_game_window(title)
        log.info(f"找到窗口: hwnd={hwnd}")

        # 初始化识别器
        log.info("初始化识别器...")
        recognizer = GridRecognizer(
            template_dir="templates",
            confidence_threshold=config["recognition"]["confidence_threshold"],
            dark_threshold=config["recognition"].get("dark_threshold", 80),
        )
        log.info(f"识别器初始化完成，加载了 {len(recognizer.templates_raw)} 个数字模板")

        # 初始化执行器
        grid_cfg = config["grid"]
        log.info(f"初始化执行器... grid_cfg={grid_cfg}")
        executor = Executor(
            hwnd=hwnd,
            grid_origin_x=grid_cfg["origin_x"],
            grid_origin_y=grid_cfg["origin_y"],
            cell_width=grid_cfg["cell_width"],
            cell_height=grid_cfg["cell_height"],
            inward_shrink=config["executor"]["inward_shrink"],
            animation_delay=config["executor"]["animation_delay"],
        )
        log.info("执行器初始化完成")

        search_depth = config["solver"]["search_depth"]
        beam_width = config["solver"].get("beam_width", 15)
        time_budget = config["solver"].get("time_budget", 10.0)
        n_simulations = config["solver"].get("n_simulations", 200)
        log.info(f"搜索参数: depth={search_depth}, beam_width={beam_width}, "
                 f"time_budget={time_budget}s, n_simulations={n_simulations}")

        total_score = 0

        # 注册退出快捷键
        hotkeys = config["hotkeys"]
        running = True

        def on_quit():
            nonlocal running
            running = False
            log.info("收到退出快捷键")

        keyboard.add_hotkey(hotkeys["quit"], on_quit)
        log.info(f"快捷键: 退出={hotkeys['quit']}")

        loop_count = 0
        while running:
            loop_count += 1
            log.info(f"\n===== 轮次 #{loop_count} =====")

            # --- 截图 ---
            print(f"\n[轮次 #{loop_count}] 请确保游戏窗口可见，按 Enter 截图...")
            input()
            if not running:
                break

            log.debug("截图中...")
            t0 = time.perf_counter()
            screenshot = capture_window(hwnd)
            t1 = time.perf_counter()
            log.info(f"截图完成: {screenshot.shape}, 耗时 {(t1 - t0) * 1000:.1f}ms")

            # 保存 debug 信息
            if loop_count == 1:
                import cv2
                import os
                os.makedirs("debug", exist_ok=True)
                cv2.imwrite("debug/screenshot.png", screenshot)
                log.info("debug 截图已保存")

            # --- 识别 ---
            log.debug("识别矩阵中...")
            t0 = time.perf_counter()
            grid = recognizer.extract_grid(
                screenshot,
                origin_x=grid_cfg["origin_x"],
                origin_y=grid_cfg["origin_y"],
                cell_width=grid_cfg["cell_width"],
                cell_height=grid_cfg["cell_height"],
                cols=grid_cfg["cols"],
                rows=grid_cfg["rows"],
            )
            t1 = time.perf_counter()
            log.info(f"识别完成，耗时 {(t1 - t0) * 1000:.1f}ms")

            # 打印矩阵
            nonzero = np.count_nonzero(grid)
            print(f"\n识别到 {nonzero} 个数字:")
            for r in range(grid_cfg["rows"]):
                row_str = " ".join(
                    f"{grid[r][c]:d}" if grid[r][c] > 0 else "."
                    for c in range(grid_cfg["cols"])
                )
                print(f"  行{r:2d}: {row_str}")

            if nonzero == 0:
                print("网格为空，等待新一轮...")
                time.sleep(2.0)
                continue

            # 保存 debug grid
            if loop_count == 1:
                np.savetxt("debug/grid.txt", grid, fmt="%d", delimiter=" ")

            # --- 规划（充分搜索）---
            print(f"\n规划中 (depth={search_depth}, beam={beam_width}, "
                  f"MC={n_simulations}, 时间上限={time_budget}s)...")
            t0 = time.perf_counter()
            moves = solve(
                grid,
                depth=search_depth,
                beam_width=beam_width,
                n_simulations=n_simulations,
                time_budget=time_budget,
            )
            t1 = time.perf_counter()

            if not moves:
                print("没有可消除的矩形。")
                continue

            # 预计消除数
            g_preview = grid.copy()
            expected_score = 0
            for r1, c1, r2, c2 in moves:
                expected_score += int(np.count_nonzero(g_preview[r1:r2 + 1, c1:c2 + 1]))
                g_preview[r1:r2 + 1, c1:c2 + 1] = 0
            remaining = int(np.count_nonzero(g_preview))

            print(f"\n规划完成: {len(moves)} 步, 预计消除 {expected_score} 格, "
                  f"剩余 {remaining} 格, 耗时 {(t1 - t0) * 1000:.0f}ms")
            for i, (r1, c1, r2, c2) in enumerate(moves):
                print(f"  步骤 {i + 1}: ({r1},{c1})->({r2},{c2})")

            # --- 等待确认 ---
            print(f"\n按 Enter 开始执行（{hotkeys['quit']} 退出）...")
            input()
            if not running:
                break

            # --- 自动聚焦窗口 + 执行 ---
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception as e:
                log.warning(f"聚焦窗口失败: {e}")
            time.sleep(0.3)  # 等窗口切换完成

            for i, (r1, c1, r2, c2) in enumerate(moves):
                if not running:
                    break
                eliminated = int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))
                total_score += eliminated
                grid[r1:r2 + 1, c1:c2 + 1] = 0
                log.info(f"步骤 {i + 1}/{len(moves)}: ({r1},{c1})->({r2},{c2}), "
                         f"消 {eliminated} 个, 总分 {total_score}")
                executor.execute_move(r1, c1, r2, c2)

            print(f"\n本轮完成，总分: {total_score}")

    except Exception as e:
        log.error(f"程序异常: {e}")
        log.error(traceback.format_exc())
    finally:
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        log.info(f"程序结束，最终得分: {total_score}")


if __name__ == "__main__":
    main()
