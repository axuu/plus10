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
        log.info(f"搜索参数: depth={search_depth}, beam_width={beam_width}")

        paused = False
        total_score = 0

        # 注册快捷键
        hotkeys = config["hotkeys"]
        log.info(f"快捷键: 暂停={hotkeys['pause']}, 退出={hotkeys['quit']}")

        running = True

        def on_quit():
            nonlocal running
            running = False
            log.info("收到退出快捷键")

        def on_pause():
            nonlocal paused
            paused = not paused
            log.info(f"{'已暂停' if paused else '已恢复'}")

        keyboard.add_hotkey(hotkeys["quit"], on_quit)
        keyboard.add_hotkey(hotkeys["pause"], on_pause)
        log.info("快捷键注册完成，开始主循环")

        loop_count = 0
        while running:
            loop_count += 1

            if paused:
                time.sleep(0.1)
                continue

            # 检查窗口是否在前台
            fg = win32gui.GetForegroundWindow()
            if fg != hwnd:
                log.debug(f"循环 #{loop_count}: 窗口不在前台 (fg={fg}, hwnd={hwnd})，等待...")
                time.sleep(0.5)
                continue

            log.info(f"===== 循环 #{loop_count} =====")

            # 截图
            log.debug("截图中...")
            t0 = time.perf_counter()
            screenshot = capture_window(hwnd)
            t1 = time.perf_counter()
            log.info(f"截图完成: {screenshot.shape}, 耗时 {(t1-t0)*1000:.1f}ms")

            # 识别矩阵
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
            log.info(f"识别完成，耗时 {(t1-t0)*1000:.1f}ms")

            # 保存 debug 信息（仅第一次循环）
            if loop_count == 1:
                import cv2
                import os
                os.makedirs("debug", exist_ok=True)
                cv2.imwrite("debug/screenshot.png", screenshot)
                np.savetxt("debug/grid.txt", grid, fmt="%d", delimiter=" ")
                log.info("debug 信息已保存到 debug/ 目录")

            # 打印矩阵
            nonzero = np.count_nonzero(grid)
            log.info(f"识别到 {nonzero} 个数字，当前总分: {total_score}")
            for r in range(grid_cfg["rows"]):
                row_str = " ".join(str(grid[r][c]) if grid[r][c] > 0 else "." for c in range(grid_cfg["cols"]))
                log.debug(f"  行{r:2d}: {row_str}")

            if nonzero == 0:
                log.info("网格为空，等待新一轮...")
                time.sleep(2.0)
                continue

            # 求解：返回完整操作序列
            log.debug(f"求解中... depth={search_depth}, beam={beam_width}")
            t0 = time.perf_counter()
            moves = solve(grid, depth=search_depth, beam_width=beam_width)
            t1 = time.perf_counter()
            log.info(f"求解完成，耗时 {(t1-t0)*1000:.1f}ms, 步数={len(moves)}")

            if not moves:
                log.info("没有可消除的矩形，等待...")
                time.sleep(2.0)
                continue

            # 执行完整序列（省去中间截图识别的时间）
            for i, (r1, c1, r2, c2) in enumerate(moves):
                if paused or not running:
                    break
                eliminated = int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))
                total_score += eliminated
                grid[r1:r2 + 1, c1:c2 + 1] = 0  # 更新本地网格
                log.info(f"步骤 {i+1}/{len(moves)}: ({r1},{c1})->({r2},{c2}), "
                         f"消 {eliminated} 个, 总分 {total_score}")
                executor.execute_move(r1, c1, r2, c2)

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
