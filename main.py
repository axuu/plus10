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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str = "config.yaml"):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def auto_calibrate_grid(screenshot: np.ndarray, config: dict) -> bool:
    """用截图自动标定网格位置，更新 config 并保存。返回是否成功。"""
    from auto_calibrate import auto_calibrate_grid as _calibrate
    ok = _calibrate(screenshot, config, save_func=save_config)
    if ok:
        log.info(f"标定完成: origin=({config['grid']['origin_x']}, {config['grid']['origin_y']}), "
                 f"cell=({config['grid']['cell_width']} x {config['grid']['cell_height']})")
    else:
        log.warning("自动标定失败")
    return ok


def main():
    try:
        log.info("========== 程序启动 ==========")

        set_dpi_aware()
        config = load_config()

        title = config["window_title"]
        hwnd = find_game_window(title)
        log.info(f"找到窗口: hwnd={hwnd}")

        # 初始化识别器
        recognizer = GridRecognizer(
            template_dir="templates",
            confidence_threshold=config["recognition"]["confidence_threshold"],
            dark_threshold=config["recognition"].get("dark_threshold", 80),
        )
        log.info(f"识别器: {len(recognizer.templates_raw)} 个数字模板")

        # 自动标定：截图并检测网格位置
        log.info("自动标定网格位置...")
        screenshot = capture_window(hwnd)
        if auto_calibrate_grid(screenshot, config):
            config = load_config()  # 重新加载更新后的配置
        else:
            log.info("使用现有配置")

        grid_cfg = config["grid"]

        # 初始化执行器
        executor = Executor(
            hwnd=hwnd,
            grid_origin_x=grid_cfg["origin_x"],
            grid_origin_y=grid_cfg["origin_y"],
            cell_width=grid_cfg["cell_width"],
            cell_height=grid_cfg["cell_height"],
            inward_shrink=config["executor"]["inward_shrink"],
            animation_delay=config["executor"]["animation_delay"],
        )

        paused = False
        total_score = 0
        running = True

        hotkeys = config["hotkeys"]

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
        log.info(f"快捷键: 暂停={hotkeys['pause']}, 退出={hotkeys['quit']}")

        loop_count = 0
        while running:
            loop_count += 1

            if paused:
                time.sleep(0.1)
                continue

            # 如果窗口不在前台，自动激活
            fg = win32gui.GetForegroundWindow()
            if fg != hwnd:
                try:
                    win32gui.SetForegroundWindow(hwnd)
                    time.sleep(0.3)
                except Exception:
                    time.sleep(0.5)
                    continue

            log.info(f"===== 循环 #{loop_count} =====")

            # 截图 + 识别
            screenshot = capture_window(hwnd)
            grid = recognizer.extract_grid(
                screenshot,
                origin_x=grid_cfg["origin_x"],
                origin_y=grid_cfg["origin_y"],
                cell_width=grid_cfg["cell_width"],
                cell_height=grid_cfg["cell_height"],
                cols=grid_cfg["cols"],
                rows=grid_cfg["rows"],
            )

            nonzero = np.count_nonzero(grid)
            log.info(f"识别到 {nonzero} 个数字")

            if nonzero == 0:
                log.info("网格为空，等待新一轮...")
                time.sleep(2.0)
                continue

            # 求解：蒙特卡洛规划完整序列
            log.info("规划中...")
            t0 = time.perf_counter()
            moves = solve(grid)
            t1 = time.perf_counter()

            # 计算预计得分
            temp_grid = grid.copy()
            predicted_score = 0
            for r1, c1, r2, c2 in moves:
                predicted_score += int(np.count_nonzero(temp_grid[r1:r2+1, c1:c2+1]))
                temp_grid[r1:r2+1, c1:c2+1] = 0

            log.info(f"规划完成: {len(moves)} 步, 预计消除 {predicted_score}/{nonzero} 个格子, 耗时 {t1-t0:.1f}s")

            if not moves:
                log.info("没有可消除的矩形，等待...")
                time.sleep(2.0)
                continue

            print(f"预计消除 {predicted_score}/{nonzero} 格, {len(moves)} 步, 耗时 {t1-t0:.1f}s")

            # 执行完整序列
            for i, (r1, c1, r2, c2) in enumerate(moves):
                if paused or not running:
                    break
                eliminated = int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))
                total_score += eliminated
                grid[r1:r2 + 1, c1:c2 + 1] = 0
                log.info(f"[{i+1}/{len(moves)}] ({r1},{c1})->({r2},{c2}) "
                         f"消{eliminated} 总分{total_score}")
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
