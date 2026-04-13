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
    import cv2
    import os
    from auto_calibrate import detect_circles_in_region, find_corner, cluster_1d

    h, w = screenshot.shape[:2]
    cols = config["grid"]["cols"]
    rows = config["grid"]["rows"]

    # 尝试用边角模板定位搜索范围
    corner_path = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = os.path.join("templates", f"0{ext}")
        if os.path.exists(candidate):
            corner_path = candidate
            break

    if corner_path:
        corner_tpl = cv2.imread(corner_path)
        if corner_tpl is not None:
            th, tw = corner_tpl.shape[:2]
            gray_shot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            gray_tpl = cv2.cvtColor(corner_tpl, cv2.COLOR_BGR2GRAY)

            pos_tl, _ = find_corner(gray_shot, gray_tpl, "左上角")
            pos_br, _ = find_corner(gray_shot, cv2.flip(cv2.flip(gray_tpl, 1), 0), "右下角")

            search_x1 = pos_tl[0]
            search_y1 = pos_tl[1]
            search_x2 = pos_br[0] + tw
            search_y2 = pos_br[1] + th
        else:
            search_x1, search_y1, search_x2, search_y2 = 0, 0, w, h
    else:
        search_x1, search_y1, search_x2, search_y2 = 0, 0, w, h

    # 在范围内检测圆形格子
    centers, _ = detect_circles_in_region(screenshot, search_x1, search_y1, search_x2, search_y2)
    log.info(f"自动标定: 检测到 {len(centers)} 个格子")

    if len(centers) < 20:
        log.warning("检测到的格子太少，跳过标定")
        return False

    x_clusters = cluster_1d([c[0] for c in centers], cols)
    y_clusters = cluster_1d([c[1] for c in centers], rows)

    if x_clusters is None or y_clusters is None or len(x_clusters) < 2 or len(y_clusters) < 2:
        log.warning("聚类失败，跳过标定")
        return False

    x_clusters.sort()
    y_clusters.sort()

    x_diffs = [x_clusters[i+1] - x_clusters[i] for i in range(len(x_clusters)-1)]
    y_diffs = [y_clusters[i+1] - y_clusters[i] for i in range(len(y_clusters)-1)]
    cell_w = float(np.median(x_diffs))
    cell_h = float(np.median(y_diffs))

    origin_x = x_clusters[0] - cell_w / 2
    origin_y = y_clusters[0] - cell_h / 2

    config["grid"]["origin_x"] = float(round(origin_x / w, 6))
    config["grid"]["origin_y"] = float(round(origin_y / h, 6))
    config["grid"]["cell_width"] = float(round(cell_w / w, 6))
    config["grid"]["cell_height"] = float(round(cell_h / h, 6))

    save_config(config)
    log.info(f"标定完成: origin=({origin_x:.0f},{origin_y:.0f}), cell={cell_w:.1f}x{cell_h:.1f}")
    return True


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

            fg = win32gui.GetForegroundWindow()
            if fg != hwnd:
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
            log.info(f"规划完成: {len(moves)} 步, 耗时 {t1-t0:.1f}s")

            if not moves:
                log.info("没有可消除的矩形，等待...")
                time.sleep(2.0)
                continue

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
