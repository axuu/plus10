# main.py
import time
import yaml
import keyboard
import numpy as np

from capture import set_dpi_aware, find_game_window, capture_window
from recognizer import GridRecognizer
from solver import solve
from executor import Executor


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    set_dpi_aware()
    config = load_config()

    # 查找窗口
    title = config["window_title"]
    print(f"正在查找窗口: {title}")
    hwnd = find_game_window(title)
    print(f"找到窗口: hwnd={hwnd}")

    # 初始化识别器
    recognizer = GridRecognizer(
        template_dir="templates",
        confidence_threshold=config["recognition"]["confidence_threshold"],
        empty_variance_threshold=config["recognition"]["empty_variance_threshold"],
    )

    # 初始化执行器
    grid_cfg = config["grid"]
    executor = Executor(
        hwnd=hwnd,
        grid_origin_x=grid_cfg["origin_x"],
        grid_origin_y=grid_cfg["origin_y"],
        cell_width=grid_cfg["cell_width"],
        cell_height=grid_cfg["cell_height"],
        inward_shrink=config["executor"]["inward_shrink"],
        animation_delay=config["executor"]["animation_delay"],
    )

    search_depth = config["solver"]["search_depth"]
    beam_width = config["solver"].get("beam_width", 15)
    paused = False
    total_score = 0

    # 注册快捷键
    hotkeys = config["hotkeys"]
    print(f"按 {hotkeys['pause']} 暂停/恢复，按 {hotkeys['quit']} 退出")

    running = True

    def on_quit():
        nonlocal running
        running = False

    def on_pause():
        nonlocal paused
        paused = not paused
        print("已暂停" if paused else "已恢复")

    keyboard.add_hotkey(hotkeys["quit"], on_quit)
    keyboard.add_hotkey(hotkeys["pause"], on_pause)

    try:
        while running:
            if paused:
                time.sleep(0.1)
                continue

            # 截图
            screenshot = capture_window(hwnd)

            # 识别矩阵
            grid = recognizer.extract_grid(
                screenshot,
                origin_x=grid_cfg["origin_x"],
                origin_y=grid_cfg["origin_y"],
                cell_width=grid_cfg["cell_width"],
                cell_height=grid_cfg["cell_height"],
                cols=grid_cfg["cols"],
                rows=grid_cfg["rows"],
            )

            # 打印当前矩阵
            nonzero = np.count_nonzero(grid)
            print(f"\n识别到 {nonzero} 个数字，当前总分: {total_score}")

            if nonzero == 0:
                print("网格为空，等待新一轮...")
                time.sleep(2.0)
                continue

            # 求解
            move = solve(grid, depth=search_depth, beam_width=beam_width)

            if move is None:
                print("没有可消除的矩形，等待...")
                time.sleep(2.0)
                continue

            r1, c1, r2, c2 = move
            eliminated = int(np.count_nonzero(grid[r1:r2 + 1, c1:c2 + 1]))
            total_score += eliminated
            print(f"消除: ({r1},{c1})->({r2},{c2}), 本次消 {eliminated} 个, 总分 {total_score}")

            # 执行鼠标操作
            executor.execute_move(r1, c1, r2, c2)

    except KeyboardInterrupt:
        print("\n手动中止")
    finally:
        keyboard.unhook_all()
        print(f"最终得分: {total_score}")


if __name__ == "__main__":
    main()
