"""标定工具：截取游戏窗口，手动标定网格区域，采集数字模板。

用法:
    python calibrate.py

流程:
    1. 截取游戏窗口
    2. 弹出窗口让用户用鼠标点击网格的左上角和右下角
    3. 根据点击计算格子尺寸
    4. 展示切分结果，让用户逐个确认数字模板
    5. 保存模板到 templates/ 目录
    6. 更新 config.yaml 中的网格参数
"""
import os
import cv2
import yaml
import numpy as np

from capture import set_dpi_aware, find_game_window, capture_window


clicks = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"点击: ({x}, {y}), 已记录 {len(clicks)} 个点")


def main():
    set_dpi_aware()

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    title = config["window_title"]
    hwnd = find_game_window(title)
    screenshot = capture_window(hwnd)

    print("请在弹出的窗口中依次点击:")
    print("  1. 网格左上角第一个格子的左上角")
    print("  2. 网格右下角最后一个格子的右下角")

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibrate", mouse_callback)
    cv2.imshow("Calibrate", screenshot)

    while len(clicks) < 2:
        cv2.waitKey(100)

    cv2.destroyAllWindows()

    x1, y1 = clicks[0]
    x2, y2 = clicks[1]
    cols = config["grid"]["cols"]
    rows = config["grid"]["rows"]

    cell_w = (x2 - x1) // cols
    cell_h = (y2 - y1) // rows

    print(f"网格区域: ({x1},{y1}) -> ({x2},{y2})")
    print(f"格子尺寸: {cell_w} x {cell_h}")

    # 更新 config
    config["grid"]["origin_x"] = x1
    config["grid"]["origin_y"] = y1
    config["grid"]["cell_width"] = cell_w
    config["grid"]["cell_height"] = cell_h

    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print("config.yaml 已更新")

    # 采集模板：让用户为每个数字点击一个样本格子
    os.makedirs("templates", exist_ok=True)
    print("\n现在采集数字模板。")
    print("将展示网格，请为每个数字(1-9)点击一个包含该数字的格子。")

    for digit in range(1, 10):
        clicks.clear()
        print(f"\n请点击一个包含数字 {digit} 的格子:")

        cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Calibrate", mouse_callback)

        # 画网格线辅助
        display = screenshot.copy()
        for r in range(rows + 1):
            y = y1 + r * cell_h
            cv2.line(display, (x1, y), (x2, y), (0, 255, 0), 1)
        for c in range(cols + 1):
            x = x1 + c * cell_w
            cv2.line(display, (x, y1), (x, y2), (0, 255, 0), 1)

        cv2.imshow("Calibrate", display)

        while len(clicks) < 1:
            cv2.waitKey(100)

        cv2.destroyAllWindows()

        cx, cy = clicks[0]
        # 定位到格子
        col = (cx - x1) // cell_w
        row = (cy - y1) // cell_h
        cell_x = x1 + col * cell_w
        cell_y = y1 + row * cell_h
        cell_img = screenshot[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]

        template_path = os.path.join("templates", f"{digit}.png")
        cv2.imwrite(template_path, cell_img)
        print(f"模板 {digit} 已保存到 {template_path}")

    print("\n标定完成！可以运行 python main.py 开始自动消除。")


if __name__ == "__main__":
    main()
