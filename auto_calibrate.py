"""自动标定：用边框角模板定位网格位置。

用法:
    python auto_calibrate.py <截图路径>

需要 templates/0.jpg (或 .png) 作为左上角边框模板。
脚本会自动翻转模板匹配四个角，计算网格位置和格子尺寸，
保存可视化结果并更新 config.yaml。
"""
import sys
import cv2
import yaml
import numpy as np


def find_corner(screenshot, template, label="corner"):
    """用模板匹配找到角的位置，返回匹配位置 (x, y) 和分数。"""
    # 转灰度匹配
    gray_shot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_shot, gray_tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"  {label}: pos={max_loc}, score={max_val:.4f}")
    return max_loc, max_val


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    screenshot_path = sys.argv[1]
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"无法读取截图: {screenshot_path}")
        return

    h, w = screenshot.shape[:2]
    print(f"截图尺寸: {w}x{h}")

    # 加载边框角模板
    import os
    corner_path = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = os.path.join("templates", f"0{ext}")
        if os.path.exists(candidate):
            corner_path = candidate
            break

    if corner_path is None:
        print("未找到边框模板 templates/0.* (jpg/png/bmp)")
        return

    corner_tpl = cv2.imread(corner_path)
    if corner_tpl is None:
        print(f"无法读取边框模板: {corner_path}")
        return

    th, tw = corner_tpl.shape[:2]
    print(f"边框模板尺寸: {tw}x{th}")

    # 生成四个角的模板
    tpl_tl = corner_tpl                              # 左上角 (原图)
    tpl_tr = cv2.flip(corner_tpl, 1)                  # 右上角 (水平翻转)
    tpl_bl = cv2.flip(corner_tpl, 0)                  # 左下角 (垂直翻转)
    tpl_br = cv2.flip(corner_tpl, -1)                 # 右下角 (水平+垂直翻转)

    print("\n匹配四个角:")
    pos_tl, score_tl = find_corner(screenshot, tpl_tl, "左上角")
    pos_tr, score_tr = find_corner(screenshot, tpl_tr, "右上角")
    pos_bl, score_bl = find_corner(screenshot, tpl_bl, "左下角")
    pos_br, score_br = find_corner(screenshot, tpl_br, "右下角")

    # 网格区域：四个角模板的内侧边界
    # 左上角：模板右下角 = 网格起点
    grid_left = pos_tl[0] + tw
    grid_top = pos_tl[1] + th
    # 右下角：模板左上角 = 网格终点
    grid_right = pos_br[0]
    grid_bottom = pos_br[1]

    # 交叉验证
    grid_left2 = pos_bl[0] + tw
    grid_top2 = pos_tr[1] + th
    grid_right2 = pos_tr[0]
    grid_bottom2 = pos_bl[1]

    print(f"\n网格区域 (主):")
    print(f"  左上: ({grid_left}, {grid_top})")
    print(f"  右下: ({grid_right}, {grid_bottom})")
    print(f"网格区域 (交叉验证):")
    print(f"  左上: ({grid_left2}, {grid_top2})")
    print(f"  右下: ({grid_right2}, {grid_bottom2})")

    # 取平均
    grid_left = (grid_left + grid_left2) // 2
    grid_top = (grid_top + grid_top2) // 2
    grid_right = (grid_right + grid_right2) // 2
    grid_bottom = (grid_bottom + grid_bottom2) // 2

    print(f"网格区域 (平均):")
    print(f"  左上: ({grid_left}, {grid_top})")
    print(f"  右下: ({grid_right}, {grid_bottom})")

    # 读取 config 获取行列数
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cols = config["grid"]["cols"]
    rows = config["grid"]["rows"]

    grid_width = grid_right - grid_left
    grid_height = grid_bottom - grid_top
    cell_width = grid_width / cols
    cell_height = grid_height / rows

    print(f"\n网格: {grid_width}x{grid_height} 像素")
    print(f"格子: {cell_width:.1f}x{cell_height:.1f} 像素 ({cols}列 x {rows}行)")

    # 转为比例值
    origin_x_ratio = grid_left / w
    origin_y_ratio = grid_top / h
    cell_w_ratio = cell_width / w
    cell_h_ratio = cell_height / h

    print(f"\n比例值:")
    print(f"  origin_x: {origin_x_ratio:.6f}")
    print(f"  origin_y: {origin_y_ratio:.6f}")
    print(f"  cell_width: {cell_w_ratio:.6f}")
    print(f"  cell_height: {cell_h_ratio:.6f}")

    # 保存可视化
    vis = screenshot.copy()

    # 画四个角的匹配位置
    for pos, label, color in [
        (pos_tl, "TL", (0, 255, 0)),
        (pos_tr, "TR", (0, 255, 255)),
        (pos_bl, "BL", (255, 0, 0)),
        (pos_br, "BR", (0, 0, 255)),
    ]:
        cv2.rectangle(vis, pos, (pos[0] + tw, pos[1] + th), color, 2)

    # 画网格
    for r in range(rows + 1):
        y = int(grid_top + r * cell_height)
        cv2.line(vis, (grid_left, y), (grid_right, y), (0, 255, 0), 1)
    for c in range(cols + 1):
        x = int(grid_left + c * cell_width)
        cv2.line(vis, (x, grid_top), (x, grid_bottom), (0, 255, 0), 1)

    out_path = "calibrate_result.png"
    cv2.imwrite(out_path, vis)
    print(f"\n可视化已保存到: {out_path}")

    # 更新 config.yaml
    config["grid"]["origin_x"] = round(origin_x_ratio, 6)
    config["grid"]["origin_y"] = round(origin_y_ratio, 6)
    config["grid"]["cell_width"] = round(cell_w_ratio, 6)
    config["grid"]["cell_height"] = round(cell_h_ratio, 6)

    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print("config.yaml 已更新")


if __name__ == "__main__":
    main()
