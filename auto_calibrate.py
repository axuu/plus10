"""自动标定：边角模板定位范围 + 圆形检测精确定位。

用法:
    python auto_calibrate.py <截图路径>

需要 templates/0.jpg (或 .png) 作为左上角边框模板。
1. 用边角模板翻转匹配四个角，确定网格大致范围
2. 在范围内检测白色圆形格子，精确定位每个格子中心
3. 从中心点推算网格起点和格子尺寸
"""
import sys
import os
import cv2
import yaml
import numpy as np


def find_corner(screenshot_gray, template_gray, label="corner"):
    """用模板匹配找到角的位置。"""
    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"  {label}: pos={max_loc}, score={max_val:.4f}")
    return max_loc, max_val


def detect_circles_in_region(screenshot, x1, y1, x2, y2):
    """在指定区域内检测白色圆形格子，返回中心点坐标（相对于整张截图）。"""
    region = screenshot[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh, rw = region.shape[:2]
    estimated_cell = rw / 10  # 大约10列
    min_area = (estimated_cell * 0.3) ** 2
    max_area = (estimated_cell * 1.2) ** 2

    centers = []
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.5:
            continue

        # 用外接矩形中心而非重心，避免阴影导致偏移
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx = bx + bw // 2 + x1  # 转回整图坐标
        cy = by + bh // 2 + y1
        centers.append((cx, cy))
        areas.append(area)

    return centers, areas


def cluster_1d(values, expected_n):
    """将一维坐标聚类，返回每组中心值。"""
    if len(values) == 0:
        return None

    values = sorted(values)
    total_range = values[-1] - values[0]
    if total_range == 0:
        return None

    estimated_gap = total_range / max(expected_n - 1, 1)
    merge_threshold = estimated_gap * 0.4

    groups = []
    current_group = [values[0]]

    for v in values[1:]:
        if v - current_group[-1] <= merge_threshold:
            current_group.append(v)
        else:
            groups.append(float(np.mean(current_group)))
            current_group = [v]
    groups.append(float(np.mean(current_group)))

    return groups


def auto_calibrate_grid(screenshot: np.ndarray, config: dict, save_func=None) -> bool:
    """用截图自动标定网格位置，更新 config dict。返回是否成功。

    save_func: 可选回调，用于保存 config（如写入 yaml）。
    """
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

    if len(centers) < 20:
        return False

    x_clusters = cluster_1d([c[0] for c in centers], cols)
    y_clusters = cluster_1d([c[1] for c in centers], rows)

    if x_clusters is None or y_clusters is None or len(x_clusters) < 2 or len(y_clusters) < 2:
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

    if save_func:
        save_func(config)

    return True


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
    corner_path = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = os.path.join("templates", f"0{ext}")
        if os.path.exists(candidate):
            corner_path = candidate
            break

    if corner_path is None:
        print("未找到边框模板 templates/0.*")
        return

    corner_tpl = cv2.imread(corner_path)
    if corner_tpl is None:
        print(f"无法读取边框模板: {corner_path}")
        return

    th, tw = corner_tpl.shape[:2]
    print(f"边框模板尺寸: {tw}x{th}")

    gray_shot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(corner_tpl, cv2.COLOR_BGR2GRAY)

    # 匹配四个角
    print("\n第一步：匹配边角模板")
    pos_tl, _ = find_corner(gray_shot, gray_tpl, "左上角")
    pos_tr, _ = find_corner(gray_shot, cv2.flip(gray_tpl, 1), "右上角")
    pos_bl, _ = find_corner(gray_shot, cv2.flip(gray_tpl, 0), "左下角")
    pos_br, _ = find_corner(gray_shot, cv2.flip(gray_tpl, -1), "右下角")

    # 边角模板包含部分网格，所以搜索区域从模板位置开始（不加偏移）
    search_x1 = pos_tl[0]
    search_y1 = pos_tl[1]
    search_x2 = pos_br[0] + tw
    search_y2 = pos_br[1] + th

    print(f"\n搜索区域: ({search_x1},{search_y1}) -> ({search_x2},{search_y2})")

    # 在范围内检测圆形格子
    print("\n第二步：检测圆形格子")
    centers, areas = detect_circles_in_region(screenshot, search_x1, search_y1, search_x2, search_y2)
    print(f"检测到 {len(centers)} 个格子")

    if len(centers) < 20:
        print("检测到的格子太少，请检查截图")
        return

    # 读取 config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cols = config["grid"]["cols"]
    rows = config["grid"]["rows"]

    # 聚类
    print("\n第三步：聚类成网格")
    x_clusters = cluster_1d([c[0] for c in centers], cols)
    y_clusters = cluster_1d([c[1] for c in centers], rows)

    if x_clusters is None or y_clusters is None:
        print("聚类失败")
        return

    x_clusters.sort()
    y_clusters.sort()

    print(f"  列: {len(x_clusters)} 组 (期望 {cols})")
    print(f"  行: {len(y_clusters)} 组 (期望 {rows})")

    # 计算间距
    x_diffs = [x_clusters[i+1] - x_clusters[i] for i in range(len(x_clusters)-1)]
    y_diffs = [y_clusters[i+1] - y_clusters[i] for i in range(len(y_clusters)-1)]
    cell_w = float(np.median(x_diffs))
    cell_h = float(np.median(y_diffs))

    # origin = 第一个中心点 - 半个格子
    origin_x = x_clusters[0] - cell_w / 2
    origin_y = y_clusters[0] - cell_h / 2

    print(f"\n检测结果:")
    print(f"  网格起点: ({origin_x:.1f}, {origin_y:.1f})")
    print(f"  格子尺寸: {cell_w:.1f} x {cell_h:.1f}")
    print(f"  网格范围: ({origin_x:.0f},{origin_y:.0f}) -> ({origin_x + cols*cell_w:.0f},{origin_y + rows*cell_h:.0f})")

    # 比例值
    ox_ratio = origin_x / w
    oy_ratio = origin_y / h
    cw_ratio = cell_w / w
    ch_ratio = cell_h / h

    print(f"\n比例值:")
    print(f"  origin_x: {ox_ratio:.6f}")
    print(f"  origin_y: {oy_ratio:.6f}")
    print(f"  cell_width: {cw_ratio:.6f}")
    print(f"  cell_height: {ch_ratio:.6f}")

    # 可视化
    vis = screenshot.copy()

    # 红点：检测到的中心
    for cx, cy in centers:
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

    # 蓝圈：聚类后的网格中心
    for r in range(len(y_clusters)):
        for c in range(len(x_clusters)):
            cv2.circle(vis, (int(x_clusters[c]), int(y_clusters[r])), 5, (255, 0, 0), 2)

    # 绿线：网格边界
    for r in range(rows + 1):
        y = int(origin_y + r * cell_h)
        cv2.line(vis, (int(origin_x), y), (int(origin_x + cols * cell_w), y), (0, 255, 0), 1)
    for c in range(cols + 1):
        x = int(origin_x + c * cell_w)
        cv2.line(vis, (x, int(origin_y)), (x, int(origin_y + rows * cell_h)), (0, 255, 0), 1)

    # 紫框：边角模板匹配位置
    for pos in [pos_tl, pos_tr, pos_bl, pos_br]:
        cv2.rectangle(vis, pos, (pos[0] + tw, pos[1] + th), (255, 0, 255), 2)

    out_path = "calibrate_result.png"
    cv2.imwrite(out_path, vis)
    print(f"\n可视化已保存到: {out_path}")

    # 更新 config.yaml
    config["grid"]["origin_x"] = float(round(ox_ratio, 6))
    config["grid"]["origin_y"] = float(round(oy_ratio, 6))
    config["grid"]["cell_width"] = float(round(cw_ratio, 6))
    config["grid"]["cell_height"] = float(round(ch_ratio, 6))

    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print("config.yaml 已更新")


if __name__ == "__main__":
    main()
