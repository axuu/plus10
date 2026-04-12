"""自动标定：检测白色圆形格子定位网格位置。

用法:
    python auto_calibrate.py <截图路径>

自动检测截图中的白色圆形格子，聚类成网格，
计算格子尺寸和起始位置，保存可视化并更新 config.yaml。
"""
import sys
import os
import cv2
import yaml
import numpy as np


def detect_grid(screenshot):
    """检测白色圆形格子，返回所有格子中心点坐标。"""
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # 白色圆形格子：高亮度区域
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # 形态学操作：去噪 + 分离相邻格子
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = screenshot.shape[:2]
    # 预估格子面积范围（假设 10 列，格子占屏幕宽度的 ~80%）
    estimated_cell_size = w * 0.08
    min_area = (estimated_cell_size * 0.3) ** 2
    max_area = (estimated_cell_size * 1.5) ** 2

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # 圆度检查
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.5:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers


def cluster_grid(centers, cols, rows):
    """将检测到的中心点聚类成规则网格，返回 (origin_x, origin_y, cell_w, cell_h)。"""
    if len(centers) < cols * 2:
        print(f"  检测到的点太少 ({len(centers)}个)，无法构建网格")
        return None

    centers = sorted(centers, key=lambda p: (p[1], p[0]))

    # 提取所有 x 和 y 坐标
    xs = sorted(set(c[0] for c in centers))
    ys = sorted(set(c[1] for c in centers))

    # 聚类 x 坐标为 cols 组
    x_clusters = cluster_1d(sorted(c[0] for c in centers), cols)
    y_clusters = cluster_1d(sorted(c[1] for c in centers), rows)

    if x_clusters is None or y_clusters is None:
        return None

    # 计算间距
    x_clusters.sort()
    y_clusters.sort()

    if len(x_clusters) < 2 or len(y_clusters) < 2:
        return None

    # 用相邻中心点间距算格子大小
    x_diffs = [x_clusters[i+1] - x_clusters[i] for i in range(len(x_clusters)-1)]
    y_diffs = [y_clusters[i+1] - y_clusters[i] for i in range(len(y_clusters)-1)]
    cell_w = np.median(x_diffs)
    cell_h = np.median(y_diffs)

    # origin = 第一个中心点 - 半个格子
    origin_x = x_clusters[0] - cell_w / 2
    origin_y = y_clusters[0] - cell_h / 2

    return origin_x, origin_y, cell_w, cell_h, x_clusters, y_clusters


def cluster_1d(values, expected_n):
    """将一维坐标聚类成 expected_n 组，返回每组的中心值。"""
    if len(values) == 0:
        return None

    # 估计间距
    values = sorted(values)
    total_range = values[-1] - values[0]
    if total_range == 0:
        return None

    estimated_gap = total_range / max(expected_n - 1, 1)
    merge_threshold = estimated_gap * 0.4

    # 合并相近的值
    groups = []
    current_group = [values[0]]

    for v in values[1:]:
        if v - current_group[-1] <= merge_threshold:
            current_group.append(v)
        else:
            groups.append(np.mean(current_group))
            current_group = [v]
    groups.append(np.mean(current_group))

    print(f"  聚类: {len(values)} 个点 -> {len(groups)} 组 (期望 {expected_n})")
    return groups


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

    # 读取 config 获取行列数
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cols = config["grid"]["cols"]
    rows = config["grid"]["rows"]

    # 检测格子中心点
    print("\n检测白色圆形格子...")
    centers = detect_grid(screenshot)
    print(f"检测到 {len(centers)} 个候选格子")

    if len(centers) == 0:
        print("未检测到格子，请检查截图")
        return

    # 聚类成网格
    print("\n聚类成网格...")
    result = cluster_grid(centers, cols, rows)

    if result is None:
        print("聚类失败")
        return

    origin_x, origin_y, cell_w, cell_h, x_clusters, y_clusters = result

    print(f"\n检测结果:")
    print(f"  网格起点: ({origin_x:.1f}, {origin_y:.1f})")
    print(f"  格子尺寸: {cell_w:.1f} x {cell_h:.1f}")
    print(f"  列中心: {len(x_clusters)} 列")
    print(f"  行中心: {len(y_clusters)} 行")

    # 转为比例值
    ox_ratio = origin_x / w
    oy_ratio = origin_y / h
    cw_ratio = cell_w / w
    ch_ratio = cell_h / h

    print(f"\n比例值:")
    print(f"  origin_x: {ox_ratio:.6f}")
    print(f"  origin_y: {oy_ratio:.6f}")
    print(f"  cell_width: {cw_ratio:.6f}")
    print(f"  cell_height: {ch_ratio:.6f}")

    # 保存可视化
    vis = screenshot.copy()

    # 画检测到的中心点
    for cx, cy in centers:
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

    # 画聚类后的网格
    for r in range(len(y_clusters)):
        for c in range(len(x_clusters)):
            x = int(x_clusters[c])
            y = int(y_clusters[r])
            cv2.circle(vis, (x, y), 5, (255, 0, 0), 2)

    # 画网格线
    grid_left = int(origin_x)
    grid_top = int(origin_y)
    grid_right = int(origin_x + cols * cell_w)
    grid_bottom = int(origin_y + rows * cell_h)

    for r in range(rows + 1):
        y = int(origin_y + r * cell_h)
        cv2.line(vis, (grid_left, y), (grid_right, y), (0, 255, 0), 1)
    for c in range(cols + 1):
        x = int(origin_x + c * cell_w)
        cv2.line(vis, (x, grid_top), (x, grid_bottom), (0, 255, 0), 1)

    out_path = "calibrate_result.png"
    cv2.imwrite(out_path, vis)
    print(f"\n可视化已保存到: {out_path}")

    # 更新 config.yaml
    config["grid"]["origin_x"] = round(ox_ratio, 6)
    config["grid"]["origin_y"] = round(oy_ratio, 6)
    config["grid"]["cell_width"] = round(cw_ratio, 6)
    config["grid"]["cell_height"] = round(ch_ratio, 6)

    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    print("config.yaml 已更新")


if __name__ == "__main__":
    main()
