"""识别精准度测试工具。

用法:
    # 模式1: 用游戏截图测试，显示识别结果网格
    python test_recognition.py screenshot <截图路径>

    # 模式2: 用标注好的单元格图片测试精准度
    #   test_data/ 目录结构:
    #     test_data/1/xxx.png  (标签为数字1的格子图片)
    #     test_data/2/xxx.png
    #     ...
    python test_recognition.py accuracy <test_data目录>

    # 模式3: 从截图中切出所有格子，保存到目录供人工标注
    python test_recognition.py extract <截图路径> <输出目录>
"""
import sys
import os
import cv2
import yaml
import numpy as np
from recognizer import GridRecognizer


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_screenshot(screenshot_path: str):
    """从截图中识别网格，打印结果并保存可视化图片。"""
    config = load_config()
    grid_cfg = config["grid"]
    recog_cfg = config["recognition"]

    recognizer = GridRecognizer(
        template_dir="templates",
        confidence_threshold=recog_cfg["confidence_threshold"],
        dark_threshold=recog_cfg["dark_threshold"],
    )

    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"无法读取截图: {screenshot_path}")
        return

    print(f"截图尺寸: {screenshot.shape[1]}x{screenshot.shape[0]}")

    grid = recognizer.extract_grid(
        screenshot,
        origin_x=grid_cfg["origin_x"],
        origin_y=grid_cfg["origin_y"],
        cell_width=grid_cfg["cell_width"],
        cell_height=grid_cfg["cell_height"],
        cols=grid_cfg["cols"],
        rows=grid_cfg["rows"],
    )

    # 打印网格
    print(f"\n识别结果 ({grid_cfg['rows']}x{grid_cfg['cols']}):")
    print("-" * (grid_cfg["cols"] * 3 + 1))
    non_zero = 0
    for r in range(grid_cfg["rows"]):
        row_str = "|"
        for c in range(grid_cfg["cols"]):
            v = grid[r][c]
            if v == 0:
                row_str += " . "
            else:
                row_str += f" {v} "
                non_zero += 1
        row_str += "|"
        print(row_str)
    print("-" * (grid_cfg["cols"] * 3 + 1))
    total = grid_cfg["rows"] * grid_cfg["cols"]
    print(f"非空格子: {non_zero}/{total}")

    # 保存可视化 (浮点运算避免累积偏移)
    h, w = screenshot.shape[:2]
    ox = grid_cfg["origin_x"] * w if grid_cfg["origin_x"] <= 1.0 else float(grid_cfg["origin_x"])
    oy = grid_cfg["origin_y"] * h if grid_cfg["origin_y"] <= 1.0 else float(grid_cfg["origin_y"])
    cw = grid_cfg["cell_width"] * w if grid_cfg["cell_width"] <= 1.0 else float(grid_cfg["cell_width"])
    ch = grid_cfg["cell_height"] * h if grid_cfg["cell_height"] <= 1.0 else float(grid_cfg["cell_height"])

    vis = screenshot.copy()
    for r in range(grid_cfg["rows"]):
        for c in range(grid_cfg["cols"]):
            x1 = int(round(ox + c * cw))
            y1 = int(round(oy + r * ch))
            x2 = int(round(ox + (c + 1) * cw))
            y2 = int(round(oy + (r + 1) * ch))

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

            v = grid[r][c]
            if v > 0:
                tx = x1 + (x2 - x1) // 2 - 8
                ty = y1 + (y2 - y1) // 2 + 8
                cv2.putText(vis, str(v), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out_path = "test_result.png"
    cv2.imwrite(out_path, vis)
    print(f"\n可视化已保存到: {out_path}")


def diagnose(screenshot_path: str):
    """诊断识别问题：显示模板加载情况和前几个格子的详细匹配信息。"""
    config = load_config()
    grid_cfg = config["grid"]
    recog_cfg = config["recognition"]

    print(f"=== 配置 ===")
    print(f"  confidence_threshold: {recog_cfg['confidence_threshold']}")
    print(f"  dark_threshold: {recog_cfg['dark_threshold']}")

    recognizer = GridRecognizer(
        template_dir="templates",
        confidence_threshold=recog_cfg["confidence_threshold"],
        dark_threshold=recog_cfg["dark_threshold"],
    )

    print(f"\n=== 模板 ===")
    print(f"  已加载: {list(recognizer.templates_raw.keys())}")
    for d, tpl in recognizer.templates_raw.items():
        print(f"  模板 {d}: shape={tpl.shape}")

    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"无法读取截图: {screenshot_path}")
        return

    h, w = screenshot.shape[:2]
    ox = int(grid_cfg["origin_x"] * w) if grid_cfg["origin_x"] <= 1.0 else int(grid_cfg["origin_x"])
    oy = int(grid_cfg["origin_y"] * h) if grid_cfg["origin_y"] <= 1.0 else int(grid_cfg["origin_y"])
    cw = int(grid_cfg["cell_width"] * w) if grid_cfg["cell_width"] <= 1.0 else int(grid_cfg["cell_width"])
    ch = int(grid_cfg["cell_height"] * h) if grid_cfg["cell_height"] <= 1.0 else int(grid_cfg["cell_height"])

    print(f"\n=== 网格参数 (像素) ===")
    print(f"  origin: ({ox}, {oy}), cell: {cw}x{ch}")

    # 诊断前 5 个格子
    from recognizer import _extract_dark_pixels
    print(f"\n=== 前5个格子诊断 ===")
    os.makedirs("debug", exist_ok=True)
    count = 0
    for r in range(grid_cfg["rows"]):
        for c in range(grid_cfg["cols"]):
            if count >= 5:
                break
            x = ox + c * cw
            y = oy + r * ch
            cell = screenshot[y:y+ch, x:x+cw]
            binary = _extract_dark_pixels(cell, recog_cfg["dark_threshold"])
            dark_ratio = np.count_nonzero(binary) / binary.size

            print(f"\n  cell({r},{c}): 原图shape={cell.shape}, dark_ratio={dark_ratio:.4f}")

            # 保存格子图片和二值化图片
            cv2.imwrite(f"debug/diag_cell_{r}_{c}.png", cell)
            cv2.imwrite(f"debug/diag_bin_{r}_{c}.png", binary)

            if dark_ratio < 0.02:
                print(f"    -> 判定为空 (dark_ratio < 0.02)")
            else:
                # 灰度匹配 + 中心裁剪
                ch_, cw_ = cell.shape[:2]
                crop = 0.2
                cx1 = int(cw_ * crop)
                cy1 = int(ch_ * crop)
                cx2 = cw_ - cx1
                cy2 = ch_ - cy1
                cell_center = cell[cy1:cy2, cx1:cx2]
                cell_gray = cv2.cvtColor(cell_center, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"debug/diag_center_{r}_{c}.png", cell_center)
                for digit, tpl_raw in recognizer.templates_raw.items():
                    tpl_resized = cv2.resize(tpl_raw, (cw_, ch_))
                    tpl_center = tpl_resized[cy1:cy2, cx1:cx2]
                    tpl_gray = cv2.cvtColor(tpl_center, cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(cell_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
                    score = result[0][0]
                    print(f"    vs 模板{digit}: score={score:.4f}")

            count += 1
        if count >= 5:
            break

    print(f"\n格子图片已保存到 debug/diag_cell_*.png 和 debug/diag_bin_*.png")


def test_accuracy(test_data_dir: str):
    """用标注好的单元格图片测试精准度。"""
    config = load_config()
    recog_cfg = config["recognition"]

    recognizer = GridRecognizer(
        template_dir="templates",
        confidence_threshold=recog_cfg["confidence_threshold"],
        dark_threshold=recog_cfg["dark_threshold"],
    )

    total = 0
    correct = 0
    confusion = {}  # (expected, predicted) -> count

    for digit_str in sorted(os.listdir(test_data_dir)):
        digit_dir = os.path.join(test_data_dir, digit_str)
        if not os.path.isdir(digit_dir):
            continue

        try:
            expected = int(digit_str)
        except ValueError:
            continue

        for fname in sorted(os.listdir(digit_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".bmp")):
                continue

            img_path = os.path.join(digit_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  跳过无法读取的文件: {img_path}")
                continue

            predicted = recognizer.recognize_cell(img)
            total += 1

            if predicted == expected:
                correct += 1
            else:
                key = (expected, predicted)
                confusion[key] = confusion.get(key, 0) + 1
                print(f"  错误: {fname} 期望={expected} 预测={predicted}")

    if total == 0:
        print("没有找到测试数据")
        return

    acc = correct / total * 100
    print(f"\n精准度: {correct}/{total} = {acc:.1f}%")

    if confusion:
        print("\n混淆矩阵 (仅错误项):")
        print(f"  {'期望':>4} -> {'预测':>4} : 次数")
        for (exp, pred), count in sorted(confusion.items()):
            print(f"  {exp:>4} -> {pred:>4} : {count}")


def extract_cells(screenshot_path: str, output_dir: str):
    """从截图中切出所有格子，按识别结果分目录保存，供人工校验和标注。"""
    config = load_config()
    grid_cfg = config["grid"]
    recog_cfg = config["recognition"]

    recognizer = GridRecognizer(
        template_dir="templates",
        confidence_threshold=recog_cfg["confidence_threshold"],
        dark_threshold=recog_cfg["dark_threshold"],
    )

    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"无法读取截图: {screenshot_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for r in range(grid_cfg["rows"]):
        for c in range(grid_cfg["cols"]):
            x = grid_cfg["origin_x"] + c * grid_cfg["cell_width"]
            y = grid_cfg["origin_y"] + r * grid_cfg["cell_height"]
            cell = screenshot[y:y + grid_cfg["cell_height"], x:x + grid_cfg["cell_width"]]

            if cell.shape[0] == 0 or cell.shape[1] == 0:
                continue

            predicted = recognizer.recognize_cell(cell, row=r, col=c)

            digit_dir = os.path.join(output_dir, str(predicted))
            os.makedirs(digit_dir, exist_ok=True)

            fname = f"r{r:02d}_c{c:02d}.png"
            cv2.imwrite(os.path.join(digit_dir, fname), cell)
            count += 1

    print(f"已保存 {count} 个格子到 {output_dir}/")
    for d in sorted(os.listdir(output_dir)):
        d_path = os.path.join(output_dir, d)
        if os.path.isdir(d_path):
            n = len([f for f in os.listdir(d_path) if f.endswith(".png")])
            print(f"  数字 {d}: {n} 个")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    mode = sys.argv[1]

    if mode == "screenshot" and len(sys.argv) >= 3:
        test_screenshot(sys.argv[2])
    elif mode == "diagnose" and len(sys.argv) >= 3:
        diagnose(sys.argv[2])
    elif mode == "accuracy" and len(sys.argv) >= 3:
        test_accuracy(sys.argv[2])
    elif mode == "extract" and len(sys.argv) >= 4:
        extract_cells(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
