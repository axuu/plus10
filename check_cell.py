"""检查指定格子的匹配分数。

用法:
    python check_cell.py <截图路径> <行> <列>

例如:
    python check_cell.py debug/screenshot.png 13 9
"""
import sys
import cv2
import yaml
from recognizer import GridRecognizer


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return

    screenshot_path = sys.argv[1]
    r = int(sys.argv[2])
    c = int(sys.argv[3])

    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"无法读取截图: {screenshot_path}")
        return

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    h, w = img.shape[:2]
    g = cfg["grid"]
    ox = g["origin_x"] * w if g["origin_x"] <= 1.0 else float(g["origin_x"])
    oy = g["origin_y"] * h if g["origin_y"] <= 1.0 else float(g["origin_y"])
    cw = g["cell_width"] * w if g["cell_width"] <= 1.0 else float(g["cell_width"])
    ch = g["cell_height"] * h if g["cell_height"] <= 1.0 else float(g["cell_height"])

    x1 = int(round(ox + c * cw))
    y1 = int(round(oy + r * ch))
    x2 = int(round(ox + (c + 1) * cw))
    y2 = int(round(oy + (r + 1) * ch))
    cell = img[y1:y2, x1:x2]

    print(f"cell({r},{c}): 像素=({x1},{y1})->({x2},{y2}), shape={cell.shape}")

    rec = GridRecognizer(
        "templates",
        confidence_threshold=0.0,
        dark_threshold=cfg["recognition"]["dark_threshold"],
    )

    # 中心裁剪
    crop = 0.2
    cy1 = int(cell.shape[0] * crop)
    cx1 = int(cell.shape[1] * crop)
    cy2 = cell.shape[0] - cy1
    cx2 = cell.shape[1] - cx1
    cell_center = cell[cy1:cy2, cx1:cx2]
    cell_gray = cv2.cvtColor(cell_center, cv2.COLOR_BGR2GRAY)

    print(f"中心裁剪: {cell_center.shape[1]}x{cell_center.shape[0]}")
    print()

    scores = {}
    for d, tpl in rec.templates_raw.items():
        t = cv2.resize(tpl, (cell.shape[1], cell.shape[0]))
        t_center = t[cy1:cy2, cx1:cx2]
        t_gray = cv2.cvtColor(t_center, cv2.COLOR_BGR2GRAY)
        score = cv2.matchTemplate(cell_gray, t_gray, cv2.TM_CCOEFF_NORMED)[0][0]
        scores[d] = score
        print(f"  模板{d}: {score:.4f}")

    best = max(scores, key=scores.get)
    print(f"\n最高: 模板{best} = {scores[best]:.4f}")

    # 保存格子图片
    cv2.imwrite(f"debug/check_cell_{r}_{c}.png", cell)
    cv2.imwrite(f"debug/check_center_{r}_{c}.png", cell_center)
    print(f"格子图片: debug/check_cell_{r}_{c}.png")


if __name__ == "__main__":
    main()
