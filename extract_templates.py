"""从截图中切出所有格子，方便人工挑选模板。

用法:
    python extract_templates.py <截图路径>

会把所有格子保存到 cells/ 目录，文件名为 r行_c列.png。
然后你从中挑选 9 个格子，复制到 templates/ 并重命名：
    copy cells\\r00_c00.png templates\\4.png
    copy cells\\r00_c02.png templates\\6.png
    ...
"""
import sys
import os
import cv2
import yaml


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

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    grid_cfg = config["grid"]
    cols = grid_cfg["cols"]
    rows = grid_cfg["rows"]

    ox = int(grid_cfg["origin_x"] * w) if grid_cfg["origin_x"] <= 1.0 else int(grid_cfg["origin_x"])
    oy = int(grid_cfg["origin_y"] * h) if grid_cfg["origin_y"] <= 1.0 else int(grid_cfg["origin_y"])
    cw = int(grid_cfg["cell_width"] * w) if grid_cfg["cell_width"] <= 1.0 else int(grid_cfg["cell_width"])
    ch = int(grid_cfg["cell_height"] * h) if grid_cfg["cell_height"] <= 1.0 else int(grid_cfg["cell_height"])

    print(f"网格: origin=({ox},{oy}), cell={cw}x{ch}, grid={cols}x{rows}")

    out_dir = "cells"
    os.makedirs(out_dir, exist_ok=True)

    for r in range(rows):
        for c in range(cols):
            x = ox + c * cw
            y = oy + r * ch
            cell = screenshot[y:y+ch, x:x+cw]
            fname = f"r{r:02d}_c{c:02d}.png"
            cv2.imwrite(os.path.join(out_dir, fname), cell)

    total = rows * cols
    print(f"\n已保存 {total} 个格子到 {out_dir}/")
    print(f"\n下一步：从 {out_dir}/ 中为每个数字(1-9)挑一个清晰的格子，复制到 templates/")
    print(f"例如:")
    print(f"  copy {out_dir}\\r00_c00.png templates\\4.png")


if __name__ == "__main__":
    main()
