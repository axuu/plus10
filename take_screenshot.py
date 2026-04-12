"""截取游戏窗口并保存到 debug/screenshot.png"""
import os
import yaml
from capture import set_dpi_aware, find_game_window, capture_window
import cv2

set_dpi_aware()

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

hwnd = find_game_window(config["window_title"])
img = capture_window(hwnd)

os.makedirs("debug", exist_ok=True)
cv2.imwrite("debug/screenshot.png", img)
print(f"截图已保存: debug/screenshot.png ({img.shape[1]}x{img.shape[0]})")
