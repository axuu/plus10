import ctypes
import logging
import numpy as np

log = logging.getLogger("capture")

try:
    import win32gui
    log.debug("win32gui 导入成功")
except ImportError as e:
    win32gui = None
    log.warning(f"win32gui 导入失败: {e}")

try:
    import mss
    log.debug("mss 导入成功")
except ImportError as e:
    mss = None
    log.warning(f"mss 导入失败: {e}")


def set_dpi_aware():
    """声明进程 DPI 感知，避免高分屏坐标错误"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        log.info("DPI Awareness 设置为 PerMonitorV2")
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            log.info("DPI Awareness 设置为 SystemAware")
        except Exception as e:
            log.warning(f"DPI Awareness 设置失败: {e}")


def find_game_window(title: str) -> int:
    log.info(f"查找窗口: '{title}'")
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        log.error(f"未找到窗口: '{title}'")
        raise RuntimeError(f"未找到窗口: {title}")
    rect = win32gui.GetWindowRect(hwnd)
    log.info(f"找到窗口: hwnd={hwnd}, rect={rect}")
    return hwnd


def capture_window(hwnd: int) -> np.ndarray:
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    log.debug(f"截图区域: left={left}, top={top}, w={width}, h={height}")

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot, dtype=np.uint8)
        img = img[:, :, :3]  # BGRA -> BGR

    log.debug(f"截图完成: shape={img.shape}")
    return img


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    return win32gui.GetWindowRect(hwnd)
