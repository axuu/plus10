import ctypes
import numpy as np

try:
    import win32gui
except ImportError:
    win32gui = None

try:
    import mss
except ImportError:
    mss = None


def set_dpi_aware():
    """声明进程 DPI 感知，避免高分屏坐标错误"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def find_game_window(title: str) -> int:
    """通过标题查找窗口句柄。

    Args:
        title: 窗口标题

    Returns:
        窗口句柄 (HWND)

    Raises:
        RuntimeError: 未找到窗口
    """
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        raise RuntimeError(f"未找到窗口: {title}")
    return hwnd


def capture_window(hwnd: int) -> np.ndarray:
    """截取指定窗口所在屏幕区域的图像。

    使用 mss 直接截取屏幕区域，兼容 GPU 渲染的窗口。
    注意：窗口不能被其他窗口遮挡。

    Args:
        hwnd: 窗口句柄

    Returns:
        BGR 格式的 numpy 数组 (H, W, 3)
    """
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot, dtype=np.uint8)
        # mss 返回 BGRA，转为 BGR
        img = img[:, :, :3]

    return img


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """获取窗口位置 (left, top, right, bottom)"""
    return win32gui.GetWindowRect(hwnd)
