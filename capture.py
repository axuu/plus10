import ctypes
import numpy as np

try:
    import win32gui
    import win32ui
    import win32con
except ImportError:
    win32gui = None
    win32ui = None
    win32con = None


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
    """截取指定窗口的图像。

    Args:
        hwnd: 窗口句柄

    Returns:
        BGR 格式的 numpy 数组 (H, W, 3)

    Raises:
        RuntimeError: 截图失败
    """
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

    bmp_info = bitmap.GetInfo()
    bmp_bits = bitmap.GetBitmapBits(True)

    img = np.frombuffer(bmp_bits, dtype=np.uint8)
    img = img.reshape((bmp_info["bmHeight"], bmp_info["bmWidth"], 4))
    img = img[:, :, :3]  # BGRA -> BGR

    # 清理资源
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return img


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """获取窗口位置 (left, top, right, bottom)"""
    return win32gui.GetWindowRect(hwnd)
