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
    import win32ui
    log.debug("win32ui 导入成功")
except ImportError as e:
    win32ui = None
    log.warning(f"win32ui 导入失败: {e}")

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
    """截取窗口客户区内容，使用 PrintWindow API（不受遮挡/任务栏影响）"""
    client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
    width = client_right - client_left
    height = client_bottom - client_top

    if width == 0 or height == 0:
        raise RuntimeError(f"窗口客户区尺寸为零: {width}x{height}")

    log.debug(f"客户区尺寸: {width}x{height}")

    # 使用 PrintWindow 直接从窗口获取内容（不受遮挡影响）
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    # PW_CLIENTONLY = 1，只截取客户区
    ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 1)

    bmp_bits = bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmp_bits, dtype=np.uint8).reshape((height, width, 4))
    img = img[:, :, :3]  # BGRA -> BGR

    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    log.debug(f"截图完成: shape={img.shape}")
    return img


def get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """返回客户区的屏幕坐标 (left, top, right, bottom)"""
    try:
        client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
        left, top = win32gui.ClientToScreen(hwnd, (client_left, client_top))
        return (left, top, left + client_right - client_left, top + client_bottom - client_top)
    except Exception:
        return win32gui.GetWindowRect(hwnd)
