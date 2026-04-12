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
    """截取窗口客户区，自动排除任务栏区域"""
    # 获取客户区屏幕坐标
    cr_left, cr_top, cr_right, cr_bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (cr_left, cr_top))
    width = cr_right - cr_left
    height = cr_bottom - cr_top

    # 获取工作区（屏幕减去任务栏）并裁剪
    try:
        import win32api
        monitor = win32api.MonitorFromWindow(hwnd, 0)
        info = win32api.GetMonitorInfo(monitor)
        work = info["Work"]  # (left, top, right, bottom)
        clipped_bottom = min(top + height, work[3])
        clipped_right = min(left + width, work[2])
        clipped_left = max(left, work[0])
        clipped_top = max(top, work[1])
        width = clipped_right - clipped_left
        height = clipped_bottom - clipped_top
        left = clipped_left
        top = clipped_top
        log.debug(f"客户区截图(裁剪到工作区): left={left}, top={top}, w={width}, h={height}, work_area={work}")
    except Exception as e:
        log.warning(f"获取工作区失败，使用完整客户区: {e}")
        log.debug(f"客户区截图: left={left}, top={top}, w={width}, h={height}")

    if width <= 0 or height <= 0:
        raise RuntimeError(f"截图区域无效: {width}x{height}")

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot, dtype=np.uint8)
        img = img[:, :, :3]  # BGRA -> BGR

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
