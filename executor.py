import time
import logging

log = logging.getLogger("executor")

try:
    import pyautogui
    import win32gui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
except ImportError as e:
    pyautogui = None
    win32gui = None
    log.warning(f"导入失败: {e}")


class Executor:
    def __init__(
        self,
        hwnd: int,
        grid_origin_x: float,
        grid_origin_y: float,
        cell_width: float,
        cell_height: float,
        inward_shrink: int = 5,
        animation_delay: float = 0.5,
    ):
        self.hwnd = hwnd
        self.grid_origin_x = grid_origin_x
        self.grid_origin_y = grid_origin_y
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.inward_shrink = inward_shrink
        self.animation_delay = animation_delay
        log.info(f"Executor 初始化: origin=({grid_origin_x},{grid_origin_y}), "
                 f"cell=({cell_width}x{cell_height}), shrink={inward_shrink}, delay={animation_delay}")

    def _cell_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        # 获取客户区屏幕坐标（不含标题栏/边框）
        try:
            cr = win32gui.GetClientRect(self.hwnd)
            client_left, client_top = win32gui.ClientToScreen(self.hwnd, (cr[0], cr[1]))
            client_w = cr[2] - cr[0]
            client_h = cr[3] - cr[1]
        except Exception:
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            client_left, client_top = left, top
            client_w = right - left
            client_h = bottom - top

        # 比例值转像素（浮点运算避免漂移）
        ox = self.grid_origin_x * client_w if self.grid_origin_x <= 1.0 else self.grid_origin_x
        oy = self.grid_origin_y * client_h if self.grid_origin_y <= 1.0 else self.grid_origin_y
        cw = self.cell_width * client_w if self.cell_width <= 1.0 else self.cell_width
        ch = self.cell_height * client_h if self.cell_height <= 1.0 else self.cell_height

        # 格子中心像素坐标
        px = int(round(client_left + ox + (col + 0.5) * cw))
        py = int(round(client_top + oy + (row + 0.5) * ch))
        log.debug(f"  cell({row},{col}) -> pixel({px},{py}) [client=({client_left},{client_top})]")
        return px, py

    def _ensure_foreground(self) -> bool:
        try:
            if not win32gui.IsWindow(self.hwnd):
                log.error("窗口已不存在")
                return False
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.05)
            return True
        except Exception as e:
            log.error(f"设置前台窗口失败: {e}")
            return False

    def execute_move(self, r1: int, c1: int, r2: int, c2: int):
        log.info(f"执行消除: ({r1},{c1})->({r2},{c2})")

        if not self._ensure_foreground():
            raise RuntimeError("游戏窗口不在前台或已关闭")

        start_x, start_y = self._cell_to_pixel(r1, c1)
        end_x, end_y = self._cell_to_pixel(r2, c2)

        shrink = self.inward_shrink
        if start_x <= end_x:
            start_x += shrink
            end_x -= shrink
        else:
            start_x -= shrink
            end_x += shrink

        if start_y <= end_y:
            start_y += shrink
            end_y -= shrink
        else:
            start_y -= shrink
            end_y += shrink

        log.info(f"  拖拽: ({start_x},{start_y}) -> ({end_x},{end_y}) [shrink={shrink}]")

        pyautogui.moveTo(start_x, start_y, duration=0)
        time.sleep(0.1)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.moveTo(end_x, end_y, duration=0.3)
        time.sleep(0.1)
        pyautogui.mouseUp()

        log.info(f"  拖拽完成，等待动画 {self.animation_delay}s")
        time.sleep(self.animation_delay)
