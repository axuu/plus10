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
        # 用 GetClientRect + ClientToScreen，与 capture_window 一致
        cr = win32gui.GetClientRect(self.hwnd)
        left, top = win32gui.ClientToScreen(self.hwnd, (cr[0], cr[1]))
        cw_total = cr[2] - cr[0]
        ch_total = cr[3] - cr[1]

        ox = self.grid_origin_x * cw_total if self.grid_origin_x <= 1.0 else self.grid_origin_x
        oy = self.grid_origin_y * ch_total if self.grid_origin_y <= 1.0 else self.grid_origin_y
        cw = self.cell_width * cw_total if self.cell_width <= 1.0 else self.cell_width
        ch = self.cell_height * ch_total if self.cell_height <= 1.0 else self.cell_height

        px = int(round(left + ox + (col + 0.5) * cw))
        py = int(round(top + oy + (row + 0.5) * ch))
        log.debug(f"  cell({row},{col}) -> pixel({px},{py}) [client=({left},{top},{cw_total}x{ch_total})]")
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
