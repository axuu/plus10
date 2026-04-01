import time

try:
    import pyautogui
    import win32gui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
except ImportError:
    pyautogui = None
    win32gui = None


class Executor:
    """鼠标拖拽操作执行器"""

    def __init__(
        self,
        hwnd: int,
        grid_origin_x: int,
        grid_origin_y: int,
        cell_width: int,
        cell_height: int,
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

    def _cell_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        """将格子坐标转换为屏幕像素坐标（格子中心）"""
        left, top, _, _ = win32gui.GetWindowRect(self.hwnd)

        px = left + self.grid_origin_x + col * self.cell_width + self.cell_width // 2
        py = top + self.grid_origin_y + row * self.cell_height + self.cell_height // 2

        return px, py

    def _ensure_foreground(self) -> bool:
        """确保游戏窗口在前台"""
        try:
            if not win32gui.IsWindow(self.hwnd):
                return False
            win32gui.SetForegroundWindow(self.hwnd)
            time.sleep(0.05)
            return True
        except Exception:
            return False

    def execute_move(self, r1: int, c1: int, r2: int, c2: int):
        """执行一次矩形消除的鼠标拖拽。

        Args:
            r1, c1: 矩形左上角格子坐标
            r2, c2: 矩形右下角格子坐标
        """
        if not self._ensure_foreground():
            raise RuntimeError("游戏窗口不在前台或已关闭")

        start_x, start_y = self._cell_to_pixel(r1, c1)
        end_x, end_y = self._cell_to_pixel(r2, c2)

        # 向矩形内部收缩，避免误触相邻格子
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

        # 执行拖拽（给游戏足够时间感知）
        pyautogui.moveTo(start_x, start_y, duration=0)
        time.sleep(0.1)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.moveTo(end_x, end_y, duration=0.3)
        time.sleep(0.1)
        pyautogui.mouseUp()

        # 等待消除动画
        time.sleep(self.animation_delay)
