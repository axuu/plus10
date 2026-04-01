# 羊了个羊：星球 — 自动消除机器人设计文档

## 概述

一个 Python 程序，自动截取游戏窗口截图，识别 10x16 数字矩阵，求解"矩形区域数字和=10"的最优消除序列，并通过鼠标拖拽自动执行消除操作，持续循环以最大化分数。

## 游戏规则

- 窗口名称：`羊了个羊：星球`
- 网格：10 列 x 16 行，每格一个数字（1-9）
- 操作：鼠标从一点按下拖到另一点释放，形成矩形选区
- 消除条件：矩形内所有数字之和 = 10（可包含空格，空格不计分）
- 消除后：被消除的格子变为空白，其余数字位置不变
- 分数 = 总消除数字数量

## 架构

```
main.py (主循环)
  ├── capture.py    (窗口截图)
  ├── recognizer.py (数字识别)
  ├── solver.py     (消除求解)
  └── executor.py   (鼠标操作)
```

配置文件：`config.yaml`
模板图片：`templates/` 目录

## 模块设计

### 1. Capture — 窗口截图

- 用 `win32gui.FindWindow(None, "羊了个羊：星球")` 查找窗口句柄
- 用 `win32gui.GetWindowRect()` 获取窗口位置和大小
- 用 `win32ui.CreateDCFromHandle()` + `BitBlt` 截取窗口内容
- 截图结果转为 numpy 数组
- 程序启动时声明 DPI Aware，处理高分屏缩放问题
- 窗口不能最小化，但可被部分遮挡

### 2. Recognizer — 数字识别

**初始化（一次性）：**
- 首次运行时从截图中裁出 1-9 每个数字的模板图，保存到 `templates/` 目录
- 记录网格起始坐标、格子宽高、间距到 `config.yaml`

**识别流程：**
1. 截图后根据配置裁出网格区域
2. 按行列切分为 160 个格子
3. 先通过像素均值/方差快速判断格子是否为空（已消除），空格跳过
4. 非空格子与 9 张模板做 `cv2.matchTemplate`，取最高匹配分数
5. 低于置信度阈值判定为空格
6. 输出：16x10 二维数组，0 表示空，1-9 表示数字

### 3. Solver — 消除求解

**枚举合法矩形：**
- 构建前缀和数组加速矩形求和
- 遍历所有 `(r1, c1, r2, c2)` 组合，约 7480 个候选
- 筛选条件：矩形内非空数字和 = 10，且至少包含 2 个非空数字

**贪心 + 前瞻搜索：**
- 前瞻深度 N = 3~5（可配置）
- DFS + 剪枝搜索最优消除序列
- 评估函数：序列总消除数字数量
- 剪枝：当前分支理论上限低于已知最优解时剪掉
- 候选按消除数字数降序排列，优先探索高收益分支
- 每轮只执行第一步消除，然后重新截图识别再求解

### 4. Executor — 鼠标操作

**坐标映射：**
- 格子 `(row, col)` 像素中心 = 网格起始坐标 + (col x 格子宽, row x 格子高) + 偏移

**精度优化：**
- 起点/终点向矩形内部收缩几像素，避免落在格子边缘误触相邻格子
- 禁用 pyautogui 移动动画，使用瞬移（duration=0）
- 操作前确认游戏窗口在前台

**执行流程：**
1. `pyautogui.moveTo()` 移到起点
2. `pyautogui.mouseDown()` 按下
3. `pyautogui.moveTo()` 移到终点
4. `pyautogui.mouseUp()` 释放
5. 等待消除动画（可配置延迟）

**安全机制：**
- `pyautogui.FAILSAFE = True` — 鼠标移到左上角紧急中止
- 每次操作前检查窗口存在且在前台
- 全局快捷键（F10）暂停/恢复

## 主循环

```
while running:
    1. 检查游戏窗口是否存在
    2. 截图
    3. 识别矩阵
    4. 求解最优消除
    5. 找到可消除矩形 → 执行鼠标操作 → 等待动画 → 回到 2
    6. 未找到 → 等待后重试（可能是新一轮）
```

## 配置项（config.yaml）

- `window_title`: 窗口标题
- `grid_origin`: 网格起始坐标 (x, y)
- `cell_width`: 格子宽度
- `cell_height`: 格子高度
- `confidence_threshold`: 模板匹配置信度阈值
- `animation_delay`: 消除动画等待时间（秒）
- `search_depth`: 前瞻搜索深度
- `hotkey_pause`: 暂停快捷键
- `hotkey_quit`: 退出快捷键

## 依赖

- `pywin32` — 窗口操作和截图
- `opencv-python` — 图像处理和模板匹配
- `numpy` — 数组计算
- `pyautogui` — 鼠标操作
- `pyyaml` — 配置文件
- `keyboard` — 全局快捷键监听

## 项目文件结构

```
miemie/
├── config.yaml
├── templates/
├── main.py
├── capture.py
├── recognizer.py
├── solver.py
├── executor.py
└── requirements.txt
```
