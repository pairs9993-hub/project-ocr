"""Portable capture geometry: never persist desktop coordinates or HWNDs."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import math
import sys


@dataclass(frozen=True)
class ClientWindow:
    handle: int
    class_name: str
    title: str
    rect: tuple[int, int, int, int]


def validate_anchor(anchor: dict) -> None:
    if not isinstance(anchor, dict) or anchor.get("kind") != "window_client_v1":
        raise ValueError("Unsupported capture anchor.")
    if not isinstance(anchor.get("class_name"), str) or not anchor["class_name"]:
        raise ValueError("Capture anchor requires a window class.")
    if not isinstance(anchor.get("title"), str):
        raise ValueError("Capture anchor requires a window title hint.")
    rect = anchor.get("normalized_rect")
    if not isinstance(rect, list) or len(rect) != 4 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in rect
    ):
        raise ValueError("Invalid normalized capture rectangle.")
    if not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1):
        raise ValueError("Capture rectangle must be inside the window client.")


def make_anchor(window: ClientWindow, selection: tuple[int, int, int, int]) -> dict:
    left, top, right, bottom = window.rect
    x1, y1, x2, y2 = selection
    anchor = {"kind": "window_client_v1", "class_name": window.class_name,
              "title": window.title,
              "normalized_rect": [(x1-left)/(right-left), (y1-top)/(bottom-top),
                                  (x2-left)/(right-left), (y2-top)/(bottom-top)]}
    validate_anchor(anchor)
    return anchor


def mapped_rect(anchor: dict, client_rect: tuple[int, int, int, int], image_size) -> tuple:
    validate_anchor(anchor)
    left, top, right, bottom = client_rect
    width, height = right-left, bottom-top
    if width <= 0 or height <= 0:
        raise ValueError("Vesta client is minimized or has no drawable area.")
    x1, y1, x2, y2 = anchor["normalized_rect"]
    rect = (left+round(x1*width), top+round(y1*height),
            left+round(x2*width), top+round(y2*height))
    w, h = rect[2]-rect[0], rect[3]-rect[1]
    if w <= 0 or h <= 0 or abs((w/h)/(image_size[0]/image_size[1])-1) > 0.02:
        raise ValueError("Vesta layout/aspect ratio changed. Restore its layout or recalibrate the preset.")
    return rect


def _user32():
    if sys.platform != "win32":
        raise RuntimeError("Vesta automation requires Windows.")
    api = ctypes.WinDLL("user32", use_last_error=True)
    # HWND is pointer-sized; ctypes' default c_int truncates it on 64-bit Windows.
    api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    api.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    api.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.IsWindowVisible.argtypes = [wintypes.HWND]
    api.IsIconic.argtypes = [wintypes.HWND]
    api.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    api.GetAncestor.restype = wintypes.HWND
    api.SetForegroundWindow.argtypes = [wintypes.HWND]
    api.WindowFromPoint.argtypes = [wintypes.POINT]
    api.WindowFromPoint.restype = wintypes.HWND
    return api


def client_windows(pid: int) -> list[ClientWindow]:
    api = _user32()
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    api.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    api.EnumChildWindows.argtypes = [wintypes.HWND, callback_type, wintypes.LPARAM]
    windows = []

    def collect(hwnd, _param):
        owner = wintypes.DWORD()
        api.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid or not api.IsWindowVisible(hwnd):
            return True
        if api.IsIconic(api.GetAncestor(hwnd, 2)):
            return True
        rect, origin = wintypes.RECT(), wintypes.POINT()
        if not api.GetClientRect(hwnd, ctypes.byref(rect)) or not api.ClientToScreen(hwnd, ctypes.byref(origin)):
            return True
        if rect.right <= 0 or rect.bottom <= 0:
            return True
        cls, title = ctypes.create_unicode_buffer(512), ctypes.create_unicode_buffer(1024)
        api.GetClassNameW(hwnd, cls, len(cls))
        api.GetWindowTextW(hwnd, title, len(title))
        windows.append(ClientWindow(hwnd, cls.value, title.value,
                       (origin.x, origin.y, origin.x+rect.right, origin.y+rect.bottom)))
        return True

    child_callback = callback_type(collect)

    def top(hwnd, param):
        collect(hwnd, param)
        owner = wintypes.DWORD()
        api.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            api.EnumChildWindows(hwnd, child_callback, 0)
        return True

    api.EnumWindows(callback_type(top), 0)
    return windows


def choose_window(windows: list[ClientWindow], anchor: dict) -> ClientWindow:
    candidates = [w for w in windows if w.class_name == anchor["class_name"]]
    exact = [w for w in candidates if w.title == anchor["title"]]
    if exact:
        candidates = exact
    if len(candidates) != 1:
        raise ValueError("Vesta capture window missing/ambiguous. Restore the window or register its screen reference again.")
    return candidates[0]


def selection_window(windows: list[ClientWindow], selection: tuple) -> ClientWindow:
    x1, y1, x2, y2 = selection
    candidates = [w for w in windows if w.rect[0] <= x1 < x2 <= w.rect[2]
                  and w.rect[1] <= y1 < y2 <= w.rect[3]]
    if not candidates:
        raise ValueError("Select only the GUI screen inside the launched Vesta window (not its title bar).")
    return min(candidates, key=lambda w: (w.rect[2]-w.rect[0])*(w.rect[3]-w.rect[1]))


def foreground(window: ClientWindow) -> None:
    api = _user32()
    api.SetForegroundWindow(api.GetAncestor(window.handle, 2))


def require_unobstructed(pid: int, rect: tuple) -> None:
    """Reject hidden/offscreen/covered screen captures, not OCR somebody else's window."""
    api = _user32()
    left, top, right, bottom = rect
    for x in (left, (left+right)//2, right-1):
        for y in (top, (top+bottom)//2, bottom-1):
            hwnd = api.WindowFromPoint(wintypes.POINT(x, y))
            owner = wintypes.DWORD()
            api.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value != pid:
                raise ValueError("Vesta GUI screen is covered or offscreen. Move other windows away and keep Vesta visible.")