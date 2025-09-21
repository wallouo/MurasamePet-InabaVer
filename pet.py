# -*- coding: utf-8 -*-
"""
MurasamePet 前端（PyQt5）
- 無邊框、透明、置頂、右下角停靠的桌寵視窗
- 頭部滑動/雙擊 => 呼叫 /pat，顯示中文字幕並播放日文語音（或 mock）
"""

import os
import sys
import requests
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia

API_PORT = os.getenv("API_PORT", "5000")
API_URL = f"http://127.0.0.1:{API_PORT}"


class PetWindow(QtWidgets.QLabel):
    def __init__(self):
        super().__init__(None)

        # ---- 視窗外觀：無邊框 + 透明 + 置頂 + 不佔工作列 ----
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        # ---- 載入角色圖（需 PNG + Alpha）----
        char_path = os.path.join(os.path.dirname(__file__), "assets", "character.png")
        if not os.path.exists(char_path):
            raise FileNotFoundError(f"角色圖不存在：{char_path}")
        base_pix = QtGui.QPixmap(char_path)
        # 縮放：寬度最多 420px，等比縮放
        target_w = min(420, base_pix.width())
        self.pixmap_now = base_pix.scaledToWidth(
            target_w, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(self.pixmap_now)
        self.resize(self.pixmap_now.size())

        # ---- 字幕條（子元件）----
        self.subtitle = QtWidgets.QLabel(self)
        self.subtitle.setStyleSheet(
            "color: white; background-color: rgba(0,0,0,160);"
            "font-size: 16px; padding: 6px; border-radius: 6px;"
        )
        self.subtitle.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.subtitle.setWordWrap(True)
        self.subtitle.setText("")
        self._layout_subtitle()

        # ---- 滑鼠交互狀態 ----
        self._dragging_head = False
        self._drag_start_x = 0
        self._moved = False
        self._player = None  # 避免 QMediaPlayer 被 GC
        self._update_head_rect()

        # ---- 置於右下角並顯示 ----
        self._move_to_bottom_right(margin_x=20, margin_y=40)
        self.show()

    # ---- 佈局與幾何 ----
    def _layout_subtitle(self):
        w, h = self.width(), self.height()
        margin_x, margin_y = 16, 12
        bar_h = max(48, int(h * 0.12))
        self.subtitle.setGeometry(
            margin_x, h - bar_h - margin_y, w - margin_x * 2, bar_h
        )

    def _update_head_rect(self):
        """頭部偵測區：視窗大小變化時重算（大約上方中間區域）。"""
        w, h = self.width(), self.height()
        self.head_rect = QtCore.QRect(int(w * 0.25), int(h * 0.05),
                                      int(w * 0.5), int(h * 0.25))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout_subtitle()
        self._update_head_rect()

    def _move_to_bottom_right(self, margin_x=20, margin_y=40):
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = geo.right() - self.width() - margin_x
        y = geo.bottom() - self.height() - margin_y
        self.move(max(geo.left(), x), max(geo.top(), y))

    # ---- 滑鼠事件：摸頭觸發 /pat ----
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.head_rect.contains(event.pos()):
            self._dragging_head = True
            self._drag_start_x = event.x()
            self._moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging_head:
            if abs(event.x() - self._drag_start_x) > 30:
                self._moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging_head:
            if self._moved:
                self.trigger_pat()
            self._dragging_head = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.head_rect.contains(event.pos()):
            self.trigger_pat()
        super().mouseDoubleClickEvent(event)

    # ---- 後端互動 ----
    def trigger_pat(self) -> None:
        """呼叫 /pat，更新字幕並播放語音（VOICEVOX 或 mock）。"""
        try:
            resp = requests.post(f"{API_URL}/pat", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            subtitle = data.get("subtitle_zh", "")
            wav_path = data.get("wav_path")

            if subtitle:
                self.subtitle.setText(subtitle)

            if wav_path and os.path.exists(wav_path):
                url = QtCore.QUrl.fromLocalFile(os.path.abspath(wav_path))
                content = QtMultimedia.QMediaContent(url)
                if self._player is None:
                    self._player = QtMultimedia.QMediaPlayer()
                self._player.setMedia(content)
                self._player.setVolume(100)
                self._player.play()
            else:
                # 沒檔案就提示一下（例如 API 只有回字幕）
                if not subtitle:
                    self.subtitle.setText("(pat success, no audio)")
        except Exception as e:
            self.subtitle.setText(f"(pat failed) {e.__class__.__name__}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    # 修正高 DPI 模糊
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    win = PetWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
