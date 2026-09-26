"""Тёмная тема окна: экран статистики и всплывающее сообщение (QSS).

Цвета - из панели игры (render): тот же фон, та же панель, тот же приглушённый текст,
жёлтый - как у лидера и рекорда, голубой - как у ползунков. Вид экрана статистики -
по образцу Нейро-змейки (карточки, таблицы, графики на карточках).
"""
from PySide6.QtGui import QColor, QPalette

import render


def _hex(colour):
    return "#%02x%02x%02x" % tuple(colour)


BG = _hex(render.BG)
CARD = _hex(render.PANEL_BG)
LINE = "#2b303c"
TEXT = _hex(render.TEXT)
MUTED = _hex(render.TEXT_DIM)
ACCENT = _hex((110, 190, 230))         # ui.ACTIVE
GOOD = _hex(render.RAY_LINE)           # зелёный: не хуже замера
WARN = "#e2b340"                       # жёлтый: хуже замера
GOLD = _hex(render.LEADER_RING)
UI_FONT = '"Segoe UI", "Noto Sans", sans-serif'

QSS = f"""
QMainWindow, QWidget#screen, QWidget#scrollBody {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-family: {UI_FONT}; font-size: 10.5pt; }}
QToolTip {{ background: {CARD}; color: {TEXT}; border: 1px solid {LINE}; padding: 4px 6px; }}

QLabel#toast {{ background: rgba(22, 25, 32, 235); color: {GOLD}; border: 1px solid #484e5d;
               border-radius: 8px; padding: 8px 14px; font-family: "Consolas", monospace;
               font-size: 15px; }}

QLabel#title {{ font-size: 34pt; font-weight: 700; color: {GOLD}; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 11pt; }}
QLabel#screenTitle {{ font-size: 20pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#section {{ color: {MUTED}; font-size: 9pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#cardValue {{ font-size: 22pt; font-weight: 700; }}
QLabel#cardNote {{ color: {MUTED}; font-size: 9pt; }}
QLabel#value {{ font-weight: 600; }}

QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; }}
QFrame#card QLabel {{ background: transparent; border: none; }}

QPushButton {{ background: #232835; color: {TEXT}; border: 1px solid #343b4a; border-radius: 10px;
              padding: 9px 18px; font-family: {UI_FONT}; font-size: 11pt; }}
QPushButton:hover {{ background: #2b3242; border-color: #46506a; }}
QPushButton:pressed {{ background: #1b2029; }}
QPushButton:focus {{ border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: #07131b; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: #8cd0f0; }}
QPushButton#primary:pressed {{ background: #5aa6cc; }}
QPushButton#primary:focus {{ border: 2px solid {GOLD}; }}

QSlider::groove:horizontal {{ height: 6px; background: #2c3240; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 18px; height: 18px; margin: -7px 0; border-radius: 9px;
                             background: {TEXT}; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}

QComboBox {{ background: #232835; border: 1px solid #343b4a; border-radius: 8px;
            padding: 6px 12px; min-width: 190px; }}
QComboBox:hover {{ border-color: #46506a; }}
QComboBox QAbstractItemView {{ background: #232835; border: 1px solid #343b4a;
                              selection-background-color: #2d5a78; outline: none; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #343b4a; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


def palette():
    """Тёмная палитра Qt: у стиля Fusion по ней рисуется всё, что не описано в QSS."""
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG, QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: CARD, QPalette.ColorRole.AlternateBase: CARD,
        QPalette.ColorRole.Text: TEXT, QPalette.ColorRole.Button: "#232835",
        QPalette.ColorRole.ButtonText: TEXT, QPalette.ColorRole.Highlight: "#2d5a78",
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.ToolTipBase: CARD,
        QPalette.ColorRole.ToolTipText: TEXT, QPalette.ColorRole.PlaceholderText: MUTED,
    }
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    return p


def apply(app):
    """Стиль и палитра всего приложения. QSS ставит само главное окно (так и в тестах)."""
    app.setStyle("Fusion")
    app.setPalette(palette())
