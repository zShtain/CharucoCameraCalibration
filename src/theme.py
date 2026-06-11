"""
Centralized color scheme and dark-theme application for the calibration GUI.

All UI colors live in COLORS; apply_dark_theme() wires them into the Qt
palette plus a small stylesheet for surfaces the palette alone can't recolor
(group-box borders, button states, list selection).
"""

from PyQt5.QtGui import QPalette, QColor

COLORS = {
    "bg": "#0B0F19",  # terminal background
    "bg_alt": "#121824",  # input / list / log background
    "surface": "#1E293B",  # buttons, selection layer
    "surface2": "#26354A",  # hover / alternative surface
    "border": "#334155",  # borders & dividers
    "muted2": "#64748B",  # placeholders, disabled, timestamps
    "muted": "#94A3B8",  # section titles / metadata
    "text2": "#E2E8F0",  # primary text
    "text": "#F8FAFC",  # high-contrast text
    "info": "#38BDF8",  # cursor / info / selection text
    "info2": "#06B6D4",  # secondary info
    "success": "#10B981",  # success status
    "warning": "#F59E0B",  # warning status
    "error": "#F43F5E",  # error status
}


def apply_dark_theme(app):
    """ Apply the slate dark scheme to a QApplication. """
    C = COLORS
    app.setStyle("Fusion")  # native styles ignore most palette colors

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(C["bg"]))
    pal.setColor(QPalette.WindowText, QColor(C["text2"]))
    pal.setColor(QPalette.Base, QColor(C["bg_alt"]))
    pal.setColor(QPalette.AlternateBase, QColor(C["surface"]))
    pal.setColor(QPalette.Text, QColor(C["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(C["muted2"]))
    pal.setColor(QPalette.Button, QColor(C["surface"]))
    pal.setColor(QPalette.ButtonText, QColor(C["text2"]))
    pal.setColor(QPalette.BrightText, QColor(C["text"]))
    pal.setColor(QPalette.ToolTipBase, QColor(C["bg_alt"]))
    pal.setColor(QPalette.ToolTipText, QColor(C["text2"]))
    pal.setColor(QPalette.Highlight, QColor(C["surface"]))
    pal.setColor(QPalette.HighlightedText, QColor(C["info"]))
    pal.setColor(QPalette.Link, QColor(C["info"]))
    pal.setColor(QPalette.LinkVisited, QColor(C["info2"]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(C["muted2"]))
    app.setPalette(pal)

    app.setStyleSheet(f"""
        QGroupBox {{ border:1px solid {C['border']}; border-radius:6px; margin-top:8px; }}
        QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; color:{C['muted']}; }}
        QPushButton {{ background:{C['surface']}; border:1px solid {C['border']};
                       border-radius:5px; padding:6px 10px; color:{C['text2']}; }}
        QPushButton:hover {{ background:{C['surface2']}; border-color:{C['info']}; }}
        QPushButton:pressed {{ background:{C['bg_alt']}; }}
        QListView, QTextEdit, QLineEdit, QComboBox {{
            background:{C['bg_alt']}; border:1px solid {C['border']}; border-radius:4px; }}
        QListView::item:selected {{ background:{C['surface']}; color:{C['info']}; }}
        QGraphicsView {{ background:{C['bg']}; border:1px solid {C['border']}; }}
    """)
