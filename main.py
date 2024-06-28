from nicegui import ui

from src.gui import setup_jit

if __name__ in {"__main__", "__mp_main__"}:
    setup_jit()
    ui.run()
