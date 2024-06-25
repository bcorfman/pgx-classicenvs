from nicegui import run, ui

from src.model import ai_move, human_move, setup_jit

setup_jit()


class Action:
    def __init__(self):
        self.value = ""


action = Action()
confirm_btn = None
img = None


@ui.refreshable:
    img.update()


def on_txt_change(e):
    try:
        if int(action.value) < 0 or int(action.value) > 6 or action.value == "":
            confirm_btn.disable()
        else:
            confirm_btn.enable()
    except ValueError:
        confirm_btn.disable()


async def on_confirm():
    is_human_move = True
    if is_human_move:
        confirm_btn.enable()
        game_over = await run.io_bound(human_move, int(action.value))
        display_svg.refresh()
        print("human_move")
        confirm_btn.disable()
    if game_over:
        return
    game_over = await run.cpu_bound(ai_move)
    display_svg.refresh()
    print("ai move")
    confirm_btn.enable()
    if game_over:
        confirm_btn.disable()


with ui.row():
    img = ui.image("state.svg")
    display_svg()
    with ui.row():
        ui.input(
            label="Enter column 1-9 where to drop your checker",
            on_change=on_txt_change,
        ).bind_value(action, "value").style("min-width: 400px; max-width: 400px;")
        confirm_btn = ui.button("Confirm", on_click=on_confirm)
        confirm_btn.disable()
