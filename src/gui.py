from nicegui import run, ui

from src.model import HUMAN, ai_move, get_player_turn, human_move, setup_jit

setup_jit()


class Action:
    def __init__(self):
        self.value = ""


action = Action()
confirm_btn = None
img = None


@ui.refreshable
def display_svg():
    img = ui.image("state.svg")
    img.force_reload()
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
    game_over = False
    if get_player_turn() == HUMAN:
        confirm_btn.enable()
        game_over = human_move(int(action.value))
        display_svg.refresh()
        print("human_move")
        confirm_btn.disable()
    if game_over:
        print("game over")
        return
    game_over = ai_move()
    print("ai move")
    display_svg.refresh()
    confirm_btn.enable()
    if game_over:
        confirm_btn.disable()


with ui.row():
    display_svg()
    with ui.row():
        ui.input(
            label="Enter column 1-9 where to drop your checker",
            on_change=on_txt_change,
        ).bind_value(action, "value").style("min-width: 400px; max-width: 400px;")
        confirm_btn = ui.button("Confirm", on_click=on_confirm)
        confirm_btn.disable()
