import asyncio

from nicegui import Tailwind, ui

from src.model import HUMAN, ai_move, get_player_turn, human_move, setup_jit


class Action:
    def __init__(self):
        self.value = ""


setup_jit()

action = Action()
confirm_btn = None
game_over = False
img = None
reset_btn = None
txt = None
wait_time = 0.5


@ui.refreshable
def display_svg():
    img = ui.image("state.svg")
    img.force_reload()
    img.update()


def set_confirm_button_state():
    global game_over
    try:
        if (
            game_over
            or int(action.value) < 1
            or int(action.value) > 7
            or action.value == ""
        ):
            confirm_btn.disable()
        else:
            confirm_btn.enable()
    except ValueError:
        confirm_btn.disable()


def on_txt_change(e):
    set_confirm_button_state()


def show_game_over_dialog():
    with ui.dialog() as dialog, ui.card():
        ui.label("Game Over!")
        ui.button("Close", on_click=dialog.close)
    dialog.open()


async def on_confirm():
    global confirm_btn, reset_btn, wait_time, game_over
    n = None
    reset_btn.disable()
    game_over = False
    if get_player_turn() == HUMAN:
        confirm_btn.enable()
        game_over = human_move(int(action.value) - 1)
        display_svg.refresh()
        confirm_btn.disable()
    if game_over:
        action.value = ""
        set_confirm_button_state()
        reset_btn.enable()
        show_game_over_dialog()
        return
    if wait_time == 0.5:
        n = ui.notification(
            message="Thinking ...", spinner=True, timeout=None, position="bottom-left"
        )
    await asyncio.sleep(wait_time)
    game_over = ai_move()
    if wait_time == 0.5:
        n.dismiss()
        wait_time = 1.0
    display_svg.refresh()
    if game_over:
        action.value = ""
        show_game_over_dialog()
    set_confirm_button_state()
    reset_btn.enable()


async def on_reset():
    global game_over, txt, wait_time
    game_over = False
    action.value = ""
    wait_time = 0.1

    txt.label = "Enter column 1-7 where to drop your checker"
    setup_jit()
    display_svg.refresh()


with ui.row():
    display_svg()
    with ui.row():
        txt = (
            ui.input(
                label="Enter column 1-7 where to drop your checker",
                on_change=on_txt_change,
            )
            .bind_value(action, "value")
            .bind_enabled_from(game_over)
            .style("min-width: 425px; max-width: 425px;")
        )
        confirm_btn = ui.button("Confirm", on_click=on_confirm)
        confirm_btn.disable()
        reset_btn = ui.button("Reset game", on_click=on_reset).style(
            "min_width: 150px; max-width: 150px;"
        )
