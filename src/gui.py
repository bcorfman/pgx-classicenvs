from pathlib import Path

import solara

from src.model import RLGame

human_action = solara.Reactive("")


@solara.component
def Page():
    game_board, set_game_board = solara.use_state(solara.Image(Path("state.svg")))
    confirm_disabled, set_confirm_disabled = solara.use_state(True)

    game = RLGame("connect_four")
    game.setup_jit()

    def update_text(txt):
        try:
            set_confirm_disabled(int(txt) < 1 or int(txt) > 9 or txt == "")
        except ValueError:
            set_confirm_disabled(True)

    def on_confirm():
        is_human_move = True
        if is_human_move:
            game_over = game.human_move(int(human_action.value))
            set_game_board(solara.Image(Path("state.svg")))
        if game_over:
            set_confirm_disabled(True)
            return
        game_over = game.ai_move()
        set_game_board(solara.Image(Path("state.svg")))
        if game_over:
            set_confirm_disabled(True)

    with solara.Row():
        solara.InputText(
            "Enter a column 1-9 where to drop your checker",
            value=human_action,
            on_value=update_text,
            continuous_update=True,
        )
        solara.Button("Confirm", on_click=on_confirm, disabled=confirm_disabled)
