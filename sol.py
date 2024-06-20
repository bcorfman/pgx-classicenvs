from pathlib import Path

import solara


@solara.component
def InputMove():
    move, _ = solara.use_state("")
    confirm_disabled, set_confirm_disabled = solara.use_state(True)

    def update_text(txt):
        try:
            set_confirm_disabled(int(txt) < 1 or int(txt) > 9 or txt == "")
        except ValueError:
            set_confirm_disabled(True)

    with solara.Row():
        solara.InputText(
            "Enter a column 1-9 where to drop your checker",
            value=move,
            on_value=update_text,
            continuous_update=True,
        )
        solara.Button("Confirm", disabled=confirm_disabled)


@solara.component
def GameBoard():
    solara.Image(Path("state.svg"))


@solara.component
def Page():
    GameBoard()
    InputMove()


Page()
