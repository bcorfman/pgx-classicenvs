import os

import solara

move = solara.Reactive(1)
confirm_disabled = solara.Reactive(False)


@solara.component
def Page():
    global move
    solara.Image(os.path.join("state.svg"))
    solara.InputText(
        "Enter a column 1-9 where to drop your checker",
        value=move,
        continuous_update=True,
    )
    try:
        confirm_disabled = (
            int(move.value) < 1 or int(move.value) > 9 or move.value == ""
        )
    except ValueError:
        confirm_disabled = True
    solara.Button("Confirm", disabled=confirm_disabled)


# The following line is required only when running the code in a Jupyter notebook:
Page()
