# PgxConnectFour

A version of this example game that uses Google's [PGX](https://github.com/sotetsuk/pgx) and [Monte-Carlo Tree Search in JAX](https://github.com/google-deepmind/mctx) to play Connect Four. The [original Jupyter notebook version](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/mcts_connect_four.ipynb#scrollTo=aPYOMRQZamsk) had stopped working over time. I reimplemented it as a real browser-based game using [NiceGUI](https://nicegui.io/). 

## To run the game
* At a command prompt, type `make install` to install the [Rye dependency manager](https://rye.astral.sh/) on your system (if needed). After Rye is installed, it will download the supporting Python libraries inside a virtual environment.
* Once the install has completed, type `make run` to launch the game.
