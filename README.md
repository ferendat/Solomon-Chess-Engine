# Solomon

This is my chess engine, and the project I'm proudest of. I built Solomon from scratch and he
taught himself to play: no opening book, no human games, just thousands of games against himself.

He's named after the chess-playing hero in Deadlock.

## How he plays

- **Seeing the board.** Every position becomes 768 numbers (12 piece types x 64 squares), always
  from the point of view of the side to move. That means Solomon plays White and Black the same way.
- **Judging a position.** A small neural network (a plain MLP, no CNN) adds a positional correction to
  the material count. I capped the correction below one pawn, so he never trades away material
  because of a hunch. Without that cap, the network learned noise and played worse than the untrained one.
- **Looking ahead.** Negamax with alpha-beta pruning, then a short capture-only search (quiescence) so
  he never stops his calculation in the middle of an exchange. That was the biggest single jump in strength.
- **Learning.** Self-play. Every finished game goes into a log, and the most recent games are replayed
  into memory at startup, so training can be stopped and picked up again whenever.

## Setup

```
pip install -r requirements.txt
```

Tkinter is needed for the board window, and it ships with most Python installs.

## Playing Solomon

```
python main.py play --depth 2
```

A window opens and you pick your side with the buttons at the bottom. Without a display (Colab, for
example) it falls back to a board drawn in the notebook, where you type your moves.
`--depth 3` makes him stronger and slower.

## Training him

```
python main.py train --games 500
```

This saves `chess_model.pth`, a game log next to it, and snapshots in `checkpoints/`.
Run it again and he continues from where he stopped.

## Measuring him

```
python main.py bench --games 20 --stockfish /path/to/stockfish --elo 1320
python main.py compare --a chess_model.pth --b fresh --games 20
python main.py compare --a chess_model.pth --b chess_model.pth --qa 2 --qb 0 --depth 1
```

- `bench` plays a random mover, or Stockfish at a limited strength if you give it a path.
- `compare` plays two models against each other. `fresh` means an untrained network. The last command
  tests quiescence on its own: the same model with and without it.

From a notebook, import the functions directly:

```python
from solomon.training import run_train
from solomon.evaluation import run_compare
from solomon.play import run_play

run_train(games=100, depth=1, model_path="chess_model.pth")
```

## Layout

```
main.py               command line
solomon/
  config.py           settings
  board.py            board encoding and game rules
  model.py            the network
  agent.py            search, learning, saving
  env.py              one self-play game
  gamelog.py          saving and replaying games
  training.py         training loop
  evaluation.py       bench and compare
  gui.py              Tkinter board
  play.py             play entry points
```

## What's next

- Piece-square tables, so he understands where pieces want to be
- A real rating from a longer Stockfish match
