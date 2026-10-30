import json
import os

import chess

from .board import encode
from .config import BUFFER_SIZE


def game_to_samples(moves, outcome):
    board = chess.Board()
    samples = []
    for uci in moves:
        x, material = encode(board)
        target = outcome if board.turn == chess.WHITE else -outcome
        samples.append((x, material, float(target)))
        board.push_uci(uci)
    return samples


def save_game(path, moves, outcome):
    with open(path, "a") as f:
        f.write(json.dumps({"moves": moves, "outcome": outcome}) + "\n")


def load_games(path, max_samples=BUFFER_SIZE):
    if not os.path.exists(path):
        return [], 0
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    chunks, count = [], 0
    for line in reversed(lines):
        if count >= max_samples:
            break
        try:
            game = json.loads(line)
            chunk = game_to_samples(game["moves"], game["outcome"])
        except Exception:
            continue
        chunks.append(chunk)
        count += len(chunk)
    chunks.reverse()
    return [s for chunk in chunks for s in chunk], len(lines)
