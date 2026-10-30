import chess

from .board import encode, real_outcome, white_material_balance
from .config import ADJUDICATE_MARGIN, MAX_PLIES


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board = chess.Board()
        return self.board

    def step(self, move):
        self.board.push(move)
        done, outcome = real_outcome(self.board)
        if not done and len(self.board.move_stack) >= MAX_PLIES:
            diff = white_material_balance(self.board)
            outcome = 1 if diff >= ADJUDICATE_MARGIN else -1 if diff <= -ADJUDICATE_MARGIN else 0
            done = True
        return done, outcome


def play_episode(agent, env):
    env.reset()
    positions = []
    while True:
        x, material = encode(env.board)
        positions.append((x, material, env.board.turn))
        move = agent.choose_move(env.board, explore=True)
        done, outcome = env.step(move)
        if done:
            break
    samples = []
    for x, material, turn in positions:
        target = outcome if turn == chess.WHITE else -outcome
        samples.append((x, material, float(target)))
    moves = [m.uci() for m in env.board.move_stack]
    return samples, outcome, len(positions), moves
