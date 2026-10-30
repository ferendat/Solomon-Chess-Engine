import chess
import numpy as np

from .config import PIECE_VALUES


def encode(board):
    me = board.turn
    x = np.zeros(768, dtype=np.float32)
    material = 0
    for square, piece in board.piece_map().items():
        mine = piece.color == me
        value = PIECE_VALUES[piece.piece_type]
        material += value if mine else -value
        if me == chess.BLACK:
            square = chess.square_mirror(square)  # always look from the mover's side
        offset = 0 if mine else 384
        x[offset + (piece.piece_type - 1) * 64 + square] = 1.0
    return x, float(material)


def white_material_balance(board):
    total = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        total += value if piece.color == chess.WHITE else -value
    return total


def real_outcome(board):
    # (done, result) with result +1 white wins, -1 black wins, 0 draw
    if board.is_checkmate():
        return True, (-1 if board.turn == chess.WHITE else 1)
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_repetition(3) or board.halfmove_clock >= 100):
        return True, 0
    return False, 0
