import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .board import encode
from .config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
    MATERIAL_SCALE,
    MATE_SCORE,
    MAX_CORRECTION,
    MODEL_FILE,
    PIECE_VALUES,
    QUIESCENCE_DEPTH,
    UPDATES_PER_GAME,
)
from .model import ChessNet

torch.set_num_threads(1)


class ChessAgent:
    def __init__(self, depth=1, qdepth=None):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.depth = depth
        self.qdepth = QUIESCENCE_DEPTH if qdepth is None else qdepth
        self.model = ChessNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START
        self.buffer = []
        self.sync_weights()

    def sync_weights(self):
        # numpy copy of the weights, the search uses this because torch calls are slow one by one
        layers = [self.model.body[0], self.model.body[2], self.model.out]
        self.np_weights = []
        for layer in layers:
            self.np_weights.append(layer.weight.detach().numpy().T.copy())
            self.np_weights.append(layer.bias.detach().numpy().copy())

    def evaluate(self, board):
        x, material = encode(board)
        w1, b1, w2, b2, w3, b3 = self.np_weights
        h = np.maximum(x @ w1 + b1, 0.0)
        h = np.maximum(h @ w2 + b2, 0.0)
        correction = MAX_CORRECTION * math.tanh(float((h @ w3 + b3)[0]))
        return math.tanh(material / MATERIAL_SCALE + correction)

    @staticmethod
    def ordered_moves(board):
        return sorted(board.legal_moves,
                      key=lambda m: board.is_capture(m) or m.promotion is not None,
                      reverse=True)

    def negamax(self, board, depth, alpha, beta, ply):
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if ply > 0 and board.is_repetition(2):
            return 0.0
        if depth == 0:
            if self.qdepth == 0:
                return self.evaluate(board)
            return self.quiescence(board, alpha, beta, self.qdepth, ply)

        best = -math.inf
        for move in self.ordered_moves(board):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    @staticmethod
    def capture_value(board, move):
        victim = board.piece_type_at(move.to_square)
        value = PIECE_VALUES[victim] if victim else 1  # en passant
        return value + (8 if move.promotion else 0)

    def quiescence(self, board, alpha, beta, qdepth, ply):
        moves = list(board.legal_moves)
        in_check = board.is_check()
        if not moves:
            return -MATE_SCORE + ply if in_check else 0.0
        if qdepth == 0:
            return self.evaluate(board)

        if in_check:
            best = -math.inf
            candidates = moves
        else:
            best = self.evaluate(board)
            if best >= beta:
                return best
            alpha = max(alpha, best)
            candidates = [m for m in moves if board.is_capture(m) or m.promotion]
            candidates.sort(key=lambda m: self.capture_value(board, m), reverse=True)

        for move in candidates:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, qdepth - 1, ply + 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def choose_move(self, board, explore=False):
        moves = self.ordered_moves(board)
        if not moves:
            return None
        if explore and random.random() < self.epsilon:
            return random.choice(moves)

        best_score = -math.inf
        best_moves = []
        for move in moves:
            board.push(move)
            score = -self.negamax(board, self.depth - 1, -math.inf, math.inf, 1)
            board.pop()
            if score > best_score + 1e-9:
                best_score, best_moves = score, [move]
            elif abs(score - best_score) <= 1e-9:
                best_moves.append(move)
        return random.choice(best_moves)

    def remember(self, samples):
        self.buffer.extend(samples)
        if len(self.buffer) > BUFFER_SIZE:
            del self.buffer[:len(self.buffer) - BUFFER_SIZE]

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return None
        total = 0.0
        for _ in range(UPDATES_PER_GAME):
            batch = random.sample(self.buffer, BATCH_SIZE)
            x = torch.from_numpy(np.stack([b[0] for b in batch]))
            material = torch.tensor([b[1] for b in batch], dtype=torch.float32)
            target = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            loss = F.mse_loss(self.model(x, material), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        self.sync_weights()
        return total / UPDATES_PER_GAME

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save(self, path=MODEL_FILE):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path=MODEL_FILE):
        if not os.path.exists(path):
            print(f"No model found at {path}, starting fresh")
            return False
        try:
            checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint.get("epsilon", EPSILON_START)
            self.sync_weights()
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with fresh model...")
            return False
