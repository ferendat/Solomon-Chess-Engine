import math
import random
import time

import chess
import chess.engine

from .agent import ChessAgent
from .board import real_outcome, white_material_balance
from .config import ADJUDICATE_MARGIN, MAX_PLIES


def run_bench(games, depth, model_path, stockfish_path=None, elo=1320, qdepth=None):
    agent = ChessAgent(depth=depth, qdepth=qdepth)
    agent.load(model_path)

    engine = None
    if stockfish_path:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        except Exception as e:
            print(f"Could not set Stockfish strength: {e}")
    opponent_name = f"Stockfish (Elo {elo})" if engine else "random mover"

    wins = draws = losses = 0
    for game in range(games):
        agent_color = chess.WHITE if game % 2 == 0 else chess.BLACK
        board = chess.Board()
        outcome = 0
        while True:
            done, outcome = real_outcome(board)
            if done or len(board.move_stack) >= 300:
                if not done:
                    outcome = 0
                break
            if board.turn == agent_color:
                move = agent.choose_move(board, explore=False)
            elif engine:
                move = engine.play(board, chess.engine.Limit(time=0.05)).move
            else:
                move = random.choice(list(board.legal_moves))
            board.push(move)
        agent_result = outcome if agent_color == chess.WHITE else -outcome
        wins += agent_result == 1
        losses += agent_result == -1
        draws += agent_result == 0
        print(f"Game {game + 1}/{games}: {'win' if agent_result == 1 else 'loss' if agent_result == -1 else 'draw'}")

    if engine:
        engine.quit()
    score = (wins + 0.5 * draws) / games
    print(f"\nvs {opponent_name}: +{wins} ={draws} -{losses} (score {score:.0%})")
    if engine:
        clipped = min(max(score, 0.02), 0.98)
        estimate = elo + 400 * math.log10(clipped / (1 - clipped))
        print(f"Rough rating estimate: {estimate:.0f} (noisy with few games)")


def run_compare(path_a, path_b, games=20, depth=2, qdepth_a=None, qdepth_b=None):
    # use "fresh" as a path for an untrained model
    agents = []
    for path, qdepth in ((path_a, qdepth_a), (path_b, qdepth_b)):
        agent = ChessAgent(depth=depth, qdepth=qdepth)
        if path != "fresh":
            agent.load(path)
        agents.append(agent)

    wins = draws = losses = 0
    start = time.time()
    for game in range(games):
        a_color = chess.WHITE if game % 2 == 0 else chess.BLACK
        board = chess.Board()
        while True:
            done, outcome = real_outcome(board)
            if not done and len(board.move_stack) >= MAX_PLIES:
                diff = white_material_balance(board)
                outcome = 1 if diff >= ADJUDICATE_MARGIN else -1 if diff <= -ADJUDICATE_MARGIN else 0
                done = True
            if done:
                break
            if len(board.move_stack) < 4:
                move = random.choice(list(board.legal_moves))
            else:
                agent = agents[0] if board.turn == a_color else agents[1]
                move = agent.choose_move(board, explore=False)
            board.push(move)
        a_result = outcome if a_color == chess.WHITE else -outcome
        wins += a_result == 1
        losses += a_result == -1
        draws += a_result == 0
        running = (wins + 0.5 * draws) / (game + 1)
        print(f"Game {game + 1}/{games}: A {'won' if a_result == 1 else 'lost' if a_result == -1 else 'drew'}"
              f" | A so far: +{wins} ={draws} -{losses} ({running:.0%}) | {time.time() - start:.0f}s")

    score = (wins + 0.5 * draws) / games
    print(f"\nA = {path_a}   B = {path_b}")
    print(f"quiescence depth: A={agents[0].qdepth}  B={agents[1].qdepth}")
    print(f"A: +{wins} ={draws} -{losses} (score {score:.0%})")
