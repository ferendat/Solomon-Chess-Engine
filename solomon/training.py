import datetime
import os

from .agent import ChessAgent
from .config import CHECKPOINT_DIR
from .env import ChessEnv, play_episode
from .gamelog import load_games, save_game


def run_train(games, depth, model_path, qdepth=0):
    agent = ChessAgent(depth=depth, qdepth=qdepth)
    agent.load(model_path)
    env = ChessEnv()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    games_file = model_path + ".games.jsonl"
    old_samples, old_games = load_games(games_file)
    agent.remember(old_samples)
    print(f"Memory restored: {len(old_samples)} positions from {old_games} stored games")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_text = "n/a"

    for game in range(1, games + 1):
        samples, outcome, plies, moves = play_episode(agent, env)
        save_game(games_file, moves, outcome)
        agent.remember(samples)
        loss = agent.learn()
        if loss is not None:
            loss_text = f"{loss:.4f}"
        agent.decay_epsilon()
        print(f"Game {game}/{games} | plies {plies} | result {outcome:+d} | "
              f"epsilon {agent.epsilon:.3f} | loss {loss_text}")
        if game % 10 == 0:
            agent.save(model_path)
        if game % 100 == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"model_{stamp}_game_{game}.pth"))
    agent.save(model_path)
    agent.save(os.path.join(CHECKPOINT_DIR, f"model_{stamp}_end.pth"))
    print(f"Done. Model saved to {model_path}")
