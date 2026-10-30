import chess

from .agent import ChessAgent
from .board import real_outcome
from .config import MODEL_FILE
from .gui import ChessGUI


def run_play_notebook(depth=2, model_path=MODEL_FILE, human_color=chess.WHITE, qdepth=None):
    import chess.svg
    from IPython.display import SVG, display, clear_output

    agent = ChessAgent(depth=depth, qdepth=qdepth)
    agent.load(model_path)
    board = chess.Board()
    last_move = None
    message = ""

    while True:
        clear_output(wait=True)
        display(SVG(chess.svg.board(board, lastmove=last_move, size=420, orientation=human_color)))
        if message:
            print(message)
            message = ""

        done, outcome = real_outcome(board)
        if done:
            print({1: "Game over - White wins", -1: "Game over - Black wins", 0: "Game over - Draw"}[outcome])
            return

        if board.turn == human_color:
            text = input("Your move (e4, Nf3, e2e4... or 'quit'): ").strip()
            if text.lower() in ("quit", "q", "exit"):
                return
            try:
                move = board.parse_san(text)
            except ValueError:
                try:
                    move = chess.Move.from_uci(text.lower())
                except ValueError:
                    move = None
            if move is None or move not in board.legal_moves:
                message = f"'{text}' is not a legal move"
                continue
            board.push(move)
            last_move = move
        else:
            print("AI is thinking...")
            move = agent.choose_move(board.copy(), explore=False)
            board.push(move)
            last_move = move
            message = f"AI played {move.uci()}"


def run_play(depth, model_path, qdepth=None):
    try:
        import tkinter
        tkinter.Tk().destroy()
    except Exception:
        print("No display found, using the notebook board")
        run_play_notebook(depth, model_path, qdepth=qdepth)
        return
    agent = ChessAgent(depth=depth, qdepth=qdepth)
    agent.load(model_path)
    ChessGUI(agent)
