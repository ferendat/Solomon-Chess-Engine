import threading

import chess

from .board import real_outcome


class ChessGUI:
    SQ = 80
    PIECES = {
        "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
        "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    }

    def __init__(self, agent):
        import tkinter as tk
        self.agent = agent
        self.human = chess.WHITE
        self.board = chess.Board()
        self.selected = None
        self.last_move = None
        self.thinking = False
        self.game_id = 0
        self.pending = None

        self.root = tk.Tk()
        self.root.title("Solomon")
        self.root.resizable(False, False)
        self.status = tk.Label(self.root, text="", font=("Arial", 12, "bold"))
        self.status.pack(pady=(8, 4))
        self.canvas = tk.Canvas(self.root, width=self.SQ * 8, height=self.SQ * 8)
        self.canvas.pack(padx=10)
        self.canvas.bind("<Button-1>", self.on_click)
        buttons = tk.Frame(self.root)
        buttons.pack(pady=8)
        tk.Button(buttons, text="New game as White",
                  command=lambda: self.new_game(chess.WHITE)).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="New game as Black",
                  command=lambda: self.new_game(chess.BLACK)).pack(side=tk.LEFT, padx=4)

        self.draw()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.mainloop()

    def square_at(self, col, row):
        if self.human == chess.WHITE:
            return chess.square(col, 7 - row)
        return chess.square(7 - col, row)

    def draw(self):
        s = self.SQ
        self.canvas.delete("all")
        targets = set()
        if self.selected is not None:
            targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}
        for row in range(8):
            for col in range(8):
                square = self.square_at(col, row)
                color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
                if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
                    color = "#cdd26a"
                if square == self.selected:
                    color = "#ffff66"
                self.canvas.create_rectangle(col * s, row * s, (col + 1) * s, (row + 1) * s,
                                             fill=color, outline="")
                if square in targets:
                    cx, cy = col * s + s // 2, row * s + s // 2
                    self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill="#6b8e4e", outline="")
                piece = self.board.piece_at(square)
                if piece:
                    self.canvas.create_text(col * s + s // 2, row * s + s // 2,
                                            text=self.PIECES[piece.symbol()],
                                            font=("Arial", 44), fill="black")
        self.status.config(text=self.status_text())

    def status_text(self):
        done, outcome = real_outcome(self.board)
        if done:
            return {1: "Game over - White wins", -1: "Game over - Black wins", 0: "Game over - Draw"}[outcome]
        if self.thinking:
            return "AI is thinking..."
        side = "White" if self.human == chess.WHITE else "Black"
        return f"Your move ({side})" if self.board.turn == self.human else "AI to move"

    def find_move(self, from_square, to_square):
        for promotion in (None, chess.QUEEN):
            move = chess.Move(from_square, to_square, promotion=promotion)
            if move in self.board.legal_moves:
                return move
        return None

    def on_click(self, event):
        if self.thinking or self.board.turn != self.human or real_outcome(self.board)[0]:
            return
        col, row = event.x // self.SQ, event.y // self.SQ
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        square = self.square_at(col, row)
        piece = self.board.piece_at(square)

        if self.selected is None:
            if piece and piece.color == self.human:
                self.selected = square
        else:
            move = self.find_move(self.selected, square)
            if move:
                self.board.push(move)
                self.last_move = move
                self.selected = None
                self.draw()
                self.start_ai_move()
                return
            self.selected = square if (piece and piece.color == self.human) else None
        self.draw()

    def start_ai_move(self):
        if real_outcome(self.board)[0]:
            self.draw()
            return
        self.thinking = True
        self.pending = None
        game_id = self.game_id
        board_copy = self.board.copy()
        threading.Thread(target=self.think, args=(board_copy, game_id), daemon=True).start()
        self.root.after(100, self.poll_ai, game_id)
        self.draw()

    def think(self, board, game_id):
        self.pending = (game_id, self.agent.choose_move(board, explore=False))

    def poll_ai(self, game_id):
        if game_id != self.game_id:
            return
        if self.pending is None:
            self.root.after(100, self.poll_ai, game_id)
            return
        _, move = self.pending
        self.pending = None
        self.thinking = False
        if move is not None:
            self.board.push(move)
            self.last_move = move
        self.draw()

    def new_game(self, color=None):
        self.game_id += 1
        if color is not None:
            self.human = color
        self.board = chess.Board()
        self.selected = None
        self.last_move = None
        self.thinking = False
        self.pending = None
        self.draw()
        if self.board.turn != self.human:
            self.start_ai_move()
