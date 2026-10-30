import argparse

from solomon.config import MODEL_FILE
from solomon.evaluation import run_bench, run_compare
from solomon.play import run_play
from solomon.training import run_train


def main():
    parser = argparse.ArgumentParser(description="Solomon, a self-taught chess engine")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--games", type=int, default=500)
    p_train.add_argument("--depth", type=int, default=1)
    p_train.add_argument("--model", default=MODEL_FILE)
    p_train.add_argument("--qdepth", type=int, default=0)

    p_play = sub.add_parser("play")
    p_play.add_argument("--depth", type=int, default=2)
    p_play.add_argument("--model", default=MODEL_FILE)
    p_play.add_argument("--qdepth", type=int, default=None)

    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--games", type=int, default=20)
    p_bench.add_argument("--depth", type=int, default=2)
    p_bench.add_argument("--model", default=MODEL_FILE)
    p_bench.add_argument("--stockfish", default=None)
    p_bench.add_argument("--elo", type=int, default=1320)
    p_bench.add_argument("--qdepth", type=int, default=None)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--a", default=MODEL_FILE)
    p_cmp.add_argument("--b", default="fresh")
    p_cmp.add_argument("--games", type=int, default=20)
    p_cmp.add_argument("--depth", type=int, default=2)
    p_cmp.add_argument("--qa", type=int, default=None)
    p_cmp.add_argument("--qb", type=int, default=None)

    args = parser.parse_args()
    if args.mode == "train":
        run_train(args.games, args.depth, args.model, args.qdepth)
    elif args.mode == "play":
        run_play(args.depth, args.model, args.qdepth)
    elif args.mode == "bench":
        run_bench(args.games, args.depth, args.model, args.stockfish, args.elo, args.qdepth)
    elif args.mode == "compare":
        run_compare(args.a, args.b, args.games, args.depth, args.qa, args.qb)


if __name__ == "__main__":
    main()
