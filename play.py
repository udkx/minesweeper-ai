"""
Прогон партий и замер винрейта. Плюс запись партии в JSON для веб-визуализации.

Запуск:
    python play.py bench --difficulty beginner --games 500
    python play.py bench --all --games 200
    python play.py record --difficulty expert --out replay.json

«Агент» здесь — просто функция (view, n_mines) -> (r, c, prob_map|None).
Такой же интерфейс потом будет у нейросети, поэтому сравнивать их можно
буквально одной и той же командой.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from minesweeper import DIFFICULTIES, Minesweeper
from solver import choose_move


def solver_agent(view: np.ndarray, n_mines: int):
    r, c, a = choose_move(view, n_mines)
    return r, c, a.prob


def random_agent(view: np.ndarray, n_mines: int):
    """Тупейший базовый уровень: жмём случайную закрытую клетку.

    Нужен как нижняя граница. Если нейросеть не бьёт этого — что-то не так
    с обучением, а не с задачей.
    """
    from minesweeper import UNKNOWN
    cells = np.argwhere(view == UNKNOWN)
    r, c = cells[np.random.default_rng().integers(len(cells))]
    return int(r), int(c), None


AGENTS = {"solver": solver_agent, "random": random_agent}


def play_game(agent, game: Minesweeper, record: bool = False,
              max_moves: int = 5000):
    """Одна партия. Если record=True, сохраняем ходы и карты вероятностей.

    В записи нет самих открытых полей — только список ходов. Визуализатор
    в браузере переигрывает партию своим движком по тем же правилам.
    Это и файл уменьшает на порядок, и заодно проверяет, что JS-движок
    ведёт себя точно как Python-движок: если правила расходятся,
    воспроизведение сломается заметным образом.
    """
    moves, probs = [], []
    while not game.over and game.n_clicks < max_moves:
        view = game.view()
        r, c, prob = agent(view, game.n_mines)
        if record:
            moves.append([int(r), int(c)])
            probs.append(None if prob is None else
                         [[None if np.isnan(x) else round(float(x), 4) for x in row]
                          for row in prob])
        game.reveal(r, c)
    return game, {"moves": moves, "probs": probs}


def benchmark(agent_name: str, difficulty: str, n_games: int, seed: int = 0,
              verbose: bool = True):
    h, w, m = DIFFICULTIES[difficulty]
    agent = AGENTS[agent_name]
    rng = np.random.default_rng(seed)
    wins = 0
    cleared = []
    t0 = time.perf_counter()
    for i in range(n_games):
        game = Minesweeper(h, w, m, rng=rng)
        play_game(agent, game)
        wins += game.won
        # доля безопасных клеток, которые удалось открыть — более мягкая метрика,
        # чем винрейт: показывает прогресс даже когда побед ещё почти нет
        cleared.append(game.revealed.sum() / (h * w - m))
        if verbose and (i + 1) % max(1, n_games // 10) == 0:
            print(f"    {i + 1}/{n_games}  винрейт {wins / (i + 1):.1%}", flush=True)
    dt = time.perf_counter() - t0
    result = {
        "agent": agent_name,
        "difficulty": difficulty,
        "games": n_games,
        "win_rate": wins / n_games,
        "avg_cleared": float(np.mean(cleared)),
        "sec_per_game": dt / n_games,
        "games_per_sec": n_games / dt,
    }
    return result


def print_result(r: dict) -> None:
    print(f"  {r['agent']:>8} / {r['difficulty']:<13} "
          f"винрейт {r['win_rate']:6.1%}   "
          f"открыто в среднем {r['avg_cleared']:5.1%}   "
          f"{r['sec_per_game'] * 1000:8.2f} мс/партия")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench")
    b.add_argument("--agent", default="solver", choices=list(AGENTS))
    b.add_argument("--difficulty", default="beginner", choices=list(DIFFICULTIES))
    b.add_argument("--games", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--all", action="store_true", help="все три уровня сложности")
    b.add_argument("--json", help="куда сохранить результаты")

    rec = sub.add_parser("record")
    rec.add_argument("--agent", default="solver", choices=list(AGENTS))
    rec.add_argument("--difficulty", default="expert", choices=list(DIFFICULTIES))
    rec.add_argument("--seed", type=int, default=0)
    rec.add_argument("--out", default="replay.json")
    rec.add_argument("--want-win", action="store_true",
                     help="искать партию, которая закончилась победой")
    rec.add_argument("--min-guesses", type=int, default=0,
                     help="искать партию, где решателю пришлось угадывать хотя бы N раз "
                          "(такие партии интереснее смотреть)")

    args = p.parse_args()

    if args.cmd == "bench":
        levels = list(DIFFICULTIES) if args.all else [args.difficulty]
        results = []
        for lvl in levels:
            print(f"\n{lvl}:")
            r = benchmark(args.agent, lvl, args.games, args.seed)
            results.append(r)
        print()
        for r in results:
            print_result(r)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    elif args.cmd == "record":
        h, w, m = DIFFICULTIES[args.difficulty]
        rng = np.random.default_rng(args.seed)
        def count_guesses(rec: dict) -> int:
            """Ход считается угадыванием, если у выбранной клетки риск > 0."""
            n = 0
            for (r, c), pm in zip(rec["moves"], rec["probs"]):
                if pm and pm[r][c]:
                    n += 1
            return n

        for attempt in range(300):
            game = Minesweeper(h, w, m, rng=rng)
            game, rec = play_game(AGENTS[args.agent], game, record=True)
            ok = (not args.want_win or game.won) and count_guesses(rec) >= args.min_guesses
            if ok:
                break
        else:
            print("подходящую партию не нашёл, беру последнюю")
        payload = {
            "difficulty": args.difficulty,
            "height": h, "width": w, "n_mines": m,
            "agent": args.agent,
            "mines": game.mines.astype(int).tolist(),
            "won": bool(game.won),
            "moves": rec["moves"],
            "probs": rec["probs"],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        print(f"записано {len(rec['moves'])} ходов в {args.out} "
              f"({'победа' if game.won else 'поражение'}, "
              f"угадываний: {count_guesses(rec)})")


if __name__ == "__main__":
    main()
