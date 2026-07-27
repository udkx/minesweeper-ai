"""
Сбор датасета: позиции из настоящих партий + правда о минах.

Запуск:
    python dataset.py make --difficulty intermediate --positions 400000 --out data_train.npz
    python dataset.py make --difficulty intermediate --positions 40000  --out data_val.npz --seed 999
    python dataset.py stats data_train.npz

Что такое одна обучающая пара:
    вход  — поле глазами игрока, (H, W), int8: -1 закрыто, 0..8 цифра
    ответ — где на самом деле мины, (H, W), bool

Обрати внимание, чего здесь НЕТ: вероятностей. Метки жёсткие, ноль или один.
Вероятности возникнут сами — об этом подробно в README.

Две вещи, которые легко сделать неправильно:

1. Позиции из одной партии идут подряд и почти совпадают (отличаются на одну
   открытую клетку). Если ссыпать их в датасет как есть, «400 тысяч примеров»
   окажутся четырьмя тысячами партий, размазанными по батчам, и модель будет
   переобучаться на них, показывая при этом отличную валидацию — потому что
   валидация набрана так же. Поэтому берём из партии не все позиции, а
   случайную часть, и в конце всё перемешиваем.

2. Обучающая и валидационная выборки должны идти от РАЗНЫХ партий, а не от
   разных позиций одних и тех же партий. Отсюда отдельный --seed для валидации.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from batch_engine import BatchBoards, shift_sum
from fast_policy import choose_moves
from minesweeper import DIFFICULTIES, UNKNOWN


def make_dataset(difficulty: str, n_positions: int, batch: int = 1024,
                 keep_prob: float = 0.15, stuck_boost: float = 1.0,
                 seed: int = 0, epsilon: float = 0.1, verbose: bool = True):
    h, w, m = DIFFICULTIES[difficulty]
    rng = np.random.default_rng(seed)
    boards = BatchBoards(batch, h, w, m, rng=rng)

    views: list[np.ndarray] = []
    mines: list[np.ndarray] = []
    stuck_flags: list[np.ndarray] = []
    collected = 0
    steps = 0
    games_finished = 0
    t0 = time.perf_counter()

    while collected < n_positions:
        view = boards.view()
        r, c, known_safe = choose_moves(view, m, rng, epsilon=epsilon)

        live = boards.started & ~boards.dead & ~boards.won()
        # «Застряла» — значит тривиальные правила не нашли ни одной безопасной
        # клетки, и политика вынуждена угадывать. Именно в таких позициях от
        # нейросети будет реальная польза, поэтому их полезно уметь набирать чаще.
        stuck = live & ~known_safe.reshape(batch, -1).any(axis=1)

        p = np.where(stuck, min(1.0, keep_prob * stuck_boost), keep_prob)
        take = live & (rng.random(batch) < p)
        idx = np.flatnonzero(take)
        if idx.size:
            need = min(idx.size, n_positions - collected)
            idx = idx[:need]
            views.append(view[idx].copy())
            mines.append(boards.mines[idx].copy())
            stuck_flags.append(stuck[idx].copy())
            collected += idx.size

        boards.reveal(r, c)
        games_finished += len(boards.reset_finished())
        steps += 1
        if verbose and steps % 100 == 0:
            rate = collected / (time.perf_counter() - t0)
            print(f"    {collected}/{n_positions} позиций, "
                  f"{games_finished} партий, {rate:.0f} позиций/с", flush=True)

    dt = time.perf_counter() - t0
    view_arr = np.concatenate(views)[:n_positions]
    mine_arr = np.concatenate(mines)[:n_positions]
    stuck_arr = np.concatenate(stuck_flags)[:n_positions]

    # Перемешиваем: иначе в файле сначала лежат все ранние позиции партий,
    # и первые батчи обучения будут состоять только из начал игры.
    order = rng.permutation(len(view_arr))
    if verbose:
        print(f"  собрано {len(view_arr)} позиций из ~{games_finished} партий "
              f"за {dt:.0f} с ({len(view_arr) / dt:.0f} позиций/с)")
    return {
        "views": view_arr[order],
        "mines": mine_arr[order],
        "stuck": stuck_arr[order],
        "height": h, "width": w, "n_mines": m, "difficulty": difficulty,
    }


def save(data: dict, path: str) -> None:
    """Сохранить компактно: правда о минах — это биты, а не байты."""
    np.savez_compressed(
        path,
        views=data["views"],
        mines_packed=np.packbits(data["mines"].reshape(len(data["views"]), -1), axis=1),
        stuck=np.packbits(data["stuck"]),
        n_samples=len(data["views"]),
        height=data["height"], width=data["width"],
        n_mines=data["n_mines"], difficulty=data["difficulty"],
    )


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    n = int(z["n_samples"])
    h, w = int(z["height"]), int(z["width"])
    mines = np.unpackbits(z["mines_packed"], axis=1, count=h * w).astype(bool)
    return {
        "views": z["views"],
        "mines": mines.reshape(n, h, w),
        "stuck": np.unpackbits(z["stuck"], count=n).astype(bool),
        "height": h, "width": w, "n_mines": int(z["n_mines"]),
        "difficulty": str(z["difficulty"]),
    }


def verify(data: dict, n_check: int = 2000) -> None:
    """Проверить целостность датасета.

    Главная проверка: каждая открытая цифра должна равняться числу мин вокруг
    по сохранённой карте мин. Если движок, сбор данных или упаковка бит где-то
    рассинхронизировались, это вылезет здесь, а не через час обучения.
    """
    views, mines = data["views"], data["mines"]
    n = min(n_check, len(views))
    sub_v, sub_m = views[:n], mines[:n]
    counts = shift_sum(sub_m)
    revealed = sub_v >= 0
    assert np.array_equal(sub_v[revealed], counts[revealed]), \
        "цифры на поле не соответствуют минам"
    assert not (revealed & sub_m).any(), "открытая клетка оказалась миной"
    per_board = sub_m.reshape(n, -1).sum(axis=1)
    assert (per_board == data["n_mines"]).all(), "неверное число мин на доске"
    print(f"  целостность: ок ({n} позиций проверено)")


def stats(data: dict) -> None:
    views, mines, stuck = data["views"], data["mines"], data["stuck"]
    n, h, w = views.shape
    unknown = views == UNKNOWN
    total_unknown = int(unknown.sum())
    mines_in_unknown = int((unknown & mines).sum())
    revealed_frac = (views >= 0).reshape(n, -1).mean(axis=1)
    safe_total = h * w - data["n_mines"]
    progress = (views >= 0).reshape(n, -1).sum(axis=1) / safe_total

    print(f"  позиций: {n}, поле {h}x{w}, мин {data['n_mines']} ({data['difficulty']})")
    print(f"  доля мин среди закрытых клеток: {mines_in_unknown / total_unknown:.3f} "
          f"(это и есть базовая вероятность, которую сеть должна уметь превзойти)")
    print(f"  закрытых клеток на позицию: {total_unknown / n:.1f} из {h * w}")
    print(f"  позиций, где тривиальные правила застряли: {stuck.mean():.1%}")
    print("  распределение по стадии партии (доля открытых безопасных клеток):")
    edges = [0, .1, .25, .5, .75, .9, 1.01]
    hist, _ = np.histogram(progress, bins=edges)
    for lo, hi, cnt in zip(edges[:-1], edges[1:], hist):
        bar = "█" * int(40 * cnt / max(hist.max(), 1))
        print(f"    {lo:4.0%}-{min(hi,1):4.0%}  {cnt:7d}  {bar}")
    # Насколько позиции разнообразны: сколько уникальных полей среди первых 20k
    sample = views[:20000].reshape(min(20000, n), -1)
    uniq = len(np.unique(sample, axis=0))
    print(f"  уникальных полей среди первых {len(sample)}: {uniq} "
          f"({uniq / len(sample):.1%}) — если сильно меньше 100%, "
          f"позиции скоррелированы")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make")
    mk.add_argument("--difficulty", default="intermediate", choices=list(DIFFICULTIES))
    mk.add_argument("--positions", type=int, default=400_000)
    mk.add_argument("--batch", type=int, default=1024)
    mk.add_argument("--keep-prob", type=float, default=0.15,
                    help="с какой вероятностью брать позицию (меньше = менее "
                         "скоррелированные данные, но дольше сбор)")
    mk.add_argument("--stuck-boost", type=float, default=1.0,
                    help="во сколько раз чаще брать позиции, где правила застряли")
    mk.add_argument("--epsilon", type=float, default=0.1)
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--out", default="data_train.npz")

    st = sub.add_parser("stats")
    st.add_argument("path")

    args = p.parse_args()
    if args.cmd == "make":
        data = make_dataset(args.difficulty, args.positions, args.batch,
                            args.keep_prob, args.stuck_boost, args.seed, args.epsilon)
        verify(data)
        stats(data)
        save(data, args.out)
        import os
        print(f"  сохранено: {args.out} ({os.path.getsize(args.out) / 1e6:.0f} МБ)")
    else:
        data = load(args.path)
        verify(data)
        stats(data)


if __name__ == "__main__":
    main()
