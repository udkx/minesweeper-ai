"""
Агент на обученной сети: играет партии и замеряет винрейт.

Два режима, и разница между ними поучительная:

  pure   — сеть решает всё сама: открывает клетку с минимальной предсказанной
           вероятностью. Честная проверка того, что она выучила.

  hybrid — сначала применяются тривиальные правила (они точные и бесплатные),
           и только когда правила встали, слово даётся сети. Так и надо
           играть на практике: доказанное не надо предсказывать.

Гибрид почти всегда сильнее, и это не поражение нейросети. Это обычная
инженерная мысль: если часть задачи решается точно, решай её точно, а модель
пусти на ту часть, где точного ответа нет.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from batch_engine import BatchBoards
from fast_policy import trivial_deductions
from minesweeper import DIFFICULTIES, UNKNOWN
from model import MineNet, encode, pick_device


@torch.no_grad()
def net_probs(net: MineNet, view: np.ndarray, n_mines: int,
              device: torch.device) -> np.ndarray:
    """Карта вероятностей мин для батча полей, (B, H, W) float32."""
    net.eval()
    v = torch.as_tensor(view, device=device)
    return torch.sigmoid(net(encode(v, n_mines))).float().cpu().numpy()


def choose_moves_net(net, view: np.ndarray, n_mines: int, device,
                     mode: str = "hybrid"):
    """Выбрать по одной клетке на каждой доске батча."""
    b, h, w = view.shape
    probs = net_probs(net, view, n_mines, device)
    score = probs.copy()

    if mode == "hybrid":
        known_mines, known_safe = trivial_deductions(view)
        score = np.where(known_safe, -1.0, score)
        score = np.where(known_mines, 2.0, score)

    invalid = view != UNKNOWN
    flat = np.where(invalid.reshape(b, -1), np.inf, score.reshape(b, -1))
    pick = np.argmin(flat, axis=1)
    return (pick // w).astype(np.int32), (pick % w).astype(np.int32), probs


def evaluate_winrate(net, difficulty: str, n_games: int, device,
                     mode: str = "hybrid", batch: int = 256, seed: int = 0,
                     verbose: bool = True):
    """Сыграть n_games партий батчами и посчитать винрейт.

    Партии идут параллельно; как только доска закончилась, результат
    записывается, а доска перезапускается — батч всё время полный.
    """
    h, w, m = DIFFICULTIES[difficulty]
    rng = np.random.default_rng(seed)
    boards = BatchBoards(min(batch, n_games * 2), h, w, m, rng=rng)

    wins, losses = 0, 0
    cleared: list[float] = []
    safe_total = h * w - m
    t0 = time.perf_counter()

    while wins + losses < n_games:
        view = boards.view()
        r, c, _ = choose_moves_net(net, view, m, device, mode)
        boards.reveal(r, c)

        done = np.flatnonzero(boards.over())
        if done.size:
            won_mask = boards.won()[done]
            wins += int(won_mask.sum())
            losses += int((~won_mask).sum())
            cleared.extend((boards.revealed[done].reshape(len(done), -1)
                            .sum(axis=1) / safe_total).tolist())
            boards.reset(done)
            if verbose and (wins + losses) % max(1, n_games // 8) < len(done):
                print(f"    {wins + losses}/{n_games}  винрейт "
                      f"{wins / max(wins + losses, 1):.1%}", flush=True)

    total = wins + losses
    dt = time.perf_counter() - t0
    return {
        "agent": f"net-{mode}",
        "difficulty": difficulty,
        "games": total,
        "win_rate": wins / total,
        "avg_cleared": float(np.mean(cleared)),
        "sec_per_game": dt / total,
    }


def make_single_agent(model_path: str, device=None, mode: str = "hybrid"):
    """Агент для play.py: функция (view, n_mines) -> (r, c, карта вероятностей).

    Нужен, чтобы записать партию сети и посмотреть её в браузере тем же
    визуализатором, которым смотрели решателя.
    """
    device = device or pick_device()
    net, meta = MineNet.load(model_path, device)

    def agent(view: np.ndarray, n_mines: int):
        r, c, probs = choose_moves_net(net, view[None], n_mines, device, mode)
        prob_map = np.where(view == UNKNOWN, probs[0], np.nan)
        return int(r[0]), int(c[0]), prob_map

    agent.meta = meta
    return agent


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="minenet.pt")
    p.add_argument("--difficulty", default="intermediate", choices=list(DIFFICULTIES))
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--mode", default="both", choices=["pure", "hybrid", "both"])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    net, meta = MineNet.load(args.model, device)
    print(f"модель: {args.model} (эпоха {meta.get('epoch')}, "
          f"val loss {meta.get('val_loss', float('nan')):.4f})")
    if meta.get("difficulty") and meta["difficulty"] != args.difficulty:
        print(f"  внимание: модель обучена на {meta['difficulty']}, "
              f"играем на {args.difficulty} — свёрточной сети размер поля "
              f"не мешает, но плотность мин другая")

    modes = ["pure", "hybrid"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"\n{mode}:")
        r = evaluate_winrate(net, args.difficulty, args.games, device,
                             mode, args.batch)
        print(f"  {r['agent']:>12} / {r['difficulty']:<13} "
              f"винрейт {r['win_rate']:6.1%}   "
              f"открыто в среднем {r['avg_cleared']:5.1%}   "
              f"{r['sec_per_game'] * 1000:7.1f} мс/партия")


if __name__ == "__main__":
    main()
