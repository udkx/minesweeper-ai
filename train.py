"""
Обучение сети. Запуск:

    python dataset.py make --positions 400000 --out data_train.npz --seed 0
    python dataset.py make --positions 40000  --out data_val.npz   --seed 999
    python train.py --train data_train.npz --val data_val.npz --epochs 12

На M1 Pro сам собой выберется MPS (это GPU, не Neural Engine).

Про метрики, потому что «лосс падает» — плохой критерий:

  loss           кросс-энтропия только по закрытым клеткам. У открытых ответ
                 известен заранее, и включать их — значит бесплатно улучшать
                 цифру, ничего не улучшая по существу.

  safe-pick      доля позиций, где клетка с минимальной предсказанной
                 вероятностью действительно НЕ мина. Это и есть игровое
                 качество: сеть ровно так и будет выбирать ход. Метрика,
                 по которой имеет смысл сравнивать модели.

  калибровка     сеть обещала 30% — мины оказались в 30% случаев? Здесь и
                 проверяется главное утверждение всей затеи: метки жёсткие,
                 нули и единицы, но кросс-энтропия минимальна тогда, когда
                 модель выдаёт ЧАСТОТУ. Значит сеть обязана прийти к
                 вероятностям, которых в разметке не было ни разу.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load
from model import MineNet, encode, pick_device


def batches(n: int, size: int, rng: np.random.Generator, shuffle: bool = True):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


@torch.no_grad()
def evaluate(net, views_t, mines_t, n_mines: int, device, batch: int = 512):
    """Лосс, доля безопасных выборов и калибровка на валидации."""
    net.eval()
    total_loss, total_cells = 0.0, 0
    safe_picks, n_positions = 0, 0
    bins = np.zeros(10)
    bin_hits = np.zeros(10)
    bin_pred = np.zeros(10)

    for i in range(0, len(views_t), batch):
        v = views_t[i:i + batch].to(device)
        m = mines_t[i:i + batch].to(device).float()
        logits = net(encode(v, n_mines))
        mask = v == -1
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], m[mask], reduction="sum")
        total_loss += loss.item()
        total_cells += int(mask.sum())

        probs = torch.sigmoid(logits)
        # ход сети: минимальная вероятность среди закрытых клеток
        masked = torch.where(mask, probs, torch.full_like(probs, 2.0))
        flat = masked.flatten(1)
        pick = flat.argmin(dim=1)
        picked_mine = m.flatten(1)[torch.arange(len(pick), device=device), pick]
        has_closed = mask.flatten(1).any(dim=1)
        safe_picks += int(((picked_mine == 0) & has_closed).sum())
        n_positions += int(has_closed.sum())

        p = probs[mask].cpu().numpy()
        y = m[mask].cpu().numpy()
        b_idx = np.clip((p * 10).astype(int), 0, 9)
        np.add.at(bins, b_idx, 1)
        np.add.at(bin_hits, b_idx, y)
        np.add.at(bin_pred, b_idx, p)

    return {
        "loss": total_loss / max(total_cells, 1),
        "safe_pick": safe_picks / max(n_positions, 1),
        "calib": [(bin_pred[i] / bins[i], bin_hits[i] / bins[i], int(bins[i]))
                  for i in range(10) if bins[i] > 50],
    }


def print_calibration(calib) -> None:
    print("      предсказано → на самом деле мин   (клеток)")
    for pred, actual, n in calib:
        off = actual - pred
        flag = "" if abs(off) < 0.02 else ("  завышает" if off < 0 else "  занижает")
        print(f"        {pred:5.1%}    →   {actual:5.1%}   ({n:>8d}){flag}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data_train.npz")
    p.add_argument("--val", default="data_val.npz")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="minenet.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="взять только N примеров")
    args = p.parse_args()

    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    tr = load(args.train)
    va = load(args.val)
    assert tr["n_mines"] == va["n_mines"] and tr["height"] == va["height"], \
        "обучение и валидация с разных уровней сложности"
    n_mines = tr["n_mines"]

    if args.limit:
        for d in (tr,):
            d["views"] = d["views"][:args.limit]
            d["mines"] = d["mines"][:args.limit]

    views_tr = torch.as_tensor(tr["views"])
    mines_tr = torch.as_tensor(tr["mines"])
    views_va = torch.as_tensor(va["views"])
    mines_va = torch.as_tensor(va["mines"])

    net = MineNet(args.channels, args.blocks).to(device)
    print(f"устройство: {device}, параметров: {net.n_params:,}, "
          f"обзор: ±{net.receptive_field} клеток")
    print(f"обучение: {len(views_tr):,} позиций, валидация: {len(views_va):,}, "
          f"поле {tr['height']}x{tr['width']} ({tr['difficulty']})")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(views_tr) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps, pct_start=0.25)

    # Базовый уровень: предсказывать всем закрытым клеткам одну и ту же общую
    # плотность мин. С этим и сравниваем — если сеть не бьёт эту цифру, она
    # не выучила ничего, кроме статистики.
    closed_va = views_va == -1
    base_rate = float((mines_va & closed_va).sum() / closed_va.sum())
    base_loss = -(base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate))
    print(f"базовый уровень (всегда {base_rate:.1%}): loss {base_loss:.4f}\n")

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.perf_counter()
        run_loss, run_cells, seen = 0.0, 0, 0
        for bi, idx in enumerate(batches(len(views_tr), args.batch, rng)):
            v = views_tr[idx].to(device)
            m = mines_tr[idx].to(device).float()
            logits = net(encode(v, n_mines))
            mask = v == -1
            loss = F.binary_cross_entropy_with_logits(logits[mask], m[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * int(mask.sum())
            run_cells += int(mask.sum())
            seen += len(idx)
            if bi % 200 == 0:
                print(f"  эпоха {epoch}  {seen}/{len(views_tr)}  "
                      f"loss {run_loss / max(run_cells,1):.4f}", flush=True)
        train_loss = run_loss / max(run_cells, 1)
        ev = evaluate(net, views_va, mines_va, n_mines, device)
        dt = time.perf_counter() - t0
        print(f"эпоха {epoch:2d}: train {train_loss:.4f}  val {ev['loss']:.4f}  "
              f"safe-pick {ev['safe_pick']:.2%}  ({dt:.0f} с)")
        if ev["loss"] < best:
            best = ev["loss"]
            net.save(args.out, meta={
                "difficulty": tr["difficulty"], "n_mines": n_mines,
                "height": tr["height"], "width": tr["width"],
                "val_loss": ev["loss"], "safe_pick": ev["safe_pick"],
                "epoch": epoch,
            })
            print(f"           сохранено в {args.out}")

    print("\nкалибровка лучшей модели:")
    net, _ = MineNet.load(args.out, device)
    ev = evaluate(net, views_va, mines_va, n_mines, device)
    print_calibration(ev["calib"])
    print(f"\nитог: val loss {ev['loss']:.4f} против базового {base_loss:.4f}, "
          f"safe-pick {ev['safe_pick']:.2%}")


if __name__ == "__main__":
    main()
