"""Проверки батчевого движка и быстрой политики. Запуск: python test_batch.py

Самый важный тест здесь — первый: батчевый движок должен вести себя ТОЧНО как
одиночный. Векторизованный каскад нулей написан совсем другим способом
(раздувание масок вместо обхода в глубину), и если он расходится хотя бы на
одной клетке, датасет будет систематически кривым, а найти это потом почти
невозможно.
"""

from __future__ import annotations

import numpy as np

from batch_engine import BatchBoards, shift_sum
from fast_policy import choose_moves, trivial_deductions
from minesweeper import Minesweeper, neighbour_count


def test_batch_matches_single_engine():
    rng = np.random.default_rng(101)
    B, H, W, M = 12, 9, 9, 10

    batch = BatchBoards(B, H, W, M, rng=rng)
    singles = []
    # Одинаковая расстановка мин в обоих движках, поставленная руками.
    for i in range(B):
        mines = np.zeros((H, W), dtype=bool)
        idx = rng.choice(H * W, size=M, replace=False)
        mines.ravel()[idx] = True
        g = Minesweeper(H, W, M, rng=rng)
        g.mines = mines
        g.counts = neighbour_count(mines)
        g.started = True
        singles.append(g)
        batch.mines[i] = mines
    batch.counts = shift_sum(batch.mines)
    batch.started[:] = True

    for step in range(60):
        r = rng.integers(0, H, size=B).astype(np.int32)
        c = rng.integers(0, W, size=B).astype(np.int32)
        active = ~batch.dead & ~batch.won()
        batch.reveal(r, c, active=active)
        for i, g in enumerate(singles):
            if active[i]:
                g.reveal(int(r[i]), int(c[i]))
            assert np.array_equal(batch.revealed[i], g.revealed), (
                f"шаг {step}, доска {i}: открытые клетки разошлись")
            assert bool(batch.dead[i]) == g.dead, f"шаг {step}, доска {i}: разный исход"
        assert np.array_equal(batch.won(), np.array([g.won for g in singles]))


def test_batch_first_click_is_safe_and_zero():
    rng = np.random.default_rng(102)
    B, H, W, M = 256, 16, 16, 40
    b = BatchBoards(B, H, W, M, rng=rng)
    r = rng.integers(0, H, size=B).astype(np.int32)
    c = rng.integers(0, W, size=B).astype(np.int32)
    hit = b.reveal(r, c)
    assert not hit.any(), "первый клик кого-то убил"
    assert (b.mines.reshape(B, -1).sum(axis=1) == M).all(), "неверное число мин"
    assert (b.counts[np.arange(B), r, c] == 0).all(), "первый клик попал не в ноль"
    assert (b.revealed.reshape(B, -1).sum(axis=1) > 1).all(), "каскад не раскрылся"
    # цифры соответствуют минам
    assert np.array_equal(b.counts, shift_sum(b.mines))
    # открытая клетка никогда не мина
    assert not (b.revealed & b.mines).any()


def test_reset_finished_keeps_batch_full():
    rng = np.random.default_rng(103)
    b = BatchBoards(64, 9, 9, 10, rng=rng)
    for _ in range(200):
        view = b.view()
        r, c, _ = choose_moves(view, b.n_mines, rng)
        b.reveal(r, c)
        b.reset_finished()
    # ни одна доска не должна остаться в законченном состоянии
    assert not b.over().any()
    # и все должны быть в осмысленном состоянии
    assert (b.mines[b.started].reshape(-1, 81).sum(axis=1) == 10).all()


def test_trivial_deductions_are_sound():
    """То, что политика считает безопасным, не должно быть миной."""
    rng = np.random.default_rng(104)
    b = BatchBoards(128, 16, 16, 40, rng=rng)
    checked = 0
    for _ in range(120):
        view = b.view()
        if b.started.any():
            known_mines, known_safe = trivial_deductions(view)
            live = b.started & ~b.dead
            assert not (known_safe[live] & b.mines[live]).any(), "мину назвали безопасной"
            assert b.mines[live][known_mines[live]].all(), "мину нашли там, где её нет"
            checked += int(live.sum())
        r, c, _ = choose_moves(view, b.n_mines, rng)
        b.reveal(r, c)
        b.reset_finished()
    assert checked > 5000, checked


def test_policy_never_clicks_open_cell():
    rng = np.random.default_rng(105)
    b = BatchBoards(64, 16, 16, 40, rng=rng)
    for _ in range(150):
        view = b.view()
        r, c, _ = choose_moves(view, b.n_mines, rng)
        live = ~b.dead & ~b.won()
        opened = b.revealed[np.arange(b.B), r, c]
        assert not opened[live].any(), "политика жмёт уже открытую клетку"
        b.reveal(r, c)
        b.reset_finished()


def test_batch_speed():
    """Не тест корректности, а замер: сколько партий в секунду выдаёт батч."""
    import time
    rng = np.random.default_rng(106)
    B = 1024
    b = BatchBoards(B, 16, 16, 40, rng=rng)
    finished = 0
    t0 = time.perf_counter()
    for _ in range(400):
        view = b.view()
        r, c, _ = choose_moves(view, b.n_mines, rng)
        b.reveal(r, c)
        finished += len(b.reset_finished())
    dt = time.perf_counter() - t0
    print(f"       ({finished} партий за {dt:.1f} с = "
          f"{finished / dt:.0f} партий/с на 16x16)")
    assert finished > 1000


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} тестов прошло")
