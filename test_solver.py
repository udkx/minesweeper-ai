"""Проверки решателя. Запуск: python test_solver.py

Главные два теста — про честность:
  * то, что решатель назвал безопасным, никогда не оказывается миной;
  * то, что он назвал миной, всегда мина.
Если это ломается — значит правила вывода написаны с ошибкой, и все
дальнейшие вероятности (а потом и обучающие метки) мусор.

Третий тест сверяет вероятности с грубой силой: на маленьком поле можно
перебрать все расстановки мин напрямую и сравнить с ответом решателя.
"""

from __future__ import annotations

import itertools

import numpy as np

from minesweeper import FLAGGED, UNKNOWN, Minesweeper, neighbour_count
from solver import analyse, choose_move


def _play(game: Minesweeper, max_moves: int = 10_000):
    """Прогнать партию решателем, проверяя его утверждения по ходу дела."""
    r, c, _ = choose_move(game.view(), game.n_mines)
    game.reveal(r, c)
    moves = 1
    while not game.over and moves < max_moves:
        view = game.view()
        a = analyse(view, game.n_mines)
        # ПРОВЕРКА: доказуемо безопасные клетки действительно без мин
        assert not (a.certain_safe & game.mines).any(), "решатель назвал мину безопасной"
        # ПРОВЕРКА: доказуемо заминированные клетки действительно с минами
        assert (game.mines[a.certain_mines]).all(), "решатель нашёл мину там, где её нет"
        r, c, _ = choose_move(view, game.n_mines)
        game.reveal(r, c)
        moves += 1
    return game


def test_deductions_are_sound_beginner():
    rng = np.random.default_rng(11)
    for _ in range(60):
        _play(Minesweeper(9, 9, 10, rng=rng))


def test_deductions_are_sound_expert():
    rng = np.random.default_rng(12)
    for _ in range(8):
        _play(Minesweeper(16, 30, 99, rng=rng))


def _brute_force_probs(view: np.ndarray, n_mines: int) -> np.ndarray:
    """Честный перебор всех расстановок мин на маленьком поле.

    Медленно и в лоб — именно поэтому это хороший эталон для проверки
    умного кода: тут ошибиться почти невозможно.
    """
    h, w = view.shape
    unknown = [(r, c) for r in range(h) for c in range(w) if view[r, c] == UNKNOWN]
    flagged = int((view == FLAGGED).sum())
    need = n_mines - flagged

    counts = np.zeros((h, w))
    total = 0
    for combo in itertools.combinations(range(len(unknown)), need):
        mines = np.zeros((h, w), dtype=bool)
        mines[view == FLAGGED] = True
        for i in combo:
            mines[unknown[i]] = True
        if not np.array_equal(neighbour_count(mines)[view >= 0], view[view >= 0]):
            continue
        total += 1
        counts += mines
    assert total > 0, "позиция без решений"
    return counts / total


def test_probabilities_match_brute_force():
    rng = np.random.default_rng(13)
    checked = 0
    for _ in range(400):
        g = Minesweeper(5, 5, 5, rng=rng)
        g.reveal(int(rng.integers(5)), int(rng.integers(5)))
        for _ in range(3):
            if g.over:
                break
            view = g.view()
            if (view == UNKNOWN).sum() > 18:  # перебор был бы слишком долгим
                break
            a = analyse(view, g.n_mines)
            if not a.exact:
                break
            expected = _brute_force_probs(view, g.n_mines)
            mask = view == UNKNOWN
            got = a.prob[mask]
            assert not np.isnan(got).any(), "остались клетки без вероятности"
            assert np.allclose(got, expected[mask], atol=1e-9), (
                f"\nсчитано:\n{np.round(a.prob, 3)}\nэталон:\n{np.round(expected, 3)}")
            checked += 1
            r, c, _ = choose_move(view, g.n_mines)
            g.reveal(r, c)
    assert checked > 50, f"проверено слишком мало позиций: {checked}"
    print(f"       (сверено {checked} позиций с полным перебором)")


def test_probabilities_are_a_distribution():
    """Сумма вероятностей по всем закрытым клеткам = числу оставшихся мин.

    Это следствие того, что мин ровно столько, сколько объявлено.
    Хорошая быстрая проверка на любом размере поля.
    """
    rng = np.random.default_rng(14)
    for _ in range(20):
        g = Minesweeper(9, 9, 10, rng=rng)
        g.reveal(0, 0)
        for _ in range(6):
            if g.over:
                break
            view = g.view()
            a = analyse(view, g.n_mines)
            if not a.exact:
                break
            mask = view == UNKNOWN
            s = float(np.nansum(a.prob[mask]))
            assert abs(s - g.mines_left()) < 1e-6, f"сумма {s}, ожидалось {g.mines_left()}"
            r, c, _ = choose_move(view, g.n_mines)
            g.reveal(r, c)


def test_subset_rule_finds_121_pattern():
    """Классический узор 1-2-1: правила первого уровня его не берут, второго — берут."""
    view = np.array([
        [1, 2, 1],
        [UNKNOWN, UNKNOWN, UNKNOWN],
        [UNKNOWN, UNKNOWN, UNKNOWN],
    ], dtype=np.int8)
    a = analyse(view, 2)
    # мины под крайними единицами, центр под двойкой безопасен
    assert a.certain_mines[1, 0] and a.certain_mines[1, 2], np.round(a.prob, 2)
    assert a.certain_safe[1, 1], np.round(a.prob, 2)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} тестов прошло")
