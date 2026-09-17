"""Проверки движка. Запуск: python test_engine.py

Тесты тут не для красоты: если движок врёт, нейросеть выучит вранье,
и ты потратишь вечер на отладку модели вместо отладки игры.
"""

import numpy as np

from minesweeper import DIFFICULTIES, FLAGGED, UNKNOWN, Minesweeper, neighbour_count


def test_neighbour_count():
    mines = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ], dtype=bool)
    expected = np.array([
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ])
    assert np.array_equal(neighbour_count(mines), expected), neighbour_count(mines)


def test_mine_count_and_first_click_safe():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = Minesweeper(9, 9, 10, rng=rng)
        r, c = int(rng.integers(9)), int(rng.integers(9))
        alive = g.reveal(r, c)
        assert alive, "первый клик не должен убивать"
        assert g.mines.sum() == 10
        # zero_start: первый клик всегда попадает в нулевую клетку
        assert g.counts[r, c] == 0
        # ...а значит раскрывается область больше одной клетки
        assert g.revealed.sum() > 1


def test_counts_match_mines():
    rng = np.random.default_rng(1)
    for _ in range(50):
        g = Minesweeper(16, 30, 99, rng=rng)
        g.reveal(8, 15)
        assert np.array_equal(g.counts, neighbour_count(g.mines))
        # открытая клетка никогда не мина
        assert not (g.revealed & g.mines).any()


def test_view_hides_mines():
    g = Minesweeper(9, 9, 10, rng=np.random.default_rng(2))
    g.reveal(4, 4)
    v = g.view()
    # в видимом поле нет ни одной клетки со скрытой правдой
    assert set(np.unique(v)) <= set(range(9)) | {UNKNOWN, FLAGGED}
    # все закрытые клетки выглядят одинаково, независимо от того, мина там или нет
    hidden = v == UNKNOWN
    assert hidden.sum() == 81 - g.revealed.sum()


def test_flag_blocks_reveal():
    g = Minesweeper(9, 9, 10, rng=np.random.default_rng(3))
    g.reveal(0, 0)
    target = np.argwhere(~g.revealed)[0]
    r, c = int(target[0]), int(target[1])
    g.toggle_flag(r, c)
    before = g.revealed.sum()
    g.reveal(r, c)
    assert g.revealed.sum() == before, "клетку с флажком нельзя открыть"
    assert not g.dead
    g.toggle_flag(r, c)
    assert not g.flags[r, c]


def test_win_condition():
    # Поле, где мины расставлены вручную: открываем все безопасные клетки.
    g = Minesweeper(4, 4, 2, rng=np.random.default_rng(4))
    g.mines = np.zeros((4, 4), dtype=bool)
    g.mines[0, 0] = True
    g.mines[3, 3] = True
    g.counts = neighbour_count(g.mines)
    g.started = True
    for r in range(4):
        for c in range(4):
            if not g.mines[r, c]:
                g.reveal(r, c)
    assert g.won, str(g)
    assert not g.dead


def test_death():
    g = Minesweeper(9, 9, 10, rng=np.random.default_rng(5))
    g.reveal(4, 4)
    mine = np.argwhere(g.mines)[0]
    alive = g.reveal(int(mine[0]), int(mine[1]))
    assert not alive and g.dead and not g.won


def test_cascade_does_not_leak_past_digits():
    """Каскад нулей останавливается на цифрах — не открывает всё поле."""
    g = Minesweeper(*DIFFICULTIES["expert"], rng=np.random.default_rng(6))
    g.reveal(8, 15)
    # На expert один клик не должен открыть всё поле целиком
    assert g.revealed.sum() < 16 * 30 - 99
    # Каждая открытая клетка либо ноль, либо сосед нуля
    zeros = g.revealed & (g.counts == 0)
    from minesweeper import _NEIGHBOURS
    ok = zeros.copy()
    for dr, dc in _NEIGHBOURS:
        shifted = np.zeros_like(zeros)
        h, w = zeros.shape
        sr0, sr1 = max(0, dr), min(h, h + dr)
        tr0, tr1 = max(0, -dr), min(h, h - dr)
        sc0, sc1 = max(0, dc), min(w, w + dc)
        tc0, tc1 = max(0, -dc), min(w, w - dc)
        shifted[tr0:tr1, tc0:tc1] = zeros[sr0:sr1, sc0:sc1]
        ok |= shifted
    assert (g.revealed <= ok).all(), "открылась клетка, не связанная с нулём"


def test_mine_density_is_uniform():
    """Мины должны распределяться равномерно, а не липнуть к одному краю.

    Если генератор смещён, сеть выучит "в углах мин не бывает" — и это
    будет работать на синтетических данных и развалится на настоящей игре.
    """
    rng = np.random.default_rng(7)
    total = np.zeros((9, 9))
    n = 3000
    for _ in range(n):
        g = Minesweeper(9, 9, 10, rng=rng, zero_start=False)
        g._place_mines(4, 4)
        total += g.mines
    freq = total / n
    # исключаем защищённую клетку первого клика
    mask = np.ones((9, 9), dtype=bool)
    mask[4, 4] = False
    expected = 10 / 80
    assert abs(freq[mask].mean() - expected) < 0.01
    assert freq[mask].std() < 0.02, f"разброс частот слишком большой: {freq}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} тестов прошло")
