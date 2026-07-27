"""
Тот же сапёр, но тысячами досок одновременно.

Зачем: обычный движок из minesweeper.py делает ~60 партий в секунду на
«среднем» уровне. Для датасета нужны сотни тысяч позиций, и в цикле по одной
партии это часы. Трюк в том, чтобы держать B досок в массивах формы (B, H, W)
и делать один ход СРАЗУ НА ВСЕХ. Тогда питоновский цикл крутится по шагам
партии (их десятки), а не по партиям (их сотни тысяч), и вся арифметика
уходит внутрь numpy, то есть в C.

Ровно так же устроены векторизованные среды в обучении с подкреплением —
приём общий, не специфичный для сапёра.
"""

from __future__ import annotations

import numpy as np

from minesweeper import UNKNOWN

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _slices(offset: int, size: int):
    """Пара срезов «откуда» и «куда» для сдвига на offset вдоль оси длины size."""
    src = slice(max(0, offset), size + min(0, offset))
    dst = slice(max(0, -offset), size + min(0, -offset))
    return src, dst


# Заметка на будущее: казалось бы, работать срезами напрямую (out[:, tr, tc] +=
# a[:, sr, sc]) дешевле, чем обкладывать массив рамкой нулей и копировать его
# целиком. Замер говорит обратное — со срезами получается в 1.8 раза МЕДЛЕННЕЕ,
# потому что сдвинутые срезы неcплошные в памяти, и numpy теряет больше на
# разорванном обходе, чем экономит на копии. Кэшировать буфер под рамку тоже
# смысла не дало (разница в пределах шума). Так что здесь оставлена простая
# версия, и это тот случай, когда очевидная оптимизация проигрывает замеру.

def shift_sum(a: np.ndarray, dtype=np.int8) -> np.ndarray:
    """Сумма по 8 соседям для каждой клетки каждой доски.

    Сдвигаем только по осям 1 и 2 (высота и ширина). Ось 0 — номер доски,
    и по ней ничего не течёт: соседние доски друг о друге не знают.
    """
    b, h, w = a.shape
    padded = np.zeros((b, h + 2, w + 2), dtype=dtype)
    padded[:, 1:-1, 1:-1] = a
    out = np.zeros((b, h, w), dtype=dtype)
    for dr, dc in _NB:
        out += padded[:, 1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
    return out


def dilate(a: np.ndarray) -> np.ndarray:
    """«Раздуть» булеву маску на 8 соседей. Нужно для каскада нулей."""
    b, h, w = a.shape
    padded = np.zeros((b, h + 2, w + 2), dtype=bool)
    padded[:, 1:-1, 1:-1] = a
    out = np.zeros((b, h, w), dtype=bool)
    for dr, dc in _NB:
        out |= padded[:, 1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
    return out


class BatchBoards:
    """B независимых партий в сапёра, живущих в одних массивах."""

    def __init__(self, batch: int, height: int, width: int, n_mines: int,
                 rng: np.random.Generator | None = None, zero_start: bool = True):
        self.B, self.H, self.W = batch, height, width
        self.n_mines = n_mines
        self.zero_start = zero_start
        self.rng = rng if rng is not None else np.random.default_rng()
        shape = (batch, height, width)
        self.mines = np.zeros(shape, dtype=bool)
        self.counts = np.zeros(shape, dtype=np.int8)
        self.revealed = np.zeros(shape, dtype=bool)
        self.dead = np.zeros(batch, dtype=bool)
        self.started = np.zeros(batch, dtype=bool)
        self.clicks = np.zeros(batch, dtype=np.int32)
        self._rows = np.arange(batch)

    # -- расстановка мин ------------------------------------------------------

    def _place(self, boards: np.ndarray, r: np.ndarray, c: np.ndarray) -> None:
        """Расставить мины на указанных досках, оберегая клетку первого клика.

        Как выбрать n случайных клеток из 480 сразу на 4096 досках без цикла:
        насыпать в каждую клетку случайное число, запретным клеткам поставить
        число заведомо большое, и взять n наименьших. argpartition делает это
        за линейное время.
        """
        if boards.size == 0:
            return
        n, h, w = len(boards), self.H, self.W
        keys = self.rng.random((n, h * w))

        radius = 1 if self.zero_start else 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                keys[np.flatnonzero(ok), (rr[ok] * w + cc[ok])] = 2.0

        idx = np.argpartition(keys, self.n_mines - 1, axis=1)[:, :self.n_mines]
        flat = np.zeros((n, h * w), dtype=bool)
        np.put_along_axis(flat, idx, True, axis=1)
        self.mines[boards] = flat.reshape(n, h, w)
        self.counts[boards] = shift_sum(self.mines[boards])
        self.started[boards] = True

    # -- ход ------------------------------------------------------------------

    def reveal(self, r: np.ndarray, c: np.ndarray,
               active: np.ndarray | None = None) -> np.ndarray:
        """Открыть по одной клетке на каждой активной доске.

        Возвращает маску досок, которые подорвались этим ходом.
        """
        if active is None:
            active = ~self.dead & ~self.won()
        active = active & ~self.dead

        fresh = active & ~self.started
        if fresh.any():
            b = np.flatnonzero(fresh)
            self._place(b, r[b], c[b])

        act = np.flatnonzero(active)
        if act.size == 0:
            return np.zeros(self.B, dtype=bool)

        hit = np.zeros(self.B, dtype=bool)
        hit[act] = self.mines[act, r[act], c[act]]
        self.dead |= hit
        self.clicks[act] += 1

        # Каскад нулей сразу на всех досках: пока есть свежеоткрытые нули,
        # раздуваем маску на соседей. Число итераций — глубина самого
        # длинного каскада в батче, обычно 20-60.
        opening = np.zeros_like(self.revealed)
        survivors = act[~hit[act]]
        if survivors.size:
            opening[survivors, r[survivors], c[survivors]] = True
        while True:
            newly = opening & ~self.revealed
            if not newly.any():
                break
            self.revealed |= newly
            zeros = newly & (self.counts == 0)
            if not zeros.any():
                break
            opening = dilate(zeros) & ~self.revealed
        return hit

    # -- состояние ------------------------------------------------------------

    def won(self) -> np.ndarray:
        open_cells = self.revealed.reshape(self.B, -1).sum(axis=1)
        return (~self.dead & self.started
                & (open_cells == self.H * self.W - self.n_mines))

    def over(self) -> np.ndarray:
        return self.dead | self.won()

    def view(self) -> np.ndarray:
        """Поле глазами игрока: (B, H, W), int8, UNKNOWN там, где закрыто."""
        out = np.full((self.B, self.H, self.W), UNKNOWN, dtype=np.int8)
        np.copyto(out, self.counts, where=self.revealed)
        return out

    def reset(self, boards: np.ndarray) -> None:
        """Начать заново на указанных досках. Остальные не трогаем."""
        if boards.size == 0:
            return
        self.mines[boards] = False
        self.counts[boards] = 0
        self.revealed[boards] = False
        self.dead[boards] = False
        self.started[boards] = False
        self.clicks[boards] = 0

    def reset_finished(self) -> np.ndarray:
        """Перезапустить все законченные партии, чтобы батч не пустел.

        Это стандартный приём: батч всегда полный, поэтому производительность
        не падает к концу прогона, когда живых партий осталось мало.
        """
        done = np.flatnonzero(self.over())
        self.reset(done)
        return done
