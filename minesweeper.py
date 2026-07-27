"""
Классический сапёр: движок без графики.

Зачем нужен свой движок, а не скриншоты настоящей игры:
нейросети нужны миллионы примеров. Скриншотами ты соберёшь их за годы,
своим движком — за минуты. Плюс движок знает правду (где мины),
а это и есть ответ, которому мы будем учить сеть.

Всё поле хранится в numpy-массивах, потому что дальше мы будем гонять
тысячи досок одновременно, и списки списков тут не подойдут.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Обозначения в "видимом" поле (то, что видит игрок и увидит нейросеть)
# ---------------------------------------------------------------------------
UNKNOWN = -1  # закрытая клетка
FLAGGED = -2  # клетка с флажком

# Классические уровни Windows: (высота, ширина, количество мин)
DIFFICULTIES = {
    "beginner": (9, 9, 10),
    "intermediate": (16, 16, 40),
    "expert": (16, 30, 99),
}

# 8 направлений: смещения по строке и столбцу
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]


def neighbour_count(mines: np.ndarray) -> np.ndarray:
    """Для каждой клетки посчитать, сколько мин вокруг неё.

    Трюк: вместо двойного цикла по клеткам складываем восемь сдвинутых
    копий массива. Это ровно та же операция, что и свёртка ядром 3x3 из
    единиц с дырой в центре — то есть первый слой нашей будущей сети
    делает буквально то же самое, чему мы его будем учить обращать.
    """
    h, w = mines.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.int8)
    padded[1:-1, 1:-1] = mines
    total = np.zeros((h, w), dtype=np.int8)
    for dr, dc in _NEIGHBOURS:
        total += padded[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
    return total


class Minesweeper:
    """Одна партия в сапёра.

    Состояние партии — четыре булевых/числовых массива одинакового размера:
      mines    — где реально мины (правда, игроку не видна)
      counts   — цифры на клетках (сколько мин вокруг)
      revealed — какие клетки уже открыты
      flags    — где игрок поставил флажки

    Мины расставляются НЕ при создании доски, а при первом клике —
    так работает настоящий сапёр, и поэтому первый клик никогда не проигрышный.
    """

    def __init__(self, height: int, width: int, n_mines: int,
                 rng: np.random.Generator | None = None,
                 zero_start: bool = True):
        if n_mines >= height * width:
            raise ValueError("мин не может быть больше, чем клеток")
        self.height = height
        self.width = width
        self.n_mines = n_mines
        self.rng = rng if rng is not None else np.random.default_rng()
        # zero_start=True — гарантируем, что первый клик попадёт в ноль и
        # раскроется большая область. Так делает современный сапёр; без этого
        # партия может начаться с одинокой цифры и сразу превратиться в угадайку.
        self.zero_start = zero_start
        self.reset()

    # -- жизненный цикл партии ------------------------------------------------

    def reset(self) -> None:
        shape = (self.height, self.width)
        self.mines = np.zeros(shape, dtype=bool)
        self.counts = np.zeros(shape, dtype=np.int8)
        self.revealed = np.zeros(shape, dtype=bool)
        self.flags = np.zeros(shape, dtype=bool)
        self.started = False
        self.dead = False
        self.n_clicks = 0

    def _place_mines(self, safe_r: int, safe_c: int) -> None:
        """Расставить мины так, чтобы первый клик был безопасным."""
        h, w = self.height, self.width

        # Клетки, куда мину ставить нельзя.
        forbidden = np.zeros((h, w), dtype=bool)
        forbidden[safe_r, safe_c] = True
        if self.zero_start:
            r0, r1 = max(0, safe_r - 1), min(h, safe_r + 2)
            c0, c1 = max(0, safe_c - 1), min(w, safe_c + 2)
            forbidden[r0:r1, c0:c1] = True

        allowed = np.flatnonzero(~forbidden.ravel())
        if len(allowed) < self.n_mines:
            # Слишком тесно (например, 5x5 с 24 минами) — жертвуем нулевым
            # стартом и защищаем только саму клетку первого клика.
            forbidden[:] = False
            forbidden[safe_r, safe_c] = True
            allowed = np.flatnonzero(~forbidden.ravel())

        chosen = self.rng.choice(allowed, size=self.n_mines, replace=False)
        flat = np.zeros(h * w, dtype=bool)
        flat[chosen] = True
        self.mines = flat.reshape(h, w)
        self.counts = neighbour_count(self.mines)
        self.started = True

    def reveal(self, r: int, c: int) -> bool:
        """Открыть клетку. Возвращает False, если подорвались.

        Если в клетке ноль мин вокруг, открываем всю связную область нулей
        вместе с её "рамкой" из цифр — это тот самый каскад, из-за которого
        один клик иногда раскрывает пол-поля.
        """
        if self.dead or self.revealed[r, c] or self.flags[r, c]:
            return not self.dead

        if not self.started:
            self._place_mines(r, c)

        self.n_clicks += 1

        if self.mines[r, c]:
            self.dead = True
            return False

        # Итеративный обход в глубину. Рекурсию не используем: на поле 16x30
        # каскад может уйти на 400+ уровней вглубь и уронить стек.
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if self.revealed[cr, cc]:
                continue
            self.revealed[cr, cc] = True
            self.flags[cr, cc] = False  # открытая клетка не может быть с флажком
            if self.counts[cr, cc] == 0:
                for dr, dc in _NEIGHBOURS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        if not self.revealed[nr, nc]:
                            stack.append((nr, nc))
        return True

    def toggle_flag(self, r: int, c: int) -> None:
        if not self.revealed[r, c]:
            self.flags[r, c] = not self.flags[r, c]

    # -- что видно снаружи ----------------------------------------------------

    @property
    def won(self) -> bool:
        """Победа = открыты все клетки, кроме мин. Флажки для победы не нужны."""
        return (not self.dead
                and self.started
                and int(self.revealed.sum()) == self.height * self.width - self.n_mines)

    @property
    def over(self) -> bool:
        return self.dead or self.won

    def view(self, with_flags: bool = True) -> np.ndarray:
        """Поле глазами игрока: цифра, UNKNOWN или FLAGGED.

        Это ровно тот массив, который пойдёт на вход нейросети.
        Никакой информации о минах здесь нет — иначе сеть просто
        выучит подглядывать, и на реальной игре развалится.
        """
        out = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)
        out[self.revealed] = self.counts[self.revealed]
        if with_flags:
            out[self.flags & ~self.revealed] = FLAGGED
        return out

    def frontier_mask(self) -> np.ndarray:
        """Закрытые клетки, у которых есть хотя бы один открытый сосед.

        Только по ним вообще есть информация. Остальные закрытые клетки —
        чистая лотерея с вероятностью "оставшиеся мины / оставшиеся клетки".
        """
        h, w = self.height, self.width
        padded = np.zeros((h + 2, w + 2), dtype=bool)
        padded[1:-1, 1:-1] = self.revealed
        has_open_neighbour = np.zeros((h, w), dtype=bool)
        for dr, dc in _NEIGHBOURS:
            has_open_neighbour |= padded[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
        return has_open_neighbour & ~self.revealed & ~self.flags

    def unknown_mask(self) -> np.ndarray:
        return ~self.revealed & ~self.flags

    def mines_left(self) -> int:
        return self.n_mines - int(self.flags.sum())

    # -- удобства -------------------------------------------------------------

    @classmethod
    def from_difficulty(cls, name: str, **kwargs) -> "Minesweeper":
        h, w, m = DIFFICULTIES[name]
        return cls(h, w, m, **kwargs)

    def __str__(self) -> str:
        glyphs = {UNKNOWN: "·", FLAGGED: "⚑", 0: " "}
        v = self.view()
        rows = []
        for row in v:
            rows.append(" ".join(glyphs.get(int(x), str(int(x))) for x in row))
        status = "проиграно" if self.dead else ("выиграно" if self.won else "идёт")
        return "\n".join(rows) + f"\n[{status}, мин осталось: {self.mines_left()}]"
