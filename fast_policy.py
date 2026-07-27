"""
Быстрая политика игры на сдвигах массивов. Нужна для сбора датасета.

Почему не взять решатель из solver.py: он точный, но медленный (790 мс на
партию «эксперта»), а нам нужны сотни тысяч позиций. Здесь только первый
уровень правил — зато сразу на тысячах досок.

И вот главная мысль этапа: датасет нельзя набирать из случайных позиций.
Если взять случайную доску и случайно приоткрыть половину клеток, получатся
картинки, которых в настоящей игре не бывает никогда — например, одинокая
открытая пятёрка посреди закрытого поля (в реальной игре к ней сначала
пришёл бы каскад). Сеть, обученная на таком, отлично решает несуществующую
задачу. Поэтому позиции берём из настоящих партий, сыгранных этой политикой.
"""

from __future__ import annotations

import numpy as np

from batch_engine import _NB, dilate, shift_sum
from minesweeper import UNKNOWN


def shift_max(a: np.ndarray) -> np.ndarray:
    """Максимум по 8 соседям. За краем поля считаем -inf."""
    b, h, w = a.shape
    padded = np.full((b, h + 2, w + 2), -np.inf, dtype=np.float32)
    padded[:, 1:-1, 1:-1] = a
    out = np.full((b, h, w), -np.inf, dtype=np.float32)
    for dr, dc in _NB:
        np.maximum(out, padded[:, 1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w], out=out)
    return out


def trivial_deductions(view: np.ndarray, max_iters: int = 40):
    """Правила первого уровня, сразу на всех досках.

    Для каждой открытой цифры:
      осталось = цифра − (найденных мин вокруг)
      если осталось == 0            → все закрытые соседи безопасны
      если осталось == (закрытых)   → все закрытые соседи это мины

    Крутим до тех пор, пока что-то находится: каждая найденная мина уменьшает
    счётчик у соседних цифр и открывает новые тривиальные случаи. Именно на этом
    держится «цепная реакция», знакомая по живой игре.
    """
    revealed = view >= 0
    known_mines = np.zeros_like(revealed)
    known_safe = np.zeros_like(revealed)
    closed = view == UNKNOWN

    for _ in range(max_iters):
        unresolved = closed & ~known_mines & ~known_safe
        n_mines_around = shift_sum(known_mines)
        n_unk_around = shift_sum(unresolved)
        need = view - n_mines_around                      # int8, может быть < 0

        useful = revealed & (n_unk_around > 0)
        src_safe = useful & (need == 0)
        src_mine = useful & (need == n_unk_around)

        new_safe = dilate(src_safe) & unresolved
        new_mine = dilate(src_mine) & unresolved
        if not (new_safe.any() or new_mine.any()):
            break
        known_safe |= new_safe
        known_mines |= new_mine

    return known_mines, known_safe


def local_risk(view: np.ndarray, known_mines: np.ndarray, n_mines: int) -> np.ndarray:
    """Грубая оценка риска для закрытых клеток.

    Для каждой открытой цифры считаем «плотность» осталось/закрытых, и клетке
    присваиваем максимум по её открытым соседям — то есть худшее из мнений о
    ней. Клеткам без открытых соседей даём общую плотность мин по полю.

    Это не настоящая вероятность (её считает solver.py перебором), но для
    сбора данных достаточно: нам нужна правдоподобная игра, а не идеальная.
    """
    b, h, w = view.shape
    revealed = view >= 0
    unresolved = (view == UNKNOWN) & ~known_mines
    n_mines_around = shift_sum(known_mines)
    n_unk_around = shift_sum(unresolved)
    need = (view - n_mines_around).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(revealed & (n_unk_around > 0),
                        need / np.maximum(n_unk_around, 1), -np.inf)
    risk = shift_max(rate.astype(np.float32))

    # Клетки вдали от открытого: одна общая лотерея на весь оставшийся массив.
    mines_left = n_mines - known_mines.reshape(b, -1).sum(axis=1)
    n_free = np.maximum(unresolved.reshape(b, -1).sum(axis=1), 1)
    global_rate = (mines_left / n_free).astype(np.float32)[:, None, None]
    risk = np.where(np.isfinite(risk), risk, np.broadcast_to(global_rate, risk.shape))
    return risk


def choose_moves(view: np.ndarray, n_mines: int, rng: np.random.Generator,
                 epsilon: float = 0.1):
    """Выбрать по клетке на каждой доске.

    Сначала доказуемо безопасные. Если таких нет — минимальный локальный риск,
    но с вероятностью epsilon вместо этого случайная закрытая клетка: без
    примеси случайности политика ходит слишком однообразно, и датасет получается
    узким.
    """
    b, h, w = view.shape
    known_mines, known_safe = trivial_deductions(view)

    # Приоритет: 0 — доказуемо безопасно, дальше по возрастанию риска.
    risk = local_risk(view, known_mines, n_mines)
    score = np.where(known_safe, -1.0, risk).astype(np.float32)

    noise = rng.random((b, h, w), dtype=np.float32)
    explore = (rng.random(b) < epsilon) & ~known_safe.reshape(b, -1).any(axis=1)
    score = np.where(explore[:, None, None], noise, score + 1e-4 * noise)

    invalid = (view != UNKNOWN) | known_mines
    # Если все закрытые клетки помечены как мины, разрешаем ходить в них:
    # иначе доска зависнет. В корректной игре это почти не случается.
    all_blocked = (~invalid).reshape(b, -1).any(axis=1) == False
    invalid = np.where(all_blocked[:, None, None], view != UNKNOWN, invalid)

    flat = np.where(invalid.reshape(b, -1), np.inf, score.reshape(b, -1))
    pick = np.argmin(flat, axis=1)
    return (pick // w).astype(np.int32), (pick % w).astype(np.int32), known_safe
