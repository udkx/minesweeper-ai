"""
Точный решатель сапёра. Это наш эталон — цифра, которую нейросеть должна догнать.

Работает в три уровня, от дешёвого к дорогому:

  1. Простые правила. Клетка с цифрой 1, у которой ровно один закрытый сосед →
     это мина. Клетка с цифрой 1, у которой мина уже найдена → остальные соседи
     безопасны. Крутим до упора.

  2. Правило подмножеств. Если ограничение A целиком внутри B, то в разнице
     B\\A ровно (мин_B − мин_A) мин. Это то, чем человек ловит узоры 1-2-1.

  3. Перечисление конфигураций. Если правила встали, честно перебираем ВСЕ
     расстановки мин, совместимые с видимыми цифрами, и считаем, в какой доле
     из них клетка оказывается миной. Это и есть настоящая вероятность —
     точнее посчитать невозможно в принципе.

Важный момент для понимания нейросети: пункт 3 даёт математически идеальный
ответ. Значит, идеально обученная сеть может только приблизиться к нему,
но не превзойти. Смысл сети в другом — она выдаёт похожий ответ за один
проход, без экспоненциального перебора.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from minesweeper import FLAGGED, UNKNOWN, _NEIGHBOURS

# Если компонента фронтира больше этого — перебор может встать надолго.
# Тогда честные вероятности для неё заменяем локальной оценкой.
MAX_COMPONENT_CELLS = 26
MAX_ENUMERATION_NODES = 400_000


@dataclass
class Analysis:
    """Что решатель понял про текущую позицию."""
    prob: np.ndarray                  # вероятность мины; NaN для открытых клеток
    certain_safe: np.ndarray          # клетки, безопасные доказуемо
    certain_mines: np.ndarray         # клетки, где мина доказуемо
    exact: bool = True                # False, если где-то пришлось приближать
    notes: list[str] = field(default_factory=list)


def _constraints_from_view(view: np.ndarray, known_mines: np.ndarray,
                           known_safe: np.ndarray | None = None):
    """Превратить видимое поле в список ограничений.

    Каждое ограничение — «в этом наборе закрытых клеток ровно N мин».
    Это и есть сапёр целиком: игра сводится к системе таких уравнений.

    Уже доказанные клетки из ограничений вычёркиваются: мины уменьшают
    счётчик, безопасные просто выкидываются. Без этого правила застревают —
    например, узор 1-2-1 раскрывается только до половины.
    """
    h, w = view.shape
    revealed = view >= 0
    unknown = (view == UNKNOWN) & ~known_mines
    if known_safe is not None:
        unknown = unknown & ~known_safe

    constraints = []
    for r in range(h):
        for c in range(w):
            if not revealed[r, c]:
                continue
            need = int(view[r, c])
            cells = []
            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if known_mines[nr, nc]:
                        need -= 1
                    elif unknown[nr, nc]:
                        cells.append(nr * w + nc)
            if cells:
                constraints.append((frozenset(cells), need))
    return constraints


def _basic_deductions(view: np.ndarray, n_mines: int):
    """Уровни 1 и 2: гоняем правила до тех пор, пока что-то находится."""
    h, w = view.shape
    known_mines = (view == FLAGGED).copy()
    known_safe = np.zeros((h, w), dtype=bool)

    while True:
        changed = False
        constraints = _constraints_from_view(view, known_mines, known_safe)

        # --- Уровень 1: тривиальные ограничения --------------------------------
        survivors = []
        for cells, need in constraints:
            if need == 0:
                for idx in cells:
                    if not known_safe[idx // w, idx % w]:
                        known_safe[idx // w, idx % w] = True
                        changed = True
            elif need == len(cells):
                for idx in cells:
                    if not known_mines[idx // w, idx % w]:
                        known_mines[idx // w, idx % w] = True
                        changed = True
            else:
                survivors.append((cells, need))
        if changed:
            continue

        # --- Уровень 2: правило подмножеств -----------------------------------
        # Сравниваем только пересекающиеся ограничения: у соседних клеток
        # наборы закрытых соседей перекрываются, у далёких — нет.
        by_cell: dict[int, list[int]] = {}
        for i, (cells, _) in enumerate(survivors):
            for idx in cells:
                by_cell.setdefault(idx, []).append(i)

        for i, (cells_a, need_a) in enumerate(survivors):
            candidates = set()
            for idx in cells_a:
                candidates.update(by_cell[idx])
            candidates.discard(i)
            for j in candidates:
                cells_b, need_b = survivors[j]
                if cells_a < cells_b:  # A строго внутри B
                    diff = cells_b - cells_a
                    need_diff = need_b - need_a
                    if need_diff == 0:
                        for idx in diff:
                            if not known_safe[idx // w, idx % w]:
                                known_safe[idx // w, idx % w] = True
                                changed = True
                    elif need_diff == len(diff):
                        for idx in diff:
                            if not known_mines[idx // w, idx % w]:
                                known_mines[idx // w, idx % w] = True
                                changed = True
            if changed:
                break

        if not changed:
            break

    # Клетка не может быть одновременно миной и безопасной; если так вышло,
    # позиция противоречива (в корректной игре не бывает).
    known_safe &= ~known_mines
    known_safe &= view == UNKNOWN
    return known_mines, known_safe


def _components(cells: list[int], constraints: list[tuple[frozenset, int]]):
    """Разбить фронтир на независимые куски.

    Два конца поля не влияют друг на друга, поэтому перебирать их вместе —
    значит умножать экспоненты. Разбиение на компоненты превращает
    2^40 в 2^12 + 2^14 + ...
    """
    parent = {c: c for c in cells}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cs, _ in constraints:
        cs = list(cs)
        for x in cs[1:]:
            union(cs[0], x)

    groups: dict[int, list[int]] = {}
    for c in cells:
        groups.setdefault(find(c), []).append(c)
    return list(groups.values())


def _enumerate_component(comp_cells: list[int],
                         comp_constraints: list[tuple[frozenset, int]]):
    """Перебрать все допустимые расстановки мин внутри одной компоненты.

    Возвращает:
      sol_by_k[k]      — сколько решений используют ровно k мин
      cell_by_k[i][k]  — в скольких из них i-я клетка компоненты оказалась миной
      exact            — False, если перебор пришлось прервать по лимиту
    """
    n = len(comp_cells)
    index = {cell: i for i, cell in enumerate(comp_cells)}

    # Ограничения в локальных индексах
    cons = [([index[c] for c in cells], need) for cells, need in comp_constraints]

    # Для каждой клетки — в какие ограничения она входит
    cell_cons: list[list[int]] = [[] for _ in range(n)]
    for ci, (cells, _) in enumerate(cons):
        for cell in cells:
            cell_cons[cell].append(ci)

    # Порядок обхода: обход в ширину по графу «клетка — ограничение — клетка».
    # Так ограничения закрываются рано и отсечение работает почти сразу.
    order: list[int] = []
    seen = [False] * n
    for start in range(n):
        if seen[start]:
            continue
        queue = [start]
        seen[start] = True
        while queue:
            cell = queue.pop(0)
            order.append(cell)
            for ci in cell_cons[cell]:
                for nb in cons[ci][0]:
                    if not seen[nb]:
                        seen[nb] = True
                        queue.append(nb)

    assigned = [0] * len(cons)      # сколько мин уже поставлено в ограничение
    remaining = [len(cells) for cells, _ in cons]  # сколько клеток ещё не решено
    needs = [need for _, need in cons]

    sol_by_k: dict[int, int] = {}
    cell_by_k = [dict() for _ in range(n)]
    state = [0] * n
    nodes = 0
    exact = True

    def dfs(pos: int, used: int):
        nonlocal nodes, exact
        nodes += 1
        if nodes > MAX_ENUMERATION_NODES:
            exact = False
            return
        if pos == len(order):
            sol_by_k[used] = sol_by_k.get(used, 0) + 1
            for i in range(n):
                if state[i]:
                    cell_by_k[i][used] = cell_by_k[i].get(used, 0) + 1
            return
        cell = order[pos]
        for value in (0, 1):
            ok = True
            touched = cell_cons[cell]
            for ci in touched:
                remaining[ci] -= 1
                assigned[ci] += value
                # мин уже больше, чем нужно, либо оставшихся клеток не хватит
                if assigned[ci] > needs[ci] or assigned[ci] + remaining[ci] < needs[ci]:
                    ok = False
            if ok:
                state[cell] = value
                dfs(pos + 1, used + value)
                state[cell] = 0
            for ci in touched:
                remaining[ci] += 1
                assigned[ci] -= value
            if not exact:
                return

    dfs(0, 0)
    return sol_by_k, cell_by_k, exact


def _convolve(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
    """Свернуть две «сколько решений на k мин» в одну.

    Компоненты независимы, поэтому число комбинаций перемножается,
    а количество мин складывается — это ровно умножение полиномов.
    Считаем в питоновских int: числа тут легко перерастают float64.
    """
    out: dict[int, int] = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] = out.get(ka + kb, 0) + va * vb
    return out


def analyse(view: np.ndarray, n_mines: int) -> Analysis:
    """Полный разбор позиции: доказуемо безопасные, доказуемо мины, вероятности."""
    h, w = view.shape
    known_mines, known_safe = _basic_deductions(view, n_mines)

    prob = np.full((h, w), np.nan)
    unknown = (view == UNKNOWN) & ~known_mines
    prob[known_mines] = 1.0
    prob[known_safe] = 0.0

    todo = unknown & ~known_safe
    if not todo.any():
        return Analysis(prob, known_safe, known_mines, True, ["хватило правил"])

    mines_left = n_mines - int(known_mines.sum())
    constraints = _constraints_from_view(view, known_mines, known_safe)

    frontier = sorted({i for cells, _ in constraints for i in cells})
    outside = [int(i) for i in np.flatnonzero(todo.ravel()) if i not in set(frontier)]
    n_outside = len(outside)

    if not frontier:
        # Информации нет вообще (например, самый первый ход): чистая лотерея.
        p = mines_left / max(1, n_outside)
        for i in outside:
            prob[i // w, i % w] = p
        return Analysis(prob, known_safe, known_mines, True, ["нет информации"])

    exact = True
    comps = _components(frontier, constraints)
    per_comp = []
    for comp in comps:
        comp_set = set(comp)
        comp_cons = [(cells, need) for cells, need in constraints
                     if cells & comp_set]
        if len(comp) > MAX_COMPONENT_CELLS:
            exact = False
            per_comp.append(None)
            continue
        sol_by_k, cell_by_k, ok = _enumerate_component(comp, comp_cons)
        if not ok or not sol_by_k:
            exact = False
            per_comp.append(None)
        else:
            per_comp.append((comp, sol_by_k, cell_by_k))

    # Компоненты, которые не осилили перебором — грубая локальная оценка:
    # средняя плотность мин по ограничениям, куда клетка входит.
    approx_cells: list[int] = []
    for comp, data in zip(comps, per_comp):
        if data is None:
            approx_cells.extend(comp)
    if approx_cells:
        for i in approx_cells:
            rates = [need / len(cells) for cells, need in constraints if i in cells]
            prob[i // w, i % w] = float(np.mean(rates)) if rates else 0.5

    good = [d for d in per_comp if d is not None]
    if not good:
        p = mines_left / max(1, len(frontier) + n_outside)
        for i in outside:
            prob[i // w, i % w] = p
        return Analysis(prob, known_safe, known_mines, False, ["перебор не осилил"])

    # Префиксные и суффиксные свёртки — чтобы для каждой компоненты быстро
    # получить «сколько комбинаций у всех остальных вместе».
    dists = [d[1] for d in good]
    k = len(dists)
    prefix = [{0: 1}]
    for d in dists:
        prefix.append(_convolve(prefix[-1], d))
    suffix = [{0: 1}]
    for d in reversed(dists):
        suffix.append(_convolve(suffix[-1], d))
    suffix.reverse()

    def comb(n: int, r: int) -> int:
        if r < 0 or r > n:
            return 0
        return math.comb(n, r)

    # Z — общее число расстановок мин, совместимых с видимой картинкой.
    # Каждая расстановка равновероятна, поэтому вероятность мины в клетке —
    # это просто доля расстановок, где там мина.
    total_all = prefix[k]
    Z = sum(cnt * comb(n_outside, mines_left - kk) for kk, cnt in total_all.items())
    if Z == 0:
        return Analysis(prob, known_safe, known_mines, False, ["противоречивая позиция"])

    expected_outside_mines = 0
    for kk, cnt in total_all.items():
        rest = mines_left - kk
        expected_outside_mines += cnt * comb(n_outside, rest) * rest
    if n_outside:
        p_outside = expected_outside_mines / Z / n_outside
        for i in outside:
            prob[i // w, i % w] = p_outside

    for ci, (comp, sol_by_k, cell_by_k) in enumerate(good):
        others = _convolve(prefix[ci], suffix[ci + 1])
        for local_i, cell in enumerate(comp):
            num = 0
            for k_here, cnt_here in cell_by_k[local_i].items():
                for k_other, cnt_other in others.items():
                    c = comb(n_outside, mines_left - k_here - k_other)
                    if c:
                        num += cnt_here * cnt_other * c
            prob[cell // w, cell % w] = num / Z

    # Перебор доказывает больше, чем правила: если клетка оказалась миной
    # во ВСЕХ совместимых расстановках — это доказанная мина, даже если
    # никакое простое правило её не увидело. И наоборот для нулей.
    # (только если перебор был точным — приближённым оценкам верить нельзя)
    if exact:
        proven = ~np.isnan(prob) & (view == UNKNOWN)
        known_mines = known_mines | (proven & (prob == 1.0))
        known_safe = known_safe | (proven & (prob == 0.0))
        known_safe &= ~known_mines

    notes = ["точный перебор"] if exact else ["частично приближённо"]
    return Analysis(prob, known_safe, known_mines, exact, notes)


def choose_move(view: np.ndarray, n_mines: int) -> tuple[int, int, Analysis]:
    """Выбрать клетку для открытия: сначала доказуемо безопасные, потом минимум риска."""
    h, w = view.shape
    if not (view >= 0).any():
        # Первый ход. Угол статистически выгоднее центра: у угловой клетки
        # меньше соседей, поэтому раскрытая область даёт более плотную информацию.
        empty = np.full((h, w), np.nan)
        return 0, 0, Analysis(empty, np.zeros((h, w), bool), np.zeros((h, w), bool),
                              True, ["первый ход"])

    a = analyse(view, n_mines)
    safe = a.certain_safe & (view == UNKNOWN)
    if safe.any():
        idx = np.argwhere(safe)[0]
        return int(idx[0]), int(idx[1]), a

    candidates = np.where(np.isnan(a.prob), np.inf, a.prob)
    candidates[view != UNKNOWN] = np.inf
    candidates[a.certain_mines] = np.inf
    if not np.isfinite(candidates).any():
        idx = np.argwhere(view == UNKNOWN)[0]
        return int(idx[0]), int(idx[1]), a

    best = float(candidates.min())
    ties = np.argwhere(candidates <= best + 1e-12)
    if len(ties) > 1:
        # Из равнорискованных берём клетку с бОльшим числом открытых соседей:
        # она даст больше новой информации на следующий ход.
        revealed = view >= 0
        def info(rc):
            r, c = int(rc[0]), int(rc[1])
            return sum(revealed[r + dr, c + dc]
                       for dr, dc in _NEIGHBOURS
                       if 0 <= r + dr < h and 0 <= c + dc < w)
        ties = sorted(ties, key=info, reverse=True)
    return int(ties[0][0]), int(ties[0][1]), a
