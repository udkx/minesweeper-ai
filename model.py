"""
Свёрточная сеть, которая смотрит на поле и говорит, где мины.

Ключевое архитектурное решение: НИКАКОГО пулинга и никаких полносвязных слоёв.
Обычные классификаторы сжимают картинку до вектора («это кошка»), а нам нужен
ответ для каждой клетки: карта поля на входе, карта вероятностей того же
размера на выходе. Такая сеть называется полностью свёрточной, и у неё есть
приятный побочный эффект — она не привязана к размеру поля. Обучил на 16x16,
запускаешь на 16x30, и это работает.

Почему свёртки вообще подходят к сапёру: правила игры локальны и одинаковы во
всех частях поля. Цифра говорит только о своих восьми соседях. Свёртка — это
буквально «одно и то же правило, применённое ко всем клеткам», то есть у
модели с самого начала заложено ровно то допущение, которое в задаче верно.
Полносвязной сети пришлось бы выучивать правило заново для каждой из 256
клеток, и данных на это ушло бы в сотни раз больше.

Дальность обзора собирается послойно: 5x5 на входе плюс шесть блоков по 3x3
дают около 17 клеток в каждую сторону. Это важно, потому что в сапёре бывают
длинные цепочки вывода вдоль границы открытого — сеть должна видеть не только
ближайших соседей.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 10 каналов на состояние клетки (закрыта + цифры 0..8), плюс два служебных
N_CHANNELS = 12


def pick_device(prefer: str = "auto") -> torch.device:
    """MPS на маках с Apple Silicon, CUDA на nvidia, иначе CPU.

    Про Neural Engine: он тут не участвует. Ни PyTorch, ни TensorFlow не умеют
    на нём обучать — ANE доступен только через CoreML и только на инференсе.
    MPS — это Metal, то есть GPU (14-16 ядер на M1 Pro).
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode(views: torch.Tensor, n_mines: int) -> torch.Tensor:
    """Поле (B, H, W) int8 → тензор (B, 12, H, W) float32.

    Почему one-hot, а не просто число в одном канале: цифры сапёра — это не
    величина, а категория. «3» не в три раза больше «1», и линейная свёртка по
    одному каналу навязала бы модели ложный порядок. С one-hot сеть учит для
    каждой цифры свой набор весов.

    Каналы:
      0     закрыта
      1..9  открыта с цифрой 0..8
      10    единицы везде — по нему сеть отличает край поля от пустоты,
            которую дописывает padding у свёрток
      11    общая плотность мин (мин всего / закрытых клеток) — то же самое,
            на что человек смотрит в счётчике мин
    """
    v = views.long().clamp(min=-1)          # флажков в датасете нет
    idx = v + 1                             # -1 → 0 (закрыта), 0..8 → 1..9
    onehot = F.one_hot(idx, num_classes=10).permute(0, 3, 1, 2).float()

    b, _, h, w = onehot.shape
    ones = torch.ones((b, 1, h, w), device=views.device)

    closed = (v == -1).reshape(b, -1).sum(dim=1).clamp(min=1).float()
    density = (n_mines / closed).view(b, 1, 1, 1).expand(b, 1, h, w)
    return torch.cat([onehot, ones, density], dim=1)


class ResBlock(nn.Module):
    """Два слоя 3x3 со сквозной связью.

    Сквозная связь (x + f(x)) нужна не для «глубины ради глубины»: без неё
    у восьмислойной сети градиент до первых слоёв доходит уже размазанным,
    и обучение идёт заметно медленнее. С ней каждый блок учится дописывать
    поправку к тому, что уже есть.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class MineNet(nn.Module):
    """Вход (B, 12, H, W) → выход (B, H, W) логитов «здесь мина»."""

    def __init__(self, ch: int = 64, blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_CHANNELS, ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        # 1x1 свёртка вместо полносвязного слоя: превращает 64 признака каждой
        # клетки в одно число, не смешивая клетки между собой.
        self.head = nn.Conv2d(ch, 1, 1)
        self.ch, self.blocks = ch, blocks

    def forward(self, x):
        return self.head(self.body(self.stem(x))).squeeze(1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def receptive_field(self) -> int:
        """Сколько клеток в каждую сторону видит одна выходная клетка."""
        return 2 + 2 * self.blocks * 1 + 0  # 5x5 даёт 2, каждый блок ещё 2

    # -- сохранение/загрузка --------------------------------------------------

    def save(self, path: str, meta: dict | None = None) -> None:
        torch.save({"state": self.state_dict(), "ch": self.ch,
                    "blocks": self.blocks, "meta": meta or {}}, path)

    @staticmethod
    def load(path: str, device=None) -> tuple["MineNet", dict]:
        blob = torch.load(path, map_location=device or "cpu", weights_only=False)
        net = MineNet(blob["ch"], blob["blocks"])
        net.load_state_dict(blob["state"])
        if device is not None:
            net.to(device)
        return net, blob.get("meta", {})


@torch.no_grad()
def predict_probs(net: MineNet, views: np.ndarray, n_mines: int,
                  device: torch.device) -> np.ndarray:
    """Карта вероятностей мин для набора полей. Открытые клетки → NaN."""
    net.eval()
    v = torch.as_tensor(views, device=device)
    if v.ndim == 2:
        v = v[None]
    logits = net(encode(v, n_mines))
    probs = torch.sigmoid(logits).cpu().numpy()
    out = np.where(views >= 0 if views.ndim == 3 else views[None] >= 0, np.nan, probs)
    return out[0] if views.ndim == 2 else out
