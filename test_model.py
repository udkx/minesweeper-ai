"""Проверки кодирования и модели. Запуск: python test_model.py

Ошибки в кодировании входа — самые коварные из всех. Сеть всё равно чему-то
обучится, лосс всё равно упадёт, и понять, что канал с цифрой «3» на самом
деле означал «4», можно будет только по тому, что качество странно низкое.
Поэтому кодирование проверяем на руках собранном примере.
"""

from __future__ import annotations

import numpy as np
import torch

from minesweeper import UNKNOWN
from model import MineNet, encode, predict_probs


def test_encode_channels():
    view = np.array([
        [UNKNOWN, 0, 1],
        [2, 8, UNKNOWN],
    ], dtype=np.int8)
    x = encode(torch.as_tensor(view)[None], n_mines=10)
    assert x.shape == (1, 12, 2, 3), x.shape

    closed = x[0, 0].numpy()
    assert np.array_equal(closed, np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32))
    # цифра d лежит в канале d+1
    assert x[0, 1, 0, 1] == 1, "нуль должен попасть в канал 1"
    assert x[0, 2, 0, 2] == 1, "единица — в канал 2"
    assert x[0, 3, 1, 0] == 1, "двойка — в канал 3"
    assert x[0, 9, 1, 1] == 1, "восьмёрка — в канал 9"
    # ровно один горячий канал состояния на клетку
    assert torch.all(x[0, :10].sum(dim=0) == 1)
    # канал единиц
    assert torch.all(x[0, 10] == 1)
    # канал плотности: 10 мин на 2 закрытые клетки
    assert torch.allclose(x[0, 11], torch.full((2, 3), 5.0))


def test_model_shapes_and_size_independence():
    net = MineNet(ch=16, blocks=2)
    for h, w in [(9, 9), (16, 16), (16, 30), (24, 40)]:
        view = np.full((2, h, w), UNKNOWN, dtype=np.int8)
        out = net(encode(torch.as_tensor(view), 40))
        assert out.shape == (2, h, w), (h, w, out.shape)
    # полносвёрточная сеть не привязана к размеру поля: обучили на одном,
    # запустили на другом — и это работает без единого изменения
    assert net.n_params > 0


def test_predict_probs_masks_open_cells():
    net = MineNet(ch=16, blocks=1)
    view = np.array([
        [UNKNOWN, 1, UNKNOWN],
        [UNKNOWN, 2, UNKNOWN],
    ], dtype=np.int8)
    p = predict_probs(net, view, 4, torch.device("cpu"))
    assert p.shape == (2, 3)
    assert np.isnan(p[0, 1]) and np.isnan(p[1, 1]), "открытые клетки должны быть NaN"
    closed = ~np.isnan(p)
    assert ((p[closed] >= 0) & (p[closed] <= 1)).all(), "вероятности вне [0,1]"


def test_gradient_flows_to_every_layer():
    """Если какой-то слой отрезан от лосса, обучаться он не будет никогда."""
    net = MineNet(ch=16, blocks=2)
    view = np.random.default_rng(0).integers(-1, 5, size=(4, 9, 9)).astype(np.int8)
    target = torch.zeros(4, 9, 9)
    logits = net(encode(torch.as_tensor(view), 10))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
    dead = [n for n, p in net.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"до этих параметров градиент не дошёл: {dead}"


def test_save_load_roundtrip(tmp_path="/tmp/test_minenet.pt"):
    net = MineNet(ch=16, blocks=2)
    net.eval()
    view = np.full((1, 9, 9), UNKNOWN, dtype=np.int8)
    with torch.no_grad():
        before = net(encode(torch.as_tensor(view), 10))
    net.save(tmp_path, meta={"hello": "world"})
    net2, meta = MineNet.load(tmp_path)
    net2.eval()
    with torch.no_grad():
        after = net2(encode(torch.as_tensor(view), 10))
    assert torch.allclose(before, after), "после загрузки модель считает иначе"
    assert meta["hello"] == "world"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} тестов прошло")
