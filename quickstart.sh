#!/usr/bin/env bash
# Полный прогон с нуля: окружение → данные → обучение → замер винрейта.
# Запуск:  bash quickstart.sh
#
# Скрипт не делает ничего волшебного, это те же команды, что в README, просто
# подряд и с проверками. Каждый шаг можно спокойно запускать отдельно.

set -e  # падать сразу, а не молча продолжать со сломанным состоянием

echo "== 1/6 окружение =="
python3 --version
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy torch

echo
echo "== 2/6 проверка, что GPU виден =="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("MPS доступен:", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    print("  → обучение пойдёт на CPU. Это работает, просто в разы медленнее.")
    print("  → MPS требует Apple Silicon и macOS 12.3+")
PY

echo
echo "== 3/6 быстрые тесты =="
python test_engine.py
python test_batch.py
python test_model.py
echo "(test_solver.py идёт ~2 минуты, он сверяет вероятности с полным перебором —"
echo " запусти отдельно, когда будет время)"

echo
echo "== 4/6 данные (~2-4 минуты) =="
# Разные seed принципиальны: обучение и валидация должны идти от РАЗНЫХ партий,
# а не от разных позиций одних и тех же партий.
python dataset.py make --difficulty intermediate --positions 400000 --seed 0   --out data_train.npz
python dataset.py make --difficulty intermediate --positions 40000  --seed 999 --out data_val.npz

echo
echo "== 5/6 обучение =="
echo "Сначала две эпохи, чтобы понять скорость на этом железе."
python train.py --epochs 2 --channels 64 --blocks 6 --out minenet.pt
echo
echo "Посмотри время эпохи выше и умножь на 12 — столько займёт полный прогон:"
echo "    python train.py --epochs 12 --channels 64 --blocks 6 --out minenet.pt"
echo "Чтобы мак не засыпал на середине:"
echo "    caffeinate -i python train.py --epochs 12 --channels 64 --blocks 6"

echo
echo "== 6/6 сколько выигрывает =="
python nn_agent.py --model minenet.pt --difficulty intermediate --games 300

echo
echo "Готово. Дальше:"
echo "  посмотреть партию сети в браузере:"
echo "    python play.py record --agent net --model minenet.pt \\"
echo "        --difficulty intermediate --want-win --min-guesses 2 --out replay_net.json"
echo "    python build_html.py решатель=replay.json сеть=replay_net.json"
echo "  сравнить с эталоном:"
echo "    python play.py bench --difficulty intermediate --games 200"
