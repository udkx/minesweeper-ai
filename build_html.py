"""
Вшить записи партий в minesweeper.html.

Браузер не может прочитать локальный json через fetch (упирается в CORS на
file://), поэтому записи зашиваются прямо в html. Файл остаётся
самодостаточным: открыл — работает, без сервера.

Запуск:
    python build_html.py решатель=replay.json сеть=replay_net.json

Метка слева от знака «=» станет подписью на переключателе записей.
Если метку не писать, возьмётся имя файла.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

HTML = pathlib.Path(__file__).parent / "minesweeper.html"
MARKER = re.compile(r"const EMBEDDED_REPLAYS = .*?;\n", re.DOTALL)


def main() -> None:
    args = sys.argv[1:] or ["replay.json"]
    replays: dict[str, dict] = {}
    for arg in args:
        label, _, path = arg.rpartition("=")
        path = path or arg
        label = label or pathlib.Path(path).stem
        replays[label] = json.loads(pathlib.Path(path).read_text())

    blob = json.dumps(replays, ensure_ascii=False, separators=(",", ":"))
    html = HTML.read_text()
    if not MARKER.search(html):
        raise SystemExit("не нашёл 'const EMBEDDED_REPLAYS = ...' в minesweeper.html")
    html = MARKER.sub(f"const EMBEDDED_REPLAYS = {blob};\n", html, count=1)
    HTML.write_text(html)

    for label, data in replays.items():
        print(f"  {label}: {len(data['moves'])} ходов, "
              f"{'победа' if data['won'] else 'поражение'}")
    print(f"вшито в minesweeper.html, {len(html.encode()) / 1024:.0f} КБ")


if __name__ == "__main__":
    main()
