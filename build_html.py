"""
Вшить запись партии в minesweeper.html.

Браузер не может прочитать локальный json через fetch (упирается в CORS
на file://), поэтому запись зашивается прямо в html. Файл остаётся
самодостаточным: открыл — работает, без сервера.

Запуск:
    python play.py record --difficulty intermediate --want-win --out replay.json
    python build_html.py replay.json
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

HTML = pathlib.Path(__file__).parent / "minesweeper.html"
MARKER = re.compile(r"const EMBEDDED_REPLAY = .*?;\n", re.DOTALL)


def main() -> None:
    replay_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "replay.json")
    data = json.loads(replay_path.read_text())
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    html = HTML.read_text()
    if not MARKER.search(html):
        raise SystemExit("не нашёл строку 'const EMBEDDED_REPLAY = ...' в minesweeper.html")
    html = MARKER.sub(f"const EMBEDDED_REPLAY = {blob};\n", html, count=1)
    HTML.write_text(html)

    kb = len(html.encode()) / 1024
    print(f"вшито: {replay_path} ({len(data['moves'])} ходов, "
          f"{'победа' if data['won'] else 'поражение'}) → minesweeper.html, {kb:.0f} КБ")


if __name__ == "__main__":
    main()
