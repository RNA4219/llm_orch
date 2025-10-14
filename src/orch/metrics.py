import os, json, time
from typing import Any

class MetricsLogger:
    def __init__(self, dirpath: str):
        self.dir = dirpath
        os.makedirs(self.dir, exist_ok=True)

    def _file(self) -> str:
        day = time.strftime("%Y%m%d")
        return os.path.join(self.dir, f"requests-{day}.jsonl")

    async def write(self, record: dict[str, Any]):
        path = self._file()
        line = json.dumps(record, ensure_ascii=False)
        # async not needed here; keep simple buffered write
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
