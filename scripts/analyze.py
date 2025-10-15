import json, statistics, pathlib, os, datetime, math
from typing import Sequence
from collections import Counter

LOG = pathlib.Path("logs/test.jsonl")
REPORT = pathlib.Path("reports/today.md")
ISSUE_OUT = pathlib.Path("reports/issue_suggestions.md")

def load_results():
    tests, durs, fails = [], [], []
    if not LOG.exists():
        return tests, durs, fails
    with LOG.open() as f:
        for line in f:
            obj = json.loads(line)
            tests.append(obj.get("name"))
            durs.append(obj.get("duration_ms", 0))
            if obj.get("status") == "fail":
                fails.append(obj.get("name"))
    return tests, durs, fails

def calculate_p95(durs: Sequence[float]) -> int:
    if not durs:
        return 0
    try:
        method = "inclusive" if len(durs) < 20 else "exclusive"
        return int(statistics.quantiles(durs, n=20, method=method)[18])
    except statistics.StatisticsError:
        sorted_durs = sorted(durs)
        idx = max(0, min(len(sorted_durs) - 1, math.ceil(0.95 * len(sorted_durs)) - 1))
        return int(sorted_durs[idx])

def main():
    tests, durs, fails = load_results()
    total = len(tests) or 1
    pass_rate = (total - len(fails)) / total
    p95 = calculate_p95(durs)
    now = datetime.datetime.utcnow().isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        f.write(f"# Reflection Report ({now})\n\n")
        f.write(f"- Total tests: {total}\n")
        f.write(f"- Pass rate: {pass_rate:.2%}\n")
        f.write(f"- Duration p95: {p95} ms\n")
        f.write(f"- Failures: {len(fails)}\n\n")
        if fails:
            f.write("## Why-Why (draft)\n")
            for name, cnt in Counter(fails).items():
                f.write(f"- {name}: 仮説=前処理の不安定/依存の競合/境界値不足\n")
    if fails:
        with ISSUE_OUT.open("w", encoding="utf-8") as f:
            f.write("### 反省TODO\n")
            for name in set(fails):
                f.write(f"- [ ] {name} の再現手順/前提/境界値を追加\n")
if __name__ == "__main__":
    main()
