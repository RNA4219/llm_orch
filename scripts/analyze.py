import json, statistics, pathlib, os, datetime
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

def main():
    tests, durs, fails = load_results()
    total = len(tests) or 1
    pass_rate = (total - len(fails)) / total
    p95 = int(statistics.quantiles(durs or [0], n=20)[18]) if durs else 0
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
