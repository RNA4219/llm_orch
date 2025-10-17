import json, statistics, pathlib, datetime, math, shutil
from collections import Counter
from typing import Sequence

LOG = pathlib.Path("logs/test.jsonl")
REPORT = pathlib.Path("reports/today.md")
ISSUE_OUT = pathlib.Path("reports/issue_suggestions.md")
FAIL_STATUSES = frozenset({"fail", "failed", "error"})

def _normalize_duration(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return 0
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return 0
        if not math.isfinite(parsed):
            return 0
        return int(parsed)
    return 0


def load_results():
    tests, durs, fails = [], [], []
    if not LOG.exists():
        return tests, durs, fails
    with LOG.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            name = obj.get("name")
            tests.append(name)
            durs.append(_normalize_duration(obj.get("duration_ms", 0)))
            status = obj.get("status")
            if isinstance(status, str):
                normalized_status = status.strip().lower()
                if normalized_status in FAIL_STATUSES:
                    fails.append(name)
    return tests, durs, fails

def compute_p95(durations: Sequence[object]) -> int:
    normalized = [_normalize_duration(value) for value in durations]
    if not normalized:
        return 0
    if len(normalized) == 1:
        return normalized[0]

    sorted_durations = sorted(normalized)
    sample_count = len(sorted_durations)

    if sample_count < 20:
        try:
            return int(
                statistics.quantiles(
                    sorted_durations, n=20, method="inclusive"
                )[18]
            )
        except statistics.StatisticsError:
            index = min(sample_count - 1, math.ceil(0.95 * sample_count) - 1)
            return int(sorted_durations[index])

    try:
        return int(statistics.quantiles(sorted_durations, n=20)[18])
    except statistics.StatisticsError:
        index = min(sample_count - 1, math.ceil(0.95 * sample_count) - 1)
        return int(sorted_durations[index])

def main():
    tests, durs, fails = load_results()
    total = len(tests)
    if total == 0:
        pass_rate_text = "未実行"
    else:
        pass_rate = (total - len(fails)) / total
        pass_rate_text = f"{pass_rate:.2%}"
    p95 = compute_p95(durs)
    now = datetime.datetime.utcnow().isoformat()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        f.write(f"# Reflection Report ({now})\n\n")
        f.write(f"- Total tests: {total}\n")
        f.write(f"- Pass rate: {pass_rate_text}\n")
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
    else:
        if ISSUE_OUT.exists():
            if ISSUE_OUT.is_file() or ISSUE_OUT.is_symlink():
                try:
                    ISSUE_OUT.unlink()
                except OSError:
                    ISSUE_OUT.write_text("", encoding="utf-8")
            else:
                shutil.rmtree(ISSUE_OUT)
if __name__ == "__main__":
    main()
