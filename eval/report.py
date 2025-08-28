import json
import sys
import statistics
import csv
from collections import defaultdict

rows = [json.loads(l) for l in sys.stdin]
by_variant = defaultdict(list)
for r in rows:
    by_variant[r["variant"]].append(r)

print("\nSummary by variant")
for v, rs in by_variant.items():
    acc = sum(1 for r in rs if r["ok"]) / len(rs)
    p50 = statistics.median(sorted(r["wall_ms"] for r in rs))
    p90 = sorted(r["wall_ms"] for r in rs)[int(0.9 * len(rs)) - 1]
    print(f"- {v}: acc={acc:.2%} p50={p50:.0f}ms p90={p90:.0f}ms n={len(rs)}")

with open("eval_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("\nWrote eval_summary.csv")
