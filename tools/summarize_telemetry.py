"""Summarise a simulate.py --log file in fixed time bins: states, cues, audio, detection rate.

    python tools/summarize_telemetry.py data/test/telemetry.jsonl [--bin 3]
"""
import argparse
import collections
import json

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--bin", type=float, default=3.0)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.log)]
vis = sum(r["box"] is not None for r in rows)
print(f"frames {len(rows)}  dog tracked {vis} ({100 * vis / max(len(rows), 1):.0f}%)  alerts {sum(bool(r['alert']) for r in rows)}")
t_end = rows[-1]["video_t"] if rows else 0
k = 0.0
while k <= t_end:
    rs = [r for r in rows if k <= r["video_t"] < k + a.bin]
    if rs:
        st = collections.Counter(r["state"] for r in rs).most_common(2)
        cues = collections.Counter(c.rsplit(" ", 1)[0] if c.endswith("Hz") else c for r in rs for c in r["cues"]).most_common(3)
        f0 = [r["audio"]["f0"] for r in rs if r["audio"]["active"]]
        v = sum(r["box"] is not None for r in rs)
        print(f"{k:4.0f}s  seen {v:2d}/{len(rs)}  {st}  {cues}" + (f"  F0~{sorted(f0)[len(f0) // 2]:.0f}Hz" if f0 else ""))
    k += a.bin
