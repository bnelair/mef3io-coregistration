# Benchmarks

Performance checks on synthetic recordings, outside the test suite.

| Script | Measures |
| --- | --- |
| `bench_r10.py` | Run time of Stage 1 (gross), Stage 2 (two anchors) and Stage 3 (drift) on a day-long synthetic pair (a 10-minute offset with a ±5 s swing over the day), against their budgets, and how well the drift recovers the known offset. The pair is rendered in memory and not timed. |

```bash
python benchmarks/bench_r10.py              # 24 h at 256 Hz
python benchmarks/bench_r10.py --hours 6 --fs 512
```
