# Fresh full pilot: main study stable, wide numerical gate withheld

The public pilot completed on clean source `36ab424c51a1e1d6f02b63046aa9c4a72ab6f09a`: 180 independent W0 preparations plus two representative wide calibrations, exit 0, 2,956.6 seconds. **G6 remains `INSENSITIVE_OR_UNSUITABLE`** because the wider calibrations retained unresolved numerical solver results. No protected namespace, freeze or confirmation was generated.

The project asks whether a small network's broad background plus a few preferred directions makes one-step adaptation more effective under fixed norm/parameter limits. The exact known-prior example supports the mechanism. This pilot establishes learning/planning readiness ingredients, not an empirical advantage.

| W0 diagnostic | Actual result |
|---|---|
| Source training/validation levels | 32 / 128 / 256, two independent blocks of 30 at each level |
| Complete finite paired preparations | 180; all tuning resolved; no W0 failures |
| Upper relative covariance-error diagnostics | 0.7729 / 0.4791 / 0.3618, against target 0.5 |
| Chosen source counts | T=V=128, by accuracy/stability/cost |
| Selected-block variances / ratio | 0.0004202 / 0.0004413; ratio 1.0501 |
| Provisional confirmation count | J=37; primary has no inner simulation count |
| Centered empirical-reference calibration | Coverage 0.9515 at J=37, 1,000 resamples per block; approximate population coverage |
| Public process peak / budget | 5,242,650,624 / 7,433,478,144 bytes |
| Projected execution / wall allowance | 2,530.8 / 10,800 seconds; resources feasible |

The main source-estimation curve improves substantially with experience: mean relative error falls from 0.733 at 32 tasks to 0.459 at 128. The source choice was not based on primary effect signs. No pilot mean is used here as evidence of superiority.

The wider calibrations expose two different numerical admission issues:

- W1 B03/B04 F: an optional polish flag vetoes an iterate whose retained feasibility, stationarity, complementarity and dual gap pass the ordinary thresholds. The original native status boolean and exact internal polish rejection were not separately retained; a boundary-rounding explanation remains an inference until reproduction.
- W2 B04 Z: a retained gain about 2.20e-17 differs from a mathematically zero inactive coordinate. A strict relative-only comparison reports 1.0 despite passing KKT checks. The inactive face has positive multipliers, providing an algebraic repair path without an absolute gain cutoff or removal of arbitrary weak directions.

The planned [numerical repair](../superpowers/specs/2026-11-02-wide-qp-repair-design.md) preserves final residual thresholds and the independent rank check. It must be tested and reviewed before execution. Any code change requires a new source identity and genuinely fresh pilot; this pilot's unsuitable verdict and all adverse records remain unchanged.

The sealed RunStore is `runs/campaign/2026-11-02-full-program/full-pilot/pilot-abbb0ec8311a5608a728`. Captures, identities, terminal/status receipts, summaries, bounded diagnosis and the 147,308-file hash manifest remain under `full-pilot-capture/` in the same campaign. The output inventory binds 3,647,495,513 bytes; its SHA256 is `829a7b6774136778783a4c5a607c269340b369eeaf838483dde0a0715482b5be`. Independent artifact review passed without overriding G6.
