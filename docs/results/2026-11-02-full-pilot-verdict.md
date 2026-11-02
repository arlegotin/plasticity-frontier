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

## Development verification of the minimal wide-QP repair

Both retained wide preparations were replayed through `pilot._preparation_replication`, the adapter over public `execution.execute_replication`, on clean reviewed source `198a74e5c220f203c33f987c441f84efae2512de`. The original exact Python-integer exploration root, W1 outer180/phases70/470 and W2 outer181/phases72/472, methods, candidates, refinement cap8 and observations were preserved. Their wide workload replacements omit W0's fixed population seed, as in the original pilot. This reuses development observations and adds **zero independent units**.

| Retained preparation | Actual public result | Direct subprocess exit / wall seconds | Process peak / fresh half-RAM budget, bytes |
|---|---|---|---|
| W1 | Completed; all 260 candidate attempts completed, tuning resolved, no execution or restoration failures | 0 / 49.131 | 649,871,360 / 5,490,221,056 |
| W2 | Completed; all 260 candidate attempts completed, tuning resolved, no execution or restoration failures | 0 / 48.515 | 647,872,512 / 4,982,824,960 |

Training/validation observation hashes, source keys, truth hashes and method orders match the originals exactly; the numerical runtime fingerprint also matches. All 336 common W1 prepared/branch maps are bitwise identical. Of 339 common W2 maps, 337 are bitwise identical; the maximum Frobenius difference is `3.2434824880287607e-17`, an ordinary float64 diagnostic. W2 now stops after refinement round1 rather than round8; all 1,092 additional historical attempts remain retained, and the diagnosed old B04 attempt1350 is explicitly compared with the same-prior/options replay attempt258.

The W1 rejected optional polish remains visible, with native success and unchanged F variables. The diagnosed W2 Z reconstruction changes two gains to exact zero, reducing that branch's canonical parameter count from520 to390; its F map is unchanged. Each replay retains four frontier diagnostics labelled `unresolved` for **exact float objective ties**, including two B04 attempts handled by the existing resource tie-break; no unequal-objective unresolved tie or failed method remains in these two preparations. These outcomes do not establish general resolution of future near ties.

The first W2 admission declined to launch when half the available RAM fell below the old cumulative pilot peak. That incomplete launcher receipt is retained. W2 then received a fresh half-RAM/disk admission using an explicitly empirical forecast from streamed W1 memory and retained file-size scaling; its actual peak passed the fresh budget. The original adapter has no hard memory stop. Complete outputs, arrays, source archive, runtime/lock/input inventory, original failures and bounded comparisons are sealed under `runs/campaign/2026-11-02-wide-qp-repair-verification/`: 61 inventoried files, 566,092,752 bytes, inventory SHA256 `f6d17286028d9bdf8642f714c72d7541c4174f9e2c0e91fb18f846c27852fcdd`. New W1/W2 raw JSON SHA256 values are `8dbf024d60b7ac49e25c19a5571f86c756488aa2caa3448b1cb0af9ab4b89acd` and `dd205ca5b568d91873f56ba17c5a4e9384e2627098c30ad823bd0f59e432d935`.

The source36 pilot and its adverse G6 verdict above remain unchanged. This local repair verification permits review of source198a74e for a genuinely fresh full G6 schedule; it is neither a new G6 result nor comparative-advantage evidence. No fresh root, protected observations, freeze or confirmation was generated. A later documentary commit is not the execution identity.
