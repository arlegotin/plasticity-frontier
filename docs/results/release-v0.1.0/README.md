# Public comparison data, v0.1.0

Compact exports from the completed 3 November campaign, executed at
`0b493b194c23456845812bea905f76ebbdd01b13`. No experiment was rerun to create this
publication. Lower squared coefficient error is better.

| File | Meaning |
| --- | --- |
| `primary_31_pairs.csv` | One row per independent source preparation; baseline and joint analytic conditional risks, plus their paired difference |
| `benchmark_2048_task_losses.csv` | One row per shared target task, with observed squared coefficient errors for all 14 methods |
| `benchmark_method_summary.csv` | Empirical means of those losses, original reported risks, and cold/warm pipeline means |
| `paired_operational_comparisons.csv` | All 364 original operational contrasts, including uncertainty and inconclusive outcomes |
| `clean-replay.json.gz` | Complete original same-machine replay receipt, compressed without modifying its JSON bytes |
| `gate_decision.json` | Original final report gate decision |
| `provenance.json` | Source and export SHA-256 hashes, counts, primary statistics and scope |

The two cohorts answer different questions. **Do not pool 31 preparations with
2,048 tasks.** The primary comparison refits two selection procedures on the same
source data for each preparation. Baseline candidates are B01/B02/B11:Z; joint
candidates are B03/B04/B11:F/B11:Z. There are 128 training and 128 validation source
tasks per preparation, d=6, n=4, depth 2, step size 1, nominal squared-norm cap 0.6
and parameter cap 21. The executed norm check uses an additive 1e−9 tolerance;
15 strict stored-float flags exceed 0.6 by at most 5 ulps. All selected roles
pass the executed rule. The percentage reduction is descriptive; the approximate
Student interval applies to the absolute paired difference.

The benchmark holds the prepared assets fixed and compares them on 2,048 common
tasks. **Use `empirical_mean_of_2048_losses` to compare all methods consistently.**
The `reported_risk_mean` column instead contains analytic conditional risks for
fixed-map methods and empirical risks for ridge and empirical Bayes. These
columns need not agree. Every B05 column is the direct operator matching its
named network; all six reproduce its task losses exactly in this cohort.

| ID | Method |
| --- | --- |
| B01 | Isotropic update |
| B02 | Final-zero-head branch, moment-based gains |
| B03 | Isotropic-floor branch F, moment-based gains |
| B04 | Joint F/Z frontier, moment-based gains |
| B11:F | Isotropic-floor branch, empirically fitted gains |
| B11:Z | Final-zero-head branch, empirically fitted gains |
| B08:ridge | Source-validation-selected ridge regression |
| B10:empirical_bayes | Empirical Bayes with estimated task structure |
| B05:* | Corresponding direct-operator equality control |

Pipeline timings average 14 balanced timing positions. They include declared
generation, selection, preparation, adaptation, serialization and prediction
work; they are **not prediction latency**. Warm values are per repetition:
B11:Z uses two repetitions, the other methods one. Repeated timing workers do
not add independent preparations or test tasks. See
[the full verdict](../2026-11-03-source0b493b1-verdict.md) for the EB tradeoff and
other limits. All 364 original contrast statuses remain in the paired CSV.

The receipt is historical evidence of a matched same-machine replay, not a
portable replay package: large raw stores, benchmark workers, frozen arrays,
local wheels and replay execution directories are not included. Its original
machine paths are retained as provenance and will not resolve in a fresh clone.
The supplied receipt does not independently verify those omitted files.

Run `python3 scripts/verify_public_results.py` from the repository root for a
standard-library consistency check. Run
`.venv/bin/python scripts/rebuild_release_figures.py --output runs/readme-figures`
to regenerate the README figures from these CSVs. These are one-update
comparisons, not multi-epoch training curves. Original data sources remain sealed.
