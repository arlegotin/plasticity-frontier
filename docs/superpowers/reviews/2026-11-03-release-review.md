# First public release review

Scope: publish the completed program from the current `main` branch, with a
reader-facing README, figures, portable comparison data, licensing and citation.
Scientific execution remains source `0b493b194c23456845812bea905f76ebbdd01b13`.

## Changes reviewed

- Replaced the 25 KB execution-oriented README with a short explanation of the
  idea, mechanism, primary result, competing methods, cost and practical limits.
  Detailed commands remain in `REPRODUCE.md`.
- Added three Matplotlib figures, distributed beside the corresponding README
  claims. PNGs were visually inspected; SVGs and the rebuild script are included.
  Method error means use the same empirical 2,048-task cohort throughout.
- Published hash-bound primary pairs, all 14 task-loss columns, method summaries,
  all 364 operational contrasts and the original compressed replay receipt.
  Omitted large raw stores are explicitly identified. No new observations.
- Matched `seti-silence-audit` and `rcu-bench` conventions: AGPL-3.0-only software,
  CC BY 4.0 original presentation, CFF citation and separate attribution scopes.
  Standard license texts were copied byte for byte from the owner's RCU Bench.
  Historical supplied-note authorship and original PDF/ZIP bytes are preserved.
- Used the requested `arlegotin/write-like-me` README skill. The GitHub copy was
  fetched and compared byte for byte with the installed skill and matched.
- Completed package metadata and included `checks` in the wheel, because public
  CLI commands import those reference modules. Added a read-only public-data CI check.

## Verification

- Full suite: `python -m unittest discover -v`, **395 tests passed**, 250.523 s
  in the recorded local environment. No scientific source code changed.
- `python3 scripts/verify_public_results.py` reproduced the primary means, SE
  and interval arithmetic; verified 31 distinct source preparations, 2,048 shared
  tasks, 14 method means, six direct equalities, export hashes and replay status.
  A deliberately corrupted CSV was refused by its hash.
- Citation validated against the official CFF 1.2.0 JSON schema.
- Wheel built successfully. Reference modules imported from its isolated extracted
  contents and all finite exact fixtures passed. The package version is 0.1.0.
- Local documentation links and whitespace checks passed. Scientific source
  SHA-256 remains `dcd81b054c543dd3e8f01edf904e923a7b72b23b06bae2f18416d70c73699423`.

Independent fresh-context release reviewer `release_science_review` inspected
the scientific framing, exports, scripts, package metadata and attribution, and
visually inspected all three figures. No blocking findings or requested edits.
The reviewer separately ran the public-data verifier successfully.

These checks validate the release presentation and compact evidence. They do not
replace protected execution, supply omitted raw packages, or add external peer
review, new independent preparations or real-world evidence.
