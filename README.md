# Prepared learning research

This repository preserves the supplied research inputs and builds a local CPU float64 research environment. The original ZIP and PDFs are inputs: do not modify them or run the source scripts in place. `doctor` stages a validated read-only source snapshot under `source_bundle/`; experiments must use copies and write outputs under `runs/`. Optional source assets do not gate equation-based fixtures.

Bootstrap on a native arm64 macOS Python compatible with the locked wheels (the recorded environment used Python 3.14.6):

```sh
python3 -m venv .venv
.venv/bin/python -m pip download --only-binary=:all: --dest wheelhouse -r requirements.lock.txt
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -v
.venv/bin/python -m prepared_learning doctor --output runs/environment
```

Verify the downloaded wheels against `wheel_manifest.json`; the wheel files themselves are local and ignored. `installer-report.json` records initial installation, and `licenses/` retains wheel notices. Lock and wheel hashes describe this machine's bootstrap; a different Python/platform needs a newly resolved, recorded environment. Hardware discovery records allowlisted model fields and command outcomes; unavailable fields remain unavailable. The doctor checks required CPU operations and emits a nonzero status on failure. Its source manifest is generated alongside the environment report. Protected scientific runs require committed code and a separately frozen scientific identity; these smoke tests are not a scientific campaign.
