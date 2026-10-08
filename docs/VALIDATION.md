# Publication-package validation

Performed on 2026-10-08 against the publication staging repository using Python 3.11.5.

| Check | Actual result |
| --- | --- |
| Copied Python source parsing | 163 files passed; zero syntax errors |
| Copies without portable default edits | Byte-identical to their original project files |
| Existing diagnostic/preprocessing/training-adapter tests | 65 passed in 26.20 s; no scientific code changes required |
| Script 69: paired sequence-aware inference | Exit 0; all 6 generated CSV tables matched archived outputs |
| Script 70: lead-time analysis | Exit 0; all 6 generated CSV tables matched archived outputs |
| Script 78: final propagation metrics | Exit 0; all 6 generated CSV tables matched archived outputs |
| Prefix experiment's guarded source files | All 5 staged module hashes match archived protocol hashes |

CSV comparisons preserve row and column ordering and allow numeric absolute and relative tolerance of 1e-12; dtype representation differences are ignored. All 18 comparisons passed, including the generated 44,800-row prediction tables. Bootstrap routines used the default 10,000 repetitions, not a reduced smoke-test count.

The existing tests were run as:

```bash
python -m pytest test_physical_diagnostics.py test_snapshot_preprocessing.py test_snapshot_training.py -q
```

Analyses used the README's commands with explicit input paths. Their outputs were written outside the Git repository to a separate validation directory, leaving archived runs unchanged. Machine-readable records are [test-results.json](test-results.json), [analysis-reproduction.json](analysis-reproduction.json), [code-validation.json](code-validation.json), and [prefix-source-hashes.json](prefix-source-hashes.json).

## Hashes and portability

The five prefix modules hashed by `train_snapshot_experiment.py` were preserved byte-for-byte, including `source45_baselines.py` and `source63_risk.py`. The portable argparse-default edits affect other scripts. The prefix protocol ID includes source hashes, input content, cohort order, exclusions, thresholds, and scientific settings. It does not directly include manifest path strings. Thus CSV path normalization and directory relocation alone do not justify rewriting a stored checkpoint protocol ID or bypassing its guards.

Archived JSON files retain historical paths and hashes as provenance. Normalizing CSV paths changes those CSVs' file-level hashes, so an archived raw manifest checksum should not be presented as the checksum of the normalized copy. Input HDF5 bytes and their source hashes are preserved. A fresh experiment should use its own output directory and satisfy the existing source/content/pairing checks.

## Scope

These results verify the released code's existing synthetic tests and the listed numerical analyses against archived predictions. They do not claim a full training rerun, fresh raw-data download, full-archive waveform preprocessing, prefix preflight over every waveform file, or regeneration of every manuscript figure. Historical check reports inside original source bundles are separate provenance records.
