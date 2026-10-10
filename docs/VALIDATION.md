# Historical publication-package validation

This document records validation performed on **2026-10-08** against the complete publication staging package using Python 3.11.5, carried with the historical **v1.0.0/v1.0.1** releases. The [software DOI 10.5281/zenodo.23273595](https://doi.org/10.5281/zenodo.23273595) identifies the existing complete v1.0.1 archive.

The current `main` branch has since been reduced to **23 core training/evaluation and dependency Python files**. The retained scientific algorithms were not changed by that cleanup. The counts and results below describe the historical complete package; they are **not a claim that its checks were rerun on the trimmed branch**. Historical releases remain unchanged by later edits to `main`.

| Historical check | Recorded result |
| --- | --- |
| Copied Python source parsing | 163 files passed; zero syntax errors |
| Copies without portable default edits | Byte-identical to their original project files |
| Existing diagnostic/preprocessing/training-adapter tests | 65 passed in 26.20 s; no scientific code changes required |
| Script 69: paired sequence-aware inference | Exit 0; all 6 generated CSV tables matched archived outputs |
| Script 70: lead-time analysis | Exit 0; all 6 generated CSV tables matched archived outputs |
| Script 78: final propagation metrics | Exit 0; all 6 generated CSV tables matched archived outputs |
| Prefix experiment's guarded source files | All 5 staged module hashes match archived protocol hashes |

CSV comparisons preserved row and column ordering and allowed numeric absolute and relative tolerance of 1e-12; dtype representation differences were ignored. All 18 comparisons passed, including generated 44,800-row prediction tables. Bootstrap routines used the default 10,000 repetitions, not a reduced smoke-test count.

The historical test invocation was:

```bash
python -m pytest test_physical_diagnostics.py test_snapshot_preprocessing.py test_snapshot_training.py -q
```

Those test modules belong to the complete historical package and are no longer included in the trimmed `main` branch. This command is preserved as a record, not as a current installation step.

Analyses used explicit input paths and the numerical-analysis commands retained in the README. Outputs were written outside the Git repository to a separate validation directory, leaving archived runs unchanged. Machine-readable historical records are [test-results.json](test-results.json), [analysis-reproduction.json](analysis-reproduction.json), [code-validation.json](code-validation.json), and [prefix-source-hashes.json](prefix-source-hashes.json).

## Hashes and portability

The five prefix modules hashed by `train_snapshot_experiment.py` were preserved byte-for-byte, including `source45_baselines.py` and `source63_risk.py`. The portable argparse-default edits affected other scripts. The prefix protocol ID includes source hashes, input content, cohort order, exclusions, thresholds, and scientific settings. It does not directly include manifest path strings. Thus CSV path normalization and directory relocation alone do not justify rewriting a stored checkpoint protocol ID or bypassing its guards.

Archived JSON files retain historical paths and hashes as provenance. Normalizing CSV paths changes those CSVs' file-level hashes, so an archived raw manifest checksum must not be presented as the checksum of the normalized copy. Input HDF5 bytes and their source hashes were preserved. A fresh experiment should use its own output directory and satisfy the existing source/content/pairing checks.

## Scope

These historical results verify the complete released package's existing synthetic tests and the listed numerical analyses against archived predictions. They do not claim a full training rerun, fresh raw-data download, full-archive waveform preprocessing, prefix preflight over every waveform file, or regeneration of every manuscript figure. Historical check reports inside original source bundles are separate provenance records. The later removal of other code from `main` does not expand this validation scope.

## Published data coverage

The published package supports numerical recalculation from archived predictions and sample inference. Waveform coverage comprises an 81-event full-record convenience sample (first event IDs in sorted order), four separately supplied spatial cases, and 1,620 snapshot sidecars. The 81 events are not a random or representative training sample. Of the snapshot events, 1,603 are ready and 17 remain incomplete/excluded for audit.

The complete 3,239-event, approximately 16.97 GiB full-record archive is not included. Snapshot sidecars require their matching original full-record label files, so downloading every published asset still does not enable full training or whole-cohort prefix preflight. The historical 65-test and 18-table results do not certify coverage of omitted waveform files. Acquisition/preprocessing code is part of the complete historical software archive, rather than the current trimmed branch.

See the [GitHub v1.0.0 data assets](https://github.com/mrp5560/ca-urc-jgr-solid-earth/releases/tag/v1.0.0) and [dataset DOI 10.5281/zenodo.23273640](https://doi.org/10.5281/zenodo.23273640). For the dataset's multipart distribution, follow its START_HERE instructions before extracting the restored archives.
