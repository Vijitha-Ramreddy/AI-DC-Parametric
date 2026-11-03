# Synthetic DC Parametric Test Dataset

This dataset is synthetic and generated for demonstrating AI-based DC test time optimization. It is not real device data. You can regenerate it with `generate_dc_synthetic.py` (options: `--lots`, `--wafers`, `--seed`).

## Files

| File | Rows | Content |
| --- | --- | --- |
| `test_catalog.csv` | 16 | One row per DC test: group, unit, low/high limit, typical test time (ms) |
| `dc_test_results.csv` | 26,460 | One row per die: 12 lots × 5 wafers × 441 die |
| `settling_sweep.csv` | 96,000 | Repeated measurements vs. wait time and sample count, for 4 tests |

## dc_test_results.csv columns

| Column | Meaning |
| --- | --- |
| `ecid` | Electronic chip ID (unique device key) |
| `lot`, `wafer`, `die_x`, `die_y` | Location; die radius 12, centre at (0, 0) |
| `tester`, `site`, `temp_C` | Tester T01–T03, site 1–4, test temperature 25 or 90 °C |
| `CONT_VDIODE` … `IOS` | Measured values for the 16 tests (units in the catalog) |
| `<test>_PF` | P/F against the catalog limits |
| `first_fail_test` | First failing test in flow order (blank = pass) |
| `sort_pass`, `hard_bin` | 1 = pass; bin 1 pass, 2 continuity, 3 leakage, 4 output, 5 IDDQ, 6 supply, 7 threshold |
| `dc_test_time_ms` | Full-flow DC test time with stop-on-first-fail |
| `final_test_fail`, `burnin_fail`, `rma` | Downstream labels (sort-pass devices only) |
| `is_latent_defect_truth` | Ground truth for latent defects; use only to score PAT models, never as a feature |

## Patterns built in (what each technique should find)

| Pattern | Where | Technique it exercises |
| --- | --- | --- |
| Near-duplicate tests | `IIL_A` vs `IIL_B` (r ≈ 0.99) | Redundancy / test elimination |
| Highly correlated vectors | `IDDQ_V1`–`V5` (r ≈ 0.99) | Predictive testing |
| Never-fail test | `VIH` (0 fails) | Never-fail detection, sampling |
| Unique-fail test | 15 devices fail only `IDDQ_V4` | Escape-risk estimate (removing V4 causes escapes) |
| Lot drift | Lots L003 and L008 have higher leakage | Adaptive flow, drift detection |
| Hot-temperature lots | L004, L008, L012 tested at 90 °C | Temperature-aware limits and settling |
| Wafer-edge effect | Leakage rises with die radius | Spatial / die-level adaptive test |
| Equipment artefact | Tester T02, site 3 reads `IIL_A` ~60 nA high | Separating tester effects from device effects |
| Latent defects | 34 in-spec parts with high IDDQ relative to leakage; ~47% fail burn-in | Multivariate outlier screening (PAT) |

## settling_sweep.csv columns

| Column | Meaning |
| --- | --- |
| `test` | `IIL_A`, `IDDQ_V1`, `VOH`, `IDD_STATIC` |
| `device`, `temp_C` | 30 devices, at 25 and 90 °C |
| `wait_ms`, `n_samples` | Wait 0.1–5 ms; 1–16 averaged samples |
| `repeat` | 10 repeats per setting |
| `reference_value` | Fully settled true value |
| `measured_value` | Reading (exponential settling error plus noise) |

Settling is slower at 90 °C (time constant × 1.4), so the shortest safe wait time should be chosen at the hot condition.
