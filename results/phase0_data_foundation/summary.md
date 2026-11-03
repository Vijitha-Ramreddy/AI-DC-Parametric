# Phase 0 data foundation summary

- Validation status: **PASS**
- Results: 26,460 devices across 12 lots
- Catalog: 16 tests
- Train split: L001, L002, L003, L004, L005, L006, L007, L008 (17,640 devices)
- Test split: L009, L010, L011, L012 (8,820 devices)
- Mean baseline DC time: 401.01 ms/device
- Sort-pass rate: 98.2842%
- Duplicate ECIDs: 0
- Forbidden truth column present in source: yes
- LATENT-DEFECT TRUTH USE: scoring-only for PAT; it was not added to any feature table.

## Assumptions
- Train and held-out devices are assigned strictly by lot from config.yaml.
- Stop-on-first-fail baseline time is retained as observed synthetic time.

## Validation issues
- None
