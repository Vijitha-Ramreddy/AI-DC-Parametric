# 04 Ordering summary

- Relay-group constraint: OFF
- Original mean holdout time: 402.12 ms/device
- Ranked mean holdout time: 401.79 ms/device
- Mean change: -0.33 ms/device
- Device pass/fail reproduced exactly: YES
- Hard-bin distribution retained in replay source: 5 bins

## Assumption
- Reordering changes stop-on-first-fail sequence but does not override measurement limits.
- If relay grouping is enabled, groups are ordered by their highest test density and internally ranked.
