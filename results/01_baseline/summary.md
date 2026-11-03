# 01 Baseline summary

- Overall first-pass yield: **98.284%** (26,006/26,460)
- Mean DC time overall: **401.01 ms/device**
- Mean passing-device time: **405.00 ms**
- Mean failing-device time: **172.65 ms**
- Highest-fail-rate test: **IDDQ_V5** (0.971%)

## Assumptions
- Catalog row order is the test-flow order.
- Baseline stop-on-first-fail time is reconstructed from cumulative catalog test times.
- Poisson upper bounds are one-sided at the configured confidence level.
- Fail limits and measurements remain authoritative; no model changes these decisions.
