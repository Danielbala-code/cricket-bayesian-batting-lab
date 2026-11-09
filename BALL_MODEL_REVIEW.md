# Independent delivery-model review

Review date: 9 November 2026. Scope: `prepare_ball_data.py`, compact data, `ball_engine.py`, `evaluate_ball_model.py`, and the shared implementation plan. The reviewer did not edit the engine.

**Leakage verdict: REVIEW REQUIRED.** Conditional on the historical recorded XIs and innings order being known before play, the inspected empirical fitting and chase construction obey strict earlier-calendar-date rules. This does not establish a fully vintage-correct real-world forecast: source revisions, recorded rosters and post-toss information have no archived availability evidence. The retrospective population also excludes matches using events observed after their starts.

**Initial correctness verdict: NOT CLEAN pending the terminal-delivery finding below.** Subsequent coordinator verification closed that finding before the frozen evaluation: schema1.1 retains boundary allowance/non_boundary, winning running runs are capped, and actions after a winning penalty are suppressed. Three new terminal-chase tests and all20 total tests pass. The original independent findings are preserved below; the closure was checked by the coordinator, not attributed retroactively to the independent reviewer. No calibrated-probability claim follows from these accounting checks.

## Findings

### P2 — Winning deliveries are processed atomically rather than ending at the winning instant

In the inspected `_advance_state`, the complete sampled run and wicket outcome is applied before checking the chase target. An independent deterministic reproduction with target 1 and outcome `[0,1,1,0,0,1]` produces team total 1 and one wicket: a stumping is retained after the winning wide. With target 1 and `[2,0,0,0,0,0]`, the model credits two completed running runs instead of stopping at the first winning run. These can affect terminal player totals and top-batter ties.

The first case can be corrected with an explicit winning-penalty ordering guard and regression test. General terminal truncation requires distinguishing boundaries from completed runs; the agreed six-field outcome representation does not preserve `runs.non_boundary`. A documented terminal-event approximation is therefore necessary unless that representation is refined. Do not call this a complete MCC-exact simulator. MCC Law 16.6 distinguishes the point a result is reached from the full allowance for a winning boundary. [Official MCC 2026 laws, Law 16.6](https://www.lords.org/getmedia/a4ef9f77-2a25-4f5b-a286-4601f08e6334/Laws-of-Cricket-2017-Code-4th-Edition-(2026)_5.pdf).

### P1 — Future-only identity collision affected an earlier forecast; fixed and independently retested

The first engine revision resolved names from every global registry entry. Adding a future-only player with an existing display name caused the earlier forecast to raise an ambiguity error. The engine author changed resolution to IDs in eligible past rosters. The independently rerun complete suite now passes 17 tests, including the future-only outcome and duplicate-name perturbation. This finding is closed for the tested perturbations; outcome-support filtering alone would not have caught it.

### P2 — Evaluation eligibility is an outcome-conditioned retrospective subset

Both team innings must satisfy delivery-model eligibility. Retired hurt, replacements, D/L, penalties, nonstandard rosters and other unsupported events can be discovered only during or after a match. Excluding them is a valid scoped diagnostic with disclosed exclusions, but performance cannot be described as covering all future matches. The evaluator records these exclusions and uses the same remaining cases for v2 and v3.

### P2 — XI and innings-order provenance remains an assumption

Historical actual XIs and which team batted first enter as assumed post-toss information. Expected order uses the median of each player's last five strictly prior observed positions, fallback 6, and lexical tie breaking. Current realised batting order is not used. Nonetheless, recorded rosters can differ from the original announced XI, and no timestamped source proves the historical availability of these inputs. The current fixture contains possible XIs, not confirmed teams.

## Checks that passed

- Eligible delivery rows are filtered by date before selecting observed outcome tuples; active tuples are unioned with a fixed nine-event base and sorted by decoded value. Global future-only vocabulary does not receive prior mass.
- Outside-NZ, NZ-other-ground and exact-ground pools are mutually exclusive. The target player's counts are subtracted from all pooled contexts; matching own-player counts enter once. Empty context cells fall back to the same context's other-player role aggregate.
- Training phase uses legal deliveries before the event divided by the scheduled 300-ball limit. Forecast phase uses the declared simulated budget. Neither uses actual final duration.
- Wides consume neither legal nor faced balls. No-balls consume faced balls but not legal balls. Running parity subtracts the wide/no-ball penalty; unsupported five-run penalties are excluded from fitting.
- Strike is rotated for running, dismissed original-player identity is preserved for run-outs, and ends swap after a completed over. The tested odd-run nonstriker run-out replaces the correct player.
- Free hits survive illegal deliveries. Ordinary dismissals are suppressed on no-balls and free hits; run-outs remain available. A free hit resets after a legal delivery.
- A replacement batter is not introduced after all-out, budget exhaustion or an attained chase target. DNB scores remain -1; a participating zero-ball nonstriker remains distinct from DNB.
- Chase targets are generated from the simulated first innings. The evaluator passes lineups, venue, date, batting-first scenario and a 50-over budget; it does not pass the actual first-innings score or duration into the simulator.
- Team total equals batter runs plus extras. Total faced balls equals legal balls plus no-ball delivery counts. Legal balls never exceed the declared budget; wickets never exceed ten. The illegal-delivery safety cap raises instead of silently truncating.

## Extraction audit

The source has 2,617 JSON files. The 1,434 files before 2016 are counted in the audit and omitted from the portable innings table. The retained metadata has 2,378 innings rows: 1,921 eligible and 457 excluded, with overlapping explicit reasons. Eligible delivery totals are 524,087 recorded events, 511,174 legal balls, 1,138 no-ball deliveries and 512,312 non-wide faced deliveries. All eligible innings reconcile batter runs, extras, legal/non-wide counts and individual batter runs/faced counts against raw extraction; compressed round-trip equality passes. Source JSON is read only. These checks establish extraction accounting, not scorecard completeness or source vintage.

## Statistical interpretation

This is a discounted categorical empirical-Bayes sampler, not a normal run model or full hierarchical posterior. Player, role, phase and context samples are observational and can be sparse or confounded. There is no fitted linear design matrix, so a classic VIF or linear multicollinearity threshold is not an appropriate pass/fail test. Relevant checks are disjoint pools, self-exclusion, effective delivery counts, fallback-cell counts and ablation stability.

The development grid must be fixed before fitting and selected only on 2019–2022. Later results remain retrospective because earlier analyses and final-period summaries have already been seen. Shortened budgets are unvalidated declared scenarios, not a weather forecast or DLS model. Probability calibration, profit and causal swing effects remain unestablished.

## Verification record

`python -m unittest discover -s outputs/cricket_app/tests -v`: 17/17 passed after the future-identity fix. Independent terminal-delivery probes reproduced the open P2 finding above. Recheck this document after any engine correction; passing the existing suite does not cover an untested terminal-law edge.
