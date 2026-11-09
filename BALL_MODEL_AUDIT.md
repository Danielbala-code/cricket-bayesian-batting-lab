# Final addition: leakage, redundancy and implementation audit

Status: computational checks pass; real-world historical availability remains REVIEW REQUIRED. Publication remains paused. This is a review of the implemented delivery model, not an assurance of calibrated forecasts.

## Decision time and target

Forecast after toss/declared XI but before the first match delivery. Targets are each team's regulation-innings top-scorer set and joint player runs/non-wide deliveries faced. The model conditions on expected batting orders, ground, first batting team and a declared initial overs limit. For evaluation, recorded XI and innings sequence are assumed known post-toss; archival snapshot evidence is unavailable.

## Leakage checks

| Item | Verdict and evidence |
|---|---|
| Training outcomes | PASS: fit_ball_model filters eligible innings date strictly before cutoff before computing support, counts, identity lookup or context priors. Same-date games are conservatively excluded. |
| Future outcome vocabulary | PASS: only strict-past observed event types plus a fixed nine-event law-valid base receive probability; a future-only category cannot acquire prior mass. Synthetic perturbation leaves earlier score arrays identical. |
| Future identity collisions | PASS: identity resolution uses people in past eligible rosters; adding a future person with the same display name cannot change the earlier forecast. Ambiguous past identities are rejected. |
| Expected order | PASS conditionally: evaluator uses last five prior positions, fallback6 and deterministic name tiebreak. Mutating target runs or realised positions does not alter forecast inputs. |
| Innings phase | PASS: running legal balls before the historical event divided by scheduled limit. Final observed innings duration is not used as a phase denominator. Simulated phase uses the declared budget. |
| Chase target | PASS: first innings is generated, then its total+1 is passed to the simulated chase. Actual first score is read only as an evaluation label, never a pre-match simulator input. |
| Short overs | PASS for declared scenarios only: limit is a user input. No realized rain reduction or DLS target is passed into a pre-match test. Only scheduled50-over cases are evaluated. |
| Tuning | PASS conditionally: four combinations h2/4 and prior60/240 deliveries are searched on2019-2022 only. Selection timestamp precedes2023+ comparison; hashes prohibit mixed engine versions. Prior exploratory use of final-period summaries prevents untouched-confirmation claims. |
| Roster/toss availability | REVIEW REQUIRED: current inputs are probable XIs; historical recorded rosters and innings order lack publication timestamps. The forecast is conditional on those inputs, not proof of an operationally reconstructed historical feed. |
| Data revisions | REVIEW REQUIRED: current corrected Cricsheet files are used; original historical vintages are unavailable. |
| Population selection | REVIEW REQUIRED: unsupported retirements, substitutions, miscounts and other events are excluded after observation. Both target and opponent must be supported; this narrows the retrospective population and is not a knowable pre-match eligibility rule. Matched v2 comparison uses the same retained cases. |

Overall leakage verdict: **REVIEW REQUIRED**, because roster/toss/vintage provenance and outcome-conditioned exclusions remain unresolved. No demonstrated future-score feature contamination was found in the reviewed computational path; that narrower statement is not an unconditional CLEAN verdict.

## Multicollinearity and repeated evidence

This sampler does not fit a linear/logistic regression with independently estimated coefficients for runs, balls, rate, position and role. A VIF computed from categorical probabilities summing to one would diagnose the simplex constraint, not model validity.

Instead, the audit checks:

- Position drives arrival/strike state; its derived coarse role chooses a pooling cell. These are not two competing free regression coefficients.
- Runs and balls are generated jointly through events. The algebraic rate runs/balls is not entered as a separate redundant predictor of runs.
- Other-player context classes are mutually exclusive: outside NZ, NZ other grounds, exact ground. For non-NZ targets, background and exact ground. Each weighted delivery enters one context class.
- The target player's counts are subtracted from every prior pool, then added once in the own-player posterior update. Empty context/phase/order cells use same-context role aggregates only when the exact cell is empty.
- Age weights, prior strength, sparse cells and player/context ablations require sensitivity and later-match evaluation. Nested categorical conditioning can still be unstable without classical multicollinearity.

These measures address redundant evidence and sparse-context identification. They do not make empirical-Bayes population priors fully Bayesian or propagate all their estimation uncertainty.

## State and arithmetic checks

The test suite covers run and end rotation, over changes, extras, free-hit persistence, ordinary dismissal suppression, nonstriker run-outs, no extra batter after the innings ends, all-out, illegal-delivery safety cap, ties/DNB, exact identities, repeatability and future perturbations. Per-simulation assertions reconcile batter runs+extras to team totals and non-wide deliveries faced to legal balls+no-ball deliveries; legal balls never exceed the declared budget.

Independent review caught terminal-delivery errors. Schema1.1 now retains boundary allowance and respects non_boundary. A winning wide/no-ball penalty stops later actions; completed running wins are capped at the required runs, later run-outs are suppressed, and a winning boundary retains its allowance. This follows the winning-hit/extras principle in MCC/ICC rules. The source does not record every within-delivery timing detail; extra-boundary classification is inferred from the recorded breakdown and no claim of complete cricket physics is made.

Reference: [MCC2026 Laws](https://www.lords.org/getmedia/a4ef9f77-2a25-4f5b-a286-4601f08e6334/Laws-of-Cricket-2017-Code-4th-Edition-%282026%29_5.pdf), result/winning hit or extras. Post-result disciplinary penalties, obstruction, deliberate short running and mid-match revisions are outside this supported model.

## Frozen scope

The requested balls/opportunity, first/chase and declared-budget addition is implemented. No bowling-identity, weather, DLS, pressure-response or new algorithm family is added after evaluation. Remaining work is verification, mathematical review and packaging. If later-match results do not support predictive improvement, v3 stays an explicitly experimental comparison and the existing default is retained.
