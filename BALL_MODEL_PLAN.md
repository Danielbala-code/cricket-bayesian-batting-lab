# Final shared-opportunity model implementation plan

**Goal:** make balls faced and scoring outcomes drive separate India/NZ top-team-batter forecasts, with physically shared innings resources, explicit pre-match timing and a controlled comparison against v2.

**Authorization:** the user approved this structural addition and explicitly called it the final addition. Publication remains on hold pending their maths greenlight. No new feature families after this pass; later work is correctness, audit, evaluation and packaging.

**Architecture:** retain v2 for comparison; add a delivery-level categorical empirical-Bayes model and a vectorized, resource-constrained two-innings simulator. Fit only past outcomes. Sample delivery outcomes, update strike/balls/wickets/runs, and generate the first-innings target before simulating the chase. Runs become a consequence of opportunity and scoring events rather than independent whole-innings totals. No normal run distribution or causal swing claim.

**Scope:** declared first batting team, expected XIs/order, ground and initial overs limit are scenario inputs. Unknown batting first is a declared50:50 scenario, not a learned toss prediction. Short-overs scenarios enforce the ball budget but are unvalidated extrapolations from standard innings. No forecast rain process, mid-match DLS revisions, bowlers or pressure-response regression is added.

## Data and model interfaces

Create `data/ball_counts.npz` with numeric array `counts` (int64 columns: innings_index, player_index, role_index, phase_index, outcome_index, count). Create `data/ball_metadata.json` with `innings`, `players`, `outcomes`, definitions and extraction audits. Count all eligible source deliveries, including wides/no-balls. Retain excluded metadata with reasons but do not train on it.

- `innings`: id, match_id, date, team, opponent, venue, host_nz, innings_order(1/2), scheduled_overs, eligible, exclusions, players[{id,name,position,runs,balls_faced}], actual_legal_balls, actual_team_runs, actual_extras. Recorded outcome/length is label/audit only.
- `players`: globally indexed stable registry `{id,name}`. Exact aliases may reuse the reviewed v2 crosswalk; never fuzzy-match different people.
- `outcomes`: tuple `[batter_runs, extras, is_wide, is_noball, running_parity, wicket_kind]`; wicket_kind0 none,1 ordinary striker dismissal,2 striker run-out,3 nonstriker run-out. Legal ball iff no wide/no-ball. Balls-faced convention is non-wide deliveries. Running parity excludes no-ball/wide penalty and five-run penalties.
- Roles0/1/2 represent positions1-3/4-7/8-11; phases0/1/2 are first20%, middle60%, last20% of scheduled legal balls. Training phase uses running past legal balls and scheduled limit, never the final actual innings duration.
- Eligibility excludes NR/DL, missing innings, non11-player rosters, super overs and unsupported retirement/penalty/multiple-wicket or nonstandard events. Export exact exclusions rather than silently discarding runs.

`ball_engine.load_ball_data() -> dict` loads metadata and compact arrays.

`ball_engine.forecast_match(data, lineups:dict[str,list[str]], venue:str, cutoff:str, batting_first:str, overs:int=50, half_life:float=2, prior_strength:float=120, n_sims:int=3000, seed:int=7, use_venue:bool=True, use_player:bool=True) -> dict`.

Result: `teams[team]` contains existing player table fields plus mean_balls, median_balls; `scores`Nx11 with DNB=-1; `balls_faced`Nx11; `batted`mask. Match result contains `team_totals`, `extras`, `legal_balls`, `wickets`, `first10_wickets` as arrays, first-team choice and budget; warnings, training_max_date, training_n and model-settings audit. Both lineups are expected orders, not verified XIs. The fit/simulator may expose smaller internal functions for deterministic testing.

## Probability construction

For each expected role, phase and innings order, use age-discounted categorical delivery counts. Outcome support must contain ONLY outcome types observed before cutoff plus a fixed documented law-valid base; future-only vocabulary cannot acquire prior probability. Sort active decoded tuples deterministically so future data perturbations cannot change forecasts.

Use disjoint other-player context pools: outside NZ / NZ other grounds / exact ground (for non-NZ targets, background / exact ground). Exclude the target player's observations from all pooled priors. Sequential smoothing is `(counts + kappa * previous_pmf)/(sum(counts)+kappa)`; apply own-player role/phase/order counts once. A weak symmetric base has total mass0.5 across active outcomes. This remains a discounted empirical-Bayes approximation; do not claim a full hierarchical posterior or independently identified context coefficients.

`prior_strength` is now measured in equivalent DELIVERIES, not innings. Candidate values60 and240 are explicit development choices; do not transfer v2's20-innings interpretation. Half-life candidates2 and4 years. Per-delivery samples use fitted posterior predictive probabilities; no unmeasured form/weather variable or new random-effect family is added in this version.

## State transitions and invariants

- Start openers at the crease, next player index2, zero runs/extras/legal balls/wickets.
- Every event adds batter runs only to the striker, extras separately, and team total=batting runs+extras.
- Wides consume neither legal balls nor faced balls; no-balls consume faced balls but no legal ball. Rotate on running parity, then replace the identified dismissed player at the appropriate end.
- Swap ends after each completed over. Run-outs may dismiss either original batter; ordinary dismissals remove striker. Preserve identity through odd running and over changes.
- No-ball triggers free hit. Free hit persists through illegal deliveries; ordinary dismissals are suppressed on a free hit, while run-outs remain possible. Reset after the next legal ball unless that event is a no-ball.
- Stop at ten wickets, declared legal-ball budget, or a reached chase target. Generate the chase target from the simulated first innings. Never read the actual first-innings score during a pre-match historical forecast.
- Mark a new batter as having batted only if play continues after their predecessor leaves. A nonstriker run-out can yield a genuine zero-ball innings. DNB remains distinct from zero.
- Use a documented safety cap on illegal-delivery loops; raise an error if reached, never silently truncate.

## Leakage and redundancy contract

Decision time: post-XI/toss, before the first match delivery. All empirical counts strictly date<cutoff. Expected historical order comes from last5 prior appearances, fallback6 and deterministic tie breaking; recorded XIs and innings order are reconstructed post-toss assumptions with no vintage proof. Use scheduled50 overs in standard-innings evaluation; realized duration/target/balls faced/current score/order are targets or simulated states, never pre-match inputs. Keep both team cases in one match together in resampling.

No linear regression design matrix is fitted, so VIF is not a meaningful pass/fail criterion for this categorical sampler. Audit disjoint context pools, player-vs-role shrinkage, sparse cells, redundant deterministic summaries and stability instead. Do not feed runs, balls and runs/ball as separately estimated redundant predictors. Label observational associations separately from causal mechanisms.

## Tasks and tests

1. **Data/model agent A:** `prepare_ball_data.py`, compact data and extraction tests. Reconcile per-innings batter runs/extras/legal/non-wide counts to raw JSON; test no-ball/wide/retirement flags and phase against scheduled limit.
2. **Engine agent B:** `ball_engine.py` and `tests/test_ball_engine.py`. First write failing deterministic state tests, then implement. Test strike, over swap, wickets/run-outs, free hits, all-out, budget, chase stopping, shared resources, ties/DNB, exact aliases, repeatability and future perturbation including a future-only outcome category. Check all probability vectors and outcome shares normalize.
3. **Interface agent C:** add explicit model comparison controls, both probable lineups, batting-first and overs scenarios. Display balls-faced distributions from the simulator and ball-budget audit. Keep original model available and mark v3 experimental until evaluation. Update maths text to exact engine behavior, not a specification that code does not implement.
4. **Coordinator:** validate contract integration; freeze four-candidate grid on2019-2022 development cases before fitting; evaluate selected candidate on the fixed116-case2023+ set where supported, reporting all exclusions and paired same-row v2 comparison. Report per-team/year Brier, opportunity/50+/100+/interval diagnostics and actual simulator runtime. Data already explored remain retrospective, not untouched confirmation. New test outcomes must not drive another parameter search.
5. **Independent review:** mathematical state transitions, same-match target leakage, future perturbation, unsupported short-overs claims, disjoint pools and multicollinarity interpretation. No publication or stronger forecasting claim without evidence and user review.

## Correctness amendment before fitting

Independent review found that terminal running runs and a wicket after a winning extra were counted atomically. Before fitting, schema1.1 added a seventh outcome element, boundary_allowance, derived from recorded boundary/non_boundary information. Tests now cap completed running wins, retain boundary allowances, and stop after a winning penalty before a later dismissal. Six-field synthetic inputs remain backward compatible. The full data were regenerated and all extraction totals reconciled before the frozen evaluation started.

## Acceptance

Correctness and leakage checks must pass independently of metric improvement. If v3 fails to beat the simpler comparator consistently, retain it only as a labelled research comparison and state failure. Shortened-innings handling means declared-budget simulation; it is not a validated DLS/weather model. Feature scope is frozen after this implementation.
