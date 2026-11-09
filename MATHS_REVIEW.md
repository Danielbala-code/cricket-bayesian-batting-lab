# Maths review before publication

Publication was authorized by the owner on9November2026 after review. The pending-publication wording below records earlier review stages; the model assumptions and limitations continue to apply. The app presents this material in a dedicated Maths tab and offers a separate Guide & glossary.

The v1/v2 sections below describe the score sampler; the final v3 section separately describes the implemented experimental delivery engine. User publication greenlight is pending. Results below are model-conditional, uncalibrated research estimates; they are not verified betting probabilities.

## 1. Information available before the forecast

X = selected probable XI in expected batting order, ground, team, cutoff date and declared model controls. Only eligible matches dated strictly before the cutoff are used. Training excludes no-results, missing innings and D/L outcomes. There are 2,096 eligible team innings in 1,048 matches. Original v1 settings h=4 years and kappa=20 were demonstration choices. When data/tuning_v2.json is present, the sidebar uses its selected_config: a six-setting comparison of h=2, 4 or infinity and kappa=5 or 20 on the earlier 2019–2022 development period. The later 2023-onward evaluation is stored separately in validation_v2.json. Infinity means no recency decay and weight 1 for every eligible observation. Read those artifacts for actual selected values and sample sizes; do not infer them from the v1 example below. Player IDs are stable; verified aliases map full names to Cricsheet names. Unknown players receive a visible role-prior fallback.

## 2. Recency, likelihood and prior

For history j, w_j = 2^[-ageYears_j/h]. Four-year-old evidence receives half the weight of current evidence. For score x and the selected role/regime, count weighted observations C_x = sum_j w_j 1(R_j=x).

The broad other-player role prior starts as P_role(x) = [C_role(x)+0.5 b(x)]/[sum_x C_role(x)+0.5], where b(x) is proportional to exp(-x/25) on integer support0 through max(300, training maximum). The0.5 mass is a small explicit exponential tail prior, not observed data.

Each subsequent location, exact-ground and player update uses:

P_next(x) = [C_next(x)+kappa P_previous(x)]/[N_next+kappa].

Own player history matches the current expected role (1-3,4-7,8-11) and regime; pooled score priors exclude that player. Location means NZ versus outside NZ, not every country separately. Turning venue off skips location and ground updates. A12th participant is not treated as an expected XI role.

This is an empirical-Bayes / discounted-likelihood approximation. Priors are learned from overlapping historical pools, and weights are fractional. It is not a fully Bayesian hierarchical model with posterior uncertainty propagated through every prior-estimation step.

## 3. Shared future scenarios

Z=1 means at least two wickets in the first ten overs; Z=0 means fewer. The prior weighted regime frequency is smoothed through location, ground and team using (weighted events+kappa previous_rate)/(weighted innings+kappa). In the original v1 worked run below, the mixture was P(Z=1|X)=0.3773267 and P(Z=0|X)=0.6226733. This is not a fixed mixture: it changes with selected controls and context. The two-wicket threshold, exponential prior scale 25 and tail mass 0.5 remain fixed structural assumptions rather than parameters selected by the six-setting grid.

Each simulated innings samples Z, then one shared batting depth from the corresponding smoothed empirical depth distribution. That draw decides which expected batting slots get to bat. Conditional score draws then use the player's role/regime PMF. DNB is stored as-1 and excluded from batting-only means and histograms.

P(A|X) = sum_z P(A|Z=z,X) P(Z=z|X).

This captures shared regime and batting-opportunity dependence. Residual score draws are conditionally independent, and the sampler does not conserve team balls or a chase target. It is not a full physical innings simulator or a causal swing model.

## 4. Original v1 reproducible conditional example

This retained v1 example will not generally match a forecast using the selected v2 defaults. With the preview's probable India XI, Seddon Park, cutoff2026-11-10, h=4, kappa=20,10,000 simulations and seed7:

- B: Rohit and Gill both bat and each scores<=20:2,045 simulations.
- A and B: Rahul reaches50+ and B occurs:491 simulations.
- P(B|X)=2045/10000=20.45%.
- P(A and B|X)=491/10000=4.91%.
- P(A|B,X)=491/2045=24.01%.

The identity P(A|B,X) P(B|X)=P(A and B|X) holds. These are simulated conditional probabilities under the model, not measured historical conditional rates or causal effects. Standard Monte Carlo error is additional to model uncertainty;10,000 draws do not create10,000 observed matches.

## 5. Top scorer, ties and EV

In simulation s, let I_si indicate that player i is tied or solely highest, and K_s be the number of tied leaders. Then:

p_top_i = mean_s I_si; q_i = mean_s(I_si/K_s); sum_i q_i=1.

q is the expected dead-heat settlement share, distinct from the probability of being among the tied leaders. At decimal odds d, expected net return per unit stake under standard dead-heat rules is d q_i-1 if DNB loses the stake. If DNB returns the stake, unconditional EV is d q_i-P(bats), and break-even q is P(bats)/d. The app lets the user select this assumption; it does not assert a bookmaker's rule.

In the original v1 illustrative standard-innings run above, Rahul q=0.11025 and P(bats)=0.8896. At the supplied7/1 price (decimal8), DNB-loses EV is8*0.11025-1=-0.118, or-11.8%; DNB-refund EV is8*0.11025-0.8896=-0.0076, or-0.76%. Neither is a recommendation. DNB-loses break-even q is1/8=12.5%; refund break-even q is0.8896/8=11.12%. The live rain-affected market may settle at25 overs, which this standard-innings model does not price.

## 6. Checks and retrospective evidence

Six engine tests pass: exact dead-heat accounting; time decay; reproducibility/cold starts/future-outcome exclusion/DNB; known-player aliases and forced regimes; early-wicket counting; and12th-participant exclusion. Streamlit checks cover initial rendering, a forecast, conditional queries, odds and invalid XIs.

Original v1 fixed retrospective diagnostic,40 recent eligible India/NZ team innings, strictly prior training and expected order reconstructed from earlier matches:

| Method | Mean multiclass Brier, lower better |
|---|---:|
| Model | 0.8289 |
| Venue-blind model | 0.8318 |
| Historical top-share baseline | 0.8754 |
| Uniform XI | 0.9091 |

These40 cases are a small retrospective diagnostic chosen from previously explored history. The venue difference is small and not established as statistically meaningful. This is not an untouched holdout, a calibration study, a causal test or a profit backtest.

The Validation tab prefers the later-period v2 evaluation when bundled, with its actual dates, selected settings and sample count. The earlier v1 diagnostic is labelled as such if v2 is absent. The separate distribution_checks_v2.json diagnostic compares supplied medians, zero/50+/100+ rates, P(bats), predictive interval coverage and role slices on its own frozen sample and declared settings. These comparisons do not establish calibration, and their sample/settings must not be conflated with the full v2 evaluation. No post-hoc recalibration is implied.

## 7. Publication conditions

Possible XIs remain unconfirmed; the app exposes that assumption. India data end2026-10-03, missing the November series. There is no weather or DLS model. V2 represents a standard innings; experimental v3 permits a declared pre-start reduced-overs budget, which is an unvalidated extrapolation rather than rain-adjusted market pricing. Balls-faced charts are descriptive and count non-wide deliveries including no-balls, with an explicit definition. Publication should label the work an exploratory research prototype and retain these limitations.

## Completed v2 evaluation: evidence does not establish an improvement

The six-setting development search selected h=2 years and kappa20 on60 earlier2019-2022 cases. All116 eligible2023+ India/NZ team cases in106 matches were then evaluated, using last-five-prior-appearance expected order. The original4-year comparator was recomputed on exactly the same cases and order assumptions.

Mean Brier: selected2-year0.83396; original4-year0.83196; selected venue-blind0.83328; selected without early-trouble conditioning0.82926. Lower is better. Selected minus original difference is+0.00200, with an exploratory paired-game95% interval[-0.00179,+0.00565]. The intervals for venue and scenario comparisons also include zero. Do not describe tuning or added context as demonstrated improvements; do not select the ablation using these same final cases.

Across1,276 player forecast rows,984 actually batted. Expected aggregate batting participation was79.8% versus77.1% observed. Conditional50+ rate was18.1% predicted versus21.1% observed;100+4.4% versus5.8%. Central80% intervals covered84.2% overall and92.3% in the expected lower-order group. Discrete intervals can exceed nominal coverage; these patterns require further modeling review and do not certify calibration.

The earlier491/2045 worked example remains ORIGINAL V1 at h=4. Across the six plausible h/kappa settings, the same future conditional question ranged approximately23%-33% in50,000 simulations per setting. This spread is parameter sensitivity, not a credible interval. More draws reduce simulation noise without addressing these modeling assumptions. Seven engine tests now pass, including an unconditioned-regime ablation test.


## Experimental v3: shared opportunity, separate from v2 evidence

The default remains **v2 score sampler**. The optional **v3 shared-opportunity (experimental)** engine is implemented in ball_engine.py. Its completed evaluation, if supplied, belongs in a separate validation_v3.json artifact. The v2 Brier, distribution and sensitivity results above do not validate v3.

Inputs are both probable XIs in expected order, ground, strict cutoff, declared batting-first team and a pre-start overs budget of 20–50. Unknown batting first draws each team with probability 0.5; this is a declared scenario, not a toss forecast. The second team's target is the **simulated first innings total + 1**. No actual first-innings target is used by the pre-match call.

### Delivery probabilities and sparse-cell fallback

For eligible history i strictly before the cutoff, w_i = 2^(-ageYears_i/h). Weighted categorical counts are C_e = sum_i w_i n_(i,e), where each event tuple contains batter runs, extras, wide flag, no-ball flag, running parity and wicket kind. Active support is past observed event tuples plus nine fixed law-valid base events (legal 0/1/2/3/4/6, a wide, a no-ball and a dot-ball ordinary wicket). It does not assign prior probability to future-only catalog categories.

Each expected batter has role/phase/innings-order cells. Roles are 1–3, 4–7 and 8–11. Phases are first 20%, middle 60% and last 20% of the **declared scheduled legal-ball budget**, never final realised innings duration. Historical phase uses scheduled history limits. For a target NZ ground, other-player pools are disjoint outside NZ, NZ other grounds and exact ground. For other targets they are background and exact ground. All pools exclude the target player.

Start with p_0(e) = [C_background(e) + 0.5/K] / [N_background + 0.5], where K is active event count. Each location/ground update and the final own-player update uses p_next(e) = [C_next(e) + kappa p_previous(e)] / [N_next + kappa]. An empty pooled context/phase/order cell falls back to that same context's aggregate for the expected role across phase/order. Own-player matching role/phase/order counts enter once; they do not use this fallback. Turning player history off skips the own-count contribution. Turning venue adjustment off pools locations.

Kappa is equivalent **deliveries**, unlike v2's equivalent innings. UI controls expose h=2 or 4 years and strengths 60, 120 or 240 deliveries. These controls are declared choices unless a separate earlier-development selection and evaluation are reported. Priors and cells are learned from data with fractional weights; this is a discounted empirical-Bayes approximation, not a complete hierarchical posterior over all context estimates.

### Simulated state and accounting

The declared legal-ball budget is 6 × overs. The state tracks striker/non-striker identity, next batter, batter runs, extras, legal balls, non-wide faced balls, wickets and free hit. Batter runs are credited to the original striker. Running parity rotates strike, the original dismissed identity is replaced at its resulting end, and a completed over swaps ends. Wides consume neither legal balls nor faced balls. No-balls consume faced balls but no legal ball and trigger a free hit; it persists through illegal deliveries. Ordinary dismissals on a no-ball/free hit are suppressed, while striker/nonstriker run-outs can remain.

An innings stops at the legal-ball budget, ten wickets or a reached simulated chase target. A replacement batter is marked batted only when play continues after the wicket. DNB is -1 in the score array; a genuine zero-ball batting appearance is retained. Conditional means/medians for runs and faced balls condition on the batted mask. P(50+) is unconditional. Event sampling has a 20 × budget + 100 safety cap and raises if reached, without silent truncation.

The UI checks each team's legal balls <= budget; sum(batter runs) + extras = team total; sum(faced balls) = legal balls + no-ball deliveries; and every chase target = the other simulated total + 1, while first-innings target is zero. Shared accounting is a correctness property, not evidence of calibrated probabilities. Delivery probabilities are fixed within the role/phase/innings-order cell: there is no fitted bowler identity, run-rate pressure response, weather or DLS process.

The v2 early-trouble fitting control is disabled for v3. Its separate what-if query filters generated first-ten-over wickets, reports matching counts/frequency, and recomputes conditional shares. It does not feed future wickets into training. Existing low-score conditional queries likewise count the simulated outcomes. The same dead-heat q and DNB-dependent EV arithmetic applies to either model.

A 1,000-simulation, 20-over India-first Streamlit check passed: both teams respected a 120-legal-ball budget, New Zealand's target equalled each simulated India total plus one, balls-faced/audit output rendered, and a malformed opponent XI was rejected. This is an integration check, not a forecasting-quality claim. Shortened innings remain unvalidated extrapolations; no weather-adjusted price or exact physical match model is claimed.

## Final v3 result and solved model-conditional example

V3 settings were selected on60 earlier2019-2022 cases: h=2 years, prior strength240 equivalent deliveries. On91 supported later team innings in82 matches, mean Brier was0.80280, compared with v2's0.82156 on exactly the same cases and expected orders. V3 minus v2 paired-game95% resampling interval was[-0.03029,-0.00608], ignoring series/era dependence and adaptive research selection. All seven available team/year point comparisons favoured v3. This is promising retrospective evidence, not untouched or prospective confirmation. Removing venue changed Brier to0.80372; its incremental contribution remains inconclusive.

V3 predictive run rates remain imperfect: conditional50+16.4% predicted versus20.9% observed;100+3.2% versus5.4%. Predicted batting participation80.8% versus77.5% observed. Central80% run intervals cover82.9%, and balls-faced intervals82.3%, on observed batters in the supported cohort. No recalibration was fitted to these test outcomes.

For the preview's India/NZ XIs at Seddon Park,50 declared overs, unknown batting first as an explicit50:50 scenario, h=2/kappa240,10,000 simulations/seed7: B=(Rohit and Gill both bat and each scores<=20) occurred1,846 times; A and B=(Rahul50+ and B) occurred521 times. Therefore P(A|B,X)=521/1846=28.22%. Unconditional simulated P(Rahul50+)=21.6%. The model explicitly generates balls faced and an opponent-generated chase target; this example is conditional on those assumptions and not a rain-adjusted betting probability. More simulations do not fix model misspecification.

Schema1.1 retains a seventh event field for boundary allowance. Correctness tests cap completed running wins, preserve full boundary allowances and suppress actions after a winning extra. All20 extraction/state/leakage tests and the v3 two-XI/20-over Streamlit check pass. Current-score, actual chase-target and realised-duration inputs are not used in a pre-match forecast. The real-world availability audit remains REVIEW REQUIRED because archived rosters/toss/vintages are reconstructed assumptions and unsupported events narrow the test population.

The final feature addition is complete and frozen. Please review this mathematics before authorizing publication of the Streamlit app and its GitHub source. No publication has been performed.

