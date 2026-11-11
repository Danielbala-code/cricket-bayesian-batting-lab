# ODI batting intelligence

A portable Streamlit app for historical ODI batting-order analysis, probable-XI simulation and transparent top-team-batter price comparisons.

The interface separates History, Forecast & scenarios, Validation, Captured prices, Maths, and Guide & glossary. Mathematical controls and formulas live in Maths; the guide provides a short walkthrough and plain-English definitions. Publication was authorized by the owner after mathematical review; the research limitations below remain in force.

## Run

Use Python 3.12. From this directory:

```sh
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, choose this repository and `app.py` as the entrypoint (or the repository-relative path if nested). All data live in `data/`; no access to the original local archive is required.

## Use

The default **v2 score sampler** remains the research comparator. **v3 shared-opportunity (experimental)** is a separate delivery model: enter both probable XIs, a declared first-batting team (or an explicit 50:50 unknown scenario), and 20–50 pre-start overs. Its prior strength is in equivalent deliveries, unlike v2's equivalent innings. Shortened-overs scenarios are unvalidated extrapolations and do not estimate rain, washouts or DLS revisions.

The v3 output includes a legal-ball budget and run/extras/balls-faced/target accounting for both teams, plus a simulated balls-faced distribution. A first-ten-over wickets query filters simulated outcomes; it does not use the future wicket count to fit the pre-start delivery probabilities. Existing validation and sensitivity panels are explicitly v2 unless a separate v3 artifact is bundled. The default model is not changed merely because the experimental model is more detailed.

The final v3 comparison is now bundled in `data/ball_validation.json`. Four delivery-model settings were selected on60 earlier2019-2022 cases, choosing a2-year half-life and240 equivalent deliveries of prior strength. On91 supported later team cases in82 matches, v3 Brier was0.80280 versus0.82156 for v2 recomputed on identical cases. The paired match-resampling interval for v3 minus v2 was[-0.03029,-0.00608]. All seven available team/year point estimates favoured v3, though individual slices are small. Venue's incremental difference remained inconclusive. This is exploratory retrospective evidence, not untouched confirmation, calibrated betting probabilities or a causal venue result.

V3 still underpredicts aggregate50+ and100+ rates and overpredicts batting participation. Correctness tests and v2/v3 Streamlit checks cover the implementation. The leakage review in [BALL_MODEL_AUDIT.md](BALL_MODEL_AUDIT.md) remains REVIEW REQUIRED for historical XI/toss/vintage provenance and outcome-conditioned exclusions. No weather, DLS, bowler or new algorithm families are added in the captured-price update. The published app remains a research prototype.

Review the probable team and batting order, enter exactly 11 unique names, and click **Run forecast**. Controls include recency half-life, shrinkage, venue adjustment and innings scenario. Results remain visible when controls change; a notice asks you to rerun.

History filters expose sample counts and distinguish run shares from fractional top-team-batter outcomes. Forecast `q` is expected dead-heat settlement share; `p_top` includes tied leaders. DNB is distinct from a zero. The conditional query makes the batting requirement explicit.

Captured prices come from the owner's existing Betfair collector for the India/New Zealand events dated 7 and 10 November 2026. The bundled sanitized artifact preserves event/market/selection IDs, recorded back/lay prices and sizes, LTP separately, actual observation times, feed-delay status and coverage. The 7 November snapshot has recorded pre-play back offers; the 10 November cutoff state is closed/in-play and is explicitly unsuitable for a pre-match price comparison. Its stored schedule changed during recording, so the saved final scheduled time is not treated as an immutable original start. Candidate fields contain more than eleven players and do not establish the actual XI. No credentials, live connection, order placement or automatic association with the current forecast is included. The manual Rahul example and bookmaker vig/de-vig controls have been removed.

The forecast now shows the distribution of how many players bat and its expected value, with DNB separate from zero. History includes all-team/NZ/exact-source-ground participant distributions split into batting first and chasing, from the same retrospectively eligible delivery-model population: 1,921 team innings in 1,023 matches, including 109 NZ innings in 59 matches. This population differs from the broader descriptive tables elsewhere in History. The NZ subset ends 22 November 2025; the overall archive ends 2 November 2026. Validation includes role/location/innings-order opportunity and tail diagnostics for the already-inspected v3 comparison. These views do not change model probabilities or establish a causal venue effect.

The capture adapter rejects duplicate identities, malformed objects, invalid numeric values and information observed after the snapshot cutoff. Tests verify LTP is never used to fill a missing offer, zero-run appearances count as batting, and closed/in-play/missing-image records are labelled unsuitable. Reproduce the source extraction from the workspace with `python outputs/collector_export/export_captures.py`; the exporter uses the existing SSH-over-SSM access helper and reads closed files without changing the collector. Raw exports remain outside the app bundle.

## Evidence and limitations

Data originate from Cricsheet. Historical tables cover male ODIs from 2016 and end on 2 November 2026; India's observed history ends 3 October 2026. Probable lineups, cutoff date, innings length and batting order remain scenario assumptions. No weather or DLS model is included. The validation tab displays only the actual bundled validation artifact and makes no invented accuracy claim.

This app supports analysis; a positive estimated EV is not evidence of a durable betting edge. Check selection, prices, bookmaker dead-heat rules, void rules and commission independently.

## What I set out to investigate

Does New Zealand venue context change which batting roles supply a team's highest scorer, and can a transparent conditional model improve probability forecasts? The project covers separate India and New Zealand team markets. The hypotheses and modelling questions were proposed by the project owner; AI agents helped inspect data, implement components and review assumptions.

Important decisions were to discount older innings, distinguish middle order from the tail, keep DNB separate from zero, pool sparse ground samples, preserve scenario dependence, handle dead heats explicitly and require mathematical review before publication. An initial archive check caught ODM data rather than ODI data, preventing an invalid training population. Full player names were matched to verified stable identities; genuine cold starts remain visible.

## Method and tests

See [MATHS_REVIEW.md](MATHS_REVIEW.md) for the implemented discounted empirical-Bayes categorical posterior predictive, shared early-wicket regime, shared batting depth, conditional queries and EV formulas. Residual player scores are conditionally independent; this is not a full physical innings simulator, fully Bayesian hierarchy, causal swing model or calibrated betting system.

Run `python -m unittest discover -s tests -v` for engine, evaluation-maths, capture-adapter and Streamlit integration checks. `python evaluate.py` reproduces the original v1 fixed-setting40-case diagnostic. `python tune_v2.py` reproduces the later development/test protocol: six candidates, h=2/4/no decay and prior strength5/20, selected using60 deterministically spread2019-2022 cases, followed by all116 eligible2023+ team cases in106 matches. Expected order uses each player's last five prior appearances, with a declared cold-start fallback.

Development selected h=2, kappa20. Later-period Brier was0.83396, versus0.83196 for the original4-year setting recomputed on identical cases and order assumptions. Venue-blind scored0.83328 and unconditioned-regime0.82926. All paired match-resampling intervals for these differences include zero. Tuning and the added context did not demonstrate an improvement. The no-regime ablation is exploratory and was not selected retrospectively as a new winner.

`python distribution_checks.py --max-targets 0` checks skew, zero scores,50+/100+ rates, batting opportunity, predictive intervals and calibration summaries. The current model underpredicts aggregate50+/100+ rates, and the lower-order intervals are conservative. `python sensitivity_check.py` shows parameter sensitivity of a future conditional scenario; its spread is not a credible interval. Full-history exploration precludes an untouched-holdout claim; no calibration, causal or betting-edge claim is made. The app uses development-selected settings as an inspectable candidate, not as a proven upgrade.

## AI collaboration and review

Work was divided into data/model, interface, source verification and coordinator/reviewer tasks. The owner challenged sample definitions, time weighting, causation claims, joint probability assumptions and settlement arithmetic. Reviews caught unknown-name fallbacks,12th participants from concussion substitutions, and the distinction between DNB-loses and DNB-refund EV. These corrections are part of the research record rather than evidence that every model assumption is solved.

## Sources and attribution

Derived historical data: [Cricsheet](https://cricsheet.org/), distributed under ODC Attribution; see [data/CRICSHEET_LICENSE.txt](data/CRICSHEET_LICENSE.txt). The archive does not guarantee complete coverage and omits some matches. Probable XIs and rain context: [Betfair preview,9 November2026](https://betting.betfair.com/cricket/india-cricket/new-zealand-v-india-third-odi-tips-back-gill-at-11-4-in-rainy-hamilton-091126-194.html). Expected-value learning reference supplied by the owner: [Priorslab](https://priorslab.com/#/learn/ev-basics). Prices require a fresh bookmaker check. No automated betting is included.
