# Captured prices and batting opportunity: release evidence

11 November 2026. This update changes presentation and source adapters, not either cricket engine or the frozen evaluation artifacts.

## Captures

The owner requested the two recent India/New Zealand top-batter captures already discussed: source events 36278925 (7 November) and 36290098 (10 November). Each has separate India/New Zealand markets and sixteen candidate selections per team. A market candidate list is not the confirmed XI.

- 7 November: the cutoff state has seven recorded India back offers and nine New Zealand back offers. Missing offers remain null. LTP and its observed-field timestamp remain separate.
- 10 November: the cutoff state is CLOSED with in-play flags. The market-definition start times changed during recording. The saved final schedule/cutoff is 06:00 UTC, while recording begins 02:20 UTC. This capture cannot establish an open pre-match price comparison; the app prominently says so and preserves the source definitions.
- Both captures come from a delayed-key collector. Recorded offer fields and sizes do not certify historical execution. Receipt time is not the underlying trade time. No ten/five/two-hour checkpoint history is claimed.
- The replay reads only closed gzip archives, restores initial images, applies size replacements/removals, and excludes updates received after the displayed cutoff. Source manifest/replay hashes and actual coverage are retained. The app contains sanitized derived records, not AWS credentials or raw stream archives.
- Captured-price selection never joins automatically to the current probable-XI forecast. There is no EV, CLV, live-feed or paper-ledger claim in this narrow release.

## Opportunity

Historical participant distributions use the retrospectively eligible delivery-model population: 1,921 team innings in 1,023 matches across 22 teams; NZ contributes 109 innings in 59 matches. Overall coverage ends 2 November 2026; the NZ subset ends 22 November 2025. The app displays actual dates and sample sizes for each selection. These are not the broader history population or the India/NZ evaluation cohort.

Participants are players whose runs are not null: zero-run and zero-ball appearances count, DNB does not. First-innings and chase distributions are separated. Source-named grounds are retained without silently merging renamed grounds. Frequency denominators are team innings; repeated summary columns in the long table are never summed.

The current forecast shows the count distribution and expected number who bat, exactly equal to the sum of player P(bats). It does not assume independent participation. V3 already models shared wickets, balls and a generated chase target; no new probability correction is added.

The retrospective v3 diagnostic shows too much batting participation and deficient aggregate score tails. Total simulated faced balls are fairly close to observed totals, with allocation differences by role. The small NZ slice does not have an overall century deficit. Global tail inflation is therefore inappropriate. A shrunk simulated balls-already-faced state is a candidate for a separate strict-past development experiment, not an accepted upgrade.

## Verification

- `python -m unittest discover -s outputs/cricket_app/tests -v`: 36 tests pass, including both capture selectors, both engines, NZ diagnostic denominators and historical-depth sample counts.
- `python -m unittest discover -s outputs/second_odi_betfair -p test_stream_prices.py -v`: six replay tests pass.
- `python -m unittest discover -s outputs/model_tail_review -p test_depth_history.py -v`: two participation-definition tests pass.
- Source hashes confirm `engine.py`, `ball_engine.py`, `data/ball_validation.json` and `data/validation_v2.json` are identical to the preceding published checkout.
- Independent review found no blocking issue for the bundled data; its malformed-schema and final-schedule-label recommendations were implemented and tested.

Live hosting verification is recorded separately from these local checks. A repository push or HTTP health response alone is not evidence of the new UI rendering on Streamlit Community Cloud.
