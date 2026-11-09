# ODI conditional probability research prototype implementation plan

Goal: a Streamlit Community Cloud research app for separate top India and top New Zealand batter markets, with auditable historical charts, conditional score scenarios and illustrative dead-heat EV.

Spec: ../probability_and_test_protocol.md and ../batting_order_charts/method_and_app_outline.md. The user approved building, requires a maths greenlight before publishing, and selected Streamlit Community Cloud only.

## Architecture and responsibilities

- Data/model agent: prepare_data.py; data/innings.json; engine.py; tests/test_engine.py; evaluate.py; data/validation.json.
- Interface agent: app.py; requirements.txt; .streamlit/config.toml; README.md.
- Source agent: data/fixture.json, with cited possible XIs and rain/market assumptions.
- Coordinator: history data/assets, licence, integration tests, maths review, packaging, GitHub and Streamlit deployment after user greenlight.

## Execution checklist

- [x] Audit raw Cricsheet and reconcile historical top-share/run-share totals.
- [x] Export history charts and preserve reproducible extraction and plotting code.
- [x] Verify article's possible XIs; label them unconfirmed and identify rain-related settlement mismatch.
- [ ] Test and implement strict-past, recency-discounted empirical-Bayes score sampler with shared regime and depth; retain conditional-independence approximation explicitly.
- [ ] Verify stable player aliases, DNB handling, ties, cold starts, future perturbation and scenario overrides.
- [ ] Render history, model/scenarios, validation and odds/EV views using packaged data only.
- [ ] Run retrospective fixed-setting comparisons and report failures; no untouched-holdout claim.
- [ ] Review actual implemented formulas and worked conditional example with the user.
- [ ] Obtain the user's explicit mathematical greenlight before publishing.
- [ ] Publish source and app only to the selected GitHub/Streamlit route; verify a live usable page.

## Constraints and review focus

Raw D-drive files are read-only. Cloud code must use packaged data, never local D-drive paths. h=4 years and kappa=20 are frozen demonstration defaults, not tuned parameters. Training excludes no-results, missing team innings and D/L; current rain risk is not forecast by this model. Scores are not a physically complete innings simulator. No demonstrated betting edge, historical profit claim or claim of causal swing effects is permitted. Source data attribution and ODC-By licence travel with the artifact.

Test focus: ties versus outright wins; data availability at the forecast cutoff; role changes and unknown players; probable XI versus confirmed team; standard-innings forecasts versus rain-affected market settlement.
