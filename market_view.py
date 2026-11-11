"""Read-only captured market and batting-opportunity views.

Captured prices have no join to the current forecast. No EV or live-price
calculation belongs in this historical browser.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def render_captured_prices(path):
    st.subheader("Recorded Betfair top-team-batter markets")
    st.info("Historical captured prices are independent of the current forecast. Match identity, venue, lineup and cutoff have not been joined; no EV or model backtest is calculated here.")
    st.caption("This is an archived collector view, not a live feed. All listed runners are market candidates, not a confirmed playing XI. Missing offers remain missing; last traded price is not substituted for an offer.")
    if not Path(path).exists():
        st.info("The historical capture artifact has not yet been bundled in this build.")
        return
    from captured_markets import load_captures, quote_quality
    try:
        events = load_captures(path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        st.error(f"The captured-price artifact could not be validated: {exc}")
        return
    if not events:
        st.info("No captured events are available.")
        return
    event_ids = [event["event_id"] for event in events]
    labels = {event["event_id"]: f"{event['event_name']} · final stored schedule {event['scheduled_start_utc']}" for event in events}
    event_id = st.selectbox("Captured match", event_ids, format_func=labels.get)
    event = next(event for event in events if event["event_id"] == event_id)
    for warning in event.get("warnings", []):
        st.warning(str(warning))
    if event.get("delayed_key"):
        st.warning("Delayed-key historical capture. Collector receipt times do not establish when an executable exchange quote was available.")
    st.caption(f"Event ID: {event_id} · final stored schedule / snapshot cutoff: {event['scheduled_start_utc']} · sanitized export: {event['exported_at_utc']} (UTC)")
    markets = event.get("markets", [])
    if not markets:
        st.info("This event has no captured batting markets.")
        return
    market_ids = [market["market_id"] for market in markets]
    market_labels = {market["market_id"]: f"{market['team']} · {market['name']}" for market in markets}
    market_id = st.selectbox("Captured team market", market_ids, format_func=market_labels.get)
    market = next(market for market in markets if market["market_id"] == market_id)
    rows = market.get("rows", [])
    table_rows = []
    for row in rows:
        table_rows.append({
            "Candidate": row.get("player"),
            "Back offer": row.get("best_back"),
            "Back size": row.get("best_back_size"),
            "Lay offer": row.get("best_lay"),
            "Lay size": row.get("best_lay_size"),
            "Last traded price (not an offer)": row.get("last_traded_price"),
            "Recorded quote status": quote_quality(row),
            "Selection ID": row.get("selection_id"),
            "LTP field last seen (UTC)": row.get("ltp_field_last_seen_at_utc"),
            "Row as-of (UTC)": row.get("as_of_utc"),
            "Market update received (UTC)": row.get("last_market_update_received_at_utc"),
            "Market status": row.get("market_status"),
            "In play": row.get("in_play"),
            "Initial image seen": row.get("initial_image_seen"),
            "Runner status": row.get("runner_status"),
        })
    st.markdown(f"**{market['team']} · {market['name']}**")
    st.caption(f"Market ID: {market_id} · {len(rows)} recorded candidate rows. Times are observation metadata, not a guarantee of quote freshness or execution.")
    statuses = pd.Series([row["Recorded quote status"] for row in table_rows], dtype=str).value_counts()
    st.write("Quote-status counts: " + " · ".join(f"{label}: {count}" for label, count in statuses.items()))
    if rows and not any(row["Recorded quote status"] == "Recorded back offer" for row in table_rows):
        if not any(row.get("market_status") == "OPEN" and row.get("in_play") is False for row in rows):
            st.warning("No open pre-start offers in this capture. The table retains raw historical fields for inspection; they are not usable price quotes.")
        else:
            st.warning("No usable pre-start back offers identified from this capture's metadata. Inspect the recorded status and source fields.")
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch", height=min(630, 38 * len(rows) + 42))
    st.caption("Back and lay prices/sizes are recorded fields. The status column checks the available capture metadata; closed, suspended, in-play, removed-runner and missing-image records must not be treated as usable pre-start offers. Sizes are exchange-recorded amounts; no currency conversion or execution guarantee is implied.")
    with st.expander("Capture coverage & source provenance"):
        coverage = event.get("coverage", {})
        st.write("First received:", coverage.get("recording_first_received_at_utc", "Not recorded"))
        st.write("Last received:", coverage.get("recording_last_received_at_utc", "Not recorded"))
        st.write("Source archive read:", event.get("source_read_at_utc", "Not recorded"))
        st.write("Records scanned:", coverage.get("records_scanned", "Not recorded"))
        st.json(coverage, expanded=False)
        for field, label in (("source_manifest_sha256", "Source manifest SHA-256"), ("replay_source_sha256", "Replay exporter SHA-256")):
            if event.get(field):
                st.write(label + ":", event[field])
        if event.get("market_time_observations"):
            st.write("Recorded market-time definitions (source observations, not a verified actual start):")
            st.dataframe(pd.DataFrame(event["market_time_observations"]), hide_index=True, width="stretch")
        st.caption("Coverage describes what the collector recorded. A scheduled start and a source filename do not prove complete pre-start price history.")


def render_participation(scores):
    from captured_markets import participation_summary
    summary = participation_summary(scores)
    st.markdown("**Batting opportunity · current simulated innings**")
    left, right = st.columns([1, 3])
    left.metric("Expected number who bat", f"{summary['expected_batters']:.2f}")
    left.caption(f"{summary['n_sims']:,} simulations. DNB = −1; a zero-run batting appearance counts as batting.")
    frequencies = dict(zip(summary["counts"], summary["probabilities"]))
    depth = pd.DataFrame({"Number of batters": range(2, 12), "Probability": [frequencies.get(k, 0.0) for k in range(2, 12)]})
    right.bar_chart(depth.set_index("Number of batters"), color="#087e8b")
    st.caption("The chart counts batters in each simulated team innings, from 2 through 11. The expectation equals the sum of player P(bats), already shown in the forecast table; it does not assume independent participation. These are unchanged model outputs, with no opportunity correction applied.")


def render_historical_depth(data_dir):
    data_dir = Path(data_dir)
    summary_path, depth_path = data_dir / "batting_depth_summary.csv", data_dir / "batting_depth_history.csv"
    if not summary_path.exists() or not depth_path.exists():
        return
    summary, depth = pd.read_csv(summary_path), pd.read_csv(depth_path)
    st.subheader("How many batters actually bat?")
    st.caption("Descriptive history from the v3-eligible ODI archive, across all 22 teams. This is a separate population from the India/New Zealand forecast and retrospective diagnostic below; these selectors do not change other History tables.")
    left, right = st.columns([2, 1])
    scope = left.selectbox("Historical depth population", summary.scope.unique())
    context = right.selectbox("Historical batting context", ["Combined", "Batting first", "Chasing"])
    selected = summary[(summary.scope == scope) & (summary.batting_context == context)]
    if selected.empty:
        st.info("No eligible innings for this population and batting context.")
        return
    row = selected.iloc[0]  # Summary denominators repeat in the long depth table; never sum them.
    cols = st.columns(3)
    cols[0].metric("Historical team innings", f"{int(row.sample_team_innings):,}")
    cols[1].metric("Mean number who batted", f"{row.mean_batters:.2f}")
    cols[2].metric("Mean team deliveries faced", f"{row.mean_team_balls_faced:.1f}")
    selected_depth = depth[(depth.scope == scope) & (depth.batting_context == context)].sort_values("n_batters")
    st.bar_chart(selected_depth.set_index("n_batters")[["frequency"]].rename_axis("Number who batted"), color="#087e8b")
    st.caption(f"{row.min_date} to {row.max_date} · {int(row.sample_unique_matches):,} unique matches · observed participants 2–11. Zero-run and zero-ball batting appearances count; DNB does not. Faced deliveries exclude wides and include no-balls.")
    st.caption("NZ, outside NZ, all-ODI and exact-ground labels retain their source populations. Ground names are not merged across renamed venues. Sample sizes may be small and both innings from one match are dependent; observed differences do not show venue causation.")
    with st.expander("Historical depth definitions & eligibility"):
        definitions_path = data_dir / "batting_depth_definitions.json"
        if definitions_path.exists():
            definitions = json.loads(definitions_path.read_text(encoding="utf-8-sig"))
            st.json({key: definitions[key] for key in ("status", "inclusion", "exclusions", "definitions", "caveats", "source_sha256") if key in definitions}, expanded=False)
        st.dataframe(selected_depth[["n_batters", "count_team_innings", "frequency"]], hide_index=True, width="stretch")


def render_opportunity_diagnostic(path):
    if not Path(path).exists():
        return
    diagnostic = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    st.markdown("**V3 batting opportunity · retrospective diagnostic**")
    st.info(diagnostic.get("status", "Retrospective diagnostic; no model adjustment"))
    st.caption("This audits the frozen shared-opportunity model comparison, not the currently selected forecast or captured-price fixture. It does not adjust either engine.")
    slices = diagnostic.get("slices", {})
    scope = st.selectbox("Opportunity diagnostic population", ["All audited rows", *slices.keys()])
    row = diagnostic.get("aggregate", {}) if scope == "All audited rows" else slices[scope]
    cols = st.columns(3)
    cols[0].metric("Player predictions", f"{int(row.get('player_predictions', 0)):,}")
    cols[1].metric("Expected batting appearances", f"{row.get('expected_batting_appearances', 0):.1f}")
    cols[2].metric("Observed batting appearances", f"{int(row.get('actual_batting_appearances', 0)):,}")
    pairs = [("P(bats)", "p_bat"), ("50+ · unconditional", "50_unconditional"), ("50+ | bats", "50_if_bats"), ("100+ · unconditional", "100_unconditional"), ("100+ | bats", "100_if_bats"), ("Faced deliveries per candidate", "balls_per_candidate"), ("Faced deliveries | bats", "balls_if_bats")]
    comparison = [{"Measure": label, "Predicted": row.get("predicted_" + key), "Observed": row.get("observed_" + key)} for label, key in pairs]
    st.dataframe(pd.DataFrame(comparison).round(4), hide_index=True, width="stretch")
    st.caption("Participation and score tails have different denominators. Unconditional rates include every candidate; conditional rates use batting appearances. Expected appearances sum P(bats). Fewer actual batting appearances do not by themselves justify multiplying 50+/100+ probabilities.")
    st.write("The v2 model pools broad ODI role evidence, then NZ/outside-NZ context, exact ground and player evidence through shrinkage. The experimental v3 model uses disjoint outside-NZ, NZ-other-ground and exact-ground delivery pools for a NZ ground, followed by the player's own cell. Sparse ground evidence is supported by broader data rather than treated as a separate certainty.")
    st.caption("These are observational retrospective slices, not a causal test that country or ground changes participation. Slice sample sizes can be small, the archive omits some recent fixtures, and this view makes no model correction or new calibration claim.")
    with st.expander("Opportunity diagnostic definitions & complete artifact"):
        st.json(diagnostic, expanded=False)
