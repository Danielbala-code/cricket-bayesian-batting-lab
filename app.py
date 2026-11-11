"""Portable Streamlit interface for historical ODI analysis and scenario forecasts."""
from datetime import date
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from market_view import render_captured_prices, render_participation, render_opportunity_diagnostic, render_historical_depth

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
HISTORY = DATA / "history"
st.set_page_config(page_title="Cricket | ODI batting intelligence", page_icon="🏏", layout="wide")
st.markdown("""<style>
.stApp {background:#f5f7fa;color:#13263c}
[data-testid="stSidebar"] {background:#eaf0f5}
h1,h2,h3 {color:#112b46;letter-spacing:-.03em}
[data-testid="stMetric"] {background:white;border:1px solid #dce5eb;border-radius:12px;padding:16px}
.eyebrow {color:#087e8b;font-size:.78rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase}
.hero {padding:18px 0 8px;font-size:1.04rem;color:#52677b;max-width:900px}
.stTabs [data-baseweb="tab"] {font-weight:650}
.stButton>button[kind="primary"] {background:#087e8b;border-color:#087e8b}
</style>""", unsafe_allow_html=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).exists() else {}


@st.cache_data
def history_csv(name):
    path = HISTORY / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def make_forecast(team, lineup, venue, cutoff, half_life, kappa, n_sims, scenario, use_venue):
    import engine
    records = engine.load_data()
    return engine.forecast(records, team=team, lineup=list(lineup), venue=venue,
                           cutoff=cutoff, half_life=half_life, kappa=kappa,
                           n_sims=n_sims, seed=7, scenario="Settled start" if scenario == "Steadier start" else scenario, use_venue=use_venue)


@st.cache_data(show_spinner=False)
def make_match_forecast(india, new_zealand, venue, cutoff, batting_first, overs, half_life, prior_strength, n_sims, use_venue, use_player):
    # Schema1.1 preserves boundary allowances when a chase ends.
    import ball_engine
    return ball_engine.forecast_match(ball_engine.load_ball_data(),
        lineups={"India": list(india), "New Zealand": list(new_zealand)}, venue=venue,
        cutoff=cutoff, batting_first=batting_first, overs=overs, half_life=half_life,
        prior_strength=prior_strength, n_sims=n_sims, seed=7,
        use_venue=use_venue, use_player=use_player)


fixture = read_json(DATA / "fixture.json")
lineups = fixture.get("lineups", {})
tuning = read_json(DATA / "tuning_v2.json")
chosen = tuning.get("selected_config", tuning.get("selected", tuning.get("selected_settings", {})))
chosen_half_life = chosen.get("half_life_years", chosen.get("half_life", 4.0))
chosen_half_life = float("inf") if chosen_half_life is None else float(chosen_half_life)
chosen_kappa = int(chosen.get("kappa", 20))
ball_tuning = read_json(DATA / "ball_tuning.json")
ball_chosen = ball_tuning.get("selected_config", {})
with st.sidebar:
    st.markdown("### Match workspace")
    model_choice = st.selectbox("Forecast engine", ["v2 score sampler", "v3 shared-opportunity (experimental)"])
    v3 = model_choice.startswith("v3")
    team = st.selectbox("Batting team", ["India", "New Zealand"])
    venue = st.text_input("Ground", value=fixture.get("venue", "Bay Oval, Mount Maunganui"))
    cutoff = st.date_input("Prediction cutoff (exclude this date onward)", value=date(2026, 11, 10))
    lineup_text = st.text_area("Probable batting order · one player per line", "\n".join(lineups.get(team, [])), height=285, key=f"lineup_{team}")
    lineup = [x.strip() for x in lineup_text.splitlines() if x.strip()]
    st.caption("Order is a scenario input. Edit it after the toss and selection announcement.")
    opponent = "New Zealand" if team == "India" else "India"
    opponent_lineup, batting_first, overs, prior_strength, use_player = [], "Unknown", 50, 120, True
    if v3:
        opponent_text = st.text_area("Opponent probable batting order · one player per line", "\n".join(lineups.get(opponent, [])), height=240, key=f"opponent_{opponent}")
        opponent_lineup = [x.strip() for x in opponent_text.splitlines() if x.strip()]
        batting_first = st.selectbox("Batting first", ["Unknown", "India", "New Zealand"])
        st.caption("Unknown is a declared 50:50 batting-first scenario, not a toss prediction.")
        overs = st.slider("Declared overs per innings", 20, 50, 50)
        st.caption("A pre-start overs scenario. Shortened-innings outputs are unvalidated extrapolations, with no weather or mid-match DLS model.")

st.markdown('<div class="eyebrow">ODI · Batting intelligence</div>', unsafe_allow_html=True)
st.title("Cricket, through the batting order")
st.markdown('<div class="hero">Explore where ODI runs come from, test a probable XI, and inspect independently captured historical batting markets.</div>', unsafe_allow_html=True)
st.warning("Probable lineups only — the XI and batting order are not confirmed. Update these inputs before interpreting a forecast.")
st.caption(f"Forecast fixture: {fixture.get('date', '10 November 2026')} · {venue} · India v New Zealand. Captured-price selections are separate.")
tab_history, tab_forecast, tab_validation, tab_odds, tab_math, tab_guide = st.tabs(["History", "Forecast & scenarios", "Validation", "Captured prices", "Maths", "Guide & glossary"])

with tab_math:
    st.subheader("Maths you can inspect and change")
    st.write("Adjust these assumptions, then select Run forecast in the sidebar. The displayed run records its settings; changing controls does not rewrite the historical data or establish an edge.")
    with st.expander("Model controls", expanded=True):
        if v3:
            half_life = st.selectbox("Delivery-model half-life (years)", [2, 4], index=[2,4].index(ball_chosen.get("half_life",2)))
            prior_strength = st.selectbox("Delivery-prior strength (equivalent deliveries)", [60, 120, 240], index=[60,120,240].index(ball_chosen.get("prior_strength",120)))
            kappa = chosen_kappa
            use_player = st.checkbox("Use player delivery history", True)
            st.caption("Delivery-model strength is measured in deliveries. Defaults use earlier-development selection when available; this model remains experimental and is evaluated separately from v2.")
        else:
            decay = st.checkbox("Use recency decay", value=bool(np.isfinite(chosen_half_life)))
            finite_h = st.slider("Recency half-life (years)", 0.5, 10.0, chosen_half_life if np.isfinite(chosen_half_life) else 4.0, 0.5, disabled=not decay)
            half_life = finite_h if decay else float("inf")
            kappa = st.slider("Role-prior strength (equivalent innings)", 5, 80, chosen_kappa, 5)
            st.caption("Defaults selected on the earlier development period; see Validation for the frozen comparison." if tuning else "Original v1 demonstration settings: h=4 years, κ=20. These defaults have not been tuned.")
        scenario = st.selectbox("Innings scenario", ["Mixed", "Early trouble", "Steadier start"], disabled=v3)
        if v3:
            scenario = "Mixed"
            st.caption("The v2 regime prior is disabled. Early-wicket scenarios can be queried after simulating v3, without using future outcomes to fit its delivery probabilities.")
        use_venue = st.checkbox("Apply venue adjustment", True)
        n_sims = st.select_slider("Simulated innings", [1000, 3000, 5000] if v3 else [2000, 5000, 10000, 20000], value=3000 if v3 else 10000)
    st.markdown("The preview reports a >70% rain chance at publication and discusses a 25-over top-batter settlement requirement. v2 represents a standard innings; v3 allows a declared pre-start overs scenario. Neither models weather, washout probability or DLS revisions, and neither establishes a rain-adjusted market price.")
    if v3:
        st.warning("v3 is an experimental delivery model. Its evaluation and implementation audit are separate from the v2 results below. Declared shortened-overs scenarios are unvalidated extrapolations.")
        st.write("Both probable XIs, batting first and the declared legal-ball budget are pre-start scenario inputs. The chase target is generated from the simulated first innings; no actual first-innings score is supplied before the match.")
        st.latex(r"w_i=2^{-\mathrm{ageYears}_i/h},\quad C_e=\sum_i w_i n_{i,e},\quad p_{0,e}=\frac{C_{\mathrm{background},e}+0.5/K}{N_{\mathrm{background}}+0.5},\quad p_{\mathrm{next},e}=\frac{C_{\mathrm{next},e}+\kappa p_{\mathrm{previous},e}}{N_{\mathrm{next}}+\kappa}")
        st.write("Only eligible deliveries strictly before the cutoff enter the counts. K event types are the union of past observed tuples and a fixed nine-event law-valid base; the 0.5 symmetric prior mass is spread across that active support. Event tuples contain batter runs, extras, wide/no-ball flags, running parity and wicket type.")
        st.write("Cells use expected role (1–3, 4–7, 8–11), phase (first 20%, middle 60%, last 20% of the declared legal-ball budget) and innings order. For a NZ ground the disjoint other-player pools are outside NZ → NZ other grounds → exact ground. Elsewhere they are background → exact ground. The target player's observations are excluded from every pooled stage and added once in the final update. An empty context cell falls back to that same context's role aggregate over phase/order. Turning venue adjustment off pools locations; turning player history off uses the pooled prior.")
        st.latex(r"B_{\mathrm{legal}}=6\,\mathrm{overs},\quad \mathrm{BF}_j=\sum_{s:\,\mathrm{striker}=j}\mathbf{1}(\mathrm{not\ wide}_s),\quad T=\sum_jR_j+\mathrm{extras},\quad \mathrm{chaseTarget}=T_{\mathrm{first}}+1")
        st.write("Events update strike, extras, legal balls and wickets; parity and completed overs rotate strike. Wides consume no legal or faced ball; no-balls consume a faced ball but no legal ball. No-balls trigger a free hit, which persists through illegal deliveries. Ordinary dismissals are suppressed on no-balls/free hits; run-outs can remain. Stop at the budget, ten wickets or a reached simulated target. The illegal-event safety cap raises an error rather than silently truncating.")
        st.caption("κ is equivalent deliveries, not v2 innings. Delivery probabilities remain fixed within role/phase/innings-order cells: no bowler, pressure-response, weather or DLS effect is fitted. This is a discounted empirical-Bayes approximation, not a complete hierarchical posterior or complete cricket-physics model. The v2 whole-score exponential prior and early-trouble fit do not apply to v3. First-ten-over filters condition on simulated outcomes only.")
    else:
        st.latex(r"w_i=2^{-(t-t_i)/(365.25h)},\quad C_{r,x}=\sum_i w_i\,\mathbf{1}(R_i=x,\ \mathrm{role}_i=r,\ Z_i=z)")
        st.write("Only eligible innings strictly before the cutoff train the forecast; no-results and D/L innings are excluded. h is the half-life in years. Roles are 1–3, 4–7 and 8–11; Z indicates observed early trouble. Player history matches the entered role, and pooled role counts exclude that player.")
        st.caption("Decay off means h=∞ and all eligible historical innings receive weight 1. When a v2 selection artifact is bundled, h and κ default to its earlier-development selection; they remain editable. The two-wickets-in-ten-overs regime threshold, exponential scale 25 and tail mass 0.5 remain fixed structural assumptions.")
        st.latex(r"b_x\propto e^{-x/25},\quad p_{\mathrm{role},x}=\frac{C_{\mathrm{role},x}+0.5b_x}{\sum_x C_{\mathrm{role},x}+0.5},\quad p_{\mathrm{next},x}=\frac{C_{\mathrm{next},x}+\kappa p_{\mathrm{prior},x}}{\sum_x C_{\mathrm{next},x}+\kappa}")
        st.write("The run support is the integers 0 through at least 300 (or the archive maximum, if larger). The Dirichlet posterior predictive mean shrinks pooled role → NZ/outside-NZ location → exact ground → player. Turning venue adjustment off skips the location and ground stages. κ is prior mass in equivalent weighted observations, not a fitted certainty level.")
        st.latex(r"P(A\mid X)=\sum_{z\in\{0,1\}}P(A\mid Z=z,X)P(Z=z\mid X)")
        st.write("Mixed uses a recency-weighted, smoothed early-trouble frequency (global → location → ground → team). Early trouble or Steadier start fixes that regime. Each simulation samples one shared regime and one shared batting-depth template; player runs are conditionally independent given that regime. This approximation does not conserve team balls or a target score. Batting depth is smoothed by regime, location, ground and team.")
    st.latex(r"\widehat{P}(A\mid B,X)=\frac{n(A\cap B)}{n(B)},\quad \widehat{P}(A\cap B\mid X)=\frac{n(A\cap B)}{N}")
    st.write("Conditional queries count the simulated innings only. X contains the selected lineup and model assumptions. A can be Rahul ≥50; B can be Rohit ≤20 and Kohli ≤20, with batting required by default. These are not reported historical conditional hit rates.")
    st.latex(r"p_j^{\mathrm{top}}=\frac1N\sum_s I_{sj},\quad q_j=\frac1N\sum_s\frac{I_{sj}}{K_s},\quad \sum_jq_j=1,\quad EV_j=d_jq_j-1\quad(d=8\Rightarrow EV=8q_j-1)")
    st.write("I marks a tied or sole top scorer and K is the number of tied leaders. q is expected dead-heat settlement share; P(top) includes ties in full. Mean and median runs condition on batting, while P(50+) is unconditional. EV=dq−1 assumes DNB loses. If DNB returns the stake, unconditional EV=dq−P(bats). Both assume standard dead-heat stake reduction and exclude commission. Forecasts are uncalibrated; no established betting edge is claimed.")

with st.sidebar:
    run = st.button("Run forecast", type="primary", width="stretch")
    st.caption("Historical archive through 2 November 2026. India coverage ends 3 October 2026.")


if run:
    if len(lineup) != 11 or len(set(lineup)) != 11:
        st.error("Enter exactly 11 different player names in the expected batting order.")
    elif v3 and (len(opponent_lineup) != 11 or len(set(opponent_lineup)) != 11 or set(lineup) & set(opponent_lineup)):
        st.error("The opponent XI must also contain exactly 11 different players, with no player listed for both teams.")
    elif not (ROOT / ("ball_engine.py" if v3 else "engine.py")).exists():
        st.error("The forecasting engine is not available in this build.")
    else:
        with st.spinner("Simulating team innings…"):
            try:
                if v3:
                    both = {team: tuple(lineup), opponent: tuple(opponent_lineup)}
                    match = make_match_forecast(both["India"], both["New Zealand"], venue, cutoff.isoformat(), batting_first, overs, half_life, prior_strength, n_sims, use_venue, use_player)
                    result = dict(match["teams"][team])
                    for field in ("warnings", "training_max_date", "training_n", "model"):
                        result.setdefault(field, match.get(field, [] if field == "warnings" else "Not recorded"))
                    st.session_state["match_forecast"] = match
                else:
                    result = make_forecast(team, tuple(lineup), venue, cutoff.isoformat(), half_life, kappa, n_sims, scenario, use_venue)
                    st.session_state.pop("match_forecast", None)
                st.session_state["forecast"] = result
                st.session_state["forecast_inputs"] = (team, tuple(lineup), venue, cutoff.isoformat(), half_life, kappa, n_sims, scenario, use_venue, model_choice, tuple(opponent_lineup), batting_first, overs, prior_strength, use_player)
            except (ValueError, KeyError, FileNotFoundError, TypeError) as exc:
                st.error(f"Forecast could not run: {exc}")

result = st.session_state.get("forecast")
current_inputs = (team, tuple(lineup), venue, cutoff.isoformat(), half_life, kappa, n_sims, scenario, use_venue, model_choice, tuple(opponent_lineup), batting_first, overs, prior_strength, use_player)
if result and current_inputs != st.session_state.get("forecast_inputs"):
    st.info("Inputs changed. Run the forecast again to update the results below.")



with tab_history:
    render_historical_depth(DATA)
    st.subheader("The shape of an ODI innings")
    st.write("Observed frequencies describe this archive. They are not forecast odds for the next match.")
    counts = history_csv("scope_counts.csv")
    roles = history_csv("by_role.csv")
    positions = history_csv("by_position.csv")
    if counts.empty:
        st.info("Historical tables will appear once the bundled data are available.")
    else:
        c1, c2 = st.columns([2, 1])
        scope = c1.selectbox("Historical population", counts["scope"].unique())
        variant = c2.selectbox("Rain sensitivity", counts["filter_variant"].unique())
        subset = counts[(counts.scope == scope) & (counts.filter_variant == variant)]
        if len(subset):
            row = subset.iloc[0]
            cols = st.columns(4)
            cols[0].metric("Settled team innings", f"{int(row.settled_team_innings):,}")
            cols[1].metric("Observed batting innings", f"{int(row.batting_innings):,}")
            cols[2].metric("Did not bat entries", f"{int(row.dnb_entries):,}")
            cols[3].metric("Archive ends", str(row.max_date))
        role_rows = roles[(roles.scope == scope) & (roles.filter_variant == variant)].copy() if len(roles) else pd.DataFrame()
        pos_rows = positions[(positions.scope == scope) & (positions.filter_variant == variant)].copy() if len(positions) else pd.DataFrame()
        if len(role_rows):
            left, right = st.columns(2)
            left.markdown("**Share of batter runs**")
            left.bar_chart(role_rows.set_index("group")[["share_of_batter_runs"]], color="#087e8b")
            right.markdown("**Share of top-team-batter outcomes**")
            right.bar_chart(role_rows.set_index("group")[["share_of_top_team_outcomes"]], color="#d6943a")
            st.caption("Top outcomes split equally among tied leaders. Run share excludes extras. These are distinct denominators.")
            display_cols = ["group", "batting_innings", "mean_runs_per_batting_innings", "median_runs", "share_of_batter_runs", "share_of_top_team_outcomes"]
            st.dataframe(role_rows[display_cols].round(3), hide_index=True, width="stretch")
        if len(pos_rows):
            with st.expander("All batting positions & downloadable tables"):
                st.dataframe(pos_rows, hide_index=True, width="stretch")
                st.download_button("Download this population by position", pos_rows.to_csv(index=False), "odi_batting_positions.csv", "text/csv")
        st.caption("DNB is separate from a zero-run innings. Batting-position means condition on batting. Positions 1 and 2 follow first observed striker/non-striker, so the opening pair's internal order is conventional.")
    st.subheader("Balls faced: New Zealand versus elsewhere")
    st.caption("Historical opportunity by batting position. Deliveries faced excluding wides; no-balls included. NZ and outside-NZ populations are disjoint. This descriptive comparison does not change the forecasting model.")
    balls_chart = DATA / "balls_faced_distribution.png"
    if not balls_chart.exists():
        balls_chart = ROOT / "balls_faced_distribution.png"
    balls_summary = DATA / "balls_faced_summary.csv"
    if balls_chart.exists():
        st.image(str(balls_chart), width="stretch")
    if balls_summary.exists():
        with st.expander("Balls-faced samples, definitions & download"):
            balls_table = pd.read_csv(balls_summary)
            st.dataframe(balls_table, hide_index=True, width="stretch")
            balls_meta = read_json(DATA / "balls_faced_metadata.json")
            if balls_meta:
                st.json(balls_meta, expanded=False)
            st.download_button("Download balls-faced summary", balls_summary.read_bytes(), "balls_faced_summary.csv", "text/csv")
    if not balls_chart.exists() and not balls_summary.exists():
        st.info("The balls-faced comparison has not yet been bundled in this build.")
    with st.expander("Original archive charts", expanded=False):
        for filename, caption in [("all_odi_batting_order.png", "All teams · full archive"), ("india_batting_order.png", "India · full archive"), ("run_distributions.png", "Historical run distributions")]:
            path = HISTORY / filename
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")

with tab_forecast:
    st.subheader("From a probable XI to a distribution")
    st.write("The score sampler combines role and recency distributions with shared batting depth. The experimental delivery model instead derives runs and balls faced from a shared legal-ball budget and a simulated chase target.")
    st.warning("Rain or a shortened match can materially change opportunity. No weather or DLS process is modelled. A v3 declared-overs scenario is an unvalidated extrapolation, not a rain forecast.")
    if result is None:
        st.info("Review the probable XI in the sidebar, then select Run forecast.")
    else:
        players = pd.DataFrame(result["players"])
        scores = np.asarray(result["scores"])
        shown = st.session_state["forecast_inputs"]
        shown_v3 = len(shown) > 9 and shown[9].startswith("v3")
        st.caption(f"Displayed run: {shown[0]} · {shown[2]} · cutoff {shown[3]} · {shown[9] if len(shown) > 9 else 'v2'} · half-life {shown[4]} years")
        st.caption(f"Declared {shown[12]} overs · batting first: {shown[11]} · prior strength {shown[13]} equivalent deliveries" if shown_v3 else f"{shown[7]} regime scenario · κ={shown[5]} equivalent innings")
        for warning in result.get("warnings", []):
            st.warning(warning)
        cols = st.columns(3)
        cols[0].metric("Simulated innings", f"{len(scores):,}")
        cols[1].metric("Sum of expected dead-heat shares", f"{players.q.sum():.4f}")
        cols[2].metric("Latest training date", str(result.get("training_max_date", "Unavailable")))
        st.markdown("**Top-team-batter share, allowing for ties**")
        st.bar_chart(players.set_index("player")[["q"]], color="#087e8b")
        st.caption("q = expected settlement share: a tied leader receives 1 / number of tied leaders. P(top) includes every tied win and can sum above 100%. q sums to 100% across the XI.")
        renamed = players.rename(columns={"player":"Player", "position":"Order", "mean_runs":"Mean runs | bats", "median_runs":"Median | bats", "p50plus":"P(50+) unconditional", "p_top":"P(top incl. ties)", "q":"Dead-heat share q", "p_bat":"P(bats)", "history_n":"History innings", "weighted_n":"Recency-weighted innings"})
        if shown_v3:
            renamed = renamed.rename(columns={"Recency-weighted innings": "Recency-weighted deliveries", "mean_balls": "Mean deliveries faced | bats", "median_balls": "Median deliveries faced | bats"})
        st.dataframe(renamed.round(4), hide_index=True, width="stretch")
        render_participation(scores)
        st.download_button("Download forecast table", players.to_csv(index=False), "odi_forecast.csv", "text/csv")
        if not shown_v3:
            st.markdown("**Scenario mixture**")
            st.dataframe(pd.DataFrame(list(result.get("scenario_weights", {}).items()), columns=["Scenario", "Weight"]), hide_index=True)
        else:
            match = st.session_state["match_forecast"]
            audit_rows = []
            for audit_team, frame in match["teams"].items():
                legal = np.asarray(match["legal_balls"][audit_team])
                totals = np.asarray(match["team_totals"][audit_team])
                extras = np.asarray(match["extras"][audit_team])
                runs = np.maximum(np.asarray(frame["scores"]), 0).sum(axis=1)
                faced = np.asarray(frame["balls_faced"]).sum(axis=1)
                no_balls = np.asarray(match["no_ball_deliveries"][audit_team])
                other_team = next(name for name in match["teams"] if name != audit_team)
                expected_target = np.where(np.asarray(match["batting_first"]) == audit_team, 0, np.asarray(match["team_totals"][other_team]) + 1)
                target_mismatch = np.asarray(match["targets"][audit_team]) != expected_target
                audit_rows.append({"Team": audit_team, "Mean legal balls": legal.mean(), "Maximum legal balls": legal.max(), "Budget": match["budget"], "Budget violations": int((legal > match["budget"]).sum()), "Run/extras discrepancies": int((totals != runs + extras).sum()), "Balls-faced discrepancies": int((faced != legal + no_balls).sum()), "Simulated-target discrepancies": int(target_mismatch.sum()), "Mean wickets": np.asarray(match["wickets"][audit_team]).mean()})
            st.markdown("**Shared-opportunity accounting · both teams**")
            st.dataframe(pd.DataFrame(audit_rows).round(2), hide_index=True, width="stretch")
            st.caption("A legal-ball budget is shared by all batters. Wides add neither legal nor faced balls; no-balls add faced balls but no legal ball. Runs plus extras must equal team totals. Chases can finish before the budget is exhausted.")
            with st.expander("Early-wicket what-if · filter simulated outcomes"):
                what_if = st.selectbox("Simulated first-ten-over condition", ["All simulated innings", "At least two wickets", "Fewer than two wickets"])
                early = np.asarray(match["first10_wickets"][shown[0]])
                subset = np.ones(len(early), dtype=bool) if what_if == "All simulated innings" else early >= 2 if what_if == "At least two wickets" else early < 2
                st.write(f"Matching simulations: {int(subset.sum()):,} / {len(subset):,} ({subset.mean():.1%})")
                if subset.any():
                    conditioned = scores[subset]
                    tied = (conditioned == conditioned.max(axis=1, keepdims=True)) & (conditioned >= 0)
                    shares = (tied / tied.sum(axis=1, keepdims=True)).mean(axis=0)
                    st.dataframe(pd.DataFrame({"Player": players.player, "Conditional dead-heat share": shares, "Conditional P(50+)": (conditioned >= 50).mean(axis=0)}).round(4), hide_index=True, width="stretch")
                if subset.sum() < 200:
                    st.warning("Few simulations match; conditional estimates are noisy.")
                st.caption("This conditions on outcomes generated after the pre-start forecast. It does not train on the future first-ten-over wickets and is not a causal intervention.")
        names = players.player.tolist()
        selected = st.selectbox("Inspect a batter's run distribution", names)
        idx = names.index(selected)
        values = scores[:, idx]
        batting = values[values >= 0]
        fig, ax = plt.subplots(figsize=(9, 2.7))
        ax.hist(batting, bins=np.arange(0, max(160, int(batting.max()) + 10) if len(batting) else 160, 10), color="#087e8b", alpha=.9)
        ax.set(xlabel="Runs, conditional on batting", ylabel="Simulated innings")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Did not bat: {(values < 0).mean():.1%}. DNB is excluded from this histogram, not converted into a duck.")
        if shown_v3:
            faced_values = np.asarray(result["balls_faced"])[:, idx]
            observed_batted = np.asarray(result["batted"])[:, idx].astype(bool)
            fig, ax = plt.subplots(figsize=(9, 2.5))
            ax.hist(faced_values[observed_batted], bins=20, color="#d6943a", alpha=.9)
            ax.set(xlabel="Deliveries faced, conditional on batting", ylabel="Simulated innings")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Simulated non-wide deliveries faced, including no-balls. Genuine zero-ball batting appearances remain; DNB is excluded. Compare with the observed History distribution, whose inclusion rules are documented separately.")
            reference_file = DATA / "balls_faced_summary.csv"
            if reference_file.exists():
                references = pd.read_csv(reference_file)
                group = "1-2" if idx < 2 else str(idx + 1)
                references = references[references.position_group.astype(str) == group]
                comparison_rows = [{"Population": "Selected simulated innings", "Batting appearances": int(observed_batted.sum()), "Mean deliveries faced": float(faced_values[observed_batted].mean()) if observed_batted.any() else None, "Median deliveries faced": float(np.median(faced_values[observed_batted])) if observed_batted.any() else None}]
                comparison_rows += [{"Population": str(row.scope) + " · historical", "Batting appearances": int(row.n), "Mean deliveries faced": float(row.mean), "Median deliveries faced": float(row.median)} for row in references.itertuples()]
                st.dataframe(pd.DataFrame(comparison_rows).round(2), hide_index=True, width="stretch")
                st.caption(f"Descriptive reference for batting position {group}, all teams. Historical reference retains D/L matches and uses its own eligibility rules; it is not the fitted player's sample or a forecast-calibration test.")
        with st.expander("Conditional scenario query"):
            target = st.selectbox("Target batter", names, index=next((i for i, name in enumerate(names) if "rahul" in name.lower()), 0), key="query_target")
            threshold = st.number_input("Target reaches at least", 0, 250, 50)
            options = [x for x in names if x != target]
            preferred = [x for x in options if "rohit" in x.lower() or "kohli" in x.lower()]
            others = st.multiselect("Condition on these batters' low scores", options, default=preferred if len(preferred) == 2 else options[:2], max_selections=2)
            maximum = st.number_input("Each selected batter scores at most", 0, 100, 20)
            require_batted = st.checkbox("Require the condition batters to have batted", value=True)
            if len(others) == 2:
                condition = np.ones(len(scores), dtype=bool)
                for name in others:
                    v = scores[:, names.index(name)]
                    condition &= (v <= maximum) & ((v >= 0) if require_batted else np.ones(len(v), dtype=bool))
                target_hit = scores[:, names.index(target)] >= threshold
                eligible = int(condition.sum())
                a, b, c = st.columns(3)
                a.metric("P(target & conditions)", f"{np.mean(condition & target_hit):.1%}")
                b.metric("P(target | conditions)", f"{target_hit[condition].mean():.1%}" if eligible else "N/A")
                c.metric("Condition-matching simulations", f"{eligible:,}")
                if eligible < 200:
                    st.warning("Few simulations match this condition; the conditional estimate is noisy. Increase simulations or broaden the condition.")
                st.caption("If the batting requirement is unchecked, DNB qualifies as a low-score condition. This is a model scenario, not causal evidence that another batter's failure helps the target.")

with tab_validation:
    st.subheader("Evidence before confidence")
    render_opportunity_diagnostic(DATA / "opportunity_diagnostic.json")
    st.caption("The parameter search, score diagnostics and sensitivity tables labelled v2 assess the score sampler. They do not validate the experimental v3 delivery model.")
    validation_v3 = read_json(DATA / "ball_validation.json") or read_json(DATA / "validation_v3.json")
    if validation_v3:
        with st.expander("Experimental v3 evaluation · separate artifact", expanded=v3):
            st.info(validation_v3.get("status", "Retrospective comparison"))
            st.caption(f"{validation_v3.get('n',0)} team innings in {validation_v3.get('unique_games',0)} matches · {validation_v3.get('min_date','')} to {validation_v3.get('max_date','')}")
            bm=validation_v3.get("mean_brier",{})
            if bm:st.dataframe(pd.DataFrame({"Method":list(bm),"Mean Brier ↓":list(bm.values())}).round(4),hide_index=True,width="stretch")
            st.write("Matched-case distribution checks:", validation_v3.get("distribution",{}))
            st.caption("Unsupported cases were excluded retrospectively; results apply to that narrower population. These are not untouched confirmation or causal evidence. The simpler model remains the default.")
            with st.expander("V3 exclusions, per-match records and audit"):
                st.json(validation_v3,expanded=False)
    elif v3:
        st.warning("A completed v3 evaluation has not been bundled. Shared-ball accounting alone does not establish predictive quality or calibration.")
    validation_v2 = read_json(DATA / "validation_v2.json")
    validation = validation_v2 or read_json(DATA / "validation.json")
    if tuning:
        st.markdown("**Earlier-period parameter selection · v2**")
        st.write("Half-life and prior strength are selected on an earlier development period, before evaluating the later period. This is a limited six-setting comparison, not a guarantee of calibration.")
        st.caption(f"Selected half-life: {'no decay (∞)' if not np.isfinite(chosen_half_life) else str(chosen_half_life) + ' years'} · κ={chosen_kappa}")
        st.caption(f"Development window: {' to '.join(tuning.get('development_window', []))} · later evaluation: {' to '.join(tuning.get('test_window', []))}")
        grid = tuning.get("grid_results", [])
        if grid:
            grid_rows = [{**row.get("config", {}), "Mean development Brier ↓": row.get("mean_brier"), "Team innings": row.get("n")} for row in grid]
            grid_table = pd.DataFrame(grid_rows)
            if "half_life_years" in grid_table:
                grid_table["half_life_years"] = grid_table["half_life_years"].astype(str)
            st.dataframe(grid_table.round(4), hide_index=True, width="stretch")
        with st.expander("Selection grid, periods and reproducibility"):
            st.json(tuning, expanded=True)
    if validation:
        st.markdown("**Later-period evaluation · v2**" if validation_v2 else "**Original v1 retrospective diagnostic · fixed h=4, κ=20**")
        st.info(validation.get("status", "Retrospective diagnostic"))
        actual_n = validation.get("n", validation.get("n_team_innings", "not recorded"))
        actual_start = validation.get("min_date", validation.get("start_date", "not recorded"))
        actual_end = validation.get("max_date", validation.get("end_date", "not recorded"))
        st.caption(f"{actual_n} team innings · {actual_start} to {actual_end}")
        metric_values = validation.get("mean_brier", {})
        if metric_values:
            st.dataframe(pd.DataFrame({"Method": list(metric_values), "Mean multiclass Brier ↓": list(metric_values.values())}).round(4), hide_index=True, width="stretch")
        for limitation in validation.get("limitations", []):
            st.write("•", limitation)
        st.caption("A Brier comparison does not establish probability calibration or profitable betting. Previously explored data are not an untouched holdout, even when prediction training strictly precedes each target.")
        with st.expander("Per-match validation records and reproducibility"):
            st.json(validation, expanded=False)
    else:
        st.info("A completed holdout evaluation has not been bundled. No calibration or predictive-accuracy claim is made.")
    distribution = read_json(DATA / "distribution_checks_v2.json")
    if distribution:
        st.markdown("**v2 distribution checks · frozen retrospective diagnostic**")
        st.info(distribution.get("status", "Retrospective distribution comparison"))
        st.caption(f"{distribution.get('n_team_innings', 'not recorded')} team innings · {distribution.get('n_player_rows', 'not recorded')} player predictions. Settings and sample may differ from the selected v2 evaluation; inspect the artifact below.")
        st.caption(f"Dates: {distribution.get('min_date', 'not recorded')} to {distribution.get('max_date', 'not recorded')} · diagnostic settings: {distribution.get('settings', {})}")
        if distribution.get("aggregate"):
            aggregate = distribution["aggregate"]
            rate_pairs = [("Batting opportunity · P(bats)", "p_bat"), ("Zero runs | bats", "zero_conditional_bat"), ("50+ runs | bats", "p50_conditional_bat"), ("100+ runs | bats", "p100_conditional_bat")]
            rate_table = [{"Check": label, "Predicted": aggregate.get(key + "_predicted"), "Observed": aggregate.get(key + "_observed")} for label, key in rate_pairs]
            st.dataframe(pd.DataFrame(rate_table).round(4), hide_index=True, width="stretch")
            check_cols = st.columns(3)
            if "central80_coverage" in aggregate:
                check_cols[0].metric("Central 80% interval coverage", f"{aggregate['central80_coverage']:.1%}")
            if "predicted_median_runs" in aggregate:
                check_cols[1].metric("Median of predictive medians", f"{aggregate['predicted_median_runs']:.1f}")
            if "observed_median_runs" in aggregate:
                check_cols[2].metric("Pooled observed batting median", f"{aggregate['observed_median_runs']:.1f}")
        if distribution.get("by_expected_role"):
            role_checks = pd.DataFrame.from_dict(distribution["by_expected_role"], orient="index").rename_axis("Expected role").reset_index()
            st.dataframe(role_checks.round(4), hide_index=True, width="stretch")
        st.caption("Zero, 50+ and 100+ frequencies marked conditional on batting exclude DNB; P(bats) has its own denominator. Central 80% coverage is the fraction of observed scores inside the predictive interval, not an 80% confidence statement about a mean. Discrete score intervals can over-cover. Role comparisons and marginal checks do not establish joint calibration.")
        st.caption("The predicted median field is the median of individual predictive medians. The observed median pools realised batting scores. They are different summaries and their difference is not a median-calibration test. Aggregate conditional predicted rates use summed event probabilities divided by summed P(bats); observed rates use observed batting counts.")
        with st.expander("Calibration bins, realised-role checks & full definitions"):
            st.json(distribution, expanded=False)
    sensitivity = read_json(DATA / "sensitivity_v2.json")
    if sensitivity:
        with st.expander("v2 parameter sensitivity · conditional scenario estimates"):
            st.write(sensitivity.get("event", "Parameter sensitivity"))
            st.caption(sensitivity.get("status", "Sensitivity only"))
            sensitivity_table = pd.DataFrame(sensitivity.get("rows", []))
            if "half_life_years" in sensitivity_table:
                sensitivity_table["half_life_years"] = sensitivity_table["half_life_years"].astype(str)
            st.dataframe(sensitivity_table.round(4), hide_index=True, width="stretch")
            st.caption(f"{sensitivity.get('n_simulations_per_setting', 'Not recorded')} simulations per setting. This is parameter sensitivity, not parameter selection, a credible/confidence interval or a complete measure of model uncertainty. Monte Carlo error describes simulation noise only.")
    st.markdown("""The forecast should be assessed on chronological holdouts against a role-only baseline. Relevant checks include expected dead-heat shares, probability calibration, error by batting role, and sensitivity to recency, lineup and venue assumptions.

The archive can omit recent fixtures. In particular, India's observed history ends 3 October 2026. Player identities and a changed batting role can produce sparse samples. Shrinkage stabilises estimates but does not create new evidence. Scenario selection, toss, weather, opposition strength and actual playing XI remain sources of uncertainty.""")

with tab_odds:
    render_captured_prices(DATA / "captured_betfair.json")

st.caption("ODI batting intelligence · Historical evidence, explicit assumptions, transparent arithmetic")


with tab_guide:
    st.subheader("Start here")
    st.markdown("""1. **Choose a model and team.** The score sampler is the simpler comparison; the shared-opportunity model generates deliveries, balls faced and chase targets.
2. **Check the inputs.** Edit the probable XI and expected order. For the delivery model, check both teams, who bats first and the declared overs limit.
3. **Explore assumptions in Maths.** Half-life changes how quickly old evidence fades; prior strength changes pooling. These controls are scenarios, not a way to select favourable odds.
4. **Select Run forecast.** Inspect score distributions, batting opportunity and conditional scenarios. A conditional result applies only to the stated event.
5. **Read Validation and Captured prices.** Compare models on matched cases, then browse the archived Betfair match/team markets independently. Capture timestamps, offer sizes and quote status describe recorded evidence, not a live price or an automatic forecast backtest.""")
    st.subheader("Small glossary")
    glossary = [
        ("Balls faced", "Non-wide deliveries faced, including no-balls in this app. A nonstriker can participate without facing a ball."),
        ("DNB", "Did not bat. Kept separate from batting and scoring zero."),
        ("Conditional probability", "The chance of A among cases where B happens: P(A|B) = P(A and B) / P(B)."),
        ("Prior", "A starting probability distribution used before adding this player's evidence."),
        ("Posterior predictive", "The distribution of a future outcome after combining the prior with eligible past data."),
        ("Half-life", "How old an observation is when it receives half the weight of recent evidence."),
        ("Shrinkage / κ", "How strongly a small player sample is pulled toward a broader prior. Units differ: innings for v2, deliveries for v3."),
        ("P(top)", "Chance of being among the highest scorers, including ties. Across players these can sum above 100%."),
        ("Dead-heat share q", "Expected share after splitting each tied win equally. These shares sum to 100% within a team."),
        ("Decimal odds", "Total return including stake per unit staked. Fractional 7/1 means decimal 8.0."),
        ("Break-even share", "The share needed for zero expected profit. With DNB losing, it is 1 / decimal odds."),
        ("Expected value / EV", "Average net return under the model and settlement assumptions. DNB loses: odds × q − 1; DNB refund: odds × q − P(bats)."),
        ("Captured offer", "A back/lay field received by the collector. Its status, size, time and delayed-feed provenance matter; it is not a live execution guarantee."),
        ("Last traded price / LTP", "The last recorded trade field, kept separate from back/lay offers. It is not substituted when an offer is missing."),
        ("Brier score", "Squared error of probability forecasts against outcomes. Lower is better; it is not a profit or accuracy percentage."),
        ("Predictive interval", "A range for a future score under the model. It differs from uncertainty about a player's average ability."),
        ("Simulation noise", "Numerical variation from using a finite number of simulated innings. More draws reduce this, not model bias."),
        ("Data leakage", "Using information that would not have been available at the prediction time."),
        ("Correlation vs causation", "A pattern between variables does not by itself show that one causes the other."),
    ]
    st.dataframe(pd.DataFrame(glossary, columns=["Term", "Meaning in this app"]), hide_index=True, width="stretch", height=540)
    st.info("Probable XIs are not confirmed. Rain, DLS and bookmaker void probabilities are not forecast. Historical comparisons remain retrospective research.")
    if fixture.get("notes"):
        with st.expander("Fixture verification & availability"):
            for note in fixture["notes"]:
                st.write(note)
            if fixture.get("source"):
                st.write("Source:", fixture["source"])

    with st.expander("Sources, definitions & limitations"):
        st.markdown("""- [Cricsheet](https://cricsheet.org/) — ball-by-ball ODI archive; historical tables bundled with this app.
    - [Betfair Exchange](https://www.betfair.com/exchange/plus/cricket) — historical top-team-batter captures from the existing delayed-key collector. Event/market/selection identities, recorded offers, sizes and observation times are bundled in a sanitized artifact; this view has no live connection or forecast association.
    - [Betfair: dead-heat rules](https://support.betfair.com/app/answers/detail/a_id/6093/) — check the applicable operator's settlement terms.
    - [PriorsLab](https://www.priorslab.com/) — expected-value background; arithmetic here is shown explicitly.

    **V2 method:** recency-weighted player history shrunk toward batting-role priors, simulated jointly through an innings regime and batting depth. Historical shares use fractional credit for tied leaders. Forecast q uses the same dead-heat share interpretation. Empirical top-batter frequencies, mean run shares and forecast settlement shares answer different questions.

    **V3 experimental method:** categorical delivery probabilities depend on role, phase and innings order. Both teams share their respective declared legal-ball budgets; the second team's target comes from the simulated first innings. The separate accounting audit checks quantities, not predictive calibration. Shortened overs remain an unvalidated pre-start scenario.

    **Limits:** no live feed, confirmed XI, injury model, weather model, DLS model or guaranteed calibration. Venue samples can be small. DNB is stored separately from zero. Scores and market outcomes concern the selected team's innings, not the highest scorer across both teams.""")
        ball_meta_path = DATA / "ball_metadata.json"
        if ball_meta_path.exists():
            ball_audit = read_json(ball_meta_path).get("audit", {})
            st.caption(f"V3 delivery extraction: {ball_audit.get('eligible_innings', 'not recorded')} eligible team innings · {ball_audit.get('eligible_source_deliveries', 'not recorded')} source deliveries · {ball_audit.get('min_date', '')} to {ball_audit.get('max_date', '')}. The forecast additionally applies its cutoff. Exclusions differ from the descriptive history and v2 training set.")
            st.json({key: ball_audit[key] for key in ("exclusion_reasons", "reconciled_innings", "all_reconciliations_passed") if key in ball_audit}, expanded=False)
