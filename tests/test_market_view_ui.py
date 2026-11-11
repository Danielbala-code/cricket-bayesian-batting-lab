"""Integration checks for historical prices and unchanged forecast opportunity."""
import json
import os
from pathlib import Path
import tempfile
import unittest

TEST_CACHE = Path(__file__).resolve().parents[1] / "work" / "ui_test_cache"
TEST_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(TEST_CACHE)
os.environ["TEMP"] = str(TEST_CACHE)
os.environ["TMP"] = str(TEST_CACHE)
tempfile.tempdir = str(TEST_CACHE)

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class MarketViewUITests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, app.exception)
        return app

    def test_both_captures_and_team_fields_without_forecast_join(self):
        app = self.app()
        self.assertIn("Captured prices", [tab.label for tab in app.tabs])
        self.assertNotIn("forecast", app.session_state)
        payload = json.loads((ROOT / "data/captured_betfair.json").read_text())
        for event in payload["events"]:
            next(x for x in app.selectbox if x.label == "Captured match").select(event["event_id"]).run()
            for market in event["markets"]:
                next(x for x in app.selectbox if x.label == "Captured team market").select(market["market_id"]).run()
                self.assertFalse(app.exception, app.exception)
                table = next(x.value for x in app.dataframe if "Recorded quote status" in x.value.columns)
                self.assertEqual(len(table), len(market["rows"]))
                self.assertEqual(set(table["Selection ID"]), {row["selection_id"] for row in market["rows"]})
                self.assertIn("Last traded price (not an offer)", table)
                if all(row["market_status"] == "CLOSED" for row in market["rows"]):
                    self.assertTrue(any("No open pre-start offers" in x.value for x in app.warning))
        self.assertFalse(any("Bookmaker / quote" in x.label for x in app.text_input))
        self.assertFalse(any("same bookmaker" in x.label for x in app.checkbox))
        self.assertTrue(any("independent of the current forecast" in x.value for x in app.info))

    def test_participation_is_derived_from_both_engines_without_adjustment(self):
        app = self.app()
        for model in ("v2 score sampler", "v3 shared-opportunity (experimental)"):
            next(x for x in app.selectbox if x.label == "Forecast engine").select(model).run()
            if model.startswith("v3"):
                next(x for x in app.slider if x.label == "Declared overs per innings").set_value(20)
                next(x for x in app.select_slider if x.label == "Simulated innings").set_value(1000)
            else:
                next(x for x in app.select_slider if x.label == "Simulated innings").set_value(2000)
            app.run()
            app.button[0].click().run(timeout=180)
            self.assertFalse(app.exception, app.exception)
            scores = np.asarray(app.session_state["forecast"]["scores"])
            expected = (scores >= 0).sum(axis=1).mean()
            metric = next(x for x in app.metric if x.label == "Expected number who bat")
            self.assertEqual(metric.value, f"{expected:.2f}")
            self.assertAlmostEqual(sum((scores >= 0).mean(axis=0)), expected)

    def test_nz_diagnostic_uses_actual_slice_denominators(self):
        app = self.app()
        next(x for x in app.selectbox if x.label == "Opportunity diagnostic population").select("host:NZ").run()
        self.assertFalse(app.exception, app.exception)
        diagnostic = json.loads((ROOT / "data/opportunity_diagnostic.json").read_text())
        row = diagnostic["slices"]["host:NZ"]
        table = next(x.value for x in app.dataframe if list(x.value.columns) == ["Measure", "Predicted", "Observed"])
        self.assertAlmostEqual(table.iloc[0]["Observed"], round(row["observed_p_bat"], 4))
        self.assertEqual(next(x.value for x in app.metric if x.label == "Player predictions"), f"{row['player_predictions']:,}")

    def test_historical_depth_uses_one_summary_denominator(self):
        app = self.app()
        summary = pd.read_csv(ROOT / "data/batting_depth_summary.csv")
        scope = "New Zealand grounds"
        next(x for x in app.selectbox if x.label == "Historical depth population").select(scope).run()
        row = summary[(summary.scope == scope) & (summary.batting_context == "Combined")].iloc[0]
        self.assertFalse(app.exception, app.exception)
        self.assertEqual(next(x.value for x in app.metric if x.label == "Historical team innings"), f"{int(row.sample_team_innings):,}")
        self.assertEqual(next(x.value for x in app.metric if x.label == "Mean number who batted"), f"{row.mean_batters:.2f}")


if __name__ == "__main__":
    unittest.main()
