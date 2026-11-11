import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from captured_markets import load_captures, participation_summary, quote_quality


def sample():
    row = dict(market_id='1.23', market_name='Top India Batter', selection_id='123',
               player='A', as_of_utc='2026-11-07T02:00:00Z', market_status='OPEN',
               in_play=False, initial_image_seen=True, runner_status='ACTIVE',
               best_back=4., best_back_size=20., best_lay=4.2, best_lay_size=10.,
               last_traded_price=3.9, ltp_field_last_seen_at_utc='2026-11-07T01:59:00Z',
               last_market_update_received_at_utc='2026-11-07T01:59:00Z')
    return dict(schema_version=1, events=[dict(event_id='1', event_name='New Zealand v India',
        scheduled_start_utc='2026-11-07T02:00:00Z', exported_at_utc='2026-11-11T13:00:00Z',
        delayed_key=True, coverage={}, markets=[dict(market_id='1.23', name='Top India Batter',
        team='India', rows=[row])])])


class CaptureTests(unittest.TestCase):
    def load(self, payload):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p = Path(d) / 'prices.json'
            p.write_text(json.dumps(payload), encoding='utf-8')
            return load_captures(p)

    def test_roundtrip_preserves_identity_and_separates_ltp(self):
        events = self.load(sample())
        r = events[0]['markets'][0]['rows'][0]
        self.assertEqual(r['selection_id'], '123')
        self.assertEqual(r['last_traded_price'], 3.9)
        self.assertEqual(r['best_back'], 4.)
        self.assertEqual(quote_quality(r), 'Recorded back offer')

    def test_duplicate_runner_or_event_fails(self):
        p = sample(); p['events'][0]['markets'][0]['rows'] *= 2
        with self.assertRaises(ValueError): self.load(p)

    def test_malformed_objects_fail_with_a_validation_error(self):
        bad = [[], {'schema_version':1, 'events':[None]}]
        for key, value in [('event_name', None), ('coverage', []), ('markets', [None])]:
            p=sample(); p['events'][0][key]=value; bad.append(p)
        p=sample(); p['events'][0]['markets'][0]['rows']=[None]; bad.append(p)
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError): self.load(payload)
        p = sample(); p['events'] *= 2
        with self.assertRaises(ValueError): self.load(p)

    def test_identity_timestamp_and_nonfinite_rejected(self):
        for field, value in [('market_id', 'wrong'), ('as_of_utc', '2026-11-07T02:00:00'),
                             ('best_back', float('nan')), ('best_back_size', -1.),
                             ('in_play', 'False'), ('last_market_update_received_at_utc', '2026-11-07T02:01:00Z')]:
            p = sample(); p['events'][0]['markets'][0]['rows'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.load(p)

    def test_ltp_is_never_a_missing_offer_fallback(self):
        r = sample()['events'][0]['markets'][0]['rows'][0]
        r.update(best_back=None, best_back_size=None)
        self.assertEqual(quote_quality(r), 'No recorded back offer')
        self.assertEqual(self.load({'schema_version':1, 'events':[]}), [])

    def test_unsafe_states_and_suspicious_books_are_labelled(self):
        base = sample()['events'][0]['markets'][0]['rows'][0]
        for changes in [dict(market_status='CLOSED'), dict(in_play=True),
                        dict(initial_image_seen=False), dict(runner_status='REMOVED'),
                        dict(best_back_size=0), dict(best_back=1.01), dict(best_lay=3.5)]:
            r = {**base, **changes}
            self.assertNotEqual(quote_quality(r), 'Recorded back offer')

    def test_participation_counts_batted_zero_and_zero_ball_appearances(self):
        scores = np.array([[0, 20, -1, -1], [10, 0, 30, -1], [1, 2, 0, 4]])
        s = participation_summary(scores)
        self.assertAlmostEqual(s['expected_batters'], 3.)
        np.testing.assert_allclose(s['p_bat'], [1, 1, 2/3, 1/3])
        self.assertEqual(s['counts'], [2, 3, 4])
        np.testing.assert_allclose(s['probabilities'], [1/3]*3)
        self.assertAlmostEqual(sum(s['p_bat']), s['expected_batters'])
        with self.assertRaises(ValueError): participation_summary([[0, float('nan')]])


if __name__ == '__main__': unittest.main()
