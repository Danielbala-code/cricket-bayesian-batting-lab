"""Validated historical market snapshots; no credentials or trading connection."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np


def instant(value):
    if not isinstance(value, str):
        raise ValueError('Expected an ISO timestamp with timezone.')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('Invalid ISO timestamp.') from exc
    if result.tzinfo is None:
        raise ValueError('Timestamp must include a timezone.')
    return result.astimezone(timezone.utc)


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Provider IDs must be nonempty strings.')


def _number(value, price=False):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Price/size must be finite numeric values or null.')
    if value < 0 or (price and value <= 1):
        raise ValueError('Invalid price/size.')


def load_captures(path):
    """Return validated events, preserving unknown/missing quote states explicitly."""
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or payload.get('schema_version') != 1 or not isinstance(payload.get('events'), list):
        raise ValueError('Unsupported capture schema.')
    event_ids, market_ids = set(), set()
    for event in payload['events']:
        if not isinstance(event, dict):
            raise ValueError('Each event must be an object.')
        _identifier(event.get('event_id'))
        if not isinstance(event.get('event_name'), str) or not event['event_name'].strip():
            raise ValueError('Missing event label.')
        if not isinstance(event.get('coverage'), dict):
            raise ValueError('Coverage must be an object.')
        if event['event_id'] in event_ids:
            raise ValueError('Duplicate event ID.')
        event_ids.add(event['event_id'])
        start = instant(event.get('scheduled_start_utc'))
        instant(event.get('exported_at_utc'))
        if not isinstance(event.get('delayed_key'), bool):
            raise ValueError('Feed-delay provenance is required.')
        if not isinstance(event.get('markets'), list):
            raise ValueError('Missing markets.')
        teams = set()
        for market in event['markets']:
            if not isinstance(market, dict):
                raise ValueError('Each market must be an object.')
            _identifier(market.get('market_id'))
            team = market.get('team')
            if team not in ('India', 'New Zealand') or market.get('name') != 'Top ' + team + ' Batter':
                raise ValueError('Only separate India/New Zealand top-batter markets are supported.')
            if market['market_id'] in market_ids or team in teams:
                raise ValueError('Duplicate market/team identity.')
            market_ids.add(market['market_id']); teams.add(team)
            if not isinstance(market.get('rows'), list):
                raise ValueError('Missing runner rows.')
            seen = set()
            for row in market['rows']:
                if not isinstance(row, dict):
                    raise ValueError('Each runner row must be an object.')
                _identifier(row.get('selection_id'))
                if row['selection_id'] in seen:
                    raise ValueError('Duplicate runner identity within market.')
                seen.add(row['selection_id'])
                if row.get('market_id') != market['market_id'] or row.get('market_name') != market['name']:
                    raise ValueError('Runner market identity mismatch.')
                if not isinstance(row.get('player'), str) or not row['player'].strip():
                    raise ValueError('Missing player label.')
                asof = instant(row.get('as_of_utc'))
                if asof > start:
                    raise ValueError('A pre-start snapshot cannot use a post-start cutoff.')
                for field in ('last_market_update_received_at_utc', 'ltp_field_last_seen_at_utc'):
                    if row.get(field) is not None and instant(row[field]) > asof:
                        raise ValueError('A snapshot contains information observed after its cutoff.')
                for field in ('in_play', 'initial_image_seen'):
                    if row.get(field) is not None and not isinstance(row[field], bool):
                        raise ValueError('Market flags must be booleans or null.')
                for field in ('best_back', 'best_lay', 'last_traded_price'):
                    _number(row.get(field), price=True)
                for field in ('best_back_size', 'best_lay_size'):
                    _number(row.get(field))
    return payload['events']


def quote_quality(row):
    """Describe source quality, never certify historical availability/execution."""
    if row.get('initial_image_seen') is not True:
        return 'Missing initial market image'
    if row.get('market_status') != 'OPEN':
        return 'Market not recorded open'
    if row.get('in_play') is not False:
        return 'Not confirmed pre-play'
    if row.get('runner_status') != 'ACTIVE':
        return 'Runner not recorded active'
    price, size = row.get('best_back'), row.get('best_back_size')
    if price is None or size is None or size <= 0:
        return 'No recorded back offer'
    if price <= 1.01:
        return 'Uninformative 1.01 book — inspect source'
    lay, lay_size = row.get('best_lay'), row.get('best_lay_size')
    if lay is not None and lay_size and price > lay:
        return 'Crossed recorded book — inspect source'
    return 'Recorded back offer'


def participation_summary(scores):
    """Count batting appearances (zero is batting; -1 is DNB) per simulated XI."""
    values = np.asarray(scores)
    if (values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all()
            or np.any(values < -1) or np.any(values != np.floor(values))):
        raise ValueError('Expected a nonempty simulations × players integer score matrix, with -1 for DNB.')
    batted = values >= 0
    depth = batted.sum(axis=1)
    counts = list(range(int(depth.min()), values.shape[1] + 1))
    probabilities = [float(np.mean(depth == k)) for k in counts]
    return dict(n_sims=len(values), expected_batters=float(depth.mean()), counts=counts,
                probabilities=probabilities, depth_distribution=dict(zip(counts, probabilities)),
                p_bat=batted.mean(axis=0).tolist())
