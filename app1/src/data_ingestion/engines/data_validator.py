#!/usr/bin/env python3
"""
Data Validator
Lightweight validation utilities for price and candle data
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for data validation thresholds"""
    max_spread: float = 0.01  # 100 pips for safety; tuned per instrument later
    max_price_jump_pct: float = 5.0  # percent between consecutive prices
    max_candle_range_pct: float = 10.0  # percent of close
    min_timestamp: datetime = datetime(2000, 1, 1)
    max_future_skew_s: int = 10  # seconds allowed in the future clock skew


class DataValidator:
    """Validates price and candle data for basic quality rules"""

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self.anomaly_counts = {
            'invalid_price': 0,
            'abnormal_spread': 0,
            'price_jump': 0,
            'timestamp_out_of_bounds': 0,
            'invalid_candle_values': 0,
            'invalid_candle_bounds': 0,
            'inconsistent_ohlc': 0,
            'excessive_candle_range': 0,
        }

    def is_finite(self, *values: float) -> bool:
        for v in values:
            if v is None or isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return False
        return True

    def validate_price(self, instrument: str, time: datetime, bid: float, ask: float,
                       previous_bid_ask: Optional[Tuple[float, float]] = None) -> bool:
        """Validate a single price tick."""
        if not self.is_finite(bid, ask):
            logger.warning(f"Invalid price (NaN/Inf) for {instrument}")
            self.anomaly_counts['invalid_price'] += 1
            return False

        if bid <= 0 or ask <= 0:
            logger.warning(f"Non-positive price for {instrument}: bid={bid}, ask={ask}")
            self.anomaly_counts['invalid_price'] += 1
            return False

        if ask < bid:
            logger.warning(f"Crossed market for {instrument}: bid={bid} > ask={ask}")
            self.anomaly_counts['invalid_price'] += 1
            return False

        spread = ask - bid
        if spread < 0 or spread > self.config.max_spread:
            logger.warning(f"Abnormal spread for {instrument}: {spread}")
            self.anomaly_counts['abnormal_spread'] += 1
            return False

        now = datetime.utcnow()
        if time < self.config.min_timestamp or time > now + timedelta(seconds=self.config.max_future_skew_s):
            logger.warning(f"Timestamp out of bounds for {instrument}: {time.isoformat()}")
            self.anomaly_counts['timestamp_out_of_bounds'] += 1
            return False

        if previous_bid_ask:
            prev_bid, prev_ask = previous_bid_ask
            prev_mid = (prev_bid + prev_ask) / 2.0
            mid = (bid + ask) / 2.0
            if prev_mid > 0:
                jump_pct = abs(mid - prev_mid) / prev_mid * 100.0
                if jump_pct > self.config.max_price_jump_pct:
                    logger.warning(f"Excessive price jump for {instrument}: {jump_pct:.2f}%")
                    self.anomaly_counts['price_jump'] += 1
                    return False

        return True

    def validate_candle(self, instrument: str, time: datetime, o: float, h: float, l: float, c: float,
                        volume: Optional[float] = None) -> bool:
        """Validate a single candle."""
        if not self.is_finite(o, h, l, c):
            logger.warning(f"Invalid candle values (NaN/Inf) for {instrument} @ {time}")
            self.anomaly_counts['invalid_candle_values'] += 1
            return False

        if l > h or min(o, c, l, h) <= 0:
            logger.warning(f"Invalid candle bounds for {instrument} @ {time}")
            self.anomaly_counts['invalid_candle_bounds'] += 1
            return False

        if not (l <= o <= h and l <= c <= h):
            logger.warning(f"O/H/L/C not consistent for {instrument} @ {time}")
            self.anomaly_counts['inconsistent_ohlc'] += 1
            return False

        # Range sanity vs close
        base = max(abs(c), 1e-9)
        range_pct = (h - l) / base * 100.0
        if range_pct > self.config.max_candle_range_pct:
            logger.warning(f"Excessive candle range {range_pct:.2f}% for {instrument} @ {time}")
            self.anomaly_counts['excessive_candle_range'] += 1
            return False

        return True


