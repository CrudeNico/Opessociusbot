from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from level_detector.indicator import DEFAULT_LEVEL_PARAMS, find_levels

Side = Literal["long", "short"]

DEFAULT_SESSION_WINDOWS: tuple[tuple[int, int], ...] = (
    (7, 16),   # London session hours in UTC
    (12, 21),  # New York session hours in UTC
)
DEFAULT_TRADE_WINDOW_DAYS = (4, 25)
MIN_STOP_DISTANCE = 0.20
MIN_TP_SL_RATIO = 0.5  # TP must be at least half the SL distance
MAX_TP_SL_RATIO = 2.0  # TP cannot be more than 2x the SL distance


@dataclass(slots=True)
class BacktestConfig:
    month: str = "2025-04"
    spread: float = 0.05
    take_profit_offset: float = 0.20
    stop_loss_offset: float = 0.20
    history_months: int = 2
    data_root: Path | None = None
    levels_csv: Path | None = None
    output_dir: Path | None = None
    trading_sessions: tuple[tuple[int, int], ...] | None = DEFAULT_SESSION_WINDOWS
    trade_start_day: int | None = DEFAULT_TRADE_WINDOW_DAYS[0]
    trade_end_day: int | None = DEFAULT_TRADE_WINDOW_DAYS[1]
    flat_before_friday_close: bool = True
    trade_month_span: int = 2
    strategy: Literal["trend", "mirror"] = "trend"


@dataclass(slots=True)
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    minute_bars: pd.DataFrame
    config: BacktestConfig

    @property
    def total_pnl(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return float(self.equity_curve["equity"].iloc[-1])


def run_backtest(config: BacktestConfig) -> BacktestResult:
    project_root = Path(__file__).resolve().parent
    data_root = config.data_root or (project_root.parent / "data")
    if config.history_months < 1:
        raise ValueError("history_months must be at least 1.")
    if config.trade_month_span < 1:
        raise ValueError("trade_month_span must be at least 1.")

    month_sequence = _data_months(config.month, config.history_months, config.trade_month_span)

    minute_dir = data_root / "minute_monthly"
    minute_paths = [minute_dir / f"CL_1m_{month}.csv" for month in month_sequence]
    missing_minute = [str(path) for path in minute_paths if not path.exists()]
    if missing_minute:
        available = ", ".join(sorted(p.name for p in minute_dir.glob("CL_1m_*.csv")))
        raise FileNotFoundError(
            f"Missing minute data file(s): {missing_minute}. Available: {available or 'none'}."
        )

    minute_df = _load_minute_bars(minute_paths)
    minute_df = _add_indicator_columns(minute_df)

    if config.levels_csv:
        if not config.levels_csv.exists():
            raise FileNotFoundError(f"Levels file {config.levels_csv} not found.")
        levels_df = _load_levels([config.levels_csv])
    else:
        levels_dir = project_root.parent / "level_detector" / "outputs" / "levels"
        level_paths = []
        computed_frames: list[pd.DataFrame] = []
        for month in month_sequence:
            path = levels_dir / f"levels_{month}.csv"
            if path.exists():
                level_paths.append(path)
            else:
                computed = _compute_levels_for_month(month, data_root)
                if not computed.empty:
                    computed_frames.append(computed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    computed.to_csv(path, index=False)
        levels_df = _load_levels(level_paths) if level_paths else pd.DataFrame(
            columns=["row", "level", "strength", "time", "level_type"]
        )
        if computed_frames:
            frames = [levels_df] if not levels_df.empty else []
            frames.extend(computed_frames)
            levels_df = pd.concat(frames, ignore_index=True)
            levels_df.sort_values(["time", "level"], inplace=True)
            levels_df.reset_index(drop=True, inplace=True)

    if minute_df.empty or levels_df.empty:
        return BacktestResult(
            trades=pd.DataFrame(columns=_trade_columns()),
            equity_curve=pd.DataFrame(columns=["timestamp", "equity"]),
            minute_bars=minute_df,
            config=config,
        )

    trades = []
    equity_curve = []
    equity = 0.0
    active_trade: dict | None = None
    trade_counter = 0

    level_records = levels_df.to_dict("records")
    pending_levels: list[dict] = []
    next_level_idx = 0
    trade_window_start, trade_window_end = _trade_window_bounds(
        config.month,
        config.trade_start_day,
        config.trade_end_day,
        config.trade_month_span,
    )

    for _, bar in minute_df.iterrows():
        ts = bar["datetime"]
        bid_high = float(bar["high"])
        bid_low = float(bar["low"])
        close_price = float(bar["close"])
        indicator_snapshot = {
            "close": close_price,
            "rsi": bar.get("rsi"),
            "ema_20": bar.get("ema_20"),
            "ema_50": bar.get("ema_50"),
            "ema_200": bar.get("ema_200"),
        }

        while next_level_idx < len(level_records) and level_records[next_level_idx]["time"] <= ts:
            level_records[next_level_idx]["consumed"] = False
            pending_levels.append(level_records[next_level_idx])
            next_level_idx += 1

        if active_trade:
            exit_reason, exit_price = _evaluate_exit(active_trade, bid_high, bid_low, config.spread)
            if not exit_reason:
                forced_reason = None
                if config.flat_before_friday_close and _is_friday_close(ts):
                    forced_reason = "weekly_flat"
                elif ts >= trade_window_end:
                    forced_reason = "trade_window_end"
                if forced_reason:
                    exit_reason = forced_reason
                    exit_price = _market_exit_price(active_trade, close_price, config.spread)
            if exit_reason:
                exit_price = _round_price(exit_price)
                pnl = _round_price(active_trade["direction"] * (exit_price - active_trade["entry_price"]))
                equity += pnl
                trades.append(
                    {
                        "trade_id": active_trade["id"],
                        "side": active_trade["side"],
                        "level_type": active_trade["level_type"],
                        "level_row": active_trade["level_row"],
                        "level_detected_at": active_trade["level_time"],
                        "entry_time": active_trade["entry_time"],
                        "exit_time": ts,
                        "entry_price": active_trade["entry_price"],
                        "exit_price": exit_price,
                        "take_profit": active_trade["take_profit"],
                        "stop_loss": active_trade["stop_loss"],
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "holding_minutes": int((ts - active_trade["entry_time"]).total_seconds() // 60),
                    }
                )
                equity_curve.append({"timestamp": ts, "equity": _round_price(equity)})
                active_trade = None

        can_trade = (
            ts >= trade_window_start
            and ts < trade_window_end
            and _in_trading_session(ts, config.trading_sessions)
        )

        if not active_trade and can_trade:
            newly_opened = _maybe_open_trade(
                pending_levels=pending_levels,
                level_records=level_records,
                bid_high=bid_high,
                bid_low=bid_low,
                timestamp=ts,
                trade_counter=trade_counter,
                config=config,
                indicators=indicator_snapshot,
            )
            if newly_opened:
                trade_counter += 1
                active_trade = newly_opened
                pending_levels = [lvl for lvl in pending_levels if not lvl.get("consumed", False)]

    if active_trade:
        last_close = float(minute_df.iloc[-1]["close"])
        last_ts = minute_df.iloc[-1]["datetime"]
        if active_trade["side"] == "long":
            exit_price = last_close
        else:
            exit_price = last_close + config.spread
        exit_price = _round_price(exit_price)
        pnl = _round_price(active_trade["direction"] * (exit_price - active_trade["entry_price"]))
        equity += pnl
        trades.append(
            {
                "trade_id": active_trade["id"],
                "side": active_trade["side"],
                "level_type": active_trade["level_type"],
                "level_row": active_trade["level_row"],
                "level_detected_at": active_trade["level_time"],
                "entry_time": active_trade["entry_time"],
                "exit_time": last_ts,
                "entry_price": active_trade["entry_price"],
                "exit_price": exit_price,
                "take_profit": active_trade["take_profit"],
                "stop_loss": active_trade["stop_loss"],
                "exit_reason": "end_of_data",
                "pnl": pnl,
                "holding_minutes": int((last_ts - active_trade["entry_time"]).total_seconds() // 60),
            }
        )
        equity_curve.append({"timestamp": last_ts, "equity": _round_price(equity)})

    trades_df = pd.DataFrame(trades, columns=_trade_columns())
    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
    return BacktestResult(
        trades=trades_df,
        equity_curve=equity_df,
        minute_bars=minute_df,
        config=config,
    )


def _maybe_open_trade(
    pending_levels: list[dict],
    level_records: list[dict],
    bid_high: float,
    bid_low: float,
    timestamp: pd.Timestamp,
    trade_counter: int,
    config: BacktestConfig,
    indicators: dict | None = None,
) -> dict | None:
    if config.strategy == "mirror":
        return _maybe_open_trade_mirror(
            pending_levels=pending_levels,
            level_records=level_records,
            bid_high=bid_high,
            bid_low=bid_low,
            timestamp=timestamp,
            trade_counter=trade_counter,
            config=config,
            indicators=indicators,
        )
    return _maybe_open_trade_trend(
        pending_levels=pending_levels,
        level_records=level_records,
        bid_high=bid_high,
        bid_low=bid_low,
        timestamp=timestamp,
        trade_counter=trade_counter,
        config=config,
        indicators=indicators,
    )


def _maybe_open_trade_trend(
    pending_levels: list[dict],
    level_records: list[dict],
    bid_high: float,
    bid_low: float,
    timestamp: pd.Timestamp,
    trade_counter: int,
    config: BacktestConfig,
    indicators: dict | None = None,
) -> dict | None:
    spread = config.spread
    ask_low = bid_low + spread
    ask_high = bid_high + spread
    if indicators and not _indicators_allow_environment_trend(indicators):
        return None
    for level in pending_levels:
        if level.get("consumed"):
            continue
        entry_price = _round_price(float(level["level"]))
        if level["level_type"] == "support":
            if ask_low <= entry_price <= ask_high:
                if not _indicators_allow_side_trend("long", indicators):
                    continue
                trade_id = trade_counter + 1
                level["consumed"] = True
                take_profit, stop_loss = _resolve_targets(
                    level_records=level_records,
                    entry_level=level,
                    side="long",
                    timestamp=timestamp,
                    config=config,
                    indicators=indicators,
                )
                return {
                    "id": trade_id,
                    "side": "long",
                    "direction": 1.0,
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "level_price": entry_price,
                    "level_type": "support",
                    "level_row": int(level.get("row", -1)),
                    "level_time": level["time"],
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                }
        else:
            if bid_low <= entry_price <= bid_high:
                if not _indicators_allow_side_trend("short", indicators):
                    continue
                trade_id = trade_counter + 1
                level["consumed"] = True
                take_profit, stop_loss = _resolve_targets(
                    level_records=level_records,
                    entry_level=level,
                    side="short",
                    timestamp=timestamp,
                    config=config,
                    indicators=indicators,
                )
                return {
                    "id": trade_id,
                    "side": "short",
                    "direction": -1.0,
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "level_price": entry_price,
                    "level_type": "resistance",
                    "level_row": int(level.get("row", -1)),
                    "level_time": level["time"],
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                }
    return None


def _maybe_open_trade_mirror(
    pending_levels: list[dict],
    level_records: list[dict],
    bid_high: float,
    bid_low: float,
    timestamp: pd.Timestamp,
    trade_counter: int,
    config: BacktestConfig,
    indicators: dict | None = None,
) -> dict | None:
    spread = config.spread
    ask_low = bid_low + spread
    ask_high = bid_high + spread
    if indicators and not _indicators_allow_environment_trend(indicators):
        return None
    for level in pending_levels:
        if level.get("consumed"):
            continue
        entry_price = _round_price(float(level["level"]))
        if level["level_type"] == "support":
            if ask_low <= entry_price <= ask_high:
                if not _indicators_allow_side_trend("long", indicators):
                    continue
                trade_id = trade_counter + 1
                level["consumed"] = True
                take_profit, stop_loss = _resolve_targets(
                    level_records=level_records,
                    entry_level=level,
                    side="short",
                    timestamp=timestamp,
                    config=config,
                    indicators=indicators,
                )
                return {
                    "id": trade_id,
                    "side": "short",
                    "direction": -1.0,
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "level_price": entry_price,
                    "level_type": "support",
                    "level_row": int(level.get("row", -1)),
                    "level_time": level["time"],
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                }
        else:
            if bid_low <= entry_price <= bid_high:
                if not _indicators_allow_side_trend("short", indicators):
                    continue
                trade_id = trade_counter + 1
                level["consumed"] = True
                take_profit, stop_loss = _resolve_targets(
                    level_records=level_records,
                    entry_level=level,
                    side="long",
                    timestamp=timestamp,
                    config=config,
                    indicators=indicators,
                )
                return {
                    "id": trade_id,
                    "side": "long",
                    "direction": 1.0,
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "level_price": entry_price,
                    "level_type": "resistance",
                    "level_row": int(level.get("row", -1)),
                    "level_time": level["time"],
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                }
    return None


def _resolve_targets(
    level_records: list[dict],
    entry_level: dict,
    side: Side,
    timestamp: pd.Timestamp,
    config: BacktestConfig,
    indicators: dict | None = None,
) -> tuple[float, float]:
    entry_price = _round_price(float(entry_level["level"]))
    if side == "long":
        tp = _closest_level_price(level_records, entry_level, "resistance", "above", timestamp)
        sl = _closest_level_price(level_records, entry_level, "support", "below", timestamp)
        default_tp = entry_price + config.take_profit_offset
        default_sl = entry_price - config.stop_loss_offset
    else:
        tp = _closest_level_price(level_records, entry_level, "support", "below", timestamp)
        sl = _closest_level_price(level_records, entry_level, "resistance", "above", timestamp)
        default_tp = entry_price - config.take_profit_offset
        default_sl = entry_price + config.stop_loss_offset

    take_profit = float(tp if tp is not None else default_tp)
    stop_loss = float(sl if sl is not None else default_sl)
    tp_dist = abs(take_profit - entry_price)
    sl_dist = abs(stop_loss - entry_price)

    if _is_opening_window(timestamp):
        tp_dist *= 0.5
        sl_dist *= 0.5

    tp_dist, sl_dist = _enforce_distance_rules(tp_dist, sl_dist)
    tp_dist, sl_dist = _apply_strategy_adjustments(
        strategy=config.strategy,
        side=side,
        tp_distance=tp_dist,
        sl_distance=sl_dist,
        indicators=indicators,
    )
    tp_dist, sl_dist = _enforce_distance_rules(tp_dist, sl_dist)

    if side == "long":
        take_profit = entry_price + tp_dist
        stop_loss = entry_price - sl_dist
    else:
        take_profit = entry_price - tp_dist
        stop_loss = entry_price + sl_dist

    return _round_price(take_profit), _round_price(stop_loss)


def _closest_level_price(
    level_records: list[dict],
    entry_level: dict,
    target_type: str,
    direction: Literal["above", "below"],
    cutoff_time: pd.Timestamp,
) -> float | None:
    entry_price = float(entry_level["level"])
    best: float | None = None
    for candidate in level_records:
        if candidate is entry_level:
            continue
        if candidate.get("level_type") != target_type:
            continue
        level_time = candidate.get("time")
        if isinstance(level_time, pd.Timestamp):
            if level_time > cutoff_time:
                continue
        elif level_time is not None:
            try:
                if pd.Timestamp(level_time) > cutoff_time:
                    continue
            except Exception:
                pass
        try:
            price = float(candidate.get("level"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price):
            continue
        if direction == "above":
            if price <= entry_price:
                continue
            if best is None or price < best:
                best = price
        else:
            if price >= entry_price:
                continue
            if best is None or price > best:
                best = price
    return best


def _enforce_distance_rules(tp_distance: float, sl_distance: float) -> tuple[float, float]:
    tp_distance = max(tp_distance, 0.0)
    sl_distance = max(sl_distance, MIN_STOP_DISTANCE)
    if sl_distance == 0:
        sl_distance = MIN_STOP_DISTANCE

    ratio = tp_distance / sl_distance if sl_distance else None
    if ratio is None:
        return tp_distance, sl_distance

    if ratio > MAX_TP_SL_RATIO:
        tp_distance = sl_distance * MAX_TP_SL_RATIO
    elif ratio < MIN_TP_SL_RATIO:
        sl_cap = tp_distance / MIN_TP_SL_RATIO if MIN_TP_SL_RATIO else sl_distance
        if sl_cap >= MIN_STOP_DISTANCE:
            sl_distance = min(sl_distance, sl_cap)
        sl_distance = max(sl_distance, MIN_STOP_DISTANCE)
        ratio = tp_distance / sl_distance if sl_distance else MIN_TP_SL_RATIO
        if ratio < MIN_TP_SL_RATIO:
            tp_distance = sl_distance * MIN_TP_SL_RATIO
    return tp_distance, sl_distance


def _apply_strategy_adjustments(
    strategy: Literal["trend", "mirror"],
    side: Side,
    tp_distance: float,
    sl_distance: float,
    indicators: dict | None,
) -> tuple[float, float]:
    if not indicators:
        return tp_distance, sl_distance
    rsi = _indicator_value(indicators, "rsi")
    if strategy == "mirror":
        strategy = "trend"

    if side == "long":
        if rsi is not None and rsi >= 60 and _price_above_all_emas(indicators):
            tp_distance *= 0.85
            sl_distance *= 1.1
        elif rsi is not None and rsi < 50:
            tp_distance *= 0.9
    else:
        if rsi is not None and rsi <= 40 and _price_below_all_emas(indicators):
            tp_distance *= 0.85
            sl_distance *= 1.1
        elif rsi is not None and rsi > 50:
            tp_distance *= 0.8
            sl_distance *= 0.9
    return tp_distance, sl_distance


def _indicator_value(indicators: dict, key: str) -> float | None:
    if not indicators:
        return None
    value = indicators.get(key)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _price_above_all_emas(indicators: dict) -> bool:
    close = _indicator_value(indicators, "close")
    if close is None:
        return False
    emas = [_indicator_value(indicators, f"ema_{span}") for span in (20, 50, 200)]
    valid = [ema for ema in emas if ema is not None]
    if not valid:
        return False
    return all(close >= ema for ema in valid)


def _price_below_all_emas(indicators: dict) -> bool:
    close = _indicator_value(indicators, "close")
    if close is None:
        return False
    emas = [_indicator_value(indicators, f"ema_{span}") for span in (20, 50, 200)]
    valid = [ema for ema in emas if ema is not None]
    if not valid:
        return False
    return all(close <= ema for ema in valid)


def _indicators_allow_environment_trend(indicators: dict) -> bool:
    rsi = _indicator_value(indicators, "rsi")
    if rsi is not None and rsi < 40 and _price_below_all_emas(indicators):
        return False
    return True


def _indicators_allow_side_trend(side: Side, indicators: dict | None) -> bool:
    if not indicators:
        return True
    if side == "long":
        return _allow_long_trend(indicators)
    return _allow_short_trend(indicators)


def _allow_long_trend(indicators: dict) -> bool:
    rsi = _indicator_value(indicators, "rsi")
    if rsi is not None:
        if rsi < 40:
            return False
        if rsi > 85:
            return False
    ema20 = _indicator_value(indicators, "ema_20")
    ema50 = _indicator_value(indicators, "ema_50")
    ema200 = _indicator_value(indicators, "ema_200")
    fast_vs_slow = False
    if ema20 is not None and ema200 is not None and ema20 > ema200:
        fast_vs_slow = True
    if ema50 is not None and ema200 is not None and ema50 > ema200:
        fast_vs_slow = True
    if not fast_vs_slow:
        return False
    close = _indicator_value(indicators, "close")
    if close is not None:
        if ema20 is not None and close < ema20:
            return False
        if ema50 is not None and close < ema50 and ema20 is None:
            return False
    return True


def _allow_short_trend(indicators: dict) -> bool:
    rsi = _indicator_value(indicators, "rsi")
    if rsi is not None and rsi > 50:
        return False
    close = _indicator_value(indicators, "close")
    ema20 = _indicator_value(indicators, "ema_20")
    ema50 = _indicator_value(indicators, "ema_50")
    ema200 = _indicator_value(indicators, "ema_200")
    if ema20 is not None and ema50 is not None:
        if not (ema20 < ema50):
            return False
    if ema50 is not None and ema200 is not None:
        if not (ema50 < ema200):
            return False
    if close is not None and ema20 is not None and close >= ema20:
        return False
    if close is not None and ema200 is not None and close >= ema200:
        return False
    return True


def _evaluate_exit(
    trade: dict,
    bid_high: float,
    bid_low: float,
    spread: float,
) -> tuple[str | None, float | None]:
    if trade["side"] == "long":
        if bid_low <= trade["stop_loss"]:
            return "stop_loss", trade["stop_loss"]
        if bid_high >= trade["take_profit"]:
            return "take_profit", trade["take_profit"]
    else:
        ask_high = bid_high + spread
        ask_low = bid_low + spread
        if ask_high >= trade["stop_loss"]:
            return "stop_loss", trade["stop_loss"]
        if ask_low <= trade["take_profit"]:
            return "take_profit", trade["take_profit"]
    return None, None


def _market_exit_price(trade: dict, close_price: float, spread: float) -> float:
    if trade["side"] == "long":
        return close_price
    return close_price + spread


def _is_friday_close(ts: pd.Timestamp) -> bool:
    return ts.weekday() == 4 and ts.hour == 23 and ts.minute == 59


def _is_opening_window(ts: pd.Timestamp) -> bool:
    return 7 <= ts.hour < 9


def _load_minute_bars(csv_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        dt_series = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["datetime"] = dt_series.dt.tz_convert(None)
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        for column in ("open", "high", "low", "close"):
            df[column] = df[column].astype(float)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("datetime", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _add_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["rsi"] = _compute_rsi(df["close"], period=14)
    return df


def _load_levels(csv_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        dt_series = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["time"] = dt_series.dt.tz_convert(None)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["row", "level", "strength", "time", "level_type"])
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["time", "level"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _compute_levels_for_month(month: str, data_root: Path) -> pd.DataFrame:
    csv_path = data_root / "monthly_30m" / f"CL_30m_{month}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"30m data {csv_path} not found while computing levels.")
    data = pd.read_csv(csv_path)
    if data.empty:
        return pd.DataFrame(columns=["row", "level", "strength", "time", "level_type"])
    dt_series = pd.to_datetime(data["datetime"], utc=True, errors="coerce")
    data["datetime"] = dt_series.dt.tz_convert(None)
    data.sort_values("datetime", inplace=True)
    data.reset_index(drop=True, inplace=True)
    levels_df = find_levels(data=data, **DEFAULT_LEVEL_PARAMS)
    if levels_df.empty:
        return pd.DataFrame(columns=["row", "level", "strength", "time", "level_type"])
    levels_df = levels_df.reset_index()
    if "row" not in levels_df.columns:
        levels_df.rename(columns={"index": "row"}, inplace=True)
    time_series = pd.to_datetime(levels_df["time"], utc=True, errors="coerce")
    levels_df["time"] = time_series.dt.tz_convert(None)
    return levels_df[["row", "level", "strength", "time", "level_type"]]


def _shift_month(dt: datetime, delta: int) -> datetime:
    year = dt.year + (dt.month - 1 + delta) // 12
    month = (dt.month - 1 + delta) % 12 + 1
    return datetime(year, month, 1)


def _month_sequence(target_month: str, history_months: int) -> list[str]:
    base = datetime.strptime(target_month, "%Y-%m")
    return [
        _shift_month(base, -offset).strftime("%Y-%m")
        for offset in range(history_months - 1, -1, -1)
    ]


def _data_months(target_month: str, history_months: int, trade_month_span: int) -> list[str]:
    base = datetime.strptime(target_month, "%Y-%m")
    offsets = range(-(history_months - 1), trade_month_span)
    seen: list[str] = []
    for offset in offsets:
        label = _shift_month(base, offset).strftime("%Y-%m")
        if label not in seen:
            seen.append(label)
    return seen


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    base = datetime.strptime(month, "%Y-%m")
    start = pd.Timestamp(base)
    end = pd.Timestamp(_shift_month(base, 1))
    return start, end


def _trade_window_bounds(
    month: str,
    start_day: int | None,
    end_day: int | None,
    month_span: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    month_start, month_end = _month_bounds(month)
    window_start = month_start
    final_month_end = pd.Timestamp(_shift_month(month_start.to_pydatetime(), month_span))
    window_end = final_month_end
    if start_day is not None:
        if start_day < 1:
            raise ValueError("trade_start_day must be at least 1.")
        window_start = month_start + pd.Timedelta(days=start_day - 1)
    if end_day is not None:
        if end_day < 1:
            raise ValueError("trade_end_day must be at least 1.")
        last_month_start = pd.Timestamp(_shift_month(month_start.to_pydatetime(), month_span - 1))
        window_end = min(last_month_start + pd.Timedelta(days=end_day), final_month_end)
    if window_start >= window_end:
        raise ValueError("Trade window start must be before its end.")
    return window_start, window_end


def _plot_window_bounds(config: BacktestConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    plot_start, _ = _month_bounds(config.month)
    plot_end = pd.Timestamp(_shift_month(plot_start.to_pydatetime(), config.trade_month_span))
    return plot_start, plot_end


def _format_month_label(start_month: str, span: int) -> str:
    if span <= 1:
        return start_month
    base = datetime.strptime(start_month, "%Y-%m")
    end_month = _shift_month(base, span - 1).strftime("%Y-%m")
    return f"{start_month} to {end_month}"


def _in_trading_session(ts: pd.Timestamp, sessions: tuple[tuple[int, int], ...] | None) -> bool:
    if not sessions:
        return True
    minute_of_day = ts.hour * 60 + ts.minute
    for start_hour, end_hour in sessions:
        start_minute = max(0, start_hour * 60)
        end_minute = min(24 * 60, end_hour * 60)
        if start_minute <= minute_of_day < end_minute:
            return True
    return False


def _round_price(value: float, decimals: int = 4) -> float:
    return round(float(value), decimals)


def _trade_columns() -> list[str]:
    return [
        "trade_id",
        "side",
        "level_type",
        "level_row",
        "level_detected_at",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "take_profit",
        "stop_loss",
        "exit_reason",
        "pnl",
        "holding_minutes",
    ]


def _prepare_output_dirs(base_dir: Path) -> tuple[Path, Path]:
    trades_dir = base_dir / "trades"
    equity_dir = base_dir / "equity"
    trades_dir.mkdir(parents=True, exist_ok=True)
    equity_dir.mkdir(parents=True, exist_ok=True)
    return trades_dir, equity_dir


def save_result(result: BacktestResult) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent
    base_dir = result.config.output_dir or (project_root / "outputs")
    trades_dir, equity_dir = _prepare_output_dirs(base_dir)
    trades_path = trades_dir / f"trades_{result.config.month}.csv"
    equity_path = equity_dir / f"equity_{result.config.month}.csv"
    result.trades.to_csv(trades_path, index=False)
    result.equity_curve.to_csv(equity_path, index=False)
    return trades_path, equity_path


def save_trade_plot(result: BacktestResult) -> Path:
    if result.minute_bars.empty:
        raise ValueError("Minute data is required to build the trade plot.")
    project_root = Path(__file__).resolve().parent
    base_dir = result.config.output_dir or (project_root / "outputs")
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_start, plot_end = _plot_window_bounds(result.config)
    plot_minutes = result.minute_bars[
        (result.minute_bars["datetime"] >= plot_start)
        & (result.minute_bars["datetime"] < plot_end)
    ].copy()
    if plot_minutes.empty:
        raise ValueError(f"No minute data within specified plot window for {result.config.month}.")
    label = _format_month_label(result.config.month, result.config.trade_month_span)
    fig = _build_trade_figure(plot_minutes, result.trades, label)
    html_path = plot_dir / f"trades_{result.config.month}.html"
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


def _build_trade_figure(minute_df: pd.DataFrame, trades_df: pd.DataFrame, label_month: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f"CL 1m trades for {label_month}", "RSI (14)"),
    )

    fig.add_trace(
        go.Candlestick(
            name="CL 1m",
            x=minute_df["datetime"],
            open=minute_df["open"],
            high=minute_df["high"],
            low=minute_df["low"],
            close=minute_df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            opacity=0.9,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    ema_settings = (
        (20, "#42a5f5"),
        (50, "#ab47bc"),
        (200, "#8d6e63"),
    )
    for period, color in ema_settings:
        ema_series = minute_df["close"].ewm(span=period, adjust=False).mean()
        fig.add_trace(
            go.Scatter(
                name=f"EMA {period}",
                x=minute_df["datetime"],
                y=ema_series,
                mode="lines",
                line=dict(color=color, width=1.2),
            ),
            row=1,
            col=1,
        )

    rsi_series = _compute_rsi(minute_df["close"], period=14)
    fig.add_trace(
        go.Scatter(
            name="RSI",
            x=minute_df["datetime"],
            y=rsi_series,
            mode="lines",
            line=dict(color="#ffd54f", width=1.5),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for level, color in ((70, "#ef5350"), (30, "#26a69a")):
        fig.add_shape(
            type="line",
            xref="x",
            yref="y2",
            x0=minute_df["datetime"].iloc[0],
            x1=minute_df["datetime"].iloc[-1],
            y0=level,
            y1=level,
            line=dict(color=color, width=1, dash="dash"),
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    if trades_df.empty:
        return fig

    trades = trades_df.copy()
    for column in ("entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column])

    long_trades = trades[trades["side"] == "long"]
    short_trades = trades[trades["side"] == "short"]

    def _add_marker_trace(df: pd.DataFrame, label: str, time_col: str, price_col: str, color: str, symbol: str) -> None:
        if df.empty:
            return
        fig.add_trace(
            go.Scatter(
                name=label,
                x=df[time_col],
                y=df[price_col],
                mode="markers",
                marker=dict(color=color, size=8, symbol=symbol, line=dict(color="black", width=0.5)),
                customdata=df[["trade_id", "pnl"]],
                hovertemplate=(
                    f"{label}<br>"
                    "Trade %{customdata[0]}<br>"
                    "Price %{y:.4f}<br>"
                    "Time %{x|%Y-%m-%d %H:%M}<br>"
                    "PnL %{customdata[1]:.4f}<extra></extra>"
                ),
            )
        )

    _add_marker_trace(long_trades, "Long Entries", "entry_time", "entry_price", "#00c853", "triangle-up")
    _add_marker_trace(short_trades, "Short Entries", "entry_time", "entry_price", "#ff8f00", "triangle-down")

    exit_styles = {
        "take_profit": ("Take-Profit Exits", "#00e676"),
        "stop_loss": ("Stop-Loss Exits", "#ff1744"),
    }
    for reason, (label, color) in exit_styles.items():
        subset = trades[trades["exit_reason"] == reason]
        _add_marker_trace(subset, label, "exit_time", "exit_price", color, "x")

    remainder = trades[~trades["exit_reason"].isin(exit_styles.keys())]
    _add_marker_trace(remainder, "Other Exits", "exit_time", "exit_price", "#cfd8dc", "circle-open")

    for _, trade in trades.iterrows():
        pnl_color = "#00c853" if trade["pnl"] > 0 else "#ef5350" if trade["pnl"] < 0 else "#cfd8dc"
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=trade["entry_time"],
            x1=trade["exit_time"],
            y0=trade["entry_price"],
            y1=trade["exit_price"],
            line=dict(color=pnl_color, width=1.5),
            opacity=0.6,
        )

    return fig


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest CL support/resistance touches using bid/ask limit logic."
    )
    parser.add_argument("--month", default="2025-04", help="Month to backtest (YYYY-MM).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data root directory (defaults to ../data).",
    )
    parser.add_argument(
        "--levels-csv",
        type=Path,
        default=None,
        help="Override the detected levels CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where equity/trade CSVs will be stored.",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=0.05,
        help="Fixed spread between bid and ask.",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.20,
        help="Distance in price units between entry and take profit.",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.20,
        help="Distance in price units between entry and stop loss.",
    )
    parser.add_argument(
        "--history-months",
        type=int,
        default=2,
        help="Number of consecutive months (including the target) to load for context.",
    )
    parser.add_argument(
        "--trade-start-day",
        type=int,
        default=DEFAULT_TRADE_WINDOW_DAYS[0],
        help="First calendar day (1-indexed) to allow new entries for the target month.",
    )
    parser.add_argument(
        "--trade-end-day",
        type=int,
        default=DEFAULT_TRADE_WINDOW_DAYS[1],
        help="Last calendar day (inclusive) to allow new entries for the target month.",
    )
    parser.add_argument(
        "--disable-session-filter",
        action="store_true",
        help="Allow entries outside the default London/New York sessions.",
    )
    parser.add_argument(
        "--keep-weekend-positions",
        action="store_true",
        help="Skip flattening positions before Friday ends.",
    )
    parser.add_argument(
        "--trade-month-span",
        type=int,
        default=2,
        help="Number of consecutive months (starting at --month) during which trades may occur.",
    )
    parser.add_argument(
        "--strategy",
        choices=("trend", "mirror"),
        default="trend",
        help="Select the trading logic: 'trend' (support/res trend-following) or 'mirror' (inverted entries).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = BacktestConfig(
        month=args.month,
        spread=args.spread,
        take_profit_offset=args.take_profit,
        stop_loss_offset=args.stop_loss,
        data_root=args.data_dir,
        levels_csv=args.levels_csv,
        output_dir=args.output_dir,
        history_months=args.history_months,
        trade_start_day=args.trade_start_day,
        trade_end_day=args.trade_end_day,
        trading_sessions=None if args.disable_session_filter else DEFAULT_SESSION_WINDOWS,
        flat_before_friday_close=not args.keep_weekend_positions,
        trade_month_span=args.trade_month_span,
        strategy=args.strategy,
    )
    result = run_backtest(cfg)
    trades_path, equity_path = save_result(result)
    plot_path = save_trade_plot(result) if not result.minute_bars.empty else None
    summary_lines = [
        f"Trades written to: {trades_path}",
        f"Equity curve written to: {equity_path}",
        f"Trade plot written to: {plot_path}" if plot_path else "Trade plot skipped (no minute data).",
        f"Total PnL: {result.total_pnl:.4f}",
        f"Trades taken: {len(result.trades)}",
    ]
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
