from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import math

import pandas as pd
import plotly.graph_objects as go

from level_detector.indicator import DEFAULT_LEVEL_PARAMS, find_levels

Side = Literal["long", "short"]

DEFAULT_SESSION_WINDOWS: tuple[tuple[int, int], ...] = (
    (7, 16),   # London session hours in UTC
    (12, 21),  # New York session hours in UTC
)
DEFAULT_TRADE_WINDOW_DAYS = (4, 25)


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

    month_sequence = _month_sequence(config.month, config.history_months)

    minute_dir = data_root / "minute_monthly"
    minute_paths = [minute_dir / f"CL_1m_{month}.csv" for month in month_sequence]
    missing_minute = [str(path) for path in minute_paths if not path.exists()]
    if missing_minute:
        available = ", ".join(sorted(p.name for p in minute_dir.glob("CL_1m_*.csv")))
        raise FileNotFoundError(
            f"Missing minute data file(s): {missing_minute}. Available: {available or 'none'}."
        )

    minute_df = _load_minute_bars(minute_paths)

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
    )

    for _, bar in minute_df.iterrows():
        ts = bar["datetime"]
        bid_high = float(bar["high"])
        bid_low = float(bar["low"])
        close_price = float(bar["close"])

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
) -> dict | None:
    spread = config.spread
    ask_low = bid_low + spread
    ask_high = bid_high + spread
    for level in pending_levels:
        if level.get("consumed"):
            continue
        entry_price = _round_price(float(level["level"]))
        if level["level_type"] == "support":
            if ask_low <= entry_price <= ask_high:
                trade_id = trade_counter + 1
                level["consumed"] = True
                take_profit, stop_loss = _resolve_targets(
                    level_records=level_records,
                    entry_level=level,
                    side="long",
                    timestamp=timestamp,
                    config=config,
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
                trade_id = trade_counter + 1
                level["consumed"] = True
                take_profit, stop_loss = _resolve_targets(
                    level_records=level_records,
                    entry_level=level,
                    side="short",
                    timestamp=timestamp,
                    config=config,
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


def _resolve_targets(
    level_records: list[dict],
    entry_level: dict,
    side: Side,
    timestamp: pd.Timestamp,
    config: BacktestConfig,
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

    take_profit = _round_price(tp if tp is not None else default_tp)
    stop_loss = _round_price(sl if sl is not None else default_sl)
    return take_profit, stop_loss


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


def _month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    base = datetime.strptime(month, "%Y-%m")
    start = pd.Timestamp(base)
    end = pd.Timestamp(_shift_month(base, 1))
    return start, end


def _trade_window_bounds(
    month: str,
    start_day: int | None,
    end_day: int | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    month_start, month_end = _month_bounds(month)
    window_start = month_start
    window_end = month_end
    if start_day is not None:
        if start_day < 1:
            raise ValueError("trade_start_day must be at least 1.")
        window_start = month_start + pd.Timedelta(days=start_day - 1)
    if end_day is not None:
        if end_day < 1:
            raise ValueError("trade_end_day must be at least 1.")
        window_end = min(month_start + pd.Timedelta(days=end_day), month_end)
    if window_start >= window_end:
        raise ValueError("Trade window start must be before its end.")
    return window_start, window_end


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
    month_start, month_end = _month_bounds(result.config.month)
    month_minutes = result.minute_bars[
        (result.minute_bars["datetime"] >= month_start)
        & (result.minute_bars["datetime"] < month_end)
    ].copy()
    if month_minutes.empty:
        raise ValueError(f"No minute data within {result.config.month}.")
    fig = _build_trade_figure(month_minutes, result.trades, result.config.month)
    html_path = plot_dir / f"trades_{result.config.month}.html"
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


def _build_trade_figure(minute_df: pd.DataFrame, trades_df: pd.DataFrame, label_month: str) -> go.Figure:
    fig = go.Figure(
        data=[
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
            )
        ]
    )

    fig.update_layout(
        title=f"CL 1m trades for {label_month}",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

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
