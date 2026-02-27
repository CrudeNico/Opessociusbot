from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

DATA_DIR = Path('data') / 'minute_monthly'
LEVEL_DIR = Path('level_detector') / 'outputs' / 'levels'
Side = Literal['long', 'short']


def _default_output_dir() -> Path:
    return Path('backtester5') / 'outputs'


@dataclass(slots=True)
class Config:
    month: str = '2025-04'
    spread: float = 0.05
    history_months: int = 1
    level_history_days: int = 30
    level_cluster_distance: float = 0.40
    take_profit_distance: float = 0.20
    stop_loss_distance: float = 0.80
    initial_equity: float = 10_000.0
    trade_start_day: int = 1
    trade_end_day: int = 30
    session_start_hour: int = 7
    session_end_hour: int = 20
    min_support_resistance_gap: float = 0.50
    extra_data_months: int = 1
    data_root: Path | None = None
    level_root: Path | None = None
    output_dir: Path | None = None


@dataclass(slots=True)
class LevelReport:
    minute_bars: pd.DataFrame
    levels: pd.DataFrame
    daily_windows: list[dict]
    clusters: pd.DataFrame
    config: Config


@dataclass(slots=True)
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    config: Config
    report: LevelReport


def run_level_report(cfg: Config) -> LevelReport:
    month_start, month_end = _month_bounds(cfg.month)
    months_to_load = 1 + max(cfg.extra_data_months, 0)
    minute_df = _load_month_minutes(cfg.month, cfg.data_root or DATA_DIR, months_to_load=months_to_load)
    if minute_df.empty:
        raise ValueError(f'No minute data found for {cfg.month}.')
    level_df = _load_levels_window(month_start, month_end, cfg.level_history_days, cfg.level_root or LEVEL_DIR)
    if level_df.empty:
        raise ValueError('No level data found for the requested window.')
    daily_windows = _rolling_daily_levels(
        level_df,
        month_start,
        month_end,
        cfg.level_history_days,
        cfg.level_cluster_distance,
    )
    cluster_rows: list[dict] = []
    for window in daily_windows:
        clusters = window.get('clusters', {})
        for level_type, entries in clusters.items():
            for entry in entries:
                cluster_rows.append(
                    {
                        'date': window['date'],
                        'level_type': level_type,
                        'level': entry['level'],
                        'count': entry['count'],
                    }
                )
    cluster_df = (
        pd.DataFrame(cluster_rows, columns=['date', 'level_type', 'level', 'count'])
        if cluster_rows
        else pd.DataFrame(columns=['date', 'level_type', 'level', 'count'])
    )
    return LevelReport(
        minute_bars=minute_df,
        levels=level_df,
        daily_windows=daily_windows,
        clusters=cluster_df,
        config=cfg,
    )


def run_strategy(report: LevelReport) -> BacktestResult:
    trades, equity = _simulate_trades(report)
    trades_df = pd.DataFrame(trades, columns=_trade_columns()) if trades else pd.DataFrame(columns=_trade_columns())
    equity_df = (
        pd.DataFrame(equity, columns=['timestamp', 'equity']) if equity else pd.DataFrame(columns=['timestamp', 'equity'])
    )
    return BacktestResult(trades=trades_df, equity=equity_df, config=report.config, report=report)


def save_report(report: LevelReport):
    base = report.config.output_dir or _default_output_dir()
    plots_dir = base / 'plots'
    levels_dir = base / 'levels'
    plots_dir.mkdir(parents=True, exist_ok=True)
    levels_dir.mkdir(parents=True, exist_ok=True)
    levels_path = levels_dir / f'levels_{report.config.month}.csv'
    report.levels.to_csv(levels_path, index=False)
    cluster_path = levels_dir / f'levels_clustered_{report.config.month}.csv'
    report.clusters.to_csv(cluster_path, index=False)
    fig = _build_levels_figure(report)
    plot_path = plots_dir / f'levels_{report.config.month}.html'
    fig.write_html(plot_path, include_plotlyjs='cdn')
    return levels_path, cluster_path, plot_path


def save_strategy(result: BacktestResult):
    base = result.config.output_dir or _default_output_dir()
    trades_dir = base / 'trades'
    equity_dir = base / 'equity'
    plots_dir = base / 'plots'
    trades_dir.mkdir(parents=True, exist_ok=True)
    equity_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    trades_path = trades_dir / f'trades_{result.config.month}.csv'
    equity_path = equity_dir / f'equity_{result.config.month}.csv'
    plot_path = plots_dir / f'trades_{result.config.month}.html'
    equity_plot_path = plots_dir / f'equity_{result.config.month}.html'
    result.trades.to_csv(trades_path, index=False)
    result.equity.to_csv(equity_path, index=False)
    fig = _build_levels_figure(result.report, trades=result.trades)
    fig.write_html(plot_path, include_plotlyjs='cdn')
    eq_fig = _build_equity_figure(result)
    eq_fig.write_html(equity_plot_path, include_plotlyjs='cdn')
    return trades_path, equity_path, plot_path, equity_plot_path


def _load_month_minutes(month: str, data_root: Path, months_to_load: int = 1) -> pd.DataFrame:
    base_dt = datetime.strptime(month, '%Y-%m')
    frames: list[pd.DataFrame] = []
    for offset in range(max(months_to_load, 1)):
        target_dt = _shift_month(base_dt, offset)
        target_month = target_dt.strftime('%Y-%m')
        csv_path = data_root / f'CL_1m_{target_month}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce').dt.tz_convert(None)
        for col in ('open', 'high', 'low', 'close'):
            df[col] = df[col].astype(float)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f'No minute data found for {month} (looked ahead {months_to_load} months).')
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values('datetime', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _load_levels_window(month_start: datetime, month_end: datetime, lookback_days: int, level_root: Path) -> pd.DataFrame:
    window_start = month_start - timedelta(days=lookback_days)
    months = _months_between(window_start, month_end)
    frames: list[pd.DataFrame] = []
    for month in months:
        csv_path = level_root / f'levels_{month}.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.tz_convert(None)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['row', 'level', 'strength', 'time', 'level_type'])
    combined = pd.concat(frames, ignore_index=True)
    mask = (combined['time'] >= window_start) & (combined['time'] < month_end)
    combined = combined.loc[mask].copy()
    combined.sort_values(['time', 'level'], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _rolling_daily_levels(
    levels: pd.DataFrame,
    month_start: datetime,
    month_end: datetime,
    lookback_days: int,
    cluster_distance: float,
) -> list[dict]:
    days = pd.date_range(month_start, month_end - timedelta(days=1), freq='D')
    windows: list[dict] = []
    for day in days:
        window_start = day - timedelta(days=lookback_days)
        window_end = day
        mask = (levels['time'] >= window_start) & (levels['time'] < window_end)
        subset = levels.loc[mask].copy()
        clusters = _cluster_levels(subset, cluster_distance)
        windows.append({'date': day.to_pydatetime().date(), 'levels': subset, 'clusters': clusters})
    return windows


def _build_levels_figure(report: LevelReport, trades: pd.DataFrame | None = None) -> go.Figure:
    cfg = report.config
    month_start, month_end = _month_bounds(cfg.month)
    minute = report.minute_bars.copy()
    extended_end = _shift_month(month_end, cfg.extra_data_months) if cfg.extra_data_months > 0 else month_end
    minute = minute[(minute['datetime'] >= month_start) & (minute['datetime'] < extended_end)]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            name='CL 1m',
            x=minute['datetime'],
            open=minute['open'],
            high=minute['high'],
            low=minute['low'],
            close=minute['close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            opacity=0.9,
            showlegend=False,
        )
    )
    if trades is not None and not trades.empty:
        trades_plot = trades.copy()
        trades_plot['entry_time'] = pd.to_datetime(trades_plot['entry_time'])
        trades_plot['exit_time'] = pd.to_datetime(trades_plot['exit_time'])
        long_trades = trades_plot[trades_plot['side'] == 'long']
        short_trades = trades_plot[trades_plot['side'] == 'short']

        def _add_marker(df: pd.DataFrame, label: str, time_col: str, price_col: str, color: str, symbol: str):
            if df.empty:
                return
            fig.add_trace(
                go.Scatter(
                    name=label,
                    x=df[time_col],
                    y=df[price_col],
                    mode='markers',
                    marker=dict(color=color, size=8, symbol=symbol, line=dict(color='black', width=0.5)),
                    customdata=df[['trade_id', 'pnl']],
                    hovertemplate=(
                        f'{label}<br>'
                        'Trade %{customdata[0]}<br>'
                        'Price %{y:.4f}<br>'
                        'Time %{x|%Y-%m-%d %H:%M}<br>'
                        'PnL %{customdata[1]:.4f}<extra></extra>'
                    ),
                )
            )

        _add_marker(long_trades, 'Long Entries', 'entry_time', 'entry_price', '#00c853', 'triangle-up')
        _add_marker(short_trades, 'Short Entries', 'entry_time', 'entry_price', '#ff8f00', 'triangle-down')

        exit_styles = {
            'take_profit': ('Take-Profit Exits', '#00e676'),
            'stop_loss': ('Stop-Loss Exits', '#ff1744'),
        }
        for reason, (label, color) in exit_styles.items():
            subset = trades_plot[trades_plot['exit_reason'] == reason]
            _add_marker(subset, label, 'exit_time', 'exit_price', color, 'x')

        remainder = trades_plot[~trades_plot['exit_reason'].isin(exit_styles.keys())]
        _add_marker(remainder, 'Other Exits', 'exit_time', 'exit_price', '#cfd8dc', 'circle-open')

        for _, trade in trades_plot.iterrows():
            pnl_color = '#00c853' if trade['pnl'] > 0 else '#ef5350' if trade['pnl'] < 0 else '#cfd8dc'
            fig.add_shape(
                type='line',
                xref='x',
                yref='y',
                x0=trade['entry_time'],
                x1=trade['exit_time'],
                y0=trade['entry_price'],
                y1=trade['exit_price'],
                line=dict(color=pnl_color, width=1.5),
                opacity=0.6,
            )

    base_traces = len(fig.data)
    day_trace_indices: list[list[int]] = []
    for idx, window in enumerate(report.daily_windows):
        cluster_info = window.get('clusters', {})
        sup_clusters = cluster_info.get('support', [])
        res_clusters = cluster_info.get('resistance', [])
        traces_for_day: list[int] = []
        if sup_clusters:
            sup_cluster_trace = _cluster_trace(
                sup_clusters,
                month_start,
                extended_end,
                f"{window['date']} Support Clusters",
                '#ffffff',
                idx == 0,
            )
            fig.add_trace(sup_cluster_trace)
            traces_for_day.append(len(fig.data) - 1)
        if res_clusters:
            res_cluster_trace = _cluster_trace(
                res_clusters,
                month_start,
                extended_end,
                f"{window['date']} Resistance Clusters",
                '#fdd835',
                idx == 0,
            )
            fig.add_trace(res_cluster_trace)
            traces_for_day.append(len(fig.data) - 1)
        day_trace_indices.append(traces_for_day)

    total_traces = len(fig.data)
    level_count = total_traces - base_traces
    base_visibility = [True] * base_traces + [False] * level_count
    steps = []
    for idx, window in enumerate(report.daily_windows):
        vis = base_visibility.copy()
        for trace_idx in day_trace_indices[idx]:
            vis[trace_idx] = True
        steps.append(
            dict(
                method='update',
                args=[{'visible': vis}],
                label=str(window['date']),
            )
        )
    if steps:
        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    currentvalue={'prefix': 'Show levels for: '},
                    pad={'t': 50},
                    steps=steps,
                )
            ]
        )
    fig.update_layout(
        title=f'CL Minute Bars with 30-day Level Lookback ({cfg.month})',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
    )
    return fig


def _simulate_trades(report: LevelReport):
    cfg = report.config
    month_start, month_end = _month_bounds(cfg.month)
    minute = report.minute_bars.copy()
    extended_end = _shift_month(month_end, cfg.extra_data_months) if cfg.extra_data_months > 0 else month_end
    minute = minute[(minute['datetime'] >= month_start) & (minute['datetime'] < extended_end)].copy()
    if minute.empty:
        return [], [{'timestamp': month_start, 'equity': _round(cfg.initial_equity)}]
    last_day = (month_end - timedelta(days=1)).day
    start_day = max(1, min(cfg.trade_start_day, last_day))
    end_day = max(start_day, min(cfg.trade_end_day, last_day))
    trade_window_start = datetime.strptime(f"{cfg.month}-{start_day:02d}", '%Y-%m-%d')
    trade_window_end = datetime.strptime(f"{cfg.month}-{end_day:02d}", '%Y-%m-%d') + timedelta(days=1)
    trade_window_start = max(trade_window_start, month_start)
    trade_window_end = min(trade_window_end, month_end)
    level_plan = _prepare_daily_extremes(report.daily_windows, cfg)

    trades: list[dict] = []
    equity_curve: list[dict] = [{'timestamp': month_start, 'equity': _round(cfg.initial_equity)}]
    equity = cfg.initial_equity
    active_trades: list[dict] = []
    trade_id = 0

    for _, bar in minute.iterrows():
        ts = bar['datetime']
        day_date = ts.date()
        weekday_idx = ts.weekday()

        # Exit handling
        still_open: list[dict] = []
        for trade in active_trades:
            exit_reason, exit_price = _check_exit_extreme(trade, bar, cfg)
            if exit_reason:
                exit_price = _round(exit_price)
                pnl = _round(trade['direction'] * (exit_price - trade['entry_price']))
                equity = _round(equity + pnl)
                trades.append(
                    {
                        'trade_id': trade['trade_id'],
                        'side': trade['side'],
                        'entry_time': trade['entry_time'],
                        'exit_time': ts,
                        'entry_price': trade['entry_price'],
                        'exit_price': exit_price,
                        'take_profit': trade['take_profit'],
                        'stop_loss': trade['stop_loss'],
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'level_type': trade['level_type'],
                        'cluster_size': trade['cluster_size'],
                        'level_order': trade['level_order'],
                    }
                )
                equity_curve.append({'timestamp': ts, 'equity': equity})
            else:
                still_open.append(trade)
        active_trades = still_open

        if not (trade_window_start <= ts < trade_window_end):
            continue
        if weekday_idx in (0, 4):  # skip Mondays (0) and Fridays (4)
            continue
        if not (cfg.session_start_hour <= ts.hour < cfg.session_end_hour):
            continue

        day_levels = level_plan.get(day_date)
        if not day_levels:
            continue

        # Support entries (longs)
        for level in day_levels['support']:
            if level['used']:
                continue
            if _support_touched(level['price'], bar, cfg):
                trade_id += 1
                entry_price = _round(level['price'])
                take_profit = _round(level['tp'])
                trade = {
                    'trade_id': trade_id,
                    'side': 'long',
                    'direction': 1.0,
                    'entry_time': ts,
                    'entry_price': entry_price,
                    'take_profit': take_profit,
                    'stop_loss': None,
                    'level_type': 'support',
                    'cluster_size': 0,
                    'level_order': level['order'],
                    'level_day': day_date,
                    'break_price': level.get('break_price'),
                    'breakeven_active': False,
                }
                active_trades.append(trade)
                level['used'] = True
                if level['order'] == 3:
                    _apply_break_even(active_trades, 'long', day_date)

        # Resistance entries (shorts)
        for level in day_levels['resistance']:
            if level['used']:
                continue
            if _resistance_touched(level['price'], bar, cfg):
                trade_id += 1
                entry_price = _round(level['price'])
                take_profit = _round(level['tp'])
                trade = {
                    'trade_id': trade_id,
                    'side': 'short',
                    'direction': -1.0,
                    'entry_time': ts,
                    'entry_price': entry_price,
                    'take_profit': take_profit,
                    'stop_loss': None,
                    'level_type': 'resistance',
                    'cluster_size': 0,
                    'level_order': level['order'],
                    'level_day': day_date,
                    'break_price': None,
                    'breakeven_active': False,
                }
                active_trades.append(trade)
                level['used'] = True
                if level['order'] == 3:
                    _apply_break_even(active_trades, 'short', day_date)

        _update_support_breakevens(active_trades, bar, cfg)

    if active_trades:
        last_close = float(minute.iloc[-1]['close'])
        last_ts = minute.iloc[-1]['datetime']
        for trade in active_trades:
            if trade['side'] == 'long':
                exit_price = _round(last_close)
            else:
                exit_price = _round(last_close + cfg.spread)
            pnl = _round(trade['direction'] * (exit_price - trade['entry_price']))
            equity = _round(equity + pnl)
            trades.append(
                {
                    'trade_id': trade['trade_id'],
                    'side': trade['side'],
                    'entry_time': trade['entry_time'],
                    'exit_time': last_ts,
                    'entry_price': trade['entry_price'],
                    'exit_price': exit_price,
                    'take_profit': trade['take_profit'],
                    'stop_loss': trade['stop_loss'],
                    'exit_reason': 'end_of_data',
                    'pnl': pnl,
                    'level_type': trade['level_type'],
                    'cluster_size': trade['cluster_size'],
                    'level_order': trade['level_order'],
                }
            )
        equity_curve.append({'timestamp': last_ts, 'equity': equity})

    return trades, equity_curve


def _prepare_daily_extremes(daily_windows: list[dict], cfg: Config) -> dict:
    plan: dict = {}
    for window in daily_windows:
        date = window['date']
        clusters = window.get('clusters', {})
        supports = _select_support_levels(clusters.get('support', []), cfg)
        resistances = _select_resistance_levels(clusters.get('resistance', []), cfg)
        plan[date] = {'support': supports, 'resistance': resistances}
    return plan


def _select_support_levels(entries: list[dict], cfg: Config) -> list[dict]:
    prices = sorted({_round(float(entry['level'])) for entry in entries})
    if not prices:
        return []
    selected = prices[:3]
    if not selected:
        return []
    entry_order = list(reversed(selected))
    first_price = entry_order[0]
    break_price = entry_order[-1] if len(entry_order) >= 3 else None
    levels: list[dict] = []
    for idx, price in enumerate(entry_order, start=1):
        tp = _round(price + cfg.take_profit_distance) if idx == 1 else _round(first_price)
        levels.append({'price': price, 'tp': tp, 'order': idx, 'used': False, 'break_price': break_price})
    return levels


def _select_resistance_levels(entries: list[dict], cfg: Config) -> list[dict]:
    prices = sorted({_round(float(entry['level'])) for entry in entries})
    if not prices:
        return []
    selected = prices[-3:]
    if not selected:
        return []
    entry_order = selected
    first_price = entry_order[0]
    levels: list[dict] = []
    for idx, price in enumerate(entry_order, start=1):
        tp = _round(price - cfg.take_profit_distance) if idx == 1 else _round(first_price)
        levels.append({'price': price, 'tp': tp, 'order': idx, 'used': False})
    return levels

def _apply_break_even(active_trades: list[dict], side: str, day_date):
    for trade in active_trades:
        if trade['side'] != side:
            continue
        if trade['level_order'] not in (1, 2):
            continue
        if trade.get('level_day') != day_date:
            continue
        _activate_support_breakeven(trade)


def _update_support_breakevens(active_trades: list[dict], bar: pd.Series, cfg: Config):
    for trade in active_trades:
        if trade['side'] != 'long':
            continue
        if trade['level_order'] >= 3:
            continue
        if trade.get('breakeven_active'):
            continue
        break_price = trade.get('break_price')
        if break_price is None:
            continue
        if bar['datetime'] <= trade['entry_time']:
            continue
        if _support_touched(break_price, bar, cfg):
            _activate_support_breakeven(trade)


def _activate_support_breakeven(trade: dict):
    trade['take_profit'] = trade['entry_price']
    trade['breakeven_active'] = True


def _support_touched(price: float, bar: pd.Series, cfg: Config) -> bool:
    bid_low = float(bar['low'])
    bid_high = float(bar['high'])
    ask_low = bid_low + cfg.spread
    ask_high = bid_high + cfg.spread
    return ask_low <= price <= ask_high


def _resistance_touched(price: float, bar: pd.Series, cfg: Config) -> bool:
    bid_low = float(bar['low'])
    bid_high = float(bar['high'])
    return bid_low <= price <= bid_high


def _check_exit_extreme(trade: dict, bar: pd.Series, cfg: Config):
    bid_low = float(bar['low'])
    bid_high = float(bar['high'])
    ask_low = bid_low + cfg.spread
    ask_high = bid_high + cfg.spread
    if trade['side'] == 'long':
        if bid_high >= trade['take_profit']:
            return 'take_profit', trade['take_profit']
    else:
        if ask_low <= trade['take_profit']:
            return 'take_profit', trade['take_profit']
    return None, None


def _check_exit(trade: dict, bar: pd.Series, cfg: Config):
    bid_low = float(bar['low'])
    bid_high = float(bar['high'])
    ask_low = bid_low + cfg.spread
    ask_high = bid_high + cfg.spread
    if trade['side'] == 'long':
        if bid_low <= trade['stop_loss']:
            return 'stop_loss', trade['stop_loss']
        if bid_high >= trade['take_profit']:
            return 'take_profit', trade['take_profit']
    else:
        if ask_high >= trade['stop_loss']:
            return 'stop_loss', trade['stop_loss']
        if ask_low <= trade['take_profit']:
            return 'take_profit', trade['take_profit']
    return None, None


def _round(value: float, decimals: int = 4) -> float:
    return round(float(value), decimals)


def _trade_columns():
    return [
        'trade_id',
        'side',
        'entry_time',
        'exit_time',
        'entry_price',
        'exit_price',
        'take_profit',
        'stop_loss',
        'exit_reason',
        'pnl',
        'level_type',
        'cluster_size',
        'level_order',
    ]


def _build_equity_figure(result: BacktestResult) -> go.Figure:
    equity_df = result.equity.copy()
    if equity_df.empty:
        equity_df = pd.DataFrame(
            [{'timestamp': datetime.strptime(result.config.month + '-01', '%Y-%m-%d'), 'equity': result.config.initial_equity}]
        )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name='Equity',
            x=pd.to_datetime(equity_df['timestamp']),
            y=equity_df['equity'],
            mode='lines+markers',
            line=dict(color='#26a69a', width=2),
            marker=dict(size=4, color='#80cbc4'),
        )
    )
    fig.update_layout(
        title=f'Backtester 3 Equity Curve ({result.config.month})',
        xaxis_title='Time',
        yaxis_title='Equity',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
    )
    return fig


def _level_trace(df: pd.DataFrame, x0: datetime, x1: datetime, name: str, color: str, visible: bool) -> go.Scatter:
    if df.empty:
        return go.Scatter(name=name, x=[], y=[], mode='lines', line=dict(color=color, width=1, dash='dot'), visible=visible, showlegend=False)
    xs = []
    ys = []
    for _, row in df.iterrows():
        xs.extend([x0, x1, None])
        ys.extend([row['level'], row['level'], None])
    return go.Scatter(
        name=name,
        x=xs,
        y=ys,
        mode='lines',
        line=dict(color=color, width=1.2, dash='dot'),
        visible=visible,
        showlegend=False,
    )


def _cluster_trace(entries: list[dict], x0: datetime, x1: datetime, name: str, color: str, visible: bool) -> go.Scatter:
    if not entries:
        return go.Scatter(
            name=name,
            x=[],
            y=[],
            mode='lines',
            line=dict(color=color, width=2.5),
            visible=visible,
            showlegend=False,
        )
    xs: list = []
    ys: list = []
    for entry in entries:
        xs.extend([x0, x1, None])
        ys.extend([entry['level'], entry['level'], None])
    return go.Scatter(
        name=name,
        x=xs,
        y=ys,
        mode='lines',
        line=dict(color=color, width=2.5),
        visible=visible,
        showlegend=False,
    )


def _cluster_levels(level_df: pd.DataFrame, max_distance: float) -> dict:
    clusters = {'support': [], 'resistance': []}
    if level_df.empty:
        return clusters
    for level_type in clusters.keys():
        values = sorted(level_df[level_df['level_type'] == level_type]['level'].astype(float).tolist())
        if not values:
            continue
        current: list[float] = [values[0]]
        cluster_min = values[0]
        cluster_max = values[0]
        for value in values[1:]:
            proposed_min = min(cluster_min, value)
            proposed_max = max(cluster_max, value)
            if proposed_max - proposed_min <= max_distance:
                current.append(value)
                cluster_min = proposed_min
                cluster_max = proposed_max
            else:
                if len(current) >= 2:
                    clusters[level_type].append({'level': sum(current) / len(current), 'count': len(current)})
                current = [value]
                cluster_min = value
                cluster_max = value
        if len(current) >= 2:
            clusters[level_type].append({'level': sum(current) / len(current), 'count': len(current)})
    return clusters


def _months_between(start: datetime, end: datetime) -> list[str]:
    months: list[str] = []
    cursor = datetime(start.year, start.month, 1)
    while cursor < end:
        months.append(cursor.strftime('%Y-%m'))
        cursor = _shift_month(cursor, 1)
    return sorted(set(months))


def _shift_month(dt: datetime, delta: int) -> datetime:
    year = dt.year + (dt.month - 1 + delta) // 12
    month = (dt.month - 1 + delta) % 12 + 1
    return datetime(year, month, 1)


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    base = datetime.strptime(month, '%Y-%m')
    start = datetime(base.year, base.month, 1)
    end = _shift_month(start, 1)
    return start, end


def parse_args():
    parser = argparse.ArgumentParser(description='Backtester 3 - cluster-based strategy')
    parser.add_argument('--month', default='2025-04', help='Target month (YYYY-MM).')
    parser.add_argument('--level-history-days', type=int, default=30, help='Rolling lookback window for levels.')
    parser.add_argument('--level-cluster-distance', type=float, default=0.40, help='Maximum distance to merge nearby levels.')
    parser.add_argument('--take-profit-distance', type=float, default=0.80, help='Take-profit distance in price units.')
    parser.add_argument('--stop-loss-distance', type=float, default=0.80, help='Stop-loss distance in price units.')
    parser.add_argument('--initial-equity', type=float, default=10_000.0, help='Starting account balance.')
    parser.add_argument('--min-support-resistance-gap', type=float, default=0.50, help='Minimum distance between support/resistance clusters to allow trades.')
    parser.add_argument('--trade-start-day', type=int, default=1, help='First calendar day to allow trades.')
    parser.add_argument('--trade-end-day', type=int, default=30, help='Last calendar day (inclusive) to allow trades.')
    parser.add_argument('--session-start-hour', type=int, default=7, help='UTC hour to start taking trades (inclusive).')
    parser.add_argument('--session-end-hour', type=int, default=20, help='UTC hour to stop taking trades (exclusive).')
    parser.add_argument('--extra-data-months', type=int, default=1, help='Additional future months of minute data to load for managing open trades.')
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--level-root', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        month=args.month,
        level_history_days=args.level_history_days,
        level_cluster_distance=args.level_cluster_distance,
        take_profit_distance=args.take_profit_distance,
        stop_loss_distance=args.stop_loss_distance,
        initial_equity=args.initial_equity,
        min_support_resistance_gap=args.min_support_resistance_gap,
        trade_start_day=args.trade_start_day,
        trade_end_day=args.trade_end_day,
        session_start_hour=args.session_start_hour,
        session_end_hour=args.session_end_hour,
        extra_data_months=args.extra_data_months,
        data_root=args.data_root,
        level_root=args.level_root,
        output_dir=args.output_dir,
    )
    report = run_level_report(cfg)
    result = run_strategy(report)
    levels_path, cluster_path, level_plot = save_report(report)
    trades_path, equity_path, trade_plot, equity_plot = save_strategy(result)
    print(f'Levels snapshot written to: {levels_path}')
    print(f'Clustered levels written to: {cluster_path}')
    print(f'Level plot written to: {level_plot}')
    print(f'Trades written to: {trades_path}')
    print(f'Equity curve written to: {equity_path}')
    print(f'Trade plot written to: {trade_plot}')
    print(f'Equity plot written to: {equity_plot}')
    total_pnl = result.trades['pnl'].sum() if not result.trades.empty else 0.0
    print(f'Total unique levels loaded: {len(report.levels)}')
    print(f'Trades taken: {len(result.trades)}')
    print(f'Total PnL: {total_pnl:.4f}')


if __name__ == '__main__':
    main()
