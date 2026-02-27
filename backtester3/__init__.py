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
    return Path('backtester3') / 'outputs'


@dataclass(slots=True)
class Config:
    month: str = '2025-04'
    spread: float = 0.05
    history_months: int = 1
    level_history_days: int = 30
    level_cluster_distance: float = 0.40
    take_profit_distance: float = 0.80
    stop_loss_distance: float = 0.80
    initial_equity: float = 10_000.0
    trade_start_day: int = 1
    trade_end_day: int = 30
    session_start_hour: int = 7
    session_end_hour: int = 20
    min_support_resistance_gap: float = 0.50
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
    minute_df = _load_month_minutes(cfg.month, cfg.data_root or DATA_DIR)
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


def _load_month_minutes(month: str, data_root: Path) -> pd.DataFrame:
    csv_path = data_root / f'CL_1m_{month}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing minute data {csv_path}')
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce').dt.tz_convert(None)
    for col in ('open', 'high', 'low', 'close'):
        df[col] = df[col].astype(float)
    df = df[(df['datetime'] >= df['datetime'].min())]
    return df


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
    minute = minute[(minute['datetime'] >= month_start) & (minute['datetime'] < month_end)]
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
                month_end,
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
                month_end,
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
    minute = minute[(minute['datetime'] >= month_start) & (minute['datetime'] < month_end)].copy()
    if minute.empty:
        return [], [{'timestamp': month_start, 'equity': _round(cfg.initial_equity)}]
    minute['ema_fast'] = minute['close'].ewm(span=60, adjust=False).mean()
    minute['ema_slow'] = minute['close'].ewm(span=200, adjust=False).mean()
    minute['trend_diff'] = minute['ema_fast'] - minute['ema_slow']
    last_day = (month_end - timedelta(days=1)).day
    start_day = max(1, min(cfg.trade_start_day, last_day))
    end_day = max(start_day, min(cfg.trade_end_day, last_day))
    trade_window_start = datetime.strptime(f"{cfg.month}-{start_day:02d}", '%Y-%m-%d')
    trade_window_end = datetime.strptime(f"{cfg.month}-{end_day:02d}", '%Y-%m-%d') + timedelta(days=1)
    trade_window_start = max(trade_window_start, month_start)
    trade_window_end = min(trade_window_end, month_end)
    clusters_by_day: dict = {}
    for window in report.daily_windows:
        clusters = window.get('clusters', {})
        clusters_by_day[window['date']] = {
            'support': [dict(level=float(entry['level']), count=int(entry.get('count', 0) or 0)) for entry in clusters.get('support', [])],
            'resistance': [
                dict(level=float(entry['level']), count=int(entry.get('count', 0) or 0)) for entry in clusters.get('resistance', [])
            ],
        }
    trades: list[dict] = []
    equity_curve: list[dict] = [{'timestamp': month_start, 'equity': _round(cfg.initial_equity)}]
    equity = cfg.initial_equity
    active_trade: dict | None = None
    trade_id = 0

    for _, bar in minute.iterrows():
        ts = bar['datetime']
        day_date = ts.date()
        weekday_idx = ts.weekday()
        if active_trade:
            exit_reason, exit_price = _check_exit(active_trade, bar, cfg)
            if exit_reason:
                exit_price = _round(exit_price)
                pnl = _round(active_trade['direction'] * (exit_price - active_trade['entry_price']))
                equity = _round(equity + pnl)
                trades.append(
                    {
                        'trade_id': active_trade['trade_id'],
                        'side': active_trade['side'],
                        'entry_time': active_trade['entry_time'],
                        'exit_time': ts,
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'take_profit': active_trade['take_profit'],
                        'stop_loss': active_trade['stop_loss'],
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'level_type': active_trade['level_type'],
                        'cluster_size': active_trade['cluster_size'],
                    }
                )
                equity_curve.append({'timestamp': ts, 'equity': equity})
                active_trade = None
                continue
        if not (trade_window_start <= ts < trade_window_end):
            continue
        if weekday_idx in (0, 4):  # skip Mondays (0) and Fridays (4)
            continue
        if not (cfg.session_start_hour <= ts.hour < cfg.session_end_hour):
            continue
        if active_trade:
            continue
        day_clusters = clusters_by_day.get(day_date)
        if not day_clusters:
            continue
        opened = _attempt_entry(day_clusters, bar, cfg)
        if opened:
            trade_id += 1
            opened['trade_id'] = trade_id
            opened['entry_time'] = ts
            active_trade = opened
    if active_trade:
        last_close = float(minute.iloc[-1]['close'])
        exit_price = last_close if active_trade['side'] == 'long' else last_close + cfg.spread
        exit_price = _round(exit_price)
        pnl = _round(active_trade['direction'] * (exit_price - active_trade['entry_price']))
        equity = _round(equity + pnl)
        trades.append(
            {
                'trade_id': active_trade['trade_id'],
                'side': active_trade['side'],
                'entry_time': active_trade['entry_time'],
                'exit_time': minute.iloc[-1]['datetime'],
                'entry_price': active_trade['entry_price'],
                'exit_price': exit_price,
                'take_profit': active_trade['take_profit'],
                'stop_loss': active_trade['stop_loss'],
                'exit_reason': 'end_of_data',
                'pnl': pnl,
                'level_type': active_trade['level_type'],
                'cluster_size': active_trade['cluster_size'],
            }
        )
        equity_curve.append({'timestamp': minute.iloc[-1]['datetime'], 'equity': equity})
    return trades, equity_curve


def _attempt_entry(day_clusters: dict, bar: pd.Series, cfg: Config):
    support_levels = day_clusters.get('support', [])
    resistance_levels = day_clusters.get('resistance', [])
    trend_diff = bar.get('trend_diff')
    allow_long = pd.notna(trend_diff) and trend_diff > 0
    allow_short = pd.notna(trend_diff) and trend_diff < 0

    if allow_short:
        idx = _level_touch_index(support_levels, bar, cfg, 'support')
        while idx is not None:
            level = support_levels[idx]
            price = float(level['level'])
            if not _has_close_opposite(price, resistance_levels, cfg.min_support_resistance_gap):
                tp = _nearest_cluster_level(price, support_levels, resistance_levels, direction='below', exclude=level)
                sl = _nearest_cluster_level(price, support_levels, resistance_levels, direction='above', exclude=level)
                if tp is not None and sl is not None:
                    support_levels.pop(idx)
                    entry_price = _round(price)
                    return {
                        'side': 'short',
                        'direction': -1.0,
                        'entry_price': entry_price,
                        'take_profit': _round(tp),
                        'stop_loss': _round(sl),
                        'level_type': 'support',
                        'cluster_size': int(level.get('count', 0)),
                    }
            support_levels.pop(idx)
            idx = _level_touch_index(support_levels, bar, cfg, 'support')
    if allow_long:
        idx = _level_touch_index(resistance_levels, bar, cfg, 'resistance')
        while idx is not None:
            level = resistance_levels[idx]
            price = float(level['level'])
            if not _has_close_opposite(price, support_levels, cfg.min_support_resistance_gap):
                tp = _nearest_cluster_level(price, support_levels, resistance_levels, direction='above', exclude=level)
                sl = _nearest_cluster_level(price, support_levels, resistance_levels, direction='below', exclude=level)
                if tp is not None and sl is not None:
                    resistance_levels.pop(idx)
                    entry_price = _round(price)
                    return {
                        'side': 'long',
                        'direction': 1.0,
                        'entry_price': entry_price,
                        'take_profit': _round(tp),
                        'stop_loss': _round(sl),
                        'level_type': 'resistance',
                        'cluster_size': int(level.get('count', 0)),
                    }
            resistance_levels.pop(idx)
            idx = _level_touch_index(resistance_levels, bar, cfg, 'resistance')
    return None


def _has_close_opposite(price: float, other_levels: list[dict], min_gap: float) -> bool:
    for entry in other_levels:
        other_price = float(entry['level'])
        if abs(price - other_price) < min_gap:
            return True
    return False


def _nearest_cluster_level(
    price: float,
    support_levels: list[dict],
    resistance_levels: list[dict],
    direction: str,
    exclude: dict | None = None,
) -> float | None:
    candidates: list[float] = []
    for entry in support_levels + resistance_levels:
        if exclude is not None and entry is exclude:
            continue
        lvl = float(entry['level'])
        if direction == 'above' and lvl > price:
            candidates.append(lvl)
        elif direction == 'below' and lvl < price:
            candidates.append(lvl)
    if not candidates:
        return None
    return min(candidates) if direction == 'above' else max(candidates)


def _level_touch_index(levels: list[dict], bar: pd.Series, cfg: Config, level_type: str) -> int | None:
    if not levels:
        return None
    bid_low = float(bar['low'])
    bid_high = float(bar['high'])
    ask_low = bid_low + cfg.spread
    ask_high = bid_high + cfg.spread
    if level_type == 'support':
        for idx, entry in enumerate(levels):
            price = float(entry['level'])
            if ask_low <= price <= ask_high:
                return idx
    else:
        for idx, entry in enumerate(levels):
            price = float(entry['level'])
            if bid_low <= price <= bid_high:
                return idx
    return None


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
