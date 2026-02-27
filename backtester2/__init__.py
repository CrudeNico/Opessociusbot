from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LEVEL_DIR = Path('level_detector') / 'outputs' / 'levels'
DATA_DIR = Path('data') / 'minute_monthly'

Side = Literal['long', 'short']


def _default_output_dir() -> Path:
    return Path('backtester2') / 'outputs'


@dataclass(slots=True)
class Config:
    month: str = '2025-04'
    spread: float = 0.05
    take_profit_offset: float = 0.25
    stop_loss_offset: float = 0.25
    history_months: int = 1
    trade_start_day: int = 1
    trade_end_day: int = 30
    data_root: Path | None = None
    level_root: Path | None = None
    output_dir: Path | None = None
    level_max_distance: float = 0.50
    session_start_hour: int = 7
    session_end_hour: int = 20


@dataclass(slots=True)
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    minute_bars: pd.DataFrame
    levels: pd.DataFrame
    config: Config


def load_minute_data(months: list[str], data_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for month in months:
        csv_path = data_root / f'CL_1m_{month}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'Missing minute data {csv_path}')
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce').dt.tz_convert(None)
        for col in ('open', 'high', 'low', 'close'):
            df[col] = df[col].astype(float)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values('datetime', inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def load_levels(months: list[str], level_root: Path) -> pd.DataFrame:
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
    combined.sort_values(['time', 'level'], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def month_sequence(target_month: str, history_months: int) -> list[str]:
    base = datetime.strptime(target_month, '%Y-%m')
    months: list[str] = []
    for offset in range(history_months - 1, -1, -1):
        dt = shift_month(base, -offset)
        months.append(dt.strftime('%Y-%m'))
    for offset in range(1, 2):
        months.append(shift_month(base, offset).strftime('%Y-%m'))
    return sorted(set(months))


def shift_month(dt: datetime, delta: int) -> datetime:
    year = dt.year + (dt.month - 1 + delta) // 12
    month = (dt.month - 1 + delta) % 12 + 1
    return datetime(year, month, 1)


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    base = datetime.strptime(month, '%Y-%m')
    start = datetime(base.year, base.month, 1)
    end = shift_month(start, 1)
    return start, end


def run_backtest(cfg: Config) -> BacktestResult:
    data_root = cfg.data_root or DATA_DIR
    level_root = cfg.level_root or LEVEL_DIR
    months = month_sequence(cfg.month, cfg.history_months)
    minute_df = load_minute_data(months, data_root)
    level_df = load_levels(months, level_root)
    if minute_df.empty or level_df.empty:
        return BacktestResult(
            trades=pd.DataFrame(columns=_trade_columns()),
            equity=pd.DataFrame(columns=['timestamp', 'equity']),
            minute_bars=minute_df,
            levels=level_df,
            config=cfg,
        )
    level_records = _prepare_levels(level_df)
    trades, equity = _simulate(minute_df, level_records, cfg)
    trades_df = pd.DataFrame(trades, columns=_trade_columns())
    equity_df = pd.DataFrame(equity, columns=['timestamp', 'equity'])
    return BacktestResult(trades=trades_df, equity=equity_df, minute_bars=minute_df, levels=level_df, config=cfg)


def _prepare_levels(level_df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for idx, row in level_df.iterrows():
        strength = float(row.get('strength', 1.0) or 1.0)
        weight = max(strength, 1.0)
        records.append(
            {
                'id': idx,
                'level': float(row['level']),
                'level_type': row['level_type'],
                'time': row['time'],
                'strength': strength,
                'weight': weight,
                'last_used': None,
            }
        )
    return records


def _simulate(minute_df: pd.DataFrame, level_records: list[dict], cfg: Config):
    trades = []
    equity_curve = []
    equity = 0.0
    pending: list[dict] = []
    next_idx = 0
    records_sorted = sorted(level_records, key=lambda x: x['time'])
    trade_window_start = datetime.strptime(cfg.month + f'-{cfg.trade_start_day:02d}', '%Y-%m-%d')
    trade_window_end = datetime.strptime(cfg.month + f'-{cfg.trade_end_day:02d}', '%Y-%m-%d')
    active_trade = None
    used_today: set[int] = set()
    current_day_start: datetime | None = None
    trade_counter = 0
    for _, bar in minute_df.iterrows():
        ts = bar['datetime']
        day_start = datetime(ts.year, ts.month, ts.day)
        if current_day_start != day_start:
            current_day_start = day_start
            used_today = set()
            for rec in records_sorted:
                rec['weight'] *= 0.98
            while next_idx < len(records_sorted) and records_sorted[next_idx]['time'] < day_start:
                pending.append(records_sorted[next_idx])
                next_idx += 1
        if active_trade:
            exit_reason, exit_price = _check_exit(active_trade, bar, cfg)
            if exit_reason:
                exit_price = _round(exit_price)
                pnl = _round(active_trade['direction'] * (exit_price - active_trade['entry_price']))
                equity += pnl
                trades.append(
                    {
                        'trade_id': active_trade['trade_id'],
                        'side': active_trade['side'],
                        'level_id': active_trade['level_id'],
                        'entry_time': active_trade['entry_time'],
                        'exit_time': ts,
                        'entry_price': active_trade['entry_price'],
                        'exit_price': exit_price,
                        'take_profit': active_trade['take_profit'],
                        'stop_loss': active_trade['stop_loss'],
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                    }
                )
                equity_curve.append({'timestamp': ts, 'equity': _round(equity)})
                active_trade = None
        if not (trade_window_start <= ts <= trade_window_end):
            continue
        if active_trade:
            continue
        if not (cfg.session_start_hour <= ts.hour < cfg.session_end_hour):
            continue
        opened = _attempt_entry(pending, bar, cfg, used_today)
        if opened:
            trade_counter += 1
            opened['trade_id'] = trade_counter
            opened['entry_time'] = ts
            active_trade = opened
            used_today.add(opened['level_id'])
    if active_trade:
        # force close at last bar
        last_close = float(minute_df.iloc[-1]['close'])
        exit_price = last_close if active_trade['side'] == 'long' else last_close + cfg.spread
        pnl = _round(active_trade['direction'] * (exit_price - active_trade['entry_price']))
        equity += pnl
        trades.append(
            {
                'trade_id': active_trade['trade_id'],
                'side': active_trade['side'],
                'level_id': active_trade['level_id'],
                'entry_time': active_trade['entry_time'],
                'exit_time': minute_df.iloc[-1]['datetime'],
                'entry_price': active_trade['entry_price'],
                'exit_price': _round(exit_price),
                'take_profit': active_trade['take_profit'],
                'stop_loss': active_trade['stop_loss'],
                'exit_reason': 'end_of_data',
                'pnl': pnl,
            }
        )
        equity_curve.append({'timestamp': minute_df.iloc[-1]['datetime'], 'equity': _round(equity)})
    return trades, equity_curve


def _attempt_entry(pending_levels: list[dict], bar: pd.Series, cfg: Config, used_today: set[int]):
    bid_low = float(bar['low'])
    bid_high = float(bar['high'])
    ask_low = bid_low + cfg.spread
    ask_high = bid_high + cfg.spread
    for level in pending_levels:
        if level['weight'] < 0.5:
            continue
        level_id = level['id']
        price = float(level['level'])
        level_type = level['level_type']
        if level_id in used_today:
            continue
        if level_type == 'support':
            if ask_low <= price <= ask_high:
                tp = _find_target(levels=pending_levels, price=price, level_type='resistance', direction='above')
                sl = _find_target(levels=pending_levels, price=price, level_type='support', direction='below')
                if tp is None or sl is None:
                    continue
                if (tp - price) > cfg.level_max_distance:
                    continue
                if (price - sl) > cfg.level_max_distance:
                    continue
                level['weight'] -= 1.0
                return {
                    'side': 'long',
                    'direction': 1.0,
                    'level_id': level_id,
                    'entry_price': price,
                    'take_profit': _round(tp),
                    'stop_loss': _round(sl),
                }
        else:
            if bid_low <= price <= bid_high:
                tp = _find_target(levels=pending_levels, price=price, level_type='support', direction='below')
                sl = _find_target(levels=pending_levels, price=price, level_type='resistance', direction='above')
                if tp is None or sl is None:
                    continue
                if (price - tp) > cfg.level_max_distance:
                    continue
                if (sl - price) > cfg.level_max_distance:
                    continue
                level['weight'] -= 1.0
                return {
                    'side': 'short',
                    'direction': -1.0,
                    'level_id': level_id,
                    'entry_price': price,
                    'take_profit': _round(tp),
                    'stop_loss': _round(sl),
                }
    return None


def _find_target(levels: list[dict], price: float, level_type: str, direction: str) -> float | None:
    candidates = [lvl for lvl in levels if lvl['level_type'] == level_type and lvl['weight'] >= 0.5]
    if direction == 'above':
        higher = [lvl for lvl in candidates if lvl['level'] > price]
        if not higher:
            return None
        return min(higher, key=lambda l: l['level'])['level']
    else:
        lower = [lvl for lvl in candidates if lvl['level'] < price]
        if not lower:
            return None
        return max(lower, key=lambda l: l['level'])['level']


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
        'level_id',
        'entry_time',
        'exit_time',
        'entry_price',
        'exit_price',
        'take_profit',
        'stop_loss',
        'exit_reason',
        'pnl',
    ]


def _build_trade_figure(
    minute_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    levels_df: pd.DataFrame,
    label_month: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f'CL 1m trades for {label_month}', 'RSI (14)'),
    )

    month_start, month_end = _month_bounds(label_month)
    fig.add_trace(
        go.Candlestick(
            name='CL 1m',
            x=minute_df['datetime'],
            open=minute_df['open'],
            high=minute_df['high'],
            low=minute_df['low'],
            close=minute_df['close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            opacity=0.9,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    rsi_series = _compute_rsi(minute_df['close'], period=14)
    fig.add_trace(
        go.Scatter(
            name='RSI',
            x=minute_df['datetime'],
            y=rsi_series,
            mode='lines',
            line=dict(color='#ffd54f', width=1.5),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for level, color in ((70, '#ef5350'), (30, '#26a69a')):
        fig.add_shape(
            type='line',
            xref='x',
            yref='y2',
            x0=minute_df['datetime'].iloc[0],
            x1=minute_df['datetime'].iloc[-1],
            y0=level,
            y1=level,
            line=dict(color=color, width=1, dash='dash'),
        )

    trades = trades_df.copy()
    long_trades = trades[trades['side'] == 'long']
    short_trades = trades[trades['side'] == 'short']

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
            ),
            row=1,
            col=1,
        )

    _add_marker(long_trades, 'Long Entries', 'entry_time', 'entry_price', '#00c853', 'triangle-up')
    _add_marker(short_trades, 'Short Entries', 'entry_time', 'entry_price', '#ff8f00', 'triangle-down')

    exit_styles = {
        'take_profit': ('Take-Profit Exits', '#00e676'),
        'stop_loss': ('Stop-Loss Exits', '#ff1744'),
    }
    for reason, (label, color) in exit_styles.items():
        subset = trades[trades['exit_reason'] == reason]
        _add_marker(subset, label, 'exit_time', 'exit_price', color, 'x')

    remainder = trades[~trades['exit_reason'].isin(exit_styles.keys())]
    _add_marker(remainder, 'Other Exits', 'exit_time', 'exit_price', '#cfd8dc', 'circle-open')

    for _, trade in trades.iterrows():
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
            row=1,
            col=1,
        )

    base_traces = len(fig.data)
    level_trace_pairs: list[tuple[int, int]] = []
    if not levels_df.empty:
        month_levels = levels_df[
            (levels_df['time'] >= month_start) & (levels_df['time'] < month_end)
        ].copy()
        if not month_levels.empty:
            month_levels['date'] = month_levels['time'].dt.date
            unique_dates = sorted(month_levels['date'].unique())

            def _level_segments(df: pd.DataFrame, start_dt: datetime, end_dt: datetime):
                xs: list = []
                ys: list = []
                for _, lvl in df.iterrows():
                    xs.extend([start_dt, end_dt, None])
                    ys.extend([lvl['level'], lvl['level'], None])
                return xs, ys

            for idx, date_value in enumerate(unique_dates):
                day_levels = month_levels[month_levels['date'] == date_value]
                day_start = datetime.combine(date_value, datetime.min.time())
                day_end = day_start + pd.Timedelta(days=1)
                support = day_levels[day_levels['level_type'] == 'support']
                resistance = day_levels[day_levels['level_type'] == 'resistance']

                sup_x, sup_y = _level_segments(support, day_start, day_end)
                res_x, res_y = _level_segments(resistance, day_start, day_end)

                sup_trace = go.Scatter(
                    name=f'{date_value} Supports',
                    x=sup_x,
                    y=sup_y,
                    mode='lines',
                    line=dict(color='#00bfa5', width=1.2, dash='dot'),
                    visible=idx == 0,
                    showlegend=False,
                )
                res_trace = go.Scatter(
                    name=f'{date_value} Resistances',
                    x=res_x,
                    y=res_y,
                    mode='lines',
                    line=dict(color='#ff7043', width=1.2, dash='dot'),
                    visible=idx == 0,
                    showlegend=False,
                )
                fig.add_trace(sup_trace, row=1, col=1)
                fig.add_trace(res_trace, row=1, col=1)
                level_trace_pairs.append((len(fig.data) - 2, len(fig.data) - 1))

            if level_trace_pairs:
                total_traces = len(fig.data)
                base_visibility = [True] * base_traces
                level_count = total_traces - base_traces
                steps = []
                for idx, date_value in enumerate(unique_dates):
                    vis = base_visibility + [False] * level_count
                    sup_idx, res_idx = level_trace_pairs[idx]
                    vis[sup_idx] = True
                    vis[res_idx] = True
                    steps.append(
                        dict(
                            method='update',
                            args=[{'visible': vis}],
                            label=str(date_value),
                        )
                    )
                fig.update_layout(
                    sliders=[
                        dict(
                            active=0,
                            steps=steps,
                            currentvalue={'prefix': 'Levels date: '},
                            x=0,
                            y=0,
                            len=1.0,
                            pad={'t': 50},
                        )
                    ]
                )

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text='RSI', row=2, col=1, range=[0, 100])
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


def save_outputs(result: BacktestResult):
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
    result.trades.to_csv(trades_path, index=False)
    result.equity.to_csv(equity_path, index=False)
    if not result.minute_bars.empty and not result.trades.empty:
        fig = _build_plot(result)
        fig.write_html(plot_path, include_plotlyjs='cdn')
    return trades_path, equity_path, plot_path


def _build_plot(result: BacktestResult) -> go.Figure:
    cfg = result.config
    start, end = _month_bounds(cfg.month)
    minute = result.minute_bars[
        (result.minute_bars['datetime'] >= start) & (result.minute_bars['datetime'] < end)
    ].copy()
    if minute.empty:
        raise ValueError(f'No minute data within {cfg.month} for plotting.')
    trades = result.trades.copy()
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    levels = result.levels.copy()
    levels['time'] = pd.to_datetime(levels['time'])
    fig = _build_trade_figure(minute, trades, levels, cfg.month)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description='Backtester 2 - level-based strategy')
    parser.add_argument('--month', default='2025-04', help='Target month (YYYY-MM).')
    parser.add_argument('--spread', type=float, default=0.05)
    parser.add_argument('--history-months', type=int, default=1)
    parser.add_argument('--trade-start-day', type=int, default=1)
    parser.add_argument('--trade-end-day', type=int, default=30)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--level-root', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--session-start-hour', type=int, default=7, help='UTC hour to start taking trades (inclusive).')
    parser.add_argument('--session-end-hour', type=int, default=20, help='UTC hour to stop taking trades (exclusive).')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        month=args.month,
        spread=args.spread,
        history_months=args.history_months,
        trade_start_day=args.trade_start_day,
        trade_end_day=args.trade_end_day,
        data_root=args.data_root,
        level_root=args.level_root,
        output_dir=args.output_dir,
        session_start_hour=args.session_start_hour,
        session_end_hour=args.session_end_hour,
    )
    result = run_backtest(cfg)
    trades_path, equity_path, plot_path = save_outputs(result)
    print(f'Trades written to: {trades_path}')
    print(f'Equity curve written to: {equity_path}')
    print(f'Trade plot written to: {plot_path}')
    pnl = result.trades['pnl'].sum() if not result.trades.empty else 0.0
    print(f'Total PnL: {pnl:.4f}')
    print(f'Trades taken: {len(result.trades)}')


if __name__ == '__main__':
    main()
