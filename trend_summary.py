from pathlib import Path
import pandas as pd
import numpy as np

base = Path('backtester3') / 'outputs'
trade_dir = base / 'trades'
data_dir = Path('data') / 'minute_monthly'
trade_paths = sorted(p for p in trade_dir.glob('trades_*.csv'))
frames = []
for path in trade_paths:
    df = pd.read_csv(path, parse_dates=['entry_time', 'exit_time'])
    df['source_file'] = path.name
    frames.append(df)
trades = pd.concat(frames, ignore_index=True)
trades['month'] = trades['entry_time'].dt.to_period('M').astype(str)
trades['weekday'] = trades['entry_time'].dt.day_name()
trades = trades[(trades['month'] != '2024-08') & (~trades['weekday'].isin(['Monday','Friday']))]
minute_cache = {}
trend_class = []
for idx, trade in trades.iterrows():
    month = trade['month']
    if month not in minute_cache:
        csv_path = data_dir / f'CL_1m_{month}.csv'
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce').dt.tz_convert(None)
        for col in ['open','high','low','close']:
            df[col] = df[col].astype(float)
        df.sort_values('datetime', inplace=True)
        df['ema_fast'] = df['close'].ewm(span=60, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=200, adjust=False).mean()
        df['trend_diff'] = df['ema_fast'] - df['ema_slow']
        minute_cache[month] = df
    minute_df = minute_cache[month]
    row = minute_df.loc[minute_df['datetime'] <= trade['entry_time']].tail(1)
    if row.empty:
        trend_class.append('Unknown')
    else:
        diff = row['trend_diff'].iloc[0]
        if diff > 0:
            trend_class.append('Uptrend')
        elif diff < 0:
            trend_class.append('Downtrend')
        else:
            trend_class.append('Flat')
trades['trend_class'] = trend_class
summary = trades.groupby(['trend_class','side']).agg(count=('pnl','count'), total_pnl=('pnl','sum')).reset_index()
print(summary)
