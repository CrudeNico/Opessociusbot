from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from backtester3 import Config, run_level_report, run_strategy

TP_SL_VALUES = [0.20, 0.40, 0.60, 0.80]
MIN_GAP_VALUES = [0.10, 0.30, 0.50]
DEFAULT_MONTHS = ['2025-02', '2025-03', '2025-04', '2025-05']


@dataclass(slots=True)
class ScenarioParams:
    take_profit: float
    stop_loss: float
    ratio: float
    min_gap: float


@dataclass(slots=True)
class ScenarioResult:
    params: ScenarioParams
    total_pnl: float
    max_drawdown: float
    sharpe: float
    total_trades: int
    months: list[str]


def generate_param_grid(tp_values: List[float], sl_values: List[float], gap_values: List[float]) -> list[ScenarioParams]:
    combos: list[ScenarioParams] = []
    for tp, sl in itertools.product(tp_values, sl_values):
        ratio = tp / sl
        if 0.5 <= ratio <= 2.0:
            for gap in gap_values:
                combos.append(ScenarioParams(take_profit=tp, stop_loss=sl, ratio=ratio, min_gap=gap))
    return combos


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_progress(current: int, total: int, start_time: float):
    elapsed = time.time() - start_time
    eta = (elapsed / current * (total - current)) if current else 0.0
    width = 40
    filled = int(width * current / total)
    bar = '#' * filled + '-' * (width - filled)
    percent = current / total * 100
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} ({percent:5.1f}%) Elapsed {format_duration(elapsed)} ETA {format_duration(eta)}"
    )
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')


def compute_metrics(equity: pd.DataFrame) -> tuple[float, float, float]:
    if equity.empty:
        return 0.0, 0.0, 0.0
    equity = equity.sort_values('timestamp').reset_index(drop=True)
    eq = equity['equity'].astype(float)
    total_pnl = eq.iloc[-1] - eq.iloc[0]
    running_max = eq.cummax()
    drawdowns = eq - running_max
    max_drawdown = drawdowns.min()
    changes = eq.diff().dropna()
    if not changes.empty and changes.std() > 0:
        sharpe = (changes.mean() / changes.std()) * (len(changes) ** 0.5)
    else:
        sharpe = 0.0
    return total_pnl, abs(max_drawdown), sharpe


def run_scenario(params: ScenarioParams, months: list[str], args) -> tuple[ScenarioResult, pd.DataFrame]:
    equity_segments: list[pd.DataFrame] = []
    base_equity = args.initial_equity
    total_trades = 0
    for month in months:
        cfg = Config(
            month=month,
            take_profit_distance=params.take_profit,
            stop_loss_distance=params.stop_loss,
            min_support_resistance_gap=params.min_gap,
            session_start_hour=args.session_start_hour,
            session_end_hour=args.session_end_hour,
            trade_start_day=args.trade_start_day,
            trade_end_day=args.trade_end_day,
            initial_equity=args.initial_equity,
            level_history_days=args.level_history_days,
            level_cluster_distance=args.level_cluster_distance,
        )
        report = run_level_report(cfg)
        result = run_strategy(report)
        total_trades += len(result.trades)
        month_equity = result.equity.copy()
        if month_equity.empty:
            month_equity = pd.DataFrame([
                {
                    'timestamp': datetime.strptime(f"{month}-01", '%Y-%m-%d'),
                    'equity': args.initial_equity,
                }
            ])
        month_equity = month_equity.sort_values('timestamp').reset_index(drop=True)
        month_equity['equity'] = month_equity['equity'].astype(float)
        month_equity['equity'] = month_equity['equity'] - args.initial_equity + base_equity
        base_equity = month_equity['equity'].iloc[-1]
        equity_segments.append(month_equity)
    combined_equity = pd.concat(equity_segments, ignore_index=True)
    total_pnl, max_drawdown, sharpe = compute_metrics(combined_equity)
    scenario_result = ScenarioResult(
        params=params,
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        total_trades=total_trades,
        months=months,
    )
    return scenario_result, combined_equity


def save_results(summary: list[ScenarioResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in summary:
        rows.append(
            {
                'take_profit': item.params.take_profit,
                'stop_loss': item.params.stop_loss,
                'ratio': item.params.ratio,
                'min_gap': item.params.min_gap,
                'total_pnl': item.total_pnl,
                'max_drawdown': item.max_drawdown,
                'sharpe': item.sharpe,
                'total_trades': item.total_trades,
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values('total_pnl', ascending=False, inplace=True)
    summary_path = output_dir / 'optimizer_summary.csv'
    df.to_csv(summary_path, index=False)
    return summary_path, df


def main():
    parser = argparse.ArgumentParser(description='Backtester 3 Optimizer (TP/SL & gap sweep)')
    parser.add_argument('--months', nargs='+', default=DEFAULT_MONTHS, help='List of trading months (YYYY-MM).')
    parser.add_argument('--tp-values', nargs='+', type=float, default=TP_SL_VALUES, help='Take-profit distances to test.')
    parser.add_argument('--sl-values', nargs='+', type=float, default=TP_SL_VALUES, help='Stop-loss distances to test.')
    parser.add_argument('--gap-values', nargs='+', type=float, default=MIN_GAP_VALUES, help='Min support/resistance gaps to test.')
    parser.add_argument('--level-history-days', type=int, default=30)
    parser.add_argument('--level-cluster-distance', type=float, default=0.40)
    parser.add_argument('--initial-equity', type=float, default=10_000.0)
    parser.add_argument('--trade-start-day', type=int, default=1)
    parser.add_argument('--trade-end-day', type=int, default=30)
    parser.add_argument('--session-start-hour', type=int, default=7)
    parser.add_argument('--session-end-hour', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, default=Path('optimizer') / 'outputs')
    args = parser.parse_args()

    params_grid = generate_param_grid(args.tp_values, args.sl_values, args.gap_values)
    total_runs = len(params_grid)
    if total_runs == 0:
        print('No parameter combinations matched the ratio constraints (0.5 to 2.0).')
        return

    month_list = ', '.join(args.months)
    print(f'Testing {total_runs} parameter combinations across {len(args.months)} months: {month_list}')
    start_time = time.time()
    summary: list[ScenarioResult] = []
    all_equities: list[pd.DataFrame] = []
    for idx, params in enumerate(params_grid, start=1):
        scenario_result, equity = run_scenario(params, args.months, args)
        summary.append(scenario_result)
        equity = equity.copy()
        equity['take_profit'] = params.take_profit
        equity['stop_loss'] = params.stop_loss
        equity['min_gap'] = params.min_gap
        all_equities.append(equity)
        print_progress(idx, total_runs, start_time)
    print('Optimization complete.')

    summary_path, summary_df = save_results(summary, args.output_dir)
    equity_path = args.output_dir / 'equity_paths.csv'
    pd.concat(all_equities, ignore_index=True).to_csv(equity_path, index=False)

    best_pnl = summary_df.iloc[0]
    best_sharpe = summary_df.sort_values('sharpe', ascending=False).iloc[0]
    best_dd = summary_df.sort_values('max_drawdown').iloc[0]

    print(f'Results saved to: {summary_path}')
    print(f'Equity paths saved to: {equity_path}')
    print('Top combinations:')
    print(
        f"  Highest PnL: TP {best_pnl['take_profit']:.2f}, SL {best_pnl['stop_loss']:.2f}, gap {best_pnl['min_gap']:.2f} -> PnL {best_pnl['total_pnl']:.2f}, Sharpe {best_pnl['sharpe']:.2f}, DD {best_pnl['max_drawdown']:.2f}, Trades {int(best_pnl['total_trades'])}"
    )
    print(
        f"  Lowest Drawdown: TP {best_dd['take_profit']:.2f}, SL {best_dd['stop_loss']:.2f}, gap {best_dd['min_gap']:.2f} -> DD {best_dd['max_drawdown']:.2f}, PnL {best_dd['total_pnl']:.2f}, Sharpe {best_dd['sharpe']:.2f}, Trades {int(best_dd['total_trades'])}"
    )
    print(
        f"  Best Sharpe: TP {best_sharpe['take_profit']:.2f}, SL {best_sharpe['stop_loss']:.2f}, gap {best_sharpe['min_gap']:.2f} -> Sharpe {best_sharpe['sharpe']:.2f}, PnL {best_sharpe['total_pnl']:.2f}, DD {best_sharpe['max_drawdown']:.2f}, Trades {int(best_sharpe['total_trades'])}"
    )


if __name__ == '__main__':
    main()
