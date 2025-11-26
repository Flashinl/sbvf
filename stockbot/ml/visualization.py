"""
Visualization Tools for Multi-Horizon Stock Prediction

Provides plotting functions for:
- Training curves (loss, learning rate)
- Prediction distributions
- Signal analysis
- Feature importance
- Portfolio performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None
):
    """
    Plot training history from JSON file

    Args:
        history_path: Path to training_results.json
        save_path: Path to save figure
    """
    with open(history_path, 'r') as f:
        results = json.load(f)

    history = results['history']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)

    # Mark best epoch
    best_epoch = results['best_epoch'] + 1
    best_val_loss = results['best_val_loss']
    ax.axvline(best_epoch, color='g', linestyle='--', alpha=0.7,
               label=f'Best Epoch ({best_epoch})')
    ax.plot(best_epoch, best_val_loss, 'g*', markersize=15)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # Learning rate
    ax = axes[1]
    ax.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    plt.suptitle(f'Training History (Best Val Loss: {best_val_loss:.4f})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    plt.show()


def plot_prediction_distribution(
    predictions_df: pd.DataFrame,
    horizon: str = '1month',
    save_path: Optional[str] = None
):
    """
    Plot distribution of predictions

    Args:
        predictions_df: DataFrame from predictor.predict_to_dataframe()
        horizon: Which horizon to plot
        save_path: Path to save figure
    """
    # Filter for horizon
    df_horizon = predictions_df[predictions_df['horizon'] == horizon]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Signal distribution
    ax = axes[0, 0]
    signal_counts = df_horizon['signal'].value_counts()
    colors = {'BUY': 'green', 'HOLD': 'gray', 'SELL': 'red'}
    signal_colors = [colors[s] for s in signal_counts.index]

    ax.bar(signal_counts.index, signal_counts.values, color=signal_colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'Signal Distribution ({horizon})')
    ax.grid(axis='y', alpha=0.3)

    # Add percentages
    total = signal_counts.sum()
    for i, (signal, count) in enumerate(signal_counts.items()):
        pct = (count / total) * 100
        ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom')

    # 2. Confidence distribution
    ax = axes[0, 1]
    ax.hist(df_horizon['confidence'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(df_horizon['confidence'].mean(), color='red', linestyle='--',
               label=f'Mean: {df_horizon["confidence"].mean():.3f}')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title(f'Confidence Distribution ({horizon})')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Expected return distribution by signal
    ax = axes[1, 0]
    for signal in ['BUY', 'HOLD', 'SELL']:
        signal_data = df_horizon[df_horizon['signal'] == signal]['expected_return']
        if len(signal_data) > 0:
            ax.hist(signal_data, bins=15, alpha=0.5, label=signal, color=colors[signal])

    ax.set_xlabel('Expected Return (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Expected Return Distribution by Signal ({horizon})')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Confidence vs Expected Return scatter
    ax = axes[1, 1]
    for signal in ['BUY', 'HOLD', 'SELL']:
        signal_data = df_horizon[df_horizon['signal'] == signal]
        ax.scatter(signal_data['confidence'], signal_data['expected_return'],
                  alpha=0.5, label=signal, color=colors[signal], s=50)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Expected Return (%)')
    ax.set_title(f'Confidence vs Expected Return ({horizon})')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Prediction Analysis - {horizon.upper()}', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction distribution plot saved to: {save_path}")

    plt.show()


def plot_multi_horizon_comparison(
    predictions_df: pd.DataFrame,
    ticker: str,
    save_path: Optional[str] = None
):
    """
    Compare predictions across horizons for a single ticker

    Args:
        predictions_df: DataFrame from predictor.predict_to_dataframe()
        ticker: Stock symbol to analyze
        save_path: Path to save figure
    """
    # Filter for ticker
    df_ticker = predictions_df[predictions_df['ticker'] == ticker]

    if df_ticker.empty:
        print(f"No predictions found for {ticker}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    horizons = ['1week', '1month', '3month']
    colors_signal = {'BUY': 'green', 'HOLD': 'gray', 'SELL': 'red'}

    for idx, horizon in enumerate(horizons):
        row = df_ticker[df_ticker['horizon'] == horizon].iloc[0]

        # Plot probabilities
        ax = axes[idx]
        signals = ['SELL', 'HOLD', 'BUY']
        probs = [row['prob_sell'], row['prob_hold'], row['prob_buy']]
        bar_colors = [colors_signal[s] for s in signals]

        bars = ax.bar(signals, probs, color=bar_colors, alpha=0.7, edgecolor='black')

        # Highlight predicted signal
        predicted_signal = row['signal']
        for i, signal in enumerate(signals):
            if signal == predicted_signal:
                bars[i].set_linewidth(3)

        # Add values on bars
        for i, (signal, prob) in enumerate(zip(signals, probs)):
            ax.text(i, prob + 0.02, f'{prob:.2%}', ha='center', fontsize=10)

        # Add expected return
        ax.text(0.5, 0.9, f"Expected Return: {row['expected_return']:+.2f}%",
                transform=ax.transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Probability')
        ax.set_title(f'{horizon}')
        ax.grid(axis='y', alpha=0.3)

    current_price = df_ticker['current_price'].iloc[0]
    plt.suptitle(f'{ticker} - Multi-Horizon Predictions (Current: ${current_price:.2f})',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-horizon comparison saved to: {save_path}")

    plt.show()


def plot_top_signals(
    predictions_df: pd.DataFrame,
    horizon: str = '1month',
    top_n: int = 10,
    signal_type: str = 'BUY',
    save_path: Optional[str] = None
):
    """
    Plot top N stocks by confidence for a given signal

    Args:
        predictions_df: DataFrame from predictor.predict_to_dataframe()
        horizon: Which horizon to analyze
        top_n: Number of top stocks to show
        signal_type: 'BUY' or 'SELL'
        save_path: Path to save figure
    """
    # Filter for horizon and signal
    df_filtered = predictions_df[
        (predictions_df['horizon'] == horizon) &
        (predictions_df['signal'] == signal_type)
    ]

    if df_filtered.empty:
        print(f"No {signal_type} signals found for {horizon}")
        return

    # Sort by confidence
    df_top = df_filtered.nlargest(top_n, 'confidence')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bar chart
    y_pos = np.arange(len(df_top))
    colors_map = {'BUY': 'green', 'SELL': 'red'}
    color = colors_map.get(signal_type, 'blue')

    bars = ax.barh(y_pos, df_top['confidence'], color=color, alpha=0.7, edgecolor='black')

    # Add expected return annotations
    for i, (idx, row) in enumerate(df_top.iterrows()):
        return_text = f"{row['expected_return']:+.1f}%"
        ax.text(row['confidence'] + 0.01, i, return_text, va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['ticker'])
    ax.set_xlabel('Confidence')
    ax.set_title(f'Top {top_n} {signal_type} Signals by Confidence ({horizon})', fontsize=14)
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Top signals plot saved to: {save_path}")

    plt.show()


def plot_prediction_summary_dashboard(
    predictions_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Create comprehensive dashboard of all predictions

    Args:
        predictions_df: DataFrame from predictor.predict_to_dataframe()
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    horizons = ['1week', '1month', '3month']
    colors_signal = {'BUY': 'green', 'HOLD': 'gray', 'SELL': 'red'}

    # Row 1: Signal distributions by horizon
    for idx, horizon in enumerate(horizons):
        ax = fig.add_subplot(gs[0, idx])
        df_h = predictions_df[predictions_df['horizon'] == horizon]
        signal_counts = df_h['signal'].value_counts()

        bars = ax.bar(signal_counts.index,
                     signal_counts.values,
                     color=[colors_signal[s] for s in signal_counts.index],
                     alpha=0.7,
                     edgecolor='black')

        ax.set_title(f'{horizon} Signals')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

        # Add percentages
        total = signal_counts.sum()
        for i, (signal, count) in enumerate(signal_counts.items()):
            pct = (count / total) * 100
            ax.text(i, count, f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)

    # Row 2: Average confidence by signal and horizon
    ax = fig.add_subplot(gs[1, :])
    pivot_data = predictions_df.pivot_table(
        values='confidence',
        index='horizon',
        columns='signal',
        aggfunc='mean'
    )

    x = np.arange(len(horizons))
    width = 0.25

    for i, signal in enumerate(['SELL', 'HOLD', 'BUY']):
        if signal in pivot_data.columns:
            values = [pivot_data.loc[h, signal] if h in pivot_data.index else 0
                     for h in horizons]
            ax.bar(x + i * width, values, width, label=signal,
                  color=colors_signal[signal], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Horizon')
    ax.set_ylabel('Average Confidence')
    ax.set_title('Average Confidence by Signal and Horizon')
    ax.set_xticks(x + width)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Row 3: Expected return distributions
    for idx, horizon in enumerate(horizons):
        ax = fig.add_subplot(gs[2, idx])
        df_h = predictions_df[predictions_df['horizon'] == horizon]

        for signal in ['BUY', 'SELL']:
            signal_data = df_h[df_h['signal'] == signal]['expected_return']
            if len(signal_data) > 0:
                ax.hist(signal_data, bins=15, alpha=0.5,
                       label=signal, color=colors_signal[signal])

        ax.set_xlabel('Expected Return (%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{horizon} Returns')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Multi-Horizon Prediction Dashboard', fontsize=16, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to: {save_path}")

    plt.show()


def create_prediction_report(
    predictions_df: pd.DataFrame,
    output_dir: str
):
    """
    Generate complete visualization report

    Args:
        predictions_df: DataFrame from predictor.predict_to_dataframe()
        output_dir: Directory to save all plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating prediction report in {output_dir}...")

    # 1. Overall dashboard
    print("  Creating dashboard...")
    plot_prediction_summary_dashboard(
        predictions_df,
        save_path=output_dir / 'dashboard.png'
    )

    # 2. Per-horizon distributions
    for horizon in ['1week', '1month', '3month']:
        print(f"  Creating {horizon} distribution...")
        plot_prediction_distribution(
            predictions_df,
            horizon=horizon,
            save_path=output_dir / f'distribution_{horizon}.png'
        )

    # 3. Top BUY signals
    for horizon in ['1week', '1month', '3month']:
        print(f"  Creating top BUY signals for {horizon}...")
        plot_top_signals(
            predictions_df,
            horizon=horizon,
            top_n=15,
            signal_type='BUY',
            save_path=output_dir / f'top_buy_{horizon}.png'
        )

    # 4. Top SELL signals
    for horizon in ['1week', '1month', '3month']:
        print(f"  Creating top SELL signals for {horizon}...")
        plot_top_signals(
            predictions_df,
            horizon=horizon,
            top_n=15,
            signal_type='SELL',
            save_path=output_dir / f'top_sell_{horizon}.png'
        )

    # 5. Save summary CSV
    summary_path = output_dir / 'predictions_summary.csv'
    predictions_df.to_csv(summary_path, index=False)
    print(f"  Saved predictions to {summary_path}")

    print(f"\n✓ Report complete! Saved to {output_dir}")


if __name__ == '__main__':
    print("Visualization module loaded")
    print("\nAvailable functions:")
    print("  - plot_training_history()")
    print("  - plot_prediction_distribution()")
    print("  - plot_multi_horizon_comparison()")
    print("  - plot_top_signals()")
    print("  - plot_prediction_summary_dashboard()")
    print("  - create_prediction_report()")
