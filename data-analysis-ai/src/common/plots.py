"""
Plotting utilities for the Data Analysis AI project
Reusable, beautiful visualizations for crypto and financial data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, List
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def setup_plot_style():
    """Configure matplotlib for beautiful plots"""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def plot_price_chart(df: pd.DataFrame, price_col: str = 'price',
                    date_col: str = 'date', title: str = "Price Chart",
                    save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a professional price chart with volume

    Args:
        df: DataFrame with price data
        price_col: Name of price column
        date_col: Name of date column
        title: Chart title
        save_path: Optional path to save chart

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])

    # Price chart
    axes[0].plot(df[date_col], df[price_col], linewidth=2, color='#2E86C1', label='Price')

    # Add moving averages if they exist
    if 'ma_7' in df.columns:
        axes[0].plot(df[date_col], df['ma_7'], alpha=0.7, color='orange', label='7-day MA')
    if 'ma_21' in df.columns:
        axes[0].plot(df[date_col], df['ma_21'], alpha=0.7, color='red', label='21-day MA')

    axes[0].set_title(title, fontweight='bold')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Volume chart (if volume column exists)
    if 'volume' in df.columns:
        axes[1].bar(df[date_col], df['volume'], alpha=0.6, color='green', width=0.8)
        axes[1].set_ylabel('Volume')
        axes[1].set_xlabel('Date')
    else:
        axes[1].text(0.5, 0.5, 'No volume data available', transform=axes[1].transAxes,
                    ha='center', va='center', fontsize=12, alpha=0.7)
        axes[1].set_xlabel('Date')

    # Format x-axis
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📊 Chart saved to {save_path}")

    return fig

def plot_returns_analysis(df: pd.DataFrame, returns_col: str = 'returns',
                         title: str = "Returns Analysis",
                         save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create returns distribution and time series analysis

    Args:
        df: DataFrame with returns data
        returns_col: Name of returns column
        title: Chart title
        save_path: Optional path to save chart

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    returns = df[returns_col].dropna()

    # Time series of returns
    axes[0, 0].plot(df.index, df[returns_col], alpha=0.7, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Returns Over Time')
    axes[0, 0].set_ylabel('Daily Returns')

    # Returns distribution histogram
    axes[0, 1].hist(returns, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 1].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3f}')
    axes[0, 1].set_title('Returns Distribution')
    axes[0, 1].set_xlabel('Daily Returns')
    axes[0, 1].legend()

    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    axes[1, 0].plot(df.index, cumulative_returns, color='green', linewidth=2)
    axes[1, 0].set_title('Cumulative Returns')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].grid(True, alpha=0.3)

    # Rolling volatility (if possible)
    if len(returns) > 7:
        rolling_vol = returns.rolling(7).std()
        axes[1, 1].plot(df.index, rolling_vol, color='orange', linewidth=2)
        axes[1, 1].set_title('7-Day Rolling Volatility')
        axes[1, 1].set_ylabel('Volatility')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor volatility analysis',
                       transform=axes[1, 1].transAxes, ha='center', va='center')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📊 Returns analysis saved to {save_path}")

    return fig

def plot_correlation_matrix(df: pd.DataFrame, numeric_only: bool = True,
                           title: str = "Correlation Matrix",
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create correlation heatmap

    Args:
        df: DataFrame
        numeric_only: Only include numeric columns
        title: Chart title
        save_path: Optional path to save chart

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()

    if numeric_only:
        df_corr = df.select_dtypes(include=[np.number])
    else:
        df_corr = df

    if df_corr.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No numeric data for correlation analysis',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig

    correlation_matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': .8}, ax=ax)

    ax.set_title(title, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📊 Correlation matrix saved to {save_path}")

    return fig

def plot_technical_indicators(df: pd.DataFrame, price_col: str = 'price',
                             date_col: str = 'date',
                             title: str = "Technical Analysis",
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot price with technical indicators (RSI, MACD, etc.)

    Args:
        df: DataFrame with price and indicator data
        price_col: Name of price column
        date_col: Name of date column
        title: Chart title
        save_path: Optional path to save chart

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()

    # Determine number of subplots based on available indicators
    indicators = ['RSI', 'MACD', 'volatility']
    available_indicators = [ind for ind in indicators if ind.lower() in df.columns]

    n_plots = 1 + len(available_indicators)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))

    if n_plots == 1:
        axes = [axes]

    # Price chart with moving averages
    axes[0].plot(df[date_col], df[price_col], linewidth=2, label='Price', color='blue')

    if 'ma_7' in df.columns:
        axes[0].plot(df[date_col], df['ma_7'], alpha=0.7, label='MA(7)', color='orange')
    if 'ma_21' in df.columns:
        axes[0].plot(df[date_col], df['ma_21'], alpha=0.7, label='MA(21)', color='red')

    axes[0].set_title(f'{title} - Price and Moving Averages')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Additional indicators
    plot_idx = 1
    if 'rsi' in df.columns and plot_idx < len(axes):
        axes[plot_idx].plot(df[date_col], df['rsi'], color='purple')
        axes[plot_idx].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        axes[plot_idx].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        axes[plot_idx].set_title('RSI (Relative Strength Index)')
        axes[plot_idx].set_ylabel('RSI')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    if 'volatility' in df.columns and plot_idx < len(axes):
        axes[plot_idx].plot(df[date_col], df['volatility'], color='orange')
        axes[plot_idx].set_title('Price Volatility')
        axes[plot_idx].set_ylabel('Volatility')
        axes[plot_idx].grid(True, alpha=0.3)

    # Format x-axis for all subplots
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📊 Technical indicators chart saved to {save_path}")

    return fig

def create_dashboard(df: pd.DataFrame, title: str = "Analysis Dashboard",
                    save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Create a comprehensive analysis dashboard

    Args:
        df: DataFrame with analysis data
        title: Dashboard title
        save_path: Optional path to save dashboard

    Returns:
        matplotlib Figure object
    """
    setup_plot_style()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main price chart
    ax1 = fig.add_subplot(gs[0, :])
    if 'price' in df.columns and 'date' in df.columns:
        ax1.plot(df['date'], df['price'], linewidth=2, color='blue', label='Price')
        if 'ma_7' in df.columns:
            ax1.plot(df['date'], df['ma_7'], alpha=0.7, color='orange', label='MA(7)')
        ax1.set_title(f'{title} - Price Overview', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Volume chart
    ax2 = fig.add_subplot(gs[1, 0])
    if 'volume' in df.columns:
        ax2.bar(df['date'], df['volume'], alpha=0.6, color='green')
        ax2.set_title('Trading Volume')
        ax2.tick_params(axis='x', rotation=45)

    # Returns distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if 'returns' in df.columns:
        returns = df['returns'].dropna()
        ax3.hist(returns, bins=20, alpha=0.7, color='purple')
        ax3.axvline(returns.mean(), color='red', linestyle='--')
        ax3.set_title('Returns Distribution')

    # Key metrics
    ax4 = fig.add_subplot(gs[1, 2])
    metrics_text = "Key Metrics:\n\n"
    if 'price' in df.columns:
        price_data = df['price'].dropna()
        metrics_text += f"Avg Price: ${price_data.mean():,.0f}\n"
        metrics_text += f"Price Range: ${price_data.min():,.0f} - ${price_data.max():,.0f}\n"
    if 'returns' in df.columns:
        returns = df['returns'].dropna()
        metrics_text += f"Avg Return: {returns.mean():.2%}\n"
        metrics_text += f"Volatility: {returns.std():.2%}\n"

    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Summary Statistics')
    ax4.axis('off')

    # Correlation heatmap (bottom section)
    ax5 = fig.add_subplot(gs[2, :])
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        im = ax5.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto')
        ax5.set_xticks(range(len(correlation_matrix.columns)))
        ax5.set_yticks(range(len(correlation_matrix.columns)))
        ax5.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax5.set_yticklabels(correlation_matrix.columns)
        ax5.set_title('Feature Correlations')

        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax5.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📊 Dashboard saved to {save_path}")

    return fig