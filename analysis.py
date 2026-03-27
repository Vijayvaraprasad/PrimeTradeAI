"""
PrimeTradeAI — Junior Data Scientist Assignment
Trader Behavior Insights: Exploring the Relationship Between
Trader Performance and Bitcoin Market Sentiment

Author: Vijay
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Professional color palette
COLORS = {
    'extreme_fear': '#DC2626',
    'fear': '#F97316',
    'neutral': '#EAB308',
    'greed': '#22C55E',
    'extreme_greed': '#059669',
    'buy': '#3B82F6',
    'sell': '#EF4444',
    'primary': '#6366F1',
    'secondary': '#8B5CF6',
    'accent': '#EC4899',
    'bg_dark': '#0F172A',
    'bg_card': '#1E293B',
    'text': '#F8FAFC',
    'text_muted': '#94A3B8',
    'grid': '#334155',
}

SENTIMENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
SENTIMENT_COLORS = [COLORS['extreme_fear'], COLORS['fear'], COLORS['neutral'], COLORS['greed'], COLORS['extreme_greed']]

# Plot style
plt.rcParams.update({
    'figure.facecolor': COLORS['bg_dark'],
    'axes.facecolor': COLORS['bg_card'],
    'axes.edgecolor': COLORS['grid'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text_muted'],
    'ytick.color': COLORS['text_muted'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'legend.facecolor': COLORS['bg_card'],
    'legend.edgecolor': COLORS['grid'],
    'legend.fontsize': 10,
})

print("=" * 70)
print("  PrimeTradeAI — Trader Behavior & Market Sentiment Analysis")
print("=" * 70)

# ============================================================
# SECTION 1: DATA LOADING & CLEANING
# ============================================================
print("\n📦 SECTION 1: Loading & Cleaning Data...")

# Load datasets
fg = pd.read_csv('fear_greed_index.csv')
hd = pd.read_csv('historical_data.csv')

# Parse dates
fg['date'] = pd.to_datetime(fg['date'])
hd['date'] = pd.to_datetime(hd['Timestamp IST'], format='%d-%m-%Y %H:%M')
hd['trade_date'] = hd['date'].dt.date
hd['trade_date'] = pd.to_datetime(hd['trade_date'])
hd['hour'] = hd['date'].dt.hour
hd['day_of_week'] = hd['date'].dt.day_name()

# Merge on date
fg_daily = fg[['date', 'value', 'classification']].copy()
fg_daily.rename(columns={'value': 'fg_value', 'classification': 'sentiment'}, inplace=True)

merged = hd.merge(fg_daily, left_on='trade_date', right_on='date', how='left', suffixes=('', '_fg'))

# Check merge quality
total_trades = len(merged)
matched_trades = merged['sentiment'].notna().sum()
print(f"  Total trades: {total_trades:,}")
print(f"  Trades with sentiment data: {matched_trades:,} ({matched_trades/total_trades*100:.1f}%)")
print(f"  Unique traders: {merged['Account'].nunique()}")
print(f"  Unique coins: {merged['Coin'].nunique()}")
print(f"  Trade date range: {hd['trade_date'].min().date()} to {hd['trade_date'].max().date()}")

# Filter to only trades with sentiment data
df = merged[merged['sentiment'].notna()].copy()
print(f"  Working dataset: {len(df):,} trades")

# Add useful columns
df['is_profitable'] = df['Closed PnL'] > 0
df['pnl_category'] = pd.cut(df['Closed PnL'], bins=[-np.inf, -100, -10, 0, 10, 100, np.inf],
                             labels=['Big Loss', 'Loss', 'Small Loss', 'Small Win', 'Win', 'Big Win'])
df['account_short'] = df['Account'].str[:8] + '...'

# Sentiment as ordered categorical
df['sentiment'] = pd.Categorical(df['sentiment'], categories=SENTIMENT_ORDER, ordered=True)

# ============================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n📊 SECTION 2: Exploratory Data Analysis...")

# --- Figure 1: Sentiment Distribution Over Time ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Exploratory Data Analysis — Market Sentiment & Trading Activity', fontsize=18, fontweight='bold', y=0.98)

# 2a. Sentiment distribution
ax = axes[0, 0]
sent_counts = df['sentiment'].value_counts().reindex(SENTIMENT_ORDER)
bars = ax.bar(SENTIMENT_ORDER, sent_counts.values, color=SENTIMENT_COLORS, edgecolor='white', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, sent_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Trade Count by Market Sentiment', fontweight='bold')
ax.set_ylabel('Number of Trades')
ax.set_xlabel('Sentiment Classification')

# 2b. Daily trade volume
ax = axes[0, 1]
daily_volume = df.groupby('trade_date')['Size USD'].sum().reset_index()
ax.fill_between(daily_volume['trade_date'], daily_volume['Size USD'] / 1e6, alpha=0.4, color=COLORS['primary'])
ax.plot(daily_volume['trade_date'], daily_volume['Size USD'] / 1e6, color=COLORS['primary'], linewidth=1.5)
ax.set_title('Daily Trading Volume (USD)', fontweight='bold')
ax.set_ylabel('Volume ($ Millions)')
ax.set_xlabel('Date')
ax.tick_params(axis='x', rotation=30)

# 2c. Top 10 Most Traded Coins
ax = axes[1, 0]
top_coins = df.groupby('Coin')['Size USD'].sum().nlargest(10).sort_values()
colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_coins)))
ax.barh(top_coins.index, top_coins.values / 1e6, color=colors_gradient, edgecolor='white', linewidth=0.5)
for i, (val, name) in enumerate(zip(top_coins.values, top_coins.index)):
    ax.text(val / 1e6 + 0.5, i, f'${val/1e6:.1f}M', va='center', fontsize=9, fontweight='bold')
ax.set_title('Top 10 Coins by Trading Volume', fontweight='bold')
ax.set_xlabel('Total Volume ($ Millions)')

# 2d. Buy vs Sell distribution
ax = axes[1, 1]
side_sent = df.groupby(['sentiment', 'Side']).size().unstack(fill_value=0)
side_sent = side_sent.reindex(SENTIMENT_ORDER)
x = np.arange(len(SENTIMENT_ORDER))
width = 0.35
ax.bar(x - width/2, side_sent['BUY'], width, label='BUY', color=COLORS['buy'], alpha=0.85, edgecolor='white', linewidth=0.5)
ax.bar(x + width/2, side_sent['SELL'], width, label='SELL', color=COLORS['sell'], alpha=0.85, edgecolor='white', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15)
ax.set_title('Buy vs Sell Orders by Sentiment', fontweight='bold')
ax.set_ylabel('Number of Trades')
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUTPUT_DIR}/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 01_eda_overview.png")


# --- Figure 2: Top Traders Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Top Trader Analysis', fontsize=18, fontweight='bold', y=1.02)

# Top traders by total volume
ax = axes[0]
trader_vol = df.groupby('account_short')['Size USD'].sum().nlargest(10).sort_values()
colors_gradient = plt.cm.plasma(np.linspace(0.3, 0.9, len(trader_vol)))
ax.barh(trader_vol.index, trader_vol.values / 1e6, color=colors_gradient, edgecolor='white', linewidth=0.5)
for i, val in enumerate(trader_vol.values):
    ax.text(val / 1e6 + 0.2, i, f'${val/1e6:.1f}M', va='center', fontsize=9, fontweight='bold')
ax.set_title('Top 10 Traders by Volume', fontweight='bold')
ax.set_xlabel('Total Volume ($ Millions)')

# Top traders by net PnL
ax = axes[1]
trader_pnl = df.groupby('account_short')['Closed PnL'].sum().sort_values()
colors_pnl = ['#DC2626' if v < 0 else '#22C55E' for v in trader_pnl.values]
# Show top 5 winners and top 5 losers
top_winners = trader_pnl.nlargest(5)
top_losers = trader_pnl.nsmallest(5)
combined = pd.concat([top_losers, top_winners])
colors_combined = ['#DC2626' if v < 0 else '#22C55E' for v in combined.values]
ax.barh(combined.index, combined.values, color=colors_combined, edgecolor='white', linewidth=0.5)
for i, val in enumerate(combined.values):
    offset = 50 if val >= 0 else -50
    ha = 'left' if val >= 0 else 'right'
    ax.text(val + offset, i, f'${val:,.0f}', va='center', ha=ha, fontsize=9, fontweight='bold')
ax.axvline(x=0, color=COLORS['text_muted'], linestyle='--', alpha=0.5)
ax.set_title('Top 5 Winners & Losers (Net PnL)', fontweight='bold')
ax.set_xlabel('Closed PnL ($)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_top_traders.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 02_top_traders.png")


# ============================================================
# SECTION 3: TRADER PERFORMANCE vs MARKET SENTIMENT
# ============================================================
print("\n🎯 SECTION 3: Trader Performance vs Market Sentiment...")

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle('Trader Performance vs Market Sentiment — Deep Dive', fontsize=18, fontweight='bold', y=0.98)

# 3a. Average PnL by Sentiment
ax = axes[0, 0]
pnl_by_sent = df.groupby('sentiment', observed=True)['Closed PnL'].mean()
bars = ax.bar(SENTIMENT_ORDER, pnl_by_sent.reindex(SENTIMENT_ORDER).values, color=SENTIMENT_COLORS,
              edgecolor='white', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, pnl_by_sent.reindex(SENTIMENT_ORDER).values):
    color = '#22C55E' if val >= 0 else '#DC2626'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if val >= 0 else -2),
            f'${val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=10, fontweight='bold', color=color)
ax.axhline(y=0, color=COLORS['text_muted'], linestyle='--', alpha=0.5)
ax.set_title('Average Closed PnL by Sentiment', fontweight='bold')
ax.set_ylabel('Average PnL ($)')
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)

# 3b. Win Rate by Sentiment
ax = axes[0, 1]
# Only consider trades with non-zero PnL for win rate
trades_with_pnl = df[df['Closed PnL'] != 0].copy()
if len(trades_with_pnl) > 0:
    win_rate = trades_with_pnl.groupby('sentiment', observed=True)['is_profitable'].mean() * 100
    bars = ax.bar(SENTIMENT_ORDER, win_rate.reindex(SENTIMENT_ORDER).values, color=SENTIMENT_COLORS,
                  edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars, win_rate.reindex(SENTIMENT_ORDER).values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(y=50, color=COLORS['accent'], linestyle='--', alpha=0.7, label='50% baseline')
    ax.legend()
ax.set_title('Win Rate by Sentiment', fontweight='bold')
ax.set_ylabel('Win Rate (%)')
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)

# 3c. Average Trade Size by Sentiment
ax = axes[0, 2]
size_by_sent = df.groupby('sentiment', observed=True)['Size USD'].mean()
bars = ax.bar(SENTIMENT_ORDER, size_by_sent.reindex(SENTIMENT_ORDER).values, color=SENTIMENT_COLORS,
              edgecolor='white', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, size_by_sent.reindex(SENTIMENT_ORDER).values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Average Trade Size by Sentiment', fontweight='bold')
ax.set_ylabel('Average Size (USD)')
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)

# 3d. Total Volume by Sentiment
ax = axes[1, 0]
vol_by_sent = df.groupby('sentiment', observed=True)['Size USD'].sum()
bars = ax.bar(SENTIMENT_ORDER, vol_by_sent.reindex(SENTIMENT_ORDER).values / 1e6, color=SENTIMENT_COLORS,
              edgecolor='white', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, vol_by_sent.reindex(SENTIMENT_ORDER).values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/1e6 + 0.5,
            f'${val/1e6:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Total Volume by Sentiment', fontweight='bold')
ax.set_ylabel('Volume ($ Millions)')
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)

# 3e. PnL Distribution by Sentiment (Box plot)
ax = axes[1, 1]
pnl_data = []
labels = []
for sent in SENTIMENT_ORDER:
    data = df[df['sentiment'] == sent]['Closed PnL']
    # Clip outliers for better visualization
    q1, q3 = data.quantile(0.05), data.quantile(0.95)
    data_clipped = data[(data >= q1) & (data <= q3)]
    pnl_data.append(data_clipped.values)
    labels.append(sent)

bp = ax.boxplot(pnl_data, labels=[s.replace(' ', '\n') for s in labels], patch_artist=True,
                medianprops={'color': 'white', 'linewidth': 2},
                whiskerprops={'color': COLORS['text_muted']},
                capprops={'color': COLORS['text_muted']},
                flierprops={'markerfacecolor': COLORS['text_muted'], 'markersize': 2})
for patch, color in zip(bp['boxes'], SENTIMENT_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(y=0, color=COLORS['accent'], linestyle='--', alpha=0.5)
ax.set_title('PnL Distribution by Sentiment (5th-95th percentile)', fontweight='bold')
ax.set_ylabel('Closed PnL ($)')

# 3f. Fee Analysis by Sentiment
ax = axes[1, 2]
fee_by_sent = df.groupby('sentiment', observed=True)['Fee'].mean()
bars = ax.bar(SENTIMENT_ORDER, fee_by_sent.reindex(SENTIMENT_ORDER).values, color=SENTIMENT_COLORS,
              edgecolor='white', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, fee_by_sent.reindex(SENTIMENT_ORDER).values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'${val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Average Fee per Trade by Sentiment', fontweight='bold')
ax.set_ylabel('Average Fee ($)')
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUTPUT_DIR}/03_performance_vs_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 03_performance_vs_sentiment.png")


# --- Statistical Tests ---
print("\n  📐 Statistical Significance Tests:")
# ANOVA test for PnL across sentiment groups
groups = [df[df['sentiment'] == s]['Closed PnL'].dropna().values for s in SENTIMENT_ORDER]
groups_valid = [g for g in groups if len(g) > 0]
if len(groups_valid) >= 2:
    f_stat, p_value = stats.f_oneway(*groups_valid)
    print(f"    ANOVA F-stat: {f_stat:.4f}, p-value: {p_value:.6f}")
    print(f"    {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} difference in PnL across sentiments (α=0.05)")

# Chi-squared test for win rate
if len(trades_with_pnl) > 0:
    contingency = pd.crosstab(trades_with_pnl['sentiment'], trades_with_pnl['is_profitable'])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    print(f"    Chi² test (win rate): χ²={chi2:.4f}, p-value={p_chi:.6f}")
    print(f"    {'✅ Significant' if p_chi < 0.05 else '❌ Not significant'} association between sentiment & profitability")


# ============================================================
# SECTION 4: HIDDEN PATTERN DISCOVERY
# ============================================================
print("\n🔍 SECTION 4: Hidden Pattern Discovery...")

# --- 4a. Contrarian vs Momentum Analysis ---
print("  Analyzing contrarian vs momentum behavior...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Hidden Patterns — Contrarian Behavior & Temporal Analysis', fontsize=18, fontweight='bold', y=0.98)

# Buy ratio by sentiment (contrarian signal)
ax = axes[0, 0]
buy_ratio = df.groupby('sentiment', observed=True).apply(lambda x: (x['Side'] == 'BUY').mean() * 100)
bars = ax.bar(SENTIMENT_ORDER, buy_ratio.reindex(SENTIMENT_ORDER).values, color=SENTIMENT_COLORS,
              edgecolor='white', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, buy_ratio.reindex(SENTIMENT_ORDER).values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(y=50, color=COLORS['accent'], linestyle='--', alpha=0.7, label='50% neutral')
ax.set_title('Buy Ratio by Sentiment (Contrarian Signal)', fontweight='bold')
ax.set_ylabel('% of Trades that are BUY')
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)
ax.legend()

# 4b. Trading Activity by Hour
ax = axes[0, 1]
hourly = df.groupby('hour').size()
ax.bar(hourly.index, hourly.values, color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=0.5)
ax.set_title('Trading Activity by Hour (IST)', fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Trades')
ax.set_xticks(range(0, 24))

# 4c. Day of Week Analysis
ax = axes[1, 0]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_counts = df.groupby('day_of_week').size().reindex(day_order)
daily_pnl = df.groupby('day_of_week')['Closed PnL'].mean().reindex(day_order)
ax2 = ax.twinx()
ax.bar(range(7), daily_counts.values, color=COLORS['primary'], alpha=0.7, label='Trade Count', edgecolor='white', linewidth=0.5)
ax2.plot(range(7), daily_pnl.values, color=COLORS['accent'], linewidth=2.5, marker='o', markersize=8, label='Avg PnL')
ax.set_xticks(range(7))
ax.set_xticklabels([d[:3] for d in day_order])
ax.set_title('Trading Activity & Performance by Day', fontweight='bold')
ax.set_ylabel('Number of Trades', color=COLORS['primary'])
ax2.set_ylabel('Average PnL ($)', color=COLORS['accent'])
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# 4d. Coin Preference by Sentiment
ax = axes[1, 1]
top5_coins = df.groupby('Coin')['Size USD'].sum().nlargest(5).index
coin_sent = df[df['Coin'].isin(top5_coins)].groupby(['Coin', 'sentiment'], observed=True)['Size USD'].sum().unstack(fill_value=0)
coin_sent = coin_sent.reindex(columns=SENTIMENT_ORDER)
coin_sent_pct = coin_sent.div(coin_sent.sum(axis=1), axis=0) * 100
coin_sent_pct.plot(kind='barh', stacked=True, ax=ax, color=SENTIMENT_COLORS, edgecolor='white', linewidth=0.5)
ax.set_title('Top 5 Coins — Volume Distribution by Sentiment', fontweight='bold')
ax.set_xlabel('% of Volume')
ax.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUTPUT_DIR}/04_hidden_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 04_hidden_patterns.png")


# --- 4e. Trader Clustering ---
print("  Clustering traders by behavior...")

trader_features = df.groupby('Account').agg(
    total_trades=('Account', 'size'),
    total_volume=('Size USD', 'sum'),
    avg_trade_size=('Size USD', 'mean'),
    total_pnl=('Closed PnL', 'sum'),
    avg_pnl=('Closed PnL', 'mean'),
    pnl_std=('Closed PnL', 'std'),
    buy_ratio=('Side', lambda x: (x == 'BUY').mean()),
    win_rate=('is_profitable', 'mean'),
    avg_fee=('Fee', 'mean'),
    unique_coins=('Coin', 'nunique'),
    avg_fg_value=('fg_value', 'mean'),
).reset_index()

# Fill NaN in pnl_std
trader_features['pnl_std'] = trader_features['pnl_std'].fillna(0)

# Sharpe-like ratio
trader_features['risk_adjusted_return'] = np.where(
    trader_features['pnl_std'] > 0,
    trader_features['avg_pnl'] / trader_features['pnl_std'],
    0
)

# Normalize features for clustering
feature_cols = ['total_trades', 'avg_trade_size', 'avg_pnl', 'buy_ratio', 'win_rate', 'unique_coins', 'pnl_std']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(trader_features[feature_cols])

# Find optimal k using elbow method
inertias = []
K_range = range(2, min(8, len(trader_features)))
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Use k=3 for interpretability
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
trader_features['cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
trader_features['pca_x'] = X_pca[:, 0]
trader_features['pca_y'] = X_pca[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Trader Clustering & Behavioral Segmentation', fontsize=18, fontweight='bold', y=1.02)

# Cluster scatter
ax = axes[0]
cluster_colors = [COLORS['primary'], COLORS['accent'], COLORS['greed']]
for c in range(optimal_k):
    mask = trader_features['cluster'] == c
    ax.scatter(trader_features.loc[mask, 'pca_x'], trader_features.loc[mask, 'pca_y'],
               c=cluster_colors[c], s=120, alpha=0.8, edgecolors='white', linewidth=1.5,
               label=f'Cluster {c}')
ax.set_title('Trader Clusters (PCA Projection)', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax.legend()

# Cluster profiles - radar chart
ax = axes[1]
cluster_profiles = trader_features.groupby('cluster')[feature_cols].mean()
cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min() + 1e-8)

x_pos = np.arange(len(feature_cols))
width = 0.25
for c in range(optimal_k):
    ax.bar(x_pos + c * width, cluster_profiles_norm.loc[c].values, width,
           label=f'Cluster {c}', color=cluster_colors[c], alpha=0.8, edgecolor='white', linewidth=0.5)
ax.set_xticks(x_pos + width)
ax.set_xticklabels([col.replace('_', '\n') for col in feature_cols], fontsize=8, rotation=30)
ax.set_title('Cluster Behavioral Profiles (Normalized)', fontweight='bold')
ax.set_ylabel('Normalized Score')
ax.legend()

# Cluster summary stats
ax = axes[2]
ax.axis('off')
summary_text = "Cluster Summary\n" + "=" * 40 + "\n\n"
cluster_labels = []
for c in range(optimal_k):
    cf = trader_features[trader_features['cluster'] == c]
    n = len(cf)
    avg_pnl = cf['total_pnl'].mean()
    avg_wr = cf['win_rate'].mean() * 100
    avg_vol = cf['total_volume'].mean() / 1e6
    avg_coins = cf['unique_coins'].mean()
    avg_buy = cf['buy_ratio'].mean() * 100

    if avg_pnl > 0 and avg_wr > 55:
        label = "🏆 Profitable Pros"
    elif avg_vol > cf['total_volume'].quantile(0.7) / 1e6:
        label = "🐋 High-Volume Whales"
    else:
        label = "📊 Moderate Traders"
    cluster_labels.append(label)

    summary_text += f"Cluster {c}: {label}\n"
    summary_text += f"  Traders: {n}\n"
    summary_text += f"  Avg Net PnL: ${avg_pnl:,.2f}\n"
    summary_text += f"  Avg Win Rate: {avg_wr:.1f}%\n"
    summary_text += f"  Avg Volume: ${avg_vol:.2f}M\n"
    summary_text += f"  Avg Coins Traded: {avg_coins:.0f}\n"
    summary_text += f"  Buy Ratio: {avg_buy:.1f}%\n\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['bg_card'], alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_trader_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 05_trader_clustering.png")


# --- 4f. PnL Streaks & Sentiment Transitions ---
print("  Analyzing sentiment transition effects...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Advanced Analysis — Sentiment Momentum & Risk', fontsize=18, fontweight='bold', y=1.02)

# Sentiment momentum: Does a change in sentiment predict PnL?
ax = axes[0]
fg_sorted = fg_daily.sort_values('date').copy()
fg_sorted['prev_sentiment'] = fg_sorted['sentiment'].shift(1)
fg_sorted['sentiment_changed'] = fg_sorted['sentiment'] != fg_sorted['prev_sentiment']
fg_sorted['fg_change'] = fg_sorted['fg_value'].diff()

df_with_change = df.merge(fg_sorted[['date', 'fg_change', 'sentiment_changed']], left_on='trade_date', right_on='date', how='left', suffixes=('', '_change'))

# PnL when sentiment improves vs deteriorates
improving = df_with_change[df_with_change['fg_change'] > 0]['Closed PnL'].mean()
worsening = df_with_change[df_with_change['fg_change'] < 0]['Closed PnL'].mean()
stable = df_with_change[df_with_change['fg_change'] == 0]['Closed PnL'].mean()

categories = ['Worsening\n(FG↓)', 'Stable\n(FG=)', 'Improving\n(FG↑)']
values = [worsening, stable, improving]
bar_colors = [COLORS['extreme_fear'], COLORS['neutral'], COLORS['extreme_greed']]
bars = ax.bar(categories, values, color=bar_colors, edgecolor='white', linewidth=0.5, alpha=0.9)
for bar, val in zip(bars, values):
    color = '#22C55E' if val >= 0 else '#DC2626'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if val >= 0 else -0.5),
            f'${val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=12, fontweight='bold', color=color)
ax.axhline(y=0, color=COLORS['text_muted'], linestyle='--', alpha=0.5)
ax.set_title('Avg PnL by Sentiment Momentum (FG Index Change)', fontweight='bold')
ax.set_ylabel('Average Closed PnL ($)')

# Risk-Return by Sentiment (scatter)
ax = axes[1]
sent_risk_return = df.groupby('sentiment', observed=True).agg(
    avg_pnl=('Closed PnL', 'mean'),
    pnl_std=('Closed PnL', 'std'),
    total_volume=('Size USD', 'sum')
).reindex(SENTIMENT_ORDER)

for i, (sent, row) in enumerate(sent_risk_return.iterrows()):
    ax.scatter(row['pnl_std'], row['avg_pnl'],
               s=row['total_volume'] / 1e5, c=SENTIMENT_COLORS[i],
               alpha=0.8, edgecolors='white', linewidth=2, zorder=5)
    ax.annotate(sent, (row['pnl_std'], row['avg_pnl']),
                textcoords="offset points", xytext=(10, 10), fontsize=10, fontweight='bold')
ax.axhline(y=0, color=COLORS['text_muted'], linestyle='--', alpha=0.3)
ax.set_title('Risk-Return Profile by Sentiment (Bubble = Volume)', fontweight='bold')
ax.set_xlabel('PnL Std Dev (Risk)')
ax.set_ylabel('Average PnL (Return)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_advanced_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 06_advanced_analysis.png")


# --- 4g. Heatmap: Trader Performance by Sentiment ---
print("  Creating trader-sentiment heatmap...")

fig, ax = plt.subplots(figsize=(14, 10))
trader_sent_pnl = df.groupby(['account_short', 'sentiment'], observed=True)['Closed PnL'].mean().unstack(fill_value=0)
trader_sent_pnl = trader_sent_pnl.reindex(columns=SENTIMENT_ORDER)
# Sort by total PnL
trader_sent_pnl['total'] = trader_sent_pnl.sum(axis=1)
trader_sent_pnl = trader_sent_pnl.sort_values('total', ascending=True)
trader_sent_pnl = trader_sent_pnl.drop('total', axis=1)

sns.heatmap(trader_sent_pnl, cmap='RdYlGn', center=0, annot=True, fmt='.1f',
            linewidths=0.5, linecolor=COLORS['grid'], ax=ax,
            cbar_kws={'label': 'Average Closed PnL ($)'},
            annot_kws={'fontsize': 8})
ax.set_title('Trader Performance Heatmap by Sentiment Regime', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Trader (Account)')
ax.set_xlabel('Market Sentiment')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_trader_sentiment_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 07_trader_sentiment_heatmap.png")


# --- 4h. Leverage & Position Sizing Analysis ---
print("  Analyzing position sizing patterns...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Position Sizing & Direction Bias Analysis', fontsize=18, fontweight='bold', y=1.02)

# Direction bias by sentiment
ax = axes[0]
dir_counts = df.groupby(['sentiment', 'Direction'], observed=True).size().unstack(fill_value=0)
dir_pct = dir_counts.div(dir_counts.sum(axis=1), axis=0) * 100
if 'Buy' in dir_pct.columns and 'Sell' in dir_pct.columns:
    dir_pct = dir_pct.reindex(SENTIMENT_ORDER)
    x = np.arange(len(SENTIMENT_ORDER))
    ax.bar(x, dir_pct['Buy'], label='Long (Buy)', color=COLORS['buy'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x, dir_pct['Sell'], bottom=dir_pct['Buy'], label='Short (Sell)', color=COLORS['sell'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)
    ax.set_ylabel('% of Trades')
    ax.legend()
elif len(dir_counts.columns) > 0:
    dir_pct = dir_pct.reindex(SENTIMENT_ORDER)
    dir_pct.plot(kind='bar', stacked=True, ax=ax, edgecolor='white', linewidth=0.5)
    ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, fontsize=9)
ax.set_title('Direction Bias by Sentiment', fontweight='bold')

# Position size distribution by sentiment
ax = axes[1]
for i, sent in enumerate(SENTIMENT_ORDER):
    data = df[df['sentiment'] == sent]['Size USD']
    data_log = np.log10(data[data > 0])
    ax.hist(data_log, bins=50, alpha=0.5, color=SENTIMENT_COLORS[i], label=sent, density=True)
ax.set_title('Position Size Distribution by Sentiment (log10 USD)', fontweight='bold')
ax.set_xlabel('log10(Position Size USD)')
ax.set_ylabel('Density')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_position_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 08_position_analysis.png")


# ============================================================
# SECTION 5: KEY INSIGHTS & STRATEGY RECOMMENDATIONS
# ============================================================
print("\n💡 SECTION 5: Generating Key Insights...")

# Compute key metrics for the summary
insights = {}

# Overall metrics
insights['total_trades'] = len(df)
insights['total_volume'] = df['Size USD'].sum()
insights['unique_traders'] = df['Account'].nunique()
insights['unique_coins'] = df['Coin'].nunique()

# PnL by sentiment
for sent in SENTIMENT_ORDER:
    sent_data = df[df['sentiment'] == sent]
    key = sent.lower().replace(' ', '_')
    insights[f'{key}_avg_pnl'] = sent_data['Closed PnL'].mean()
    insights[f'{key}_win_rate'] = (sent_data['Closed PnL'] > 0).mean() * 100 if len(sent_data[sent_data['Closed PnL'] != 0]) > 0 else 0
    insights[f'{key}_avg_size'] = sent_data['Size USD'].mean()
    insights[f'{key}_trade_count'] = len(sent_data)

# Overall buy ratio
insights['overall_buy_ratio'] = (df['Side'] == 'BUY').mean() * 100

# Best performing sentiment
best_sent = max(SENTIMENT_ORDER, key=lambda s: df[df['sentiment'] == s]['Closed PnL'].mean())
worst_sent = min(SENTIMENT_ORDER, key=lambda s: df[df['sentiment'] == s]['Closed PnL'].mean())
insights['best_sentiment'] = best_sent
insights['worst_sentiment'] = worst_sent

# Sentiment momentum
insights['pnl_improving_sentiment'] = improving
insights['pnl_worsening_sentiment'] = worsening

# Cluster info
for c in range(optimal_k):
    cf = trader_features[trader_features['cluster'] == c]
    insights[f'cluster_{c}_size'] = len(cf)
    insights[f'cluster_{c}_label'] = cluster_labels[c]
    insights[f'cluster_{c}_avg_pnl'] = cf['total_pnl'].mean()
    insights[f'cluster_{c}_win_rate'] = cf['win_rate'].mean() * 100

# Save insights as JSON for the dashboard
with open(f'{OUTPUT_DIR}/insights.json', 'w') as f:
    json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in insights.items()}, f, indent=2)

print("  ✅ Saved: insights.json")

# --- Summary Figure ---
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

summary = f"""
{'='*70}
  KEY FINDINGS & STRATEGY RECOMMENDATIONS
{'='*70}

📊 DATASET OVERVIEW
  • {insights['total_trades']:,} trades analyzed across {insights['unique_traders']} traders and {insights['unique_coins']} coins
  • Total volume: ${insights['total_volume']/1e6:.1f}M USD
  • Overall buy ratio: {insights['overall_buy_ratio']:.1f}%

🎯 SENTIMENT-PERFORMANCE RELATIONSHIP
  • Best performing sentiment: {insights['best_sentiment']}
    (Avg PnL: ${insights[insights['best_sentiment'].lower().replace(' ','_') + '_avg_pnl']:.2f})
  • Worst performing sentiment: {insights['worst_sentiment']}
    (Avg PnL: ${insights[insights['worst_sentiment'].lower().replace(' ','_') + '_avg_pnl']:.2f})

📈 SENTIMENT MOMENTUM EFFECT
  • PnL during improving sentiment: ${insights['pnl_improving_sentiment']:.2f}
  • PnL during worsening sentiment: ${insights['pnl_worsening_sentiment']:.2f}
  • {"Improving sentiment correlates with better returns" if improving > worsening else "Worsening sentiment shows higher returns (contrarian edge)"}

🏆 TRADER SEGMENTS (K-Means Clustering)
  • Cluster 0 ({cluster_labels[0]}): {insights['cluster_0_size']} traders, Avg PnL ${insights['cluster_0_avg_pnl']:,.2f}
  • Cluster 1 ({cluster_labels[1]}): {insights['cluster_1_size']} traders, Avg PnL ${insights['cluster_1_avg_pnl']:,.2f}
  • Cluster 2 ({cluster_labels[2]}): {insights['cluster_2_size']} traders, Avg PnL ${insights['cluster_2_avg_pnl']:,.2f}

💡 STRATEGY RECOMMENDATIONS
  1. SENTIMENT-AWARE SIZING: Adjust position sizes based on sentiment regime
  2. CONTRARIAN OPPORTUNITIES: Monitor for extreme readings for reversal trades
  3. MOMENTUM FOLLOWING: Track sentiment changes, not just levels
  4. RISK MANAGEMENT: Tighten stops during high-volatility sentiment transitions
"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor=COLORS['bg_card'], alpha=0.9, edgecolor=COLORS['grid']))

plt.savefig(f'{OUTPUT_DIR}/09_key_insights.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 09_key_insights.png")


# ============================================================
# EXPORT DATA FOR DASHBOARD
# ============================================================
print("\n📤 Exporting data for dashboard...")

dashboard_data = {
    'sentiment_distribution': df['sentiment'].value_counts().reindex(SENTIMENT_ORDER).to_dict(),
    'pnl_by_sentiment': df.groupby('sentiment', observed=True)['Closed PnL'].mean().reindex(SENTIMENT_ORDER).to_dict(),
    'win_rate_by_sentiment': (trades_with_pnl.groupby('sentiment', observed=True)['is_profitable'].mean() * 100).reindex(SENTIMENT_ORDER).to_dict() if len(trades_with_pnl) > 0 else {},
    'volume_by_sentiment': (df.groupby('sentiment', observed=True)['Size USD'].sum() / 1e6).reindex(SENTIMENT_ORDER).to_dict(),
    'avg_size_by_sentiment': df.groupby('sentiment', observed=True)['Size USD'].mean().reindex(SENTIMENT_ORDER).to_dict(),
    'buy_ratio_by_sentiment': df.groupby('sentiment', observed=True).apply(lambda x: (x['Side'] == 'BUY').mean() * 100).reindex(SENTIMENT_ORDER).to_dict(),
    'hourly_activity': df.groupby('hour').size().to_dict(),
    'daily_volume': {str(k): v for k, v in (df.groupby('trade_date')['Size USD'].sum() / 1e6).to_dict().items()},
    'top_coins_volume': (df.groupby('Coin')['Size USD'].sum().nlargest(10) / 1e6).to_dict(),
    'top_traders_pnl': df.groupby('account_short')['Closed PnL'].sum().sort_values().to_dict(),
    'fee_by_sentiment': df.groupby('sentiment', observed=True)['Fee'].mean().reindex(SENTIMENT_ORDER).to_dict(),
    'sentiment_momentum': {'Worsening': worsening, 'Stable': stable, 'Improving': improving},
    'side_by_sentiment': {
        'BUY': df[df['Side'] == 'BUY'].groupby('sentiment', observed=True).size().reindex(SENTIMENT_ORDER).fillna(0).to_dict(),
        'SELL': df[df['Side'] == 'SELL'].groupby('sentiment', observed=True).size().reindex(SENTIMENT_ORDER).fillna(0).to_dict(),
    },
    'daily_pnl_by_day': {d: float(v) for d, v in df.groupby('day_of_week')['Closed PnL'].mean().reindex(day_order).to_dict().items()},
    'cluster_summary': [
        {
            'label': cluster_labels[c],
            'size': int(insights[f'cluster_{c}_size']),
            'avg_pnl': float(insights[f'cluster_{c}_avg_pnl']),
            'win_rate': float(insights[f'cluster_{c}_win_rate']),
        }
        for c in range(optimal_k)
    ],
    'insights': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in insights.items()},
}

# Convert int64 keys to int
def convert_keys(obj):
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.floating)) else k: convert_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

dashboard_data = convert_keys(dashboard_data)

with open(f'{OUTPUT_DIR}/dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str)

print("  ✅ Saved: dashboard_data.json")

# Save trader features for reference
trader_features.to_csv(f'{OUTPUT_DIR}/trader_features.csv', index=False)
print("  ✅ Saved: trader_features.csv")

print("\n" + "=" * 70)
print("  ✅ Analysis complete! All outputs saved to ./output/")
print("=" * 70)
