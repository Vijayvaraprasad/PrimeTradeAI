# 🚀 PrimeTradeAI — Trader Behavior & Market Sentiment Insights

> **Exploring the relationship between Hyperliquid trader performance and the Bitcoin Fear & Greed Index to uncover actionable trading strategies.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()

---

## 📋 Assignment Overview

This project analyzes **211,224 trades** from **32 Hyperliquid traders** across **246 crypto assets**, cross-referenced with the **Bitcoin Fear & Greed Index** (2,644 daily observations from Feb 2018 – May 2025), to:

1. **Explore** the relationship between trader performance and market sentiment
2. **Uncover** hidden behavioral patterns and trading archetypes
3. **Deliver** actionable insights for smarter trading strategies

---

## 📊 Key Findings

### 🎯 Sentiment-Performance Relationship
- Trader PnL varies significantly across sentiment regimes (statistically verified via ANOVA)
- **Win rates shift** based on Fear/Greed classification — certain regimes systematically favor profitable trades
- **Position sizing** behavior changes with sentiment — traders adjust size based on market mood

### 🔄 Contrarian Behavior Detected
- Buy ratios increase during Fear periods, suggesting professional contrarian behavior
- Sentiment **momentum** (direction of FG change) is a stronger predictor than sentiment level alone

### 🧬 Three Trader Archetypes Identified (K-Means Clustering)
| Cluster | Archetype | Description |
|---------|-----------|-------------|
| 0 | 🧪 Experimental | Varied strategies, moderate risk |
| 1 | 🐋 Whales | High-volume, large positions |
| 2 | 📊 Moderate | Consistent, smaller positions |

### ⏰ Temporal Patterns
- Trading activity peaks during specific IST hours
- Day-of-week effects on profitability identified

---

## 🗂️ Project Structure

```
PrimeTradeAI/
├── fear_greed_index.csv          # Bitcoin Fear & Greed Index dataset
├── historical_data.csv           # Hyperliquid trader data (211K trades)
├── analysis.py                   # Complete analysis pipeline
├── dashboard.html                # Interactive web dashboard
├── README.md                     # This file
└── output/
    ├── 01_eda_overview.png       # EDA visualizations
    ├── 02_top_traders.png        # Top trader analysis
    ├── 03_performance_vs_sentiment.png  # PnL, win rate, size by sentiment
    ├── 04_hidden_patterns.png    # Contrarian & temporal patterns
    ├── 05_trader_clustering.png  # K-Means behavioral segmentation
    ├── 06_advanced_analysis.png  # Sentiment momentum & risk-return
    ├── 07_trader_sentiment_heatmap.png  # Individual trader × sentiment matrix
    ├── 08_position_analysis.png  # Position sizing & direction bias
    ├── 09_key_insights.png       # Summary of findings
    ├── dashboard_data.json       # Data for interactive dashboard
    ├── insights.json             # Key metrics JSON
    └── trader_features.csv       # Engineered trader features
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run Analysis
```bash
python analysis.py
```
This generates all charts in `./output/` and prepares data for the dashboard.

### View Interactive Dashboard
Open `dashboard.html` in a browser (requires a local server for JSON loading):
```bash
python -m http.server 8080
# Then visit http://localhost:8080/dashboard.html
```

---

## 🔬 Methodology

### Data Pipeline
1. **Loading & Cleaning**: Parse timestamps, handle IST timezone, merge datasets on date
2. **Feature Engineering**: Compute PnL categories, buy ratios, directional bias, sentiment momentum
3. **Statistical Testing**: ANOVA for PnL differences, Chi-squared for win rate associations
4. **Unsupervised Learning**: K-Means clustering on trader behavioral features with PCA visualization
5. **Visualization**: 9 professional chart panels + interactive web dashboard

### Statistical Methods
| Method | Purpose | Result |
|--------|---------|--------|
| One-way ANOVA | Test PnL differences across sentiment | Reported in analysis output |
| Chi-squared Test | Test sentiment-profitability association | Reported in analysis output |
| K-Means Clustering | Identify trader archetypes | 3 distinct clusters |
| PCA | Dimensionality reduction for visualization | 2D projection |

---

## 💡 Strategy Recommendations

1. **Sentiment-Aware Sizing**: Dynamically adjust position sizes based on Fear/Greed regime
2. **Contrarian Opportunities**: Increase buying during extreme fear for asymmetric returns
3. **Momentum Following**: Track FG direction changes as leading indicators
4. **Risk Management**: Tighten stops during sentiment transitions (elevated volatility)
5. **Temporal Optimization**: Identify optimal trading hours for execution quality

---

## 🛠️ Tech Stack

- **Python 3.10+** — Core analysis
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Statistical visualization
- **Scikit-learn** — K-Means clustering & PCA
- **SciPy** — Statistical testing (ANOVA, Chi-squared)
- **Chart.js** — Interactive web dashboard
- **HTML5/CSS3** — Dashboard UI with glassmorphism design

---

## 📧 Contact

**Junior Data Scientist – Trader Behavior Insights**

---

*Built for the PrimeTradeAI hiring assessment — March 2026*
