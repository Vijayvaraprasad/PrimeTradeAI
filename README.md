# PrimeTradeAI — Trader Behavior & Market Sentiment Insights

## Overview

This project analyzes 211,224 trades from 32 Hyperliquid traders across 246 crypto assets, cross-referenced with the Bitcoin Fear & Greed Index (2,644 daily observations from Feb 2018 to May 2025).

The goal is to explore how market sentiment (fear vs greed) affects trader behavior and profitability, uncover hidden patterns in trading data, and propose actionable strategies based on the findings.


## Datasets

- **Bitcoin Fear & Greed Index** — Daily sentiment classification (Extreme Fear, Fear, Neutral, Greed, Extreme Greed) with numerical values
- **Hyperliquid Historical Trades** — Individual trade records including account, coin, execution price, size, side (buy/sell), closed PnL, fees, timestamps


## Key Findings

**Sentiment and Profitability:**
- Traders perform best during Extreme Greed periods, averaging $67.89 PnL per trade
- Neutral sentiment shows the weakest performance at $34.31 average PnL
- Win rates are highest during Extreme Greed (89.2%) and lowest during Extreme Fear (76.2%)

**Contrarian Behavior:**
- Buy ratios shift meaningfully across sentiment levels — certain traders increase buying during Fear periods, a classic contrarian pattern
- Sentiment momentum (direction of index change) is a stronger predictor of PnL than the sentiment level itself

**Trader Segments (K-Means Clustering):**
Three distinct trader archetypes were identified through behavioral clustering:
- A large group of 19 traders with moderate activity and average returns
- A small group of 5 high-PnL traders with lower win rates but larger gains per trade
- A group of 8 diversified traders active across many coins with moderate results

**Temporal Patterns:**
- Trading activity and profitability vary by hour and day of week
- BTC dominates volume at $644M, followed by HYPE and SOL


## Project Structure

```
PrimeTradeAI/
    fear_greed_index.csv          -- Bitcoin sentiment data
    historical_data.csv           -- Hyperliquid trade data
    analysis.py                   -- Full analysis pipeline
    dashboard.html                -- Interactive web dashboard
    README.md
    output/
        01_eda_overview.png       -- Exploratory data analysis charts
        02_top_traders.png        -- Top trader analysis
        03_performance_vs_sentiment.png
        04_hidden_patterns.png    -- Contrarian and temporal patterns
        05_trader_clustering.png  -- K-Means segmentation
        06_advanced_analysis.png  -- Sentiment momentum and risk-return
        07_trader_sentiment_heatmap.png
        08_position_analysis.png
        09_key_insights.png
        dashboard_data.json       -- Data for interactive dashboard
        insights.json
        trader_features.csv       -- Engineered trader features
```


## How to Run

Install dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Run the analysis:
```
python analysis.py
```
This generates all charts in the output folder and prepares data for the dashboard.

To view the interactive dashboard, start a local server:
```
python -m http.server 8080
```
Then open http://localhost:8080/dashboard.html in your browser.


## Methodology

1. **Data Preparation** — Parsed timestamps, merged trade data with daily sentiment on date, engineered features like win rate, buy ratio, PnL categories
2. **Statistical Testing** — One-way ANOVA to test PnL differences across sentiment groups, Chi-squared test for sentiment-profitability association
3. **Clustering** — K-Means on standardized trader behavioral features (trade count, avg size, PnL, buy ratio, win rate, coin diversity, risk), visualized with PCA
4. **Visualization** — 9 static chart panels with matplotlib/seaborn, 13+ interactive charts in the web dashboard using Chart.js


## Strategy Recommendations

1. **Sentiment-aware sizing** — Scale position sizes based on the current Fear/Greed regime, with larger allocations during historically favorable conditions
2. **Contrarian entries** — Use extreme fear readings as potential buying opportunities, as the data supports higher average returns for contrarian traders
3. **Track momentum, not just levels** — Changes in sentiment direction appear to be a leading indicator for trade outcomes
4. **Tighter risk management during transitions** — PnL variance increases during sentiment regime changes, suggesting tighter stops are warranted


## Tech Stack

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy)
- HTML/CSS/JavaScript with Chart.js for the interactive dashboard
