# Smart Stock Scanner V7

V7 keeps the V6 broad-universe architecture and adds a historical probability layer for the priority stocks.

## Core scan
- Broad NSE/BSE universe: configurable minimum (default 1000)
- Exact 7 fundamental filters from the public Screener screen
- Technical calculations from Yahoo Finance: 20/50/200 EMA, Wilder RSI(14), Wilder ATR(14), volume/20D, 5D/20D momentum
- Current/latest same-day 15-minute Yahoo bar is used for the displayed Price when available; daily history remains the basis for EMA/RSI/ATR and momentum
- Swing Score is strictly 0-100 with 5 components and a minimum 4/5 data coverage requirement
- NSE `.NS`, BSE numeric `.BO`, and NSE-to-BSE fallback

## V7 historical probability layer
For a limited priority set (Final Pass first, then Fundamental Pass, then Technical Pass/score), V7 downloads up to 5 years of daily history and finds nearest historical setups using the current daily technical feature profile.

It estimates, from those historical analogs:
- probability of reaching +5%, +10%, +15%, +20%, +30%, +40% within 5 trading days
- probability of reaching the same upside thresholds within 10 trading days
- probability of hitting -5% within 5/10 trading days
- median and interquartile historical maximum move
- number of historical analogs and a simple confidence label
- Historical Edge Score (not a probability)

### Important methodology rule
Historical features use only information available at the historical date. Forward high/low prices are used only as outcomes after that date. No future data is used to construct the historical setup. This is an empirical historical-frequency estimate, not a guaranteed prediction or investment recommendation.

## Run
```bash
streamlit run app.py
```
