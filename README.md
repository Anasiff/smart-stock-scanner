# Smart Stock Scanner

Mobile-friendly Streamlit screener for Indian equities, designed for BTST / 2–4 day swing research.

## Default filters

- P/E < 30
- ROE > 25%
- EPS > 0
- Profit growth > 50%
- Sales growth > 50%
- Debt/Equity < 0.5
- Promoter/Insider > 50% (see important limitation below)
- Current price > 200 EMA

## Important Yahoo Finance limitation

The app uses `yfinance` for free market/fundamental data.

Yahoo Finance commonly exposes `heldPercentInsiders`, which is NOT guaranteed to equal the official NSE/BSE promoter shareholding percentage. The app therefore labels this field `Promoter/Insider %` and warns the user.

If strict promoter holding is required, replace this field with a licensed/verified shareholding data provider.

## 200 EMA

At least 210 daily observations are required. The app calculates the standard exponential moving average from daily closes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Go to Streamlit Community Cloud.
4. Connect GitHub.
5. Choose the repository and `app.py`.
6. Deploy.

## Notes

- Yahoo/yfinance can rate-limit large scans.
- On free hosting, scan a manageable number of symbols at a time.
- For a complete NSE+BSE institutional-grade universe and exact promoter holding, use a licensed data provider.
- No fake/demo results are generated.
