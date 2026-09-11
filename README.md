# Smart Stock Scanner V6

Free Streamlit scanner for Indian NSE/BSE BTST and 2–4 day swing research.

## What changed in V6
- **Broad technical universe:** default minimum 1000 stocks, configurable up to 5000.
- The app loads a public Screener **all-companies** screen and scans a broad universe first.
- All stocks from the exact 7-filter fundamental screen are added to the technical universe even if they are outside the first N broad stocks.
- Technical history is downloaded in **Yahoo Finance batches** rather than one request per stock.
- NSE symbols use `.NS`; numeric/BSE symbols use `.BO`; missing NSE data gets a `.BO` fallback.
- 200 EMA, RSI(14), volume/20D, 5D and 20D momentum are calculated locally.
- The exact fundamental hard filters remain delegated to Screener:
  - P/E < 30
  - ROE > 25%
  - EPS > 0
  - Profit growth > 50%
  - Sales growth > 50%
  - Debt/Equity < 0.5
  - Promoter holding > 50%
- Screener's DMA-200 condition is intentionally not used; V6 calculates 200 EMA locally.
- Missing technical data is reported and never treated as a pass.
- Final ranking uses the existing Swing Score only after the fundamental + technical intersection.

## Default sources
Broad universe:
`https://www.screener.in/screens/509570/all-companies/?order=desc`

Fundamental screen:
`https://www.screener.in/screens/3635525/1/`

## Important limitation
Screener officially says it does not provide a public developer API. V6 therefore reads public screen HTML pages. Yahoo Finance is used for price history/technical calculations and may be incomplete or rate-limited for some BSE/newly listed securities.

## Deployment
Replace `app.py` in the existing GitHub repo and keep the existing `requirements.txt`.
