# Smart Stock Scanner V4

V4 deliberately changes the data architecture.

## Fundamental source
The default public Screener.in screen contains the original hard filters:

Price to Earning < 30
Return on equity > 25
EPS > 0
Profit growth > 50
Sales growth > 50
Debt to equity < 0.5
Promoter holding > 50
Current price > DMA 200

V4 reads the public result table instead of reconstructing these fields from Yahoo Finance.

## Technical source
Yahoo Finance is used only for:
- Price
- 200 EMA
- RSI 14
- Volume / 20-day average

## Important
Screener.in does not provide a public API. V4 therefore reads a public screen's HTML. If the public page is blocked or its layout changes, upload a Screener CSV instead.

No fake/demo stocks are generated and missing data is never silently converted into a passing fundamental value.
