
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

st.set_page_config(page_title="Smart Stock Scanner", page_icon="📈", layout="wide")

st.title("📈 Smart Stock Scanner")
st.caption("NSE • Fundamental filters + 200 EMA • Designed for BTST / 2–4 day swing research")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_nse_universe():
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if "SYMBOL" in df.columns:
                cols = ["SYMBOL"]
                if "NAME OF COMPANY" in df.columns:
                    cols.append("NAME OF COMPANY")
                df = df[cols].copy()
                df.columns = ["symbol", "company"] if len(cols) == 2 else ["symbol"]
                if "company" not in df:
                    df["company"] = df["symbol"]
                df["symbol"] = df["symbol"].astype(str).str.strip()
                df["ticker"] = df["symbol"] + ".NS"
                return df.drop_duplicates("symbol")[["symbol", "company", "ticker"]]
        except Exception:
            continue
    return pd.DataFrame(columns=["symbol", "company", "ticker"])


def clean_num(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def pct_value(x):
    """Convert Yahoo decimal percentages such as 0.55 -> 55."""
    x = clean_num(x)
    if pd.isna(x):
        return np.nan
    return x * 100 if abs(x) <= 2 else x


def get_200_ema(close):
    close = pd.Series(close).dropna()
    if len(close) < 210:
        return np.nan
    return float(close.ewm(span=200, adjust=False).mean().iloc[-1])


def calc_rsi(close, period=14):
    close = pd.Series(close).dropna()
    if len(close) < period + 1:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan


def price_scan(tickers, batch_size=80):
    """Batch-download prices first. This avoids 1 API request per stock for history."""
    rows = {}
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        try:
            data = yf.download(
                batch,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if data is None or data.empty:
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        close = data["Close"].dropna()
                        volume = data["Volume"].dropna() if "Volume" in data else pd.Series(dtype=float)
                    else:
                        if ticker not in data.columns.get_level_values(0):
                            continue
                        close = data[ticker]["Close"].dropna()
                        volume = data[ticker]["Volume"].dropna() if "Volume" in data[ticker] else pd.Series(dtype=float)

                    if len(close) < 210:
                        continue

                    price = float(close.iloc[-1])
                    ema200 = get_200_ema(close)
                    if pd.isna(ema200):
                        continue

                    avg20 = float(volume.tail(20).mean()) if len(volume) else np.nan
                    last_vol = float(volume.iloc[-1]) if len(volume) else np.nan
                    vol_ratio = last_vol / avg20 if avg20 > 0 else np.nan

                    rows[ticker] = {
                        "Price": price,
                        "200 EMA": ema200,
                        "Above 200 EMA %": (price - ema200) / ema200 * 100,
                        "RSI 14": calc_rsi(close),
                        "Volume / 20D Avg": vol_ratio,
                        "Last Data": close.index[-1].strftime("%Y-%m-%d"),
                    }
                except Exception:
                    continue
        except Exception:
            # One failed batch should not kill the complete scan.
            continue
    return rows


def get_fundamentals(row, price_data, retries=2):
    """Fetch fundamentals with retry. Missing fields are reported, not silently rejected."""
    ticker = row["ticker"]
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            pe = clean_num(info.get("trailingPE"))
            eps = clean_num(info.get("trailingEps"))
            roe = pct_value(info.get("returnOnEquity"))
            profit_growth = pct_value(info.get("earningsGrowth"))
            sales_growth = pct_value(info.get("revenueGrowth"))
            debt_equity_raw = clean_num(info.get("debtToEquity"))
            promoter = pct_value(info.get("heldPercentInsiders"))

            # Yahoo's debtToEquity is normally a percentage-like number
            # (e.g. 35.0 = 0.35 ratio), while some feeds may already return a ratio.
            if pd.notna(debt_equity_raw):
                debt_equity = debt_equity_raw / 100 if debt_equity_raw > 10 else debt_equity_raw
            else:
                debt_equity = np.nan

            p = price_data
            result = {
                "Company": row["company"],
                "NSE Symbol": row["symbol"],
                "Price": p["Price"],
                "P/E": pe,
                "ROE %": roe,
                "EPS": eps,
                "Profit Growth %": profit_growth,
                "Sales Growth %": sales_growth,
                "Debt/Equity": debt_equity,
                "Promoter/Insider %": promoter,
                "200 EMA": p["200 EMA"],
                "Above 200 EMA %": p["Above 200 EMA %"],
                "RSI 14": p["RSI 14"],
                "Volume / 20D Avg": p["Volume / 20D Avg"],
                "Last Data": p["Last Data"],
                "_ticker": ticker,
            }

            # Keep the result even when fundamentals are missing so the UI can
            # show exactly which field is unavailable.
            return result
        except Exception:
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
            else:
                return None
    return None


def filter_state(r, filters):
    checks = {
        "P/E": pd.notna(r["P/E"]) and r["P/E"] < filters["pe"],
        "ROE": pd.notna(r["ROE %"]) and r["ROE %"] > filters["roe"],
        "EPS": pd.notna(r["EPS"]) and r["EPS"] > filters["eps"],
        "Profit Growth": pd.notna(r["Profit Growth %"]) and r["Profit Growth %"] > filters["profit"],
        "Sales Growth": pd.notna(r["Sales Growth %"]) and r["Sales Growth %"] > filters["sales"],
        "Debt/Equity": pd.notna(r["Debt/Equity"]) and r["Debt/Equity"] < filters["de"],
        "Promoter/Insider": pd.notna(r["Promoter/Insider %"]) and r["Promoter/Insider %"] > filters["promoter"],
        "Price > 200 EMA": pd.notna(r["Price"]) and pd.notna(r["200 EMA"]) and r["Price"] > r["200 EMA"],
    }
    return checks


def passes(r, filters):
    return all(filter_state(r, filters).values())


def score(r):
    def n(x):
        return 0 if pd.isna(x) else x
    s = 0
    s += min(max((n(r["ROE %"]) - 25) / 50, 0), 1) * 15
    s += min(max((n(r["Profit Growth %"]) - 50) / 150, 0), 1) * 20
    s += min(max((n(r["Sales Growth %"]) - 50) / 150, 0), 1) * 20
    s += min(max((30 - n(r["P/E"])) / 25, 0), 1) * 10
    s += min(max((0.5 - n(r["Debt/Equity"])) / 0.5, 0), 1) * 10
    s += min(max((n(r["Promoter/Insider %"]) - 50) / 40, 0), 1) * 10
    s += min(max(n(r["Above 200 EMA %"]) / 30, 0), 1) * 15
    return round(s, 1)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Screening Filters")

pe = st.sidebar.number_input("P/E < ", min_value=0.0, value=30.0, step=1.0)
roe = st.sidebar.number_input("ROE % > ", min_value=-100.0, value=25.0, step=1.0)
eps = st.sidebar.number_input("EPS > ", value=0.0, step=1.0)
profit = st.sidebar.number_input("Profit Growth % > ", value=50.0, step=5.0)
sales = st.sidebar.number_input("Sales Growth % > ", value=50.0, step=5.0)
de = st.sidebar.number_input("Debt/Equity < ", min_value=0.0, value=0.5, step=0.1)
promoter = st.sidebar.number_input("Promoter/Insider % > ", min_value=0.0, value=50.0, step=5.0)

st.sidebar.divider()
max_stocks = st.sidebar.slider("Maximum stocks to scan", 100, 2500, 1000, 100)
workers = st.sidebar.slider("Parallel fundamental requests", 1, 8, 3)

st.sidebar.divider()
st.sidebar.subheader("Swing extras")
min_rsi = st.sidebar.number_input("Optional RSI minimum", 0.0, 100.0, 0.0, 1.0)
min_vol = st.sidebar.number_input("Optional volume / 20D average", 0.0, 20.0, 0.0, 0.1)

filters = {
    "pe": pe, "roe": roe, "eps": eps, "profit": profit,
    "sales": sales, "de": de, "promoter": promoter
}

# -----------------------------
# Universe
# -----------------------------
st.subheader("Universe")
nse = load_nse_universe()

if nse.empty:
    st.error("Could not download the NSE equity list right now. You can upload a CSV with columns: symbol, company.")
    st.stop()

st.success(f"NSE universe loaded: {len(nse):,} symbols")

uploaded = st.file_uploader("Optional: upload your own universe CSV (symbol, company, ticker)", type=["csv"])
if uploaded:
    try:
        custom = pd.read_csv(uploaded)
        required = {"symbol", "company"}
        if not required.issubset(custom.columns):
            st.error("CSV needs at least: symbol, company")
        else:
            if "ticker" not in custom.columns:
                custom["ticker"] = custom["symbol"].astype(str).str.strip() + ".NS"
            nse = custom[["symbol", "company", "ticker"]].copy()
            st.success(f"Custom universe loaded: {len(nse):,}")
    except Exception as e:
        st.error(f"CSV error: {e}")

universe = nse.head(max_stocks).copy()

st.info(
    "Free Yahoo Finance data is used for market/fundamental fields. "
    "Promoter/Insider is only a Yahoo proxy and is NOT guaranteed to equal NSE/BSE promoter holding. "
    "Missing fundamentals are now shown as missing instead of silently eliminating the stock."
)

if st.button("🔎 RUN FULL SCAN", type="primary", use_container_width=True):
    if universe.empty:
        st.error("No stocks available to scan.")
        st.stop()

    # Phase 1: batch price/EMA scan for the selected universe.
    st.write("### Phase 1/2 — Price + 200 EMA scan")
    tickers = universe["ticker"].tolist()
    price_map = price_scan(tickers)

    # Only fetch expensive fundamentals for stocks with usable 200 EMA data.
    price_candidates = []
    for _, row in universe.iterrows():
        if row["ticker"] in price_map:
            price_candidates.append((row, price_map[row["ticker"]]))

    st.success(f"Price history available for {len(price_candidates):,} / {len(universe):,} stocks")

    if not price_candidates:
        st.error("Yahoo Finance returned no usable daily history. Try again later; this is a data-provider issue, not a 'zero stocks qualify' result.")
        st.stop()

    # Phase 2: fundamentals with low parallelism to reduce Yahoo rate limiting.
    st.write("### Phase 2/2 — Fundamental data scan")
    results = []
    failures = 0
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(get_fundamentals, row, pdata): row["symbol"]
            for row, pdata in price_candidates
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            symbol = futures[fut]
            status.write(f"Fetching fundamentals {done}/{len(futures)}: {symbol}")
            progress.progress(done / len(futures))
            try:
                r = fut.result()
                if r is None:
                    failures += 1
                else:
                    results.append(r)
            except Exception:
                failures += 1

    progress.empty()
    status.empty()

    st.info(f"Fundamental records received: {len(results):,} | Provider failures: {failures:,}")

    if not results:
        st.error("No fundamental records were returned by Yahoo Finance. Try again later or reduce parallel requests to 1.")
        st.stop()

    all_df = pd.DataFrame(results)

    # Diagnostic: count how many stocks survive each individual hard filter.
    st.write("### Filter diagnostics")
    diagnostics = []
    for name in ["P/E", "ROE", "EPS", "Profit Growth", "Sales Growth", "Debt/Equity", "Promoter/Insider", "Price > 200 EMA"]:
        count = 0
        for _, r in all_df.iterrows():
            if filter_state(r, filters).get(name, False):
                count += 1
        diagnostics.append({"Filter": name, "Passed": count, "Total records": len(all_df)})

    diag_df = pd.DataFrame(diagnostics)
    st.dataframe(diag_df, hide_index=True, use_container_width=True)

    # Explicit missing-data report.
    missing_cols = ["P/E", "ROE %", "EPS", "Profit Growth %", "Sales Growth %", "Debt/Equity", "Promoter/Insider %"]
    missing = all_df[missing_cols].isna().sum().reset_index()
    missing.columns = ["Field", "Missing records"]

    with st.expander("Data availability (important)"):
        st.dataframe(missing, hide_index=True, use_container_width=True)
        st.caption(
            "A missing Yahoo fundamental is not treated as a pass. It is shown here so a zero-result scan "
            "cannot be mistaken for proof that no NSE stock qualifies."
        )

    final = []
    for _, r in all_df.iterrows():
        if passes(r, filters):
            if min_rsi > 0 and (pd.isna(r["RSI 14"]) or r["RSI 14"] < min_rsi):
                continue
            if min_vol > 0 and (pd.isna(r["Volume / 20D Avg"]) or r["Volume / 20D Avg"] < min_vol):
                continue
            r["Smart Score"] = score(r)
            final.append(r)

    df = pd.DataFrame(final)
    st.session_state["results"] = df
    st.session_state["scan_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p IST")

# -----------------------------
# Results
# -----------------------------
if "results" in st.session_state:
    df = st.session_state["results"]

    st.divider()
    if df.empty:
        st.warning("No stock passed ALL hard filters with the available Yahoo data.")
    else:
        df = df.sort_values(["Smart Score", "Above 200 EMA %"], ascending=False).reset_index(drop=True)
        st.success(f"✅ Stocks Passing All Filters: {len(df)}")

        show_cols = [
            "Company", "NSE Symbol", "Price", "P/E", "ROE %",
            "EPS", "Profit Growth %", "Sales Growth %",
            "Debt/Equity", "Promoter/Insider %",
            "200 EMA", "Above 200 EMA %", "RSI 14",
            "Volume / 20D Avg", "Smart Score", "Last Data"
        ]
        out = df[show_cols].copy()

        st.dataframe(
            out,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="₹%.2f"),
                "P/E": st.column_config.NumberColumn(format="%.2f"),
                "ROE %": st.column_config.NumberColumn(format="%.2f%%"),
                "EPS": st.column_config.NumberColumn(format="₹%.2f"),
                "Profit Growth %": st.column_config.NumberColumn(format="%.2f%%"),
                "Sales Growth %": st.column_config.NumberColumn(format="%.2f%%"),
                "Debt/Equity": st.column_config.NumberColumn(format="%.2f"),
                "Promoter/Insider %": st.column_config.NumberColumn(format="%.2f%%"),
                "200 EMA": st.column_config.NumberColumn(format="₹%.2f"),
                "Above 200 EMA %": st.column_config.NumberColumn(format="%.2f%%"),
                "RSI 14": st.column_config.NumberColumn(format="%.2f"),
                "Volume / 20D Avg": st.column_config.NumberColumn(format="%.2fx"),
                "Smart Score": st.column_config.NumberColumn(format="%.1f"),
            }
        )

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results CSV", csv, "stock_screen_results.csv", "text/csv")

st.divider()
st.caption(
    "Research tool only — not investment advice. Yahoo Finance data can be delayed, incomplete, rate-limited, "
    "or inconsistent across securities. Verify financial/shareholding data independently."
)
