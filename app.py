import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import io

st.set_page_config(page_title="Smart Stock Scanner", page_icon="📈", layout="wide")

st.title("📈 Smart Stock Scanner")
st.caption("NSE + BSE • Fundamental filters + 200 EMA • Designed for BTST / 2–4 day swing research")

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
                df = df[["SYMBOL", "NAME OF COMPANY"]].copy()
                df.columns = ["symbol", "company"]
                df["symbol"] = df["symbol"].astype(str).str.strip()
                df["ticker"] = df["symbol"] + ".NS"
                return df.drop_duplicates("symbol")
        except Exception:
            pass
    return pd.DataFrame(columns=["symbol", "company", "ticker"])

def clean_num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def get_200_ema(hist):
    if hist is None or hist.empty or "Close" not in hist:
        return np.nan
    close = hist["Close"].dropna()
    if len(close) < 210:
        return np.nan
    return float(close.ewm(span=200, adjust=False).mean().iloc[-1])

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else np.nan

def analyze_ticker(row, period="1y"):
    symbol = row["symbol"]
    ticker = row["ticker"]
    company = row["company"]

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Yahoo fields. Different tickers can expose different subsets.
        pe = clean_num(info.get("trailingPE"))
        roe = clean_num(info.get("returnOnEquity"))
        eps = clean_num(info.get("trailingEps"))
        profit_growth = clean_num(info.get("earningsGrowth"))
        sales_growth = clean_num(info.get("revenueGrowth"))
        debt_equity = clean_num(info.get("debtToEquity"))

        # Yahoo's insiders percentage is the closest commonly available free field.
        # It is NOT guaranteed to equal promoter holding.
        promoter = clean_num(info.get("heldPercentInsiders"))

        hist = t.history(period=period, interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None

        hist = hist.dropna(subset=["Close"])
        price = float(hist["Close"].iloc[-1])
        ema200 = get_200_ema(hist)
        if pd.isna(ema200):
            return None

        rsi = calc_rsi(hist["Close"])
        avg20vol = float(hist["Volume"].tail(20).mean()) if "Volume" in hist else np.nan
        volume = float(hist["Volume"].iloc[-1]) if "Volume" in hist else np.nan
        vol_ratio = volume / avg20vol if avg20vol and not pd.isna(avg20vol) else np.nan

        # Yahoo may return decimal growth fields (0.55 = 55%)
        growth_pct = lambda x: x * 100 if pd.notna(x) and abs(x) < 5 else x

        return {
            "Company": company,
            "NSE Symbol": symbol,
            "Price": price,
            "P/E": pe,
            "ROE %": roe * 100 if pd.notna(roe) and abs(roe) <= 2 else roe,
            "EPS": eps,
            "Profit Growth %": growth_pct(profit_growth),
            "Sales Growth %": growth_pct(sales_growth),
            "Debt/Equity": debt_equity / 100 if pd.notna(debt_equity) and debt_equity > 20 else debt_equity,
            "Promoter/Insider %": promoter * 100 if pd.notna(promoter) and promoter <= 1 else promoter,
            "200 EMA": ema200,
            "Above 200 EMA %": (price - ema200) / ema200 * 100,
            "RSI 14": rsi,
            "Volume / 20D Avg": vol_ratio,
            "Last Data": hist.index[-1].strftime("%Y-%m-%d"),
            "_ticker": ticker,
        }
    except Exception:
        return None

def passes(r, filters):
    vals = [
        r["P/E"] < filters["pe"],
        r["ROE %"] > filters["roe"],
        r["EPS"] > filters["eps"],
        r["Profit Growth %"] > filters["profit"],
        r["Sales Growth %"] > filters["sales"],
        r["Debt/Equity"] < filters["de"],
        r["Promoter/Insider %"] > filters["promoter"],
        r["Price"] > r["200 EMA"],
    ]
    return all(bool(x) for x in vals)

def score(r):
    # Ranking only after hard filters pass.
    s = 0
    s += min(max((r["ROE %"] - 25) / 50, 0), 1) * 15
    s += min(max((r["Profit Growth %"] - 50) / 150, 0), 1) * 20
    s += min(max((r["Sales Growth %"] - 50) / 150, 0), 1) * 20
    s += min(max((30 - r["P/E"]) / 25, 0), 1) * 10
    s += min(max((0.5 - r["Debt/Equity"]) / 0.5, 0), 1) * 10
    s += min(max((r["Promoter/Insider %"] - 50) / 40, 0), 1) * 10
    s += min(max(r["Above 200 EMA %"] / 30, 0), 1) * 15
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
max_stocks = st.sidebar.slider("Maximum stocks to scan", 25, 1500, 250, 25)
workers = st.sidebar.slider("Parallel requests", 2, 10, 5)
only_nse = st.sidebar.checkbox("NSE only", value=True)
only_bse = st.sidebar.checkbox("BSE only", value=False)

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
else:
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

# Keep scan size practical on free infrastructure.
universe = nse.head(max_stocks).copy()

st.info(
    "Yahoo Finance/yfinance is used for market and fundamental fields. "
    "The free Yahoo field 'heldPercentInsiders' is used as a proxy only; "
    "it is NOT guaranteed to be identical to NSE/BSE promoter holding. "
    "For strict promoter screening, use a licensed/verified shareholding data source."
)

if st.button("🔎 RUN FULL SCAN", type="primary", use_container_width=True):
    if universe.empty:
        st.error("No stocks available to scan.")
        st.stop()

    results = []
    progress = st.progress(0)
    status = st.empty()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyze_ticker, row): row["symbol"] for _, row in universe.iterrows()}
        done = 0
        for fut in as_completed(futures):
            done += 1
            status.write(f"Scanning {done}/{len(futures)}: {futures[fut]}")
            progress.progress(done / len(futures))
            r = fut.result()
            if r is not None:
                try:
                    if passes(r, filters):
                        if min_rsi > 0 and (pd.isna(r["RSI 14"]) or r["RSI 14"] < min_rsi):
                            continue
                        if min_vol > 0 and (pd.isna(r["Volume / 20D Avg"]) or r["Volume / 20D Avg"] < min_vol):
                            continue
                        r["Smart Score"] = score(r)
                        results.append(r)
                except Exception:
                    pass

    progress.empty()
    status.empty()

    df = pd.DataFrame(results)
    if df.empty:
        st.warning("No stock passed all filters. This is a valid result; no fake/demo stocks are inserted.")
    else:
        df = df.sort_values(["Smart Score", "Above 200 EMA %"], ascending=False).reset_index(drop=True)
        st.session_state["results"] = df
        st.session_state["scan_time"] = datetime.now().strftime("%d %b %Y, %I:%M %p IST")

if "results" in st.session_state:
    df = st.session_state["results"]
    st.divider()
    st.subheader(f"✅ Stocks Passing All Filters: {len(df)}")
    st.caption(f"Scan time: {st.session_state.get('scan_time', '')}")

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
    "Research tool only — not investment advice. Yahoo Finance data may be delayed, incomplete, "
    "rate-limited, or inconsistent across securities. Verify financial/shareholding data independently."
)
