
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

st.set_page_config(page_title="Smart Stock Scanner", page_icon="📈", layout="wide")

st.title("📈 Smart Stock Scanner")
st.caption("NSE • Fundamentals + 200 EMA • BTST / 2–4 day swing research")

# =========================================================
# Generic helpers
# =========================================================
def num(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def pct(x):
    x = num(x)
    if pd.isna(x):
        return np.nan
    return x * 100 if abs(x) <= 2 else x

def first_valid(df, names):
    if df is None or df.empty:
        return np.nan, None
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if len(s):
                return float(s.iloc[0]), name
    return np.nan, None

def first_two_valid(df, names):
    if df is None or df.empty:
        return [], None
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if len(s) >= 2:
                return [float(s.iloc[0]), float(s.iloc[1])], name
    return [], None

def latest_yoy(df, names):
    """Latest available quarter compared with ~4 quarters earlier."""
    if df is None or df.empty:
        return np.nan, None
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if len(s) >= 5:
                latest = float(s.iloc[0])
                prior = float(s.iloc[4])
                if prior != 0:
                    return (latest / prior - 1) * 100, name
    return np.nan, None

def get_200_ema(close):
    close = pd.Series(close).dropna()
    if len(close) < 210:
        return np.nan
    return float(close.ewm(span=200, adjust=False).mean().iloc[-1])

def calc_rsi(close, period=14):
    close = pd.Series(close).dropna()
    if len(close) < period + 1:
        return np.nan
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    x = 100 - (100 / (1 + rs))
    return float(x.iloc[-1]) if pd.notna(x.iloc[-1]) else np.nan


# =========================================================
# NSE universe
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False)
def load_nse():
    for url in [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]:
        try:
            d = pd.read_csv(url)
            if "SYMBOL" not in d.columns:
                continue
            out = pd.DataFrame({
                "symbol": d["SYMBOL"].astype(str).str.strip(),
                "company": d["NAME OF COMPANY"].astype(str).str.strip()
                    if "NAME OF COMPANY" in d.columns else d["SYMBOL"].astype(str),
            })
            out["ticker"] = out["symbol"] + ".NS"
            return out.drop_duplicates("symbol")
        except Exception:
            pass
    return pd.DataFrame(columns=["symbol", "company", "ticker"])


# =========================================================
# Batch price / EMA
# =========================================================
def scan_prices(tickers, batch=80):
    result = {}
    for i in range(0, len(tickers), batch):
        group = tickers[i:i+batch]
        try:
            d = yf.download(
                group, period="1y", interval="1d",
                auto_adjust=False, progress=False,
                group_by="ticker", threads=True
            )
            if d is None or d.empty:
                continue

            for t in group:
                try:
                    if len(group) == 1:
                        close = d["Close"].dropna()
                        vol = d["Volume"].dropna()
                    else:
                        if t not in d.columns.get_level_values(0):
                            continue
                        close = d[t]["Close"].dropna()
                        vol = d[t]["Volume"].dropna()

                    if len(close) < 210:
                        continue
                    price = float(close.iloc[-1])
                    ema = get_200_ema(close)
                    if pd.isna(ema):
                        continue
                    avg20 = float(vol.tail(20).mean()) if len(vol) else np.nan
                    lastv = float(vol.iloc[-1]) if len(vol) else np.nan
                    result[t] = {
                        "Price": price,
                        "200 EMA": ema,
                        "Above 200 EMA %": (price-ema)/ema*100,
                        "RSI 14": calc_rsi(close),
                        "Volume / 20D Avg": lastv/avg20 if avg20 > 0 else np.nan,
                        "Last Data": close.index[-1].strftime("%Y-%m-%d"),
                    }
                except Exception:
                    continue
        except Exception:
            continue
    return result


# =========================================================
# Fundamental extraction
# =========================================================
@st.cache_data(ttl=21600, show_spinner=False)
def fundamentals(ticker):
    """
    Build fundamentals from multiple Yahoo sources.

    Priority:
      1) info/default statistics for valuation/EPS/promoter
      2) quarterly statements for latest-quarter YoY growth
      3) balance sheet for ROE and Debt/Equity
      4) fallback to info where available

    This deliberately exposes source fields so a result is auditable.
    """
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # ---------- Fast quote/info fields ----------
        pe = num(info.get("trailingPE"))
        eps = num(info.get("trailingEps"))
        promoter = pct(info.get("heldPercentInsiders"))

        # ---------- Statements ----------
        qinc = pd.DataFrame()
        qbs = pd.DataFrame()
        try:
            qinc = t.get_income_stmt(freq="quarterly")
        except Exception:
            try:
                qinc = t.quarterly_income_stmt
            except Exception:
                pass

        try:
            qbs = t.get_balance_sheet(freq="quarterly")
        except Exception:
            try:
                qbs = t.quarterly_balance_sheet
            except Exception:
                pass

        # Annual balance sheet fallback
        absheet = pd.DataFrame()
        try:
            absheet = t.get_balance_sheet(freq="yearly")
        except Exception:
            try:
                absheet = t.balance_sheet
            except Exception:
                pass

        # ---------- Profit + sales growth: latest quarter YoY ----------
        profit_growth, profit_row = latest_yoy(
            qinc,
            [
                "Net Income Common Stockholders",
                "Net Income",
                "Net Income Including Noncontrolling Interests",
            ],
        )
        sales_growth, sales_row = latest_yoy(
            qinc,
            [
                "Total Revenue",
                "Operating Revenue",
                "Total Revenues",
            ],
        )

        # Fallback to Yahoo info growth fields
        if pd.isna(profit_growth):
            profit_growth = pct(info.get("earningsGrowth"))
            profit_source = "Yahoo info earningsGrowth" if pd.notna(profit_growth) else None
        else:
            profit_source = f"quarterly_income_stmt:{profit_row} YoY"

        if pd.isna(sales_growth):
            sales_growth = pct(info.get("revenueGrowth"))
            sales_source = "Yahoo info revenueGrowth" if pd.notna(sales_growth) else None
        else:
            sales_source = f"quarterly_income_stmt:{sales_row} YoY"

        # ---------- ROE: TTM-ish/latest income divided by avg equity ----------
        # Prefer latest annual/quarterly balance values. Use average of latest
        # two equity observations when available.
        equity_vals, equity_row = first_two_valid(
            qbs,
            [
                "Stockholders Equity",
                "Common Stock Equity",
                "Total Equity Gross Minority Interest",
                "Total Equity",
            ],
        )
        equity_source = "quarterly_balance_sheet"
        if len(equity_vals) < 2:
            equity_vals, equity_row = first_two_valid(
                absheet,
                [
                    "Stockholders Equity",
                    "Common Stock Equity",
                    "Total Equity Gross Minority Interest",
                    "Total Equity",
                ],
            )
            equity_source = "annual_balance_sheet"

        net_income_vals, ni_row = first_two_valid(
            qinc,
            [
                "Net Income Common Stockholders",
                "Net Income",
                "Net Income Including Noncontrolling Interests",
            ],
        )

        roe = np.nan
        roe_source = None

        if equity_vals and len(net_income_vals):
            # Latest-quarter annualization when only quarterly statements exist.
            # If four quarters are available, use TTM net income / average equity.
            for name in [
                "Net Income Common Stockholders",
                "Net Income",
                "Net Income Including Noncontrolling Interests",
            ]:
                if name in qinc.index:
                    s = pd.to_numeric(qinc.loc[name], errors="coerce").dropna()
                    if len(s) >= 4 and len(equity_vals) >= 2:
                        ttm_ni = float(s.iloc[:4].sum())
                        avg_eq = float(np.mean(equity_vals[:2]))
                        if avg_eq > 0:
                            roe = ttm_ni / avg_eq * 100
                            roe_source = f"TTM {name} / avg equity"
                            break

        if pd.isna(roe):
            roe = pct(info.get("returnOnEquity"))
            roe_source = "Yahoo info returnOnEquity" if pd.notna(roe) else None

        # ---------- Debt / Equity ----------
        debt, debt_row = first_valid(
            qbs,
            [
                "Total Debt",
                "TotalDebt",
            ],
        )
        equity, eq_row = first_valid(
            qbs,
            [
                "Stockholders Equity",
                "Common Stock Equity",
                "Total Equity Gross Minority Interest",
                "Total Equity",
            ],
        )

        de = np.nan
        de_source = None
        if pd.notna(debt) and pd.notna(equity) and equity > 0:
            de = debt / equity
            de_source = f"quarterly_balance_sheet:{debt_row}/{eq_row}"

        if pd.isna(de):
            debt_raw = num(info.get("debtToEquity"))
            if pd.notna(debt_raw):
                de = debt_raw / 100 if debt_raw > 10 else debt_raw
                de_source = "Yahoo info debtToEquity"

        # ---------- EPS fallback ----------
        if pd.isna(eps):
            eps, eps_row = first_valid(
                qinc,
                ["Diluted EPS", "Basic EPS", "Diluted EPS Other GAAP"],
            )
        else:
            eps_row = "Yahoo info trailingEps"

        return {
            "P/E": pe,
            "ROE %": roe,
            "EPS": eps,
            "Profit Growth %": profit_growth,
            "Sales Growth %": sales_growth,
            "Debt/Equity": de,
            "Promoter/Insider %": promoter,
            "ROE Source": roe_source,
            "Profit Growth Source": profit_source,
            "Sales Growth Source": sales_source,
            "Debt/Equity Source": de_source,
            "EPS Source": eps_row if isinstance(eps_row, str) else "quarterly income statement",
        }
    except Exception as e:
        return {
            "P/E": np.nan, "ROE %": np.nan, "EPS": np.nan,
            "Profit Growth %": np.nan, "Sales Growth %": np.nan,
            "Debt/Equity": np.nan, "Promoter/Insider %": np.nan,
            "ROE Source": None, "Profit Growth Source": None,
            "Sales Growth Source": None, "Debt/Equity Source": None,
            "EPS Source": None,
        }


def make_row(meta, p, f):
    r = {
        "Company": meta["company"],
        "NSE Symbol": meta["symbol"],
        "Price": p["Price"],
        "P/E": f["P/E"],
        "ROE %": f["ROE %"],
        "EPS": f["EPS"],
        "Profit Growth %": f["Profit Growth %"],
        "Sales Growth %": f["Sales Growth %"],
        "Debt/Equity": f["Debt/Equity"],
        "Promoter/Insider %": f["Promoter/Insider %"],
        "200 EMA": p["200 EMA"],
        "Above 200 EMA %": p["Above 200 EMA %"],
        "RSI 14": p["RSI 14"],
        "Volume / 20D Avg": p["Volume / 20D Avg"],
        "Last Data": p["Last Data"],
        "ROE Source": f["ROE Source"],
        "Profit Growth Source": f["Profit Growth Source"],
        "Sales Growth Source": f["Sales Growth Source"],
        "Debt/Equity Source": f["Debt/Equity Source"],
        "EPS Source": f["EPS Source"],
    }
    return r


def checks(r, cfg):
    return {
        "P/E": pd.notna(r["P/E"]) and r["P/E"] < cfg["pe"],
        "ROE": pd.notna(r["ROE %"]) and r["ROE %"] > cfg["roe"],
        "EPS": pd.notna(r["EPS"]) and r["EPS"] > cfg["eps"],
        "Profit Growth": pd.notna(r["Profit Growth %"]) and r["Profit Growth %"] > cfg["profit"],
        "Sales Growth": pd.notna(r["Sales Growth %"]) and r["Sales Growth %"] > cfg["sales"],
        "Debt/Equity": pd.notna(r["Debt/Equity"]) and r["Debt/Equity"] < cfg["de"],
        "Promoter/Insider": pd.notna(r["Promoter/Insider %"]) and r["Promoter/Insider %"] > cfg["promoter"],
        "Price > 200 EMA": pd.notna(r["Price"]) and pd.notna(r["200 EMA"]) and r["Price"] > r["200 EMA"],
    }


def smart_score(r):
    def z(x):
        return 0 if pd.isna(x) else x
    s = 0
    s += min(max((z(r["ROE %"])-25)/50,0),1)*15
    s += min(max((z(r["Profit Growth %"])-50)/150,0),1)*20
    s += min(max((z(r["Sales Growth %"])-50)/150,0),1)*20
    s += min(max((30-z(r["P/E"]))/25,0),1)*10
    s += min(max((0.5-z(r["Debt/Equity"]))/0.5,0),1)*10
    s += min(max((z(r["Promoter/Insider %"])-50)/40,0),1)*10
    s += min(max(z(r["Above 200 EMA %"])/30,0),1)*15
    return round(s,1)


# =========================================================
# UI
# =========================================================
st.sidebar.header("Hard Filters")
pe = st.sidebar.number_input("P/E <", 0.0, 500.0, 30.0, 1.0)
roe = st.sidebar.number_input("ROE % >", -100.0, 500.0, 25.0, 1.0)
eps = st.sidebar.number_input("EPS >", -100000.0, 100000.0, 0.0, 1.0)
profit = st.sidebar.number_input("Profit Growth % >", -500.0, 1000.0, 50.0, 5.0)
sales = st.sidebar.number_input("Sales Growth % >", -500.0, 1000.0, 50.0, 5.0)
de = st.sidebar.number_input("Debt/Equity <", 0.0, 20.0, 0.5, 0.1)
promoter = st.sidebar.number_input("Promoter/Insider % >", 0.0, 100.0, 50.0, 5.0)

st.sidebar.divider()
max_stocks = st.sidebar.slider("Maximum stocks to scan", 100, 2500, 1000, 100)
workers = st.sidebar.slider("Parallel fundamental requests", 1, 5, 3)
min_rsi = st.sidebar.number_input("Optional RSI minimum", 0.0, 100.0, 0.0, 1.0)
min_vol = st.sidebar.number_input("Optional volume / 20D avg", 0.0, 20.0, 0.0, 0.1)

cfg = {"pe":pe,"roe":roe,"eps":eps,"profit":profit,"sales":sales,"de":de,"promoter":promoter}

nse = load_nse()
st.subheader("Universe")
if nse.empty:
    st.error("NSE universe could not be loaded.")
    st.stop()

st.success(f"NSE universe loaded: {len(nse):,} symbols")

upload = st.file_uploader("Optional custom universe CSV (symbol, company, ticker)", type=["csv"])
if upload:
    try:
        custom = pd.read_csv(upload)
        if not {"symbol","company"}.issubset(custom.columns):
            st.error("CSV requires symbol and company columns.")
        else:
            if "ticker" not in custom.columns:
                custom["ticker"] = custom["symbol"].astype(str).str.strip()+".NS"
            nse = custom[["symbol","company","ticker"]]
            st.success(f"Custom universe loaded: {len(nse):,}")
    except Exception as e:
        st.error(f"CSV error: {e}")

st.info(
    "V3 uses Yahoo financial statements as a fallback/primary calculation source for ROE, "
    "latest-quarter YoY profit/sales growth and Debt/Equity. Yahoo insider holding remains only a proxy "
    "for promoter holding and is not official NSE/BSE shareholding."
)

if st.button("🔎 RUN FULL SCAN", type="primary", use_container_width=True):
    universe = nse.head(max_stocks).copy()
    tickers = universe["ticker"].tolist()

    st.write("### Phase 1/2 — Price + 200 EMA")
    price_map = scan_prices(tickers)
    candidates = [(row, price_map[row["ticker"]]) for _, row in universe.iterrows() if row["ticker"] in price_map]
    st.success(f"Usable price history: {len(candidates):,} / {len(universe):,}")

    if not candidates:
        st.error("No usable price history returned by Yahoo. Try again later.")
        st.stop()

    st.write("### Phase 2/2 — Fundamental reconstruction")
    progress = st.progress(0)
    status = st.empty()
    results = []
    failures = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fundamentals, row["ticker"]): (row, price_map[row["ticker"]])
            for row, _ in candidates
        }
        total = len(futures)
        for done, fut in enumerate(as_completed(futures), 1):
            row, p = futures[fut]
            status.write(f"Reconstructing fundamentals {done}/{total}: {row['symbol']}")
            progress.progress(done/total)
            try:
                f = fut.result()
                results.append(make_row(row, p, f))
            except Exception:
                failures += 1

    progress.empty()
    status.empty()

    df_all = pd.DataFrame(results)
    st.info(f"Fundamental records: {len(df_all):,} | Failures: {failures:,}")

    # Diagnostics
    diag = []
    for label in ["P/E","ROE","EPS","Profit Growth","Sales Growth","Debt/Equity","Promoter/Insider","Price > 200 EMA"]:
        n = sum(checks(r,cfg).get(label,False) for _,r in df_all.iterrows())
        diag.append({"Filter":label,"Passed":n,"Total":len(df_all)})
    st.subheader("Filter diagnostics")
    st.dataframe(pd.DataFrame(diag), hide_index=True, use_container_width=True)

    # Missing data
    miss_fields = ["P/E","ROE %","EPS","Profit Growth %","Sales Growth %","Debt/Equity","Promoter/Insider %"]
    miss = df_all[miss_fields].isna().sum().reset_index()
    miss.columns = ["Field","Missing records"]
    with st.expander("Data availability / missing fields"):
        st.dataframe(miss, hide_index=True, use_container_width=True)

    # Final hard filter
    final = []
    for _, r in df_all.iterrows():
        c = checks(r,cfg)
        if not all(c.values()):
            continue
        if min_rsi > 0 and (pd.isna(r["RSI 14"]) or r["RSI 14"] < min_rsi):
            continue
        if min_vol > 0 and (pd.isna(r["Volume / 20D Avg"]) or r["Volume / 20D Avg"] < min_vol):
            continue
        r["Smart Score"] = smart_score(r)
        final.append(r)

    out = pd.DataFrame(final)
    st.session_state["results_v3"] = out

if "results_v3" in st.session_state:
    out = st.session_state["results_v3"]
    st.divider()
    if out.empty:
        st.warning("No stock passed ALL hard filters with the available reconstructed data.")
    else:
        out = out.sort_values(["Smart Score","Above 200 EMA %"], ascending=False)
        st.success(f"✅ {len(out)} stocks passed ALL hard filters")
        cols = [
            "Company","NSE Symbol","Price","P/E","ROE %","EPS",
            "Profit Growth %","Sales Growth %","Debt/Equity",
            "Promoter/Insider %","200 EMA","Above 200 EMA %",
            "RSI 14","Volume / 20D Avg","Smart Score","Last Data"
        ]
        st.dataframe(out[cols], hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Download results CSV",
            out[cols].to_csv(index=False).encode(),
            "smart_stock_scanner_results.csv",
            "text/csv"
        )

st.caption(
    "Research tool only — not investment advice. Yahoo Finance can be incomplete or rate-limited. "
    "Verify financials and promoter/shareholding data independently."
)
