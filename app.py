
import io
import re
import time
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Smart Stock Scanner V4.1", page_icon="📈", layout="wide")

DEFAULT_SCREENER_URL = "https://www.screener.in/screens/3928301/cheetah/"
DEFAULT_QUERY = (
    "Price to Earning < 30 AND Return on equity > 25 AND EPS > 0 AND "
    "Profit growth > 50 AND Sales growth > 50 AND Debt to equity < 0.5 AND "
    "Promoter holding > 50 AND Current price > DMA 200"
)

st.title("📈 Smart Stock Scanner V4.1")
st.caption("Screener.in fundamentals + Yahoo Finance technicals • BTST / 2–4 day swing research")

st.info(
    "V4.1 fixes Screener symbol extraction; fundamentals remain sourced from Screener: fundamental hard filters are taken from a public "
    "Screener.in screen instead of reconstructing ROE/growth/debt from Yahoo. "
    "Yahoo is used only for price/200 EMA/RSI/volume checks. No stock is invented "
    "when a source is unavailable."
)

with st.sidebar:
    st.header("Fundamental source")
    screener_url = st.text_input(
        "Public Screener.in screen URL",
        value=DEFAULT_SCREENER_URL,
        help="Paste any public Screener.in screen URL. The screen itself must contain your desired fundamental filters."
    )
    st.caption("Default screen uses your original 8 hard filters.")
    st.divider()

    st.header("Technical filters")
    require_above_200 = st.checkbox("Price > 200 EMA", value=True)
    min_rsi = st.number_input("Minimum RSI (optional)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    max_rsi = st.number_input("Maximum RSI (optional)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    min_volume_ratio = st.number_input("Min volume / 20D avg (optional)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
    max_candidates = st.number_input("Max candidates for Yahoo technical check", min_value=1, max_value=100, value=50, step=1)

    st.divider()
    st.header("Source diagnostics")
    st.caption("Screener public pages do not provide an official API. V4 reads the public HTML page and parses the displayed result table. If the source blocks requests, use the CSV upload fallback below.")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_screener(url: str):
    if "screener.in/screens/" not in url:
        raise ValueError("Please provide a public Screener.in screen URL.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Screener screen"

    query = ""
    qnode = soup.select_one(".query-text, .query")
    if qnode:
        query = qnode.get_text(" ", strip=True)

    rows = []
    table = soup.select_one("table.data-table")

    if table:
        # Screener's CSS classes can change. Extract the company link
        # generically from each result row instead of relying on td.name.
        for tr in table.select("tbody tr, tr"):
            a = tr.select_one('a[href*="/company/"]')
            if not a:
                continue
            name = a.get_text(" ", strip=True)
            href = a.get("href", "")
            m = re.search(r"/company/([^/?#]+)/?", href)
            symbol = m.group(1).upper() if m else ""
            if symbol:
                rows.append({"Company": name, "NSE Symbol": symbol})

    # Fallback: collect company links directly from the whole page.
    if not rows:
        seen = set()
        for a in soup.select('a[href*="/company/"]'):
            href = a.get("href", "")
            m = re.search(r"/company/([^/?#]+)/?", href)
            if not m:
                continue
            symbol = m.group(1).upper()
            name = a.get_text(" ", strip=True)
            if symbol and name and symbol not in seen:
                seen.add(symbol)
                rows.append({"Company": name, "NSE Symbol": symbol})

    if not rows:
        raise ValueError(
            "Screener result table was found, but NSE company links could not be extracted. "
            "V4.1 stops here rather than sending zero/blank symbols to Yahoo. "
            "Use a public Screener URL or upload a Screener CSV."
        )


    if not rows:
        raise ValueError(
            "Could not read the Screener result table. The page may be private, "
            "blocked, or its HTML layout may have changed."
        )

    df = pd.DataFrame(rows)
    if "_cells" in df.columns:
        df = df.drop(columns=["_cells"])
    return title, query, df, r.url

def load_uploaded_csv(uploaded):
    raw = pd.read_csv(uploaded)
    cols = {c.lower().strip(): c for c in raw.columns}
    sym = None
    for k in ["nse symbol", "nse code", "symbol", "ticker"]:
        if k in cols:
            sym = cols[k]
            break
    if sym is None:
        raise ValueError("CSV must contain an NSE Symbol/Symbol/Ticker column.")
    raw["NSE Symbol"] = raw[sym].astype(str).str.replace(".NS", "", regex=False).str.strip().str.upper()
    if "Company" not in raw.columns:
        raw["Company"] = raw["NSE Symbol"]
    return raw

@st.cache_data(ttl=900, show_spinner=False)
def technical_check(symbols):
    results = []
    for sym in symbols:
        ticker = sym if sym.endswith(".NS") else sym + ".NS"
        try:
            hist = yf.download(
                ticker, period="1y", interval="1d",
                auto_adjust=False, progress=False, threads=False
            )
            if hist is None or hist.empty:
                results.append({"NSE Symbol": sym, "Price": np.nan, "200 EMA": np.nan, "Above 200 EMA": False,
                                "RSI 14": np.nan, "Volume/20D": np.nan, "Technical Status": "No price data"})
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            volume = pd.to_numeric(hist.get("Volume", pd.Series(index=hist.index, dtype=float)), errors="coerce")
            if len(close) < 210:
                results.append({"NSE Symbol": sym, "Price": float(close.iloc[-1]) if len(close) else np.nan,
                                "200 EMA": np.nan, "Above 200 EMA": False, "RSI 14": np.nan,
                                "Volume/20D": np.nan, "Technical Status": "Insufficient history"})
                continue

            ema200 = close.ewm(span=200, adjust=False).mean()
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            vol_avg = volume.rolling(20).mean()
            last_price = float(close.iloc[-1])
            last_ema = float(ema200.iloc[-1])
            last_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
            last_vr = float(volume.iloc[-1] / vol_avg.iloc[-1]) if pd.notna(vol_avg.iloc[-1]) and vol_avg.iloc[-1] else np.nan

            results.append({
                "NSE Symbol": sym,
                "Price": last_price,
                "200 EMA": last_ema,
                "Above 200 EMA": last_price > last_ema,
                "RSI 14": last_rsi,
                "Volume/20D": last_vr,
                "Technical Status": "OK",
            })
        except Exception as e:
            results.append({"NSE Symbol": sym, "Price": np.nan, "200 EMA": np.nan, "Above 200 EMA": False,
                            "RSI 14": np.nan, "Volume/20D": np.nan, "Technical Status": f"Error: {type(e).__name__}"})
        time.sleep(0.05)
    return pd.DataFrame(results)

st.subheader("1. Fundamental screen")
uploaded = st.file_uploader(
    "Optional fallback: upload a Screener CSV exported from your account",
    type=["csv"],
    help="Use this only if Screener's public page cannot be fetched. Export must contain NSE Symbol/Symbol/Ticker."
)

if uploaded:
    try:
        fund = load_uploaded_csv(uploaded)
        source_title = "Uploaded Screener CSV"
        source_query = "CSV source"
        source_url = ""
        st.success(f"Loaded {len(fund)} fundamental candidates from CSV.")
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    try:
        source_title, source_query, fund, source_url = fetch_screener(screener_url)
        st.success(f"{source_title}: {len(fund)} candidates loaded from Screener.")
    except Exception as e:
        st.error(f"Screener source could not be loaded: {e}")
        st.warning(
            "This is deliberately fail-safe: V4 will NOT replace missing fundamentals with Yahoo estimates. "
            "Either paste a working public Screener URL or upload a Screener CSV."
        )
        st.stop()

if source_query:
    st.caption("Fundamental query detected:")
    st.code(source_query if len(source_query) < 1500 else source_query[:1500])

if source_url:
    st.markdown(f"[Open source Screener screen]({source_url})")

# Clean and de-duplicate
fund["NSE Symbol"] = fund["NSE Symbol"].astype(str).str.upper().str.strip()
if fund.empty:
    st.error("No NSE symbols were extracted from Screener. Scan stopped safely.")
    st.stop()
fund = fund[fund["NSE Symbol"].ne("") & fund["NSE Symbol"].ne("NAN")].drop_duplicates("NSE Symbol")
fund = fund.head(int(max_candidates)).copy()

st.write(f"**Candidates sent to Yahoo technical check: {len(fund)}**")

if st.button("🔎 RUN V4.1 SCAN", type="primary", use_container_width=True):
    with st.spinner("Checking price, 200 EMA, RSI and volume for Screener-qualified stocks..."):
        tech = technical_check(fund["NSE Symbol"].tolist())

    out = fund.merge(tech, on="NSE Symbol", how="left")
    out["Technical Pass"] = True
    if require_above_200:
        out["Technical Pass"] &= out["Above 200 EMA"].fillna(False)
    if min_rsi > 0:
        out["Technical Pass"] &= out["RSI 14"].ge(min_rsi).fillna(False)
    if max_rsi < 100:
        out["Technical Pass"] &= out["RSI 14"].le(max_rsi).fillna(False)
    if min_volume_ratio > 0:
        out["Technical Pass"] &= out["Volume/20D"].ge(min_volume_ratio).fillna(False)

    out = out.sort_values(["Technical Pass", "Above 200 EMA", "RSI 14"], ascending=[False, False, False])

    st.subheader("V4.1 Results")
    passed = out[out["Technical Pass"]].copy()
    st.success(f"{len(passed)} stocks passed the Screener fundamentals + selected technical filters.")

    display_cols = [c for c in [
        "Company", "NSE Symbol", "Price", "200 EMA", "Above 200 EMA",
        "RSI 14", "Volume/20D", "Technical Pass"
    ] if c in out.columns]
    st.dataframe(out[display_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download V4 results CSV",
        out.to_csv(index=False).encode(),
        "smart_stock_scanner_v4_results.csv",
        "text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Why V4.1 is different")
    st.write(
        "Fundamental qualification is delegated to the Screener screen instead of silently "
        "rejecting stocks because Yahoo returned missing ROE/growth/debt fields. Yahoo is only "
        "used for technical confirmation. If a source is unavailable, V4 reports the failure "
        "instead of inventing or imputing a fundamental value."
    )

st.caption(
    "Research tool only — not investment advice. Screener data is subject to its own update schedule; "
    "Yahoo Finance can be incomplete or rate-limited. Verify final fundamentals independently."
)
