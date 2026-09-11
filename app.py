
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

st.set_page_config(page_title="Smart Stock Scanner V5", page_icon="📈", layout="wide")

DEFAULT_SCREENER_URL = "https://www.screener.in/screens/3635525/1/"
DEFAULT_QUERY = (
    "Price to Earning < 30 AND Return on equity > 25 AND EPS > 0 AND "
    "Profit growth > 50 AND Sales growth > 50 AND Debt to equity < 0.5 AND "
    "Promoter holding > 50"
)

st.title("📈 Smart Stock Scanner V5")
st.caption("Screener.in fundamentals + Yahoo Finance technicals • BTST / 2–4 day swing research")

st.info(
    "V5 uses the exact 7 fundamental hard filters from a public Screener screen, then performs the 200 EMA/RSI/volume/momentum analysis itself. "
    "The Screener 200-DMA condition is intentionally removed so the app can calculate the required 200 EMA locally. "
    "NSE/BSE-aware Yahoo tickers are used for technical data; no fundamental value is invented or reconstructed from Yahoo."
)

with st.sidebar:
    st.header("Fundamental source")
    screener_url = st.text_input(
        "Public Screener.in screen URL",
        value=DEFAULT_SCREENER_URL,
        help="Paste any public Screener.in screen URL. The screen itself must contain your desired fundamental filters."
    )
    st.caption("Default screen uses your 7 fundamental hard filters. 200 EMA is calculated locally by V5.")
    st.divider()

    st.header("Technical filters")
    require_above_200 = st.checkbox("Price > 200 EMA", value=True)
    min_rsi = st.number_input("Minimum RSI (optional)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    max_rsi = st.number_input("Maximum RSI (optional)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    min_volume_ratio = st.number_input("Min volume / 20D avg (optional)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
    max_candidates = st.number_input("Max candidates for Yahoo technical check", min_value=1, max_value=100, value=100, step=1)

    st.divider()
    st.header("V5 source diagnostics")
    st.caption("Screener public pages do not provide an official API. V4 reads the public HTML page and parses the displayed result table. If the source blocks requests, use the CSV upload fallback below.")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_screener(url: str):
    """Fetch all public Screener result pages, not just page 1."""
    if "screener.in/screens/" not in url:
        raise ValueError("Please provide a public Screener.in screen URL.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def fetch_page(page_no):
        params = {} if page_no == 1 else {"page": page_no}
        r = requests.get(url, headers=headers, params=params, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return r, soup

    r, soup = fetch_page(1)
    title = soup.title.get_text(" ", strip=True) if soup.title else "Screener screen"
    query = ""
    qnode = soup.select_one(".query-text, .query")
    if qnode:
        query = qnode.get_text(" ", strip=True)

    def parse_rows(soup):
        rows = []
        table = soup.select_one("table.data-table")
        if table:
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
        return rows

    all_rows = []
    seen_symbols = set()
    max_pages = 20
    page = 1
    while page <= max_pages:
        if page == 1:
            page_rows = parse_rows(soup)
        else:
            rp, sp = fetch_page(page)
            page_rows = parse_rows(sp)
            if not page_rows:
                break
        added = 0
        for row in page_rows:
            sym = row["NSE Symbol"]
            if sym not in seen_symbols:
                seen_symbols.add(sym)
                all_rows.append(row)
                added += 1
        if added == 0:
            break
        page += 1

    if not all_rows:
        raise ValueError(
            "Screener result table was found, but company links could not be extracted. "
            "Use a public Screener URL or upload a Screener CSV."
        )

    return title, query, pd.DataFrame(all_rows), r.url

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

    def candidates_for(raw):
        s = str(raw).strip().upper()
        if s.endswith(".NS") or s.endswith(".BO"):
            return [s]
        # BSE scrip codes are numeric; NSE equity symbols are normally alpha.
        if s.isdigit():
            return [s + ".BO"]
        return [s + ".NS", s + ".BO"]

    def calc_from_history(hist, sym, yahoo_symbol):
        if hist is None or hist.empty:
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if "Close" not in hist.columns:
            return None

        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        volume = pd.to_numeric(
            hist.get("Volume", pd.Series(index=hist.index, dtype=float)),
            errors="coerce"
        )
        if len(close) < 210:
            return {
                "NSE Symbol": sym,
                "Yahoo Ticker": yahoo_symbol,
                "Exchange": "BSE" if yahoo_symbol.endswith(".BO") else "NSE",
                "Price": float(close.iloc[-1]) if len(close) else np.nan,
                "200 EMA": np.nan,
                "EMA Distance %": np.nan,
                "Above 200 EMA": False,
                "RSI 14": np.nan,
                "Volume/20D": np.nan,
                "5D %": np.nan,
                "20D %": np.nan,
                "Technical Status": "Insufficient history",
            }

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
        last_vr = (
            float(volume.iloc[-1] / vol_avg.iloc[-1])
            if pd.notna(vol_avg.iloc[-1]) and vol_avg.iloc[-1]
            else np.nan
        )
        ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else np.nan
        ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else np.nan
        ema_distance = float((last_price / last_ema - 1) * 100) if last_ema else np.nan

        return {
            "NSE Symbol": sym,
            "Yahoo Ticker": yahoo_symbol,
            "Exchange": "BSE" if yahoo_symbol.endswith(".BO") else "NSE",
            "Price": last_price,
            "200 EMA": last_ema,
            "EMA Distance %": ema_distance,
            "Above 200 EMA": last_price > last_ema,
            "RSI 14": last_rsi,
            "Volume/20D": last_vr,
            "5D %": ret_5d,
            "20D %": ret_20d,
            "Technical Status": "OK",
        }

    for sym in symbols:
        found = None
        attempted = []
        for yahoo_symbol in candidates_for(sym):
            attempted.append(yahoo_symbol)
            try:
                hist = yf.download(
                    yahoo_symbol,
                    period="1y",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False
                )
                found = calc_from_history(hist, sym, yahoo_symbol)
                if found is not None and found["Technical Status"] in ("OK", "Insufficient history"):
                    break
            except Exception:
                found = None
            time.sleep(0.08)

        if found is None:
            found = {
                "NSE Symbol": sym,
                "Yahoo Ticker": attempted[-1] if attempted else "",
                "Exchange": "BSE" if attempted and attempted[-1].endswith(".BO") else "NSE",
                "Price": np.nan,
                "200 EMA": np.nan,
                "EMA Distance %": np.nan,
                "Above 200 EMA": False,
                "RSI 14": np.nan,
                "Volume/20D": np.nan,
                "5D %": np.nan,
                "20D %": np.nan,
                "Technical Status": "No Yahoo price data",
            }
        results.append(found)

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

if st.button("🔎 RUN V4.2 SCAN", type="primary", use_container_width=True):
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

    # Transparent swing score: momentum + trend + participation, used only for ranking.
    def clip_score(x, lo, hi):
        if pd.isna(x):
            return np.nan
        return max(0.0, min(100.0, (x - lo) / (hi - lo) * 100.0))

    trend_score = out["EMA Distance %"].apply(lambda x: clip_score(x, 0, 20))
    rsi_score = out["RSI 14"].apply(lambda x: 100 - abs(x - 55) * 2 if pd.notna(x) else np.nan).clip(lower=0, upper=100)
    vol_score = out["Volume/20D"].apply(lambda x: clip_score(x, 0.7, 2.0))
    momentum_score = out["5D %"].apply(lambda x: clip_score(x, -2, 8))
    out["Swing Score"] = (0.35 * trend_score + 0.25 * rsi_score + 0.20 * vol_score + 0.20 * momentum_score).round(1)

    out = out.sort_values(["Technical Pass", "Swing Score", "5D %"], ascending=[False, False, False], na_position="last")

    st.subheader("V5 Results")
    passed = out[out["Technical Pass"]].copy()
    st.success(f"{len(passed)} stocks passed the 7 fundamental filters + selected technical filters.")

    display_cols = [c for c in [
        "Company", "NSE Symbol", "Exchange", "Yahoo Ticker",
        "Price", "200 EMA", "EMA Distance %", "Above 200 EMA",
        "RSI 14", "Volume/20D", "5D %", "20D %", "Swing Score", "Technical Pass"
    ] if c in out.columns]
    st.dataframe(out[display_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download V5 results CSV",
        out.to_csv(index=False).encode(),
        "smart_stock_scanner_v5_results.csv",
        "text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Why V5 is different")
    st.write(
        "The 7 fundamental hard filters are delegated to the public Screener screen. V5 deliberately "
        "removes Screener's DMA-200 condition because the app calculates the required 200 EMA itself. "
        "All Screener result pages are collected, including page 2/3/etc. Yahoo is only used for "
        "technical data. Missing source data is reported rather than invented."
    )

st.caption(
    "Research tool only — not investment advice. Screener data is subject to its own update schedule; "
    "Yahoo Finance can be incomplete or rate-limited. Verify final fundamentals independently."
)
