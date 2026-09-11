import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Smart Stock Scanner V6", page_icon="📈", layout="wide")

DEFAULT_FUNDAMENTAL_URL = "https://www.screener.in/screens/3635525/1/"
DEFAULT_UNIVERSE_URL = "https://www.screener.in/screens/509570/all-companies/?order=desc"
DEFAULT_QUERY = (
    "Price to Earning < 30 AND Return on equity > 25 AND EPS > 0 AND "
    "Profit growth > 50 AND Sales growth > 50 AND Debt to equity < 0.5 AND "
    "Promoter holding > 50"
)

st.title("📈 Smart Stock Scanner V6.2")
st.caption("1000+ stock technical scan • exact 7 fundamental filters • robust 5-factor Swing Score • BTST / 2–4 day swing research")

st.info(
    "V6 changes the scan order: it first builds a broad 1000+ stock universe and scans technical price history "
    "in bulk, then intersects those results with the exact 7 fundamental filters from Screener. This avoids "
    "the old 50-stock technical bottleneck. No fundamental values are invented."
)

with st.sidebar:
    st.header("Universe")
    universe_size = st.number_input(
        "Minimum broad stocks to scan",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="V6 scans at least this many broad listed companies for technical data. Fundamental-qualified stocks are added too."
    )
    universe_url = st.text_input("Public Screener all-companies URL", value=DEFAULT_UNIVERSE_URL)
    st.caption("Default is a public all-listed-companies screen sorted by market cap descending.")
    st.divider()

    st.header("Fundamental source")
    fund_url = st.text_input("Public Screener fundamental screen URL", value=DEFAULT_FUNDAMENTAL_URL)
    st.caption("Default screen contains the exact 7 hard filters. 200 EMA is NOT part of this source filter.")
    st.divider()

    st.header("Technical filters")
    require_above_200 = st.checkbox("Price > 200 EMA", value=True)
    min_rsi = st.number_input("Minimum RSI (optional)", 0.0, 100.0, 0.0, 1.0)
    max_rsi = st.number_input("Maximum RSI (optional)", 0.0, 100.0, 100.0, 1.0)
    min_volume_ratio = st.number_input("Min volume / 20D avg (optional)", 0.0, 20.0, 0.0, 0.1)
    max_workers = st.number_input("Technical download workers", 1, 8, 4, 1)
    batch_size = st.number_input("Yahoo batch size", 25, 200, 100, 25)

    st.divider()
    st.header("Diagnostics")
    st.caption("Price uses the latest same-day 15-minute Yahoo bar when available; 200 EMA remains based on daily history. Yahoo may still be delayed.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def set_page(url, page_no):
    p = urlparse(url)
    qs = parse_qs(p.query)
    if page_no <= 1:
        qs.pop("page", None)
    else:
        qs["page"] = [str(page_no)]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(qs, doseq=True), p.fragment))


def parse_company_rows(soup):
    rows = []
    table = soup.select_one("table.data-table")
    if table:
        for tr in table.select("tbody tr, tr"):
            a = tr.select_one('a[href*="/company/"]')
            if not a:
                continue
            href = a.get("href", "")
            m = re.search(r"/company/([^/?#]+)/?", href)
            if not m:
                continue
            symbol = m.group(1).strip().upper()
            name = a.get_text(" ", strip=True)
            if symbol:
                rows.append({"Company": name, "NSE Symbol": symbol, "Company URL": urljoin("https://www.screener.in", href)})
    if not rows:
        seen = set()
        for a in soup.select('a[href*="/company/"]'):
            href = a.get("href", "")
            m = re.search(r"/company/([^/?#]+)/?", href)
            if not m:
                continue
            symbol = m.group(1).strip().upper()
            name = a.get_text(" ", strip=True)
            if symbol and symbol not in seen:
                seen.add(symbol)
                rows.append({"Company": name, "NSE Symbol": symbol, "Company URL": urljoin("https://www.screener.in", href)})
    return rows


@st.cache_data(ttl=900, show_spinner=False)
def fetch_public_screen(url, max_pages=20, stop_after=None):
    if "screener.in/screens/" not in url:
        raise ValueError("Please provide a public Screener.in screen URL.")

    all_rows = []
    seen = set()
    title = "Screener screen"
    query = ""
    last_url = url

    for page in range(1, max_pages + 1):
        page_url = set_page(url, page)
        r = requests.get(page_url, headers=HEADERS, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        last_url = r.url
        if page == 1:
            title = soup.title.get_text(" ", strip=True) if soup.title else title
            qnode = soup.select_one(".query-text, .query")
            if qnode:
                query = qnode.get_text(" ", strip=True)
        page_rows = parse_company_rows(soup)
        if not page_rows:
            break
        added = 0
        for row in page_rows:
            sym = row["NSE Symbol"]
            if sym not in seen:
                seen.add(sym)
                all_rows.append(row)
                added += 1
        if added == 0:
            break
        if stop_after and len(all_rows) >= stop_after:
            all_rows = all_rows[:stop_after]
            break
        if len(page_rows) < 25:
            break

    if not all_rows:
        raise ValueError("No company rows were extracted from the public Screener page.")
    return title, query, pd.DataFrame(all_rows), last_url


@st.cache_data(ttl=900, show_spinner=False)
def load_universe(url, minimum):
    pages = int(np.ceil(minimum / 25)) + 2
    return fetch_public_screen(url, max_pages=min(220, pages), stop_after=int(minimum))


@st.cache_data(ttl=900, show_spinner=False)
def load_fundamentals(url):
    return fetch_public_screen(url, max_pages=20, stop_after=None)


def yahoo_candidates(symbol):
    s = str(symbol).strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return [s]
    if s.isdigit():
        return [s + ".BO"]
    return [s + ".NS", s + ".BO"]


def normalize_yahoo_frame(hist, ticker):
    if hist is None or hist.empty:
        return None
    h = hist.copy()
    if isinstance(h.columns, pd.MultiIndex):
        # Single ticker download may still have a multi-index.
        if ticker in h.columns.get_level_values(-1):
            try:
                h = h.xs(ticker, axis=1, level=-1)
            except Exception:
                h.columns = h.columns.get_level_values(0)
        else:
            h.columns = h.columns.get_level_values(0)
    if "Close" not in h.columns:
        return None
    close = pd.to_numeric(h["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    volume = pd.to_numeric(h.get("Volume", pd.Series(index=h.index, dtype=float)), errors="coerce")
    return close, volume


def calc_technical(sym, ticker, hist, current_price=np.nan):
    norm = normalize_yahoo_frame(hist, ticker)
    if norm is None:
        return None
    close, volume = norm
    historical_close = float(close.iloc[-1])
    last_price = float(current_price) if pd.notna(current_price) and float(current_price) > 0 else historical_close
    if len(close) < 210:
        return {
            "NSE Symbol": sym, "Yahoo Ticker": ticker,
            "Exchange": "BSE" if ticker.endswith(".BO") else "NSE",
            "Price": last_price, "Historical Close": historical_close, "200 EMA": np.nan, "EMA Distance %": np.nan,
            "Above 200 EMA": False, "RSI 14": np.nan, "Volume/20D": np.nan,
            "5D %": np.nan, "20D %": np.nan, "Technical Status": "Insufficient history",
        }

    ema200 = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    vol_avg = volume.rolling(20).mean()
    last_ema = float(ema200.iloc[-1])
    last_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan
    vr = float(volume.iloc[-1] / vol_avg.iloc[-1]) if pd.notna(vol_avg.iloc[-1]) and vol_avg.iloc[-1] else np.nan
    ret5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else np.nan
    ret20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else np.nan
    dist = float((last_price / last_ema - 1) * 100) if last_ema else np.nan
    return {
        "NSE Symbol": sym, "Yahoo Ticker": ticker,
        "Exchange": "BSE" if ticker.endswith(".BO") else "NSE",
        "Price": last_price, "Historical Close": historical_close, "200 EMA": last_ema, "EMA Distance %": dist,
        "Above 200 EMA": bool(last_price > last_ema), "RSI 14": last_rsi,
        "Volume/20D": vr, "5D %": ret5, "20D %": ret20, "Technical Status": "OK",
    }


def batch_download(tickers, batch_size, workers):
    """Download many Yahoo tickers in parallel batches. Returns {ticker: history_frame}."""
    chunks = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    output = {}

    def one(chunk):
        try:
            data = yf.download(
                tickers=chunk,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
            return chunk, data
        except Exception:
            return chunk, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, c) for c in chunks]
        for fut in as_completed(futures):
            chunk, data = fut.result()
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                level0 = set(data.columns.get_level_values(0))
                # yfinance normally returns PriceField x Ticker for multi-symbol downloads.
                for ticker in chunk:
                    try:
                        sub = data.xs(ticker, axis=1, level=1, drop_level=True)
                    except Exception:
                        try:
                            sub = data[ticker]
                        except Exception:
                            continue
                    output[ticker] = sub
            else:
                # Single-ticker fallback.
                output[chunk[0]] = data
    return output


@st.cache_data(ttl=300, show_spinner=False)
def latest_intraday_prices(tickers, batch_size=50, workers=4):
    """Get same-day/latest intraday price for display and EMA-distance calculations.
    Falls back to daily history when Yahoo has no intraday data.
    Yahoo quotes can still be delayed; this avoids using an old daily close when a newer
    same-day bar is available.
    """
    tickers = list(dict.fromkeys(tickers))
    out = {}
    chunks = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    def one(chunk):
        try:
            data = yf.download(
                tickers=chunk, period="5d", interval="15m", auto_adjust=False,
                progress=False, threads=True, group_by="column", prepost=False
            )
            return chunk, data
        except Exception:
            return chunk, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, c) for c in chunks]
        for fut in as_completed(futures):
            chunk, data = fut.result()
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in chunk:
                    try:
                        sub = data.xs(ticker, axis=1, level=1, drop_level=True)
                    except Exception:
                        try:
                            sub = data[ticker]
                        except Exception:
                            continue
                    if "Close" in sub.columns:
                        c = pd.to_numeric(sub["Close"], errors="coerce").dropna()
                        if not c.empty:
                            out[ticker] = float(c.iloc[-1])
            else:
                if len(chunk) == 1 and "Close" in data.columns:
                    c = pd.to_numeric(data["Close"], errors="coerce").dropna()
                    if not c.empty:
                        out[chunk[0]] = float(c.iloc[-1])
    return out


@st.cache_data(ttl=900, show_spinner=False)
def technical_scan(symbols, batch_size=100, workers=4):
    symbols = [str(x).strip().upper() for x in symbols]
    primary = {}
    fallback_needed = []
    ticker_to_symbol = {}

    # First pass: NSE alpha symbols use .NS; BSE numeric symbols use .BO.
    for sym in symbols:
        candidates = yahoo_candidates(sym)
        t = candidates[0]
        ticker_to_symbol[t] = sym
        primary[sym] = t

    first_tickers = list(ticker_to_symbol.keys())
    # Download in batches.
    hist_map = batch_download(first_tickers, batch_size, workers)
    intraday_map = latest_intraday_prices(first_tickers, max(25, min(int(batch_size), 75)), workers)
    results = []
    for sym in symbols:
        t = primary[sym]
        hist = hist_map.get(t)
        row = calc_technical(sym, t, hist, intraday_map.get(t)) if hist is not None else None
        if row is not None:
            results.append(row)
        elif not sym.isdigit() and not sym.endswith(".BO"):
            fallback_needed.append(sym)

    # Second pass only for missing NSE symbols: try .BO.
    if fallback_needed:
        fallback_tickers = [s + ".BO" for s in fallback_needed]
        fallback_map = batch_download(fallback_tickers, batch_size, workers)
        fallback_intraday = latest_intraday_prices(fallback_tickers, max(25, min(int(batch_size), 75)), workers)
        existing = {r["NSE Symbol"]: r for r in results}
        for sym in fallback_needed:
            t = sym + ".BO"
            row = calc_technical(sym, t, fallback_map.get(t), fallback_intraday.get(t)) if t in fallback_map else None
            if row is not None:
                existing[sym] = row
            elif sym not in existing:
                existing[sym] = {
                    "NSE Symbol": sym, "Yahoo Ticker": t,
                    "Exchange": "NSE/BSE fallback attempted", "Price": np.nan,
                    "200 EMA": np.nan, "EMA Distance %": np.nan, "Above 200 EMA": False,
                    "RSI 14": np.nan, "Volume/20D": np.nan, "5D %": np.nan,
                    "20D %": np.nan, "Technical Status": "No Yahoo price data",
                }
        results = list(existing.values())

    by_sym = {r["NSE Symbol"]: r for r in results}
    final = []
    for sym in symbols:
        if sym in by_sym:
            final.append(by_sym[sym])
        else:
            final.append({
                "NSE Symbol": sym, "Yahoo Ticker": yahoo_candidates(sym)[0],
                "Exchange": "BSE" if sym.isdigit() else "NSE",
                "Price": np.nan, "200 EMA": np.nan, "EMA Distance %": np.nan,
                "Above 200 EMA": False, "RSI 14": np.nan, "Volume/20D": np.nan,
                "5D %": np.nan, "20D %": np.nan, "Technical Status": "No Yahoo price data",
            })
    return pd.DataFrame(final)


def score_rows(out):
    """Calculate a conservative 0-100 technical swing score.

    Five components are scored: EMA trend, RSI, volume, 5D momentum and 20D
    momentum. A score is only considered valid when at least 4/5 components
    have real data; missing values are never invented and never produce a
    misleading perfect score through aggressive re-weighting.
    """
    def clip_score(x, lo, hi):
        if pd.isna(x):
            return np.nan
        return max(0.0, min(100.0, (x - lo) / (hi - lo) * 100.0))

    components = pd.DataFrame({
        "trend": out["EMA Distance %"].apply(lambda x: clip_score(x, 0, 20)),
        "rsi": out["RSI 14"].apply(
            lambda x: 100 - abs(x - 55) * 2 if pd.notna(x) else np.nan
        ).clip(lower=0, upper=100),
        "volume": out["Volume/20D"].apply(lambda x: clip_score(x, 0.7, 2.0)),
        "momentum5": out["5D %"].apply(lambda x: clip_score(x, -2, 8)),
        "momentum20": out["20D %"].apply(lambda x: clip_score(x, -5, 20)),
    }, index=out.index)

    # 30% trend, 20% RSI, 15% volume, 15% 5D momentum, 20% 20D momentum.
    weights = pd.Series({
        "trend": 0.30,
        "rsi": 0.20,
        "volume": 0.15,
        "momentum5": 0.15,
        "momentum20": 0.20,
    })

    coverage = components.notna().sum(axis=1)
    weighted = components.mul(weights, axis=1)
    available_weight = components.notna().mul(weights, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, min_count=1).div(
        available_weight.replace(0, np.nan)
    )

    # Do not publish a ranking score with fewer than 4 of 5 real components.
    score = score.where(coverage >= 4, np.nan)
    return score.round(1)


# --------------------------- LOAD SOURCES ---------------------------
st.subheader("1. Build broad universe")
try:
    u_title, u_query, universe, u_source_url = load_universe(universe_url, int(universe_size))
    st.success(f"Broad universe loaded: {len(universe)} stocks.")
except Exception as e:
    st.error(f"Broad universe could not be loaded: {e}")
    st.stop()

st.subheader("2. Load exact fundamental candidates")
try:
    f_title, f_query, fundamentals, f_source_url = load_fundamentals(fund_url)
    st.success(f"Fundamental-qualified candidates loaded: {len(fundamentals)} stocks.")
except Exception as e:
    st.error(f"Fundamental source could not be loaded: {e}")
    st.stop()

# Ensure every fundamental candidate is technically checked even if it falls outside the first N broad stocks.
combined = pd.concat([universe, fundamentals], ignore_index=True)
combined["NSE Symbol"] = combined["NSE Symbol"].astype(str).str.upper().str.strip()
combined = combined[combined["NSE Symbol"].ne("") & combined["NSE Symbol"].ne("NAN")].drop_duplicates("NSE Symbol")
fund_syms = set(fundamentals["NSE Symbol"].astype(str).str.upper().str.strip())
combined["Fundamental Pass"] = combined["NSE Symbol"].isin(fund_syms)

st.write(f"**Technical universe sent to Yahoo: {len(combined)} stocks** (minimum broad universe {int(universe_size)} + any fundamental-qualified additions).")
st.caption("This is the key V6 change: technical analysis is no longer limited to the 50 Screener-qualified names.")

if st.button("🚀 RUN V6.4 — SCAN 1000+ STOCKS", type="primary", use_container_width=True):
    with st.spinner(f"Scanning {len(combined)} stocks for price, 200 EMA, RSI, volume and momentum..."):
        tech = technical_scan(combined["NSE Symbol"].tolist(), int(batch_size), int(max_workers))

    out = combined.merge(tech, on="NSE Symbol", how="left")
    out["Technical Pass"] = out["Technical Status"].eq("OK")
    if require_above_200:
        out["Technical Pass"] &= out["Above 200 EMA"].fillna(False)
    if min_rsi > 0:
        out["Technical Pass"] &= out["RSI 14"].ge(min_rsi).fillna(False)
    if max_rsi < 100:
        out["Technical Pass"] &= out["RSI 14"].le(max_rsi).fillna(False)
    if min_volume_ratio > 0:
        out["Technical Pass"] &= out["Volume/20D"].ge(min_volume_ratio).fillna(False)

    out["Swing Score"] = score_rows(out)
    out["Score Coverage"] = out[["EMA Distance %", "RSI 14", "Volume/20D", "5D %", "20D %"]].notna().sum(axis=1)
    out["Final Pass"] = out["Fundamental Pass"] & out["Technical Pass"]

    # Final ranking is only for stocks satisfying the exact fundamental filters.
    out = out.sort_values(
        ["Final Pass", "Fundamental Pass", "Technical Pass", "Swing Score", "5D %"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )

    ok_tech = int(out["Technical Status"].eq("OK").sum())
    fund_count = int(out["Fundamental Pass"].sum())
    final_count = int(out["Final Pass"].sum())
    missing = int((~out["Technical Status"].eq("OK")).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", len(out))
    c2.metric("Technical data OK", ok_tech)
    c3.metric("Fundamental pass", fund_count)
    c4.metric("Final pass", final_count)

    st.subheader("V6.2 Final Ranked Results")
    st.success(f"{final_count} stocks passed all 7 fundamental filters + selected technical filters.")
    if missing:
        st.warning(f"{missing} stocks had insufficient/missing Yahoo price history. They are NOT treated as passes.")

    display_cols = [c for c in [
        "Company", "NSE Symbol", "Exchange", "Yahoo Ticker", "Fundamental Pass",
        "Price", "Historical Close", "200 EMA", "EMA Distance %", "Above 200 EMA", "RSI 14",
        "Volume/20D", "5D %", "20D %", "Swing Score", "Score Coverage", "Technical Status",
        "Technical Pass", "Final Pass"
    ] if c in out.columns]

    st.dataframe(out[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download V6.2 full scan CSV",
        out.to_csv(index=False).encode(),
        "smart_stock_scanner_v6_2_full_scan.csv",
        "text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("How V6 works")
    st.write(
        "1) Scan a broad 1000+ listed-stock universe for technical history. "
        "2) Add all stocks from the exact 7-filter Screener result so none of the fundamental-qualified names are lost. "
        "3) Calculate 200 EMA, RSI(14), volume/20D and momentum locally. "
        "4) Apply technical filters. "
        "5) Keep only the intersection of the exact fundamental screen and technical pass, then rank by Swing Score."
    )

st.caption(
    "Research tool only — not investment advice. Screener does not provide a public developer API; this app reads public screen pages. "
    "Yahoo Finance data can be incomplete or rate-limited. Verify final fundamentals and price data independently."
)
