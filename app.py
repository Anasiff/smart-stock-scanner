import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Smart Stock Scanner V8", page_icon="📈", layout="wide")

DEFAULT_FUNDAMENTAL_URL = "https://www.screener.in/screens/3635525/1/"
DEFAULT_UNIVERSE_URL = "https://www.screener.in/screens/509570/all-companies/?order=desc"
DEFAULT_QUERY = (
    "Price to Earning < 30 AND Return on equity > 25 AND EPS > 0 AND "
    "Profit growth > 50 AND Sales growth > 50 AND Debt to equity < 0.5 AND "
    "Promoter holding > 50"
)

# ---- V8 forecast engine constants ----
LOOKAHEAD_DAYS = 5                 # forecast horizon (trading days)
MIN_TECH_BARS = 210                # bars required for a reliable 200 EMA
MIN_ANALOG_CANDIDATE_ROWS = 60     # minimum valid historical feature rows before we even try
MIN_VALID_ANALOGS = 10             # below this -> NO FORECAST (never fabricate a forecast from <10 samples)
FEATURE_COLS = [
    "dist20", "dist50", "dist200", "slope20", "slope50", "slope200",
    "rsi", "rsi_slope", "atr_pct", "ret5", "ret10", "ret20",
    "volratio", "pos20", "pos60", "vola20", "dist_high20", "dist_low20",
]

st.title("📈 Smart Stock Scanner V8 — Historical Analog Forecast Engine")
st.caption(
    "Complete NSE universe scan • exact 7 fundamental filters • full technical indicator suite • "
    "walk-forward, no-look-ahead 3–5 day historical analog forecast • Swing Conviction • Swing Call"
)

st.info(
    "V8 replaces the old weighted Swing Score with a historical-analog / walk-forward forecast engine. "
    "For every stock with enough history, today's technical setup is compared against its own past setups "
    "(no future data is ever used to build a historical snapshot), and the engine reports how often similar "
    "past setups actually went on to gain or lose over the next 5 trading days. This is a historical-pattern "
    "estimate, not a guaranteed prediction."
)

with st.sidebar:
    st.header("Universe")
    universe_size = st.number_input(
        "Minimum broad stocks to scan",
        min_value=100,
        max_value=5000,
        value=2000,
        step=100,
        help="V8 scans at least this many broad listed companies. Fundamental-qualified stocks are added too, and pagination continues until the source is exhausted or this minimum is met."
    )
    universe_url = st.text_input("Public Screener all-companies URL", value=DEFAULT_UNIVERSE_URL)
    st.caption("Default is a public all-listed-companies screen sorted by market cap descending.")
    st.divider()

    st.header("Fundamental source")
    fund_url = st.text_input("Public Screener fundamental screen URL", value=DEFAULT_FUNDAMENTAL_URL)
    st.caption("Default screen contains the exact 7 hard filters. This is used for the Fundamental Pass column only — it never removes a stock from the technical/forecast scan.")
    st.divider()

    st.header("Technical filters (optional, display only)")
    require_above_200 = st.checkbox("Price > 200 EMA (used for Technical Pass)", value=True)
    min_rsi = st.number_input("Minimum RSI (optional)", 0.0, 100.0, 0.0, 1.0)
    max_rsi = st.number_input("Maximum RSI (optional)", 0.0, 100.0, 100.0, 1.0)
    min_volume_ratio = st.number_input("Min volume / 20D avg (optional)", 0.0, 20.0, 0.0, 0.1)
    max_workers = st.number_input("Download workers", 1, 8, 4, 1, help="Capped at 4 internally regardless of this value, to avoid hammering Yahoo.")
    batch_size = st.number_input("Yahoo batch size", 25, 200, 100, 25)

    st.divider()
    st.header("Historical analog forecast")
    forecast_years = st.number_input("Historical lookback (years)", 2, 5, 3, 1)
    analog_count = st.number_input("Historical analogs (K)", 30, 100, 60, 10)
    forecast_limit = st.number_input(
        "Limit forecast to top-N stocks (0 = forecast entire universe)",
        min_value=0, max_value=5000, value=0, step=100,
        help="Default 0 forecasts every stock with sufficient history, as required. Set a limit only if you need a faster scan; limited stocks are still shown in the table, just without a forecast."
    )
    min_turnover = st.number_input(
        "Minimum 20D avg turnover (₹) for VERY STRONG liquidity gate", 0, 100000000, 5000000, 500000
    )

    st.divider()
    st.header("Diagnostics")
    st.caption("Price uses the latest same-day 15-minute Yahoo bar when available; all EMAs/RSI/ATR remain based on daily history. Yahoo may still be delayed.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


# =====================================================================
# 1. UNIVERSE / FUNDAMENTAL SOURCE LOADING (unchanged architecture)
# =====================================================================

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
def _get_with_retry(url, headers, timeout=25, max_retries=5, base_delay=2.0):
    """GET with exponential backoff on 429 / transient errors, honoring Retry-After when present.
    Screener.in rate-limits fast, tight-loop pagination — this is what was throwing
    '429 Client Error: Too Many Requests'."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else base_delay * (2 ** attempt)
                time.sleep(min(wait, 30))
                continue
            if r.status_code >= 500:
                time.sleep(base_delay * (2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(base_delay * (2 ** attempt))
    if last_exc:
        raise last_exc
    raise requests.exceptions.RequestException(f"Failed to fetch {url} after {max_retries} retries (rate limited).")


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
        if page > 1:
            time.sleep(1.2)  # pace requests so Screener.in doesn't rate-limit the scan
        r = _get_with_retry(page_url, HEADERS)
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
    # Continue pagination until the source is exhausted or the minimum is met — never stop artificially at 1000.
    pages = int(np.ceil(minimum / 25)) + 2
    return fetch_public_screen(url, max_pages=min(400, pages), stop_after=int(minimum))


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


# =====================================================================
# 2. BATCH DOWNLOAD (single long-history pull feeds BOTH technicals and
#    the forecast engine — no per-stock / per-analog web requests)
# =====================================================================

def _split_multiindex(data, chunk):
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in chunk:
            try:
                sub = data.xs(ticker, axis=1, level=1, drop_level=True)
            except Exception:
                try:
                    sub = data[ticker]
                except Exception:
                    continue
            out[ticker] = sub
    elif len(chunk) == 1:
        out[chunk[0]] = data
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def download_history_batch(tickers, period_years, batch_size, workers):
    """Download `period_years` of daily history for many tickers in small batches.
    2-4 workers max, retries with exponential backoff, graceful failure per chunk.
    This single download is reused for both current technical indicators and the
    full historical-analog forecast, so analogs never require their own web request.
    """
    tickers = list(dict.fromkeys(tickers))
    workers = max(1, min(int(workers), 4))
    chunks = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    out = {}

    def one(chunk):
        last_err = None
        for attempt in range(3):
            try:
                data = yf.download(
                    tickers=chunk, period=f"{int(period_years)}y", interval="1d",
                    auto_adjust=False, progress=False, threads=False, group_by="column",
                )
                if data is not None and not data.empty:
                    return chunk, data
            except Exception as e:
                last_err = e
            time.sleep((2 ** attempt) * 0.75)
        return chunk, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, c) for c in chunks]
        for fut in as_completed(futures):
            chunk, data = fut.result()
            if data is None or data.empty:
                continue
            out.update(_split_multiindex(data, chunk))
    return out


@st.cache_data(ttl=300, show_spinner=False)
def latest_intraday_prices(tickers, batch_size=50, workers=4):
    """Latest same-day intraday price + timestamp. Falls back silently to daily close
    (handled by the caller) when Yahoo has no intraday bars."""
    tickers = list(dict.fromkeys(tickers))
    workers = max(1, min(int(workers), 4))
    out = {}
    chunks = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    def one(chunk):
        try:
            data = yf.download(
                tickers=chunk, period="5d", interval="15m", auto_adjust=False,
                progress=False, threads=False, group_by="column", prepost=False
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
            split = _split_multiindex(data, chunk)
            for ticker, sub in split.items():
                if "Close" in sub.columns:
                    c = pd.to_numeric(sub["Close"], errors="coerce").dropna()
                    if not c.empty:
                        out[ticker] = (float(c.iloc[-1]), c.index[-1])
    return out


# =====================================================================
# 3. FEATURE ENGINEERING (technical indicators, computed once per stock
#    from the same downloaded frame, no look-ahead)
# =====================================================================

def normalize_ohlcv(hist, ticker):
    if hist is None or hist.empty:
        return None
    h = hist.copy()
    if isinstance(h.columns, pd.MultiIndex):
        try:
            h = h.xs(ticker, axis=1, level=-1, drop_level=True)
        except Exception:
            h.columns = h.columns.get_level_values(0)
    if "Close" not in h.columns:
        return None
    close = pd.to_numeric(h["Close"], errors="coerce")
    high = pd.to_numeric(h.get("High", close), errors="coerce")
    low = pd.to_numeric(h.get("Low", close), errors="coerce")
    volume = pd.to_numeric(h.get("Volume", pd.Series(index=h.index, dtype=float)), errors="coerce")
    frame = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume}).dropna(subset=["Close"])
    if frame.empty:
        return None
    return frame


def compute_feature_frame(frame):
    """All technical indicators, computed with data available ONLY up to each row's own date.
    Every rolling/ewm/diff/shift operation here is causal by construction, so this frame can be
    safely used both for 'today' and for constructing historical analog snapshots."""
    close, high, low, volume = frame["Close"], frame["High"], frame["Low"], frame["Volume"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    dist20 = (close / ema20 - 1) * 100
    dist50 = (close / ema50 - 1) * 100
    dist200 = (close / ema200 - 1) * 100

    slope20 = ema20.pct_change(5) * 100
    slope50 = ema50.pct_change(10) * 100
    slope200 = ema200.pct_change(20) * 100

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_slope = rsi.diff(5)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_pct = (atr / close) * 100

    vol_avg20 = volume.rolling(20).mean()
    volratio = volume / vol_avg20.replace(0, np.nan)
    turnover = close * volume
    turnover_avg20 = turnover.rolling(20).mean()

    ret5 = close.pct_change(5) * 100
    ret10 = close.pct_change(10) * 100
    ret20 = close.pct_change(20) * 100

    daily_ret = close.pct_change()
    vola5 = daily_ret.rolling(5).std() * 100
    vola20 = daily_ret.rolling(20).std() * 100

    high20 = high.rolling(20).max()
    low20 = low.rolling(20).min()
    pos20 = ((close - low20) / (high20 - low20).replace(0, np.nan)) * 100
    dist_high20 = (close / high20 - 1) * 100
    dist_low20 = (close / low20 - 1) * 100

    high60 = high.rolling(60).max()
    low60 = low.rolling(60).min()
    pos60 = ((close - low60) / (high60 - low60).replace(0, np.nan)) * 100

    close_strength = ((close - low) / (high - low).replace(0, np.nan)) * 100

    feats = pd.DataFrame({
        "close": close, "ema20": ema20, "ema50": ema50, "ema200": ema200,
        "dist20": dist20, "dist50": dist50, "dist200": dist200,
        "slope20": slope20, "slope50": slope50, "slope200": slope200,
        "rsi": rsi, "rsi_slope": rsi_slope,
        "atr": atr, "atr_pct": atr_pct,
        "vol_avg20": vol_avg20, "volratio": volratio, "turnover_avg20": turnover_avg20,
        "ret5": ret5, "ret10": ret10, "ret20": ret20,
        "vola5": vola5, "vola20": vola20,
        "pos20": pos20, "pos60": pos60,
        "dist_high20": dist_high20, "dist_low20": dist_low20,
        "close_strength": close_strength,
    }, index=frame.index)
    return feats


def rsi_trend_label(slope):
    if pd.isna(slope):
        return "N/A"
    if slope > 1:
        return "Rising"
    if slope < -1:
        return "Falling"
    return "Flat"


def ema_slope_label(slope):
    if pd.isna(slope):
        return "N/A"
    if slope > 0.2:
        return "Up"
    if slope < -0.2:
        return "Down"
    return "Flat"


# =====================================================================
# 4. HISTORICAL ANALOG / WALK-FORWARD FORECAST ENGINE (no look-ahead)
# =====================================================================

def historical_analog_forecast(frame, feats, analog_k, min_turnover_rs):
    """Compare today's feature snapshot against the stock's own historical setups and
    measure what ACTUALLY happened next. No future information is used to build the
    current snapshot, and no future information is used to select or weight analogs —
    only the standardized distance between past and present feature values."""
    n_rows = len(frame)
    if n_rows < MIN_TECH_BARS:
        return {"Forecast Status": "NO DATA", "Historical Analog Count": 0, "Forecast Confidence": "INSUFFICIENT"}

    # The most recent LOOKAHEAD_DAYS rows cannot be used as analog *sources* because their
    # forward-looking outcome window is not yet complete.
    usable_end = n_rows - LOOKAHEAD_DAYS - 1
    if usable_end < MIN_ANALOG_CANDIDATE_ROWS:
        return {"Forecast Status": "NO FORECAST", "Historical Analog Count": 0, "Forecast Confidence": "INSUFFICIENT"}

    current = feats.iloc[-1]
    valid_current = current[FEATURE_COLS].notna()
    if valid_current.sum() < 6:
        return {"Forecast Status": "NO FORECAST", "Historical Analog Count": 0, "Forecast Confidence": "INSUFFICIENT"}

    candidates = feats.iloc[:usable_end][FEATURE_COLS]
    candidates = candidates.loc[candidates.notna().sum(axis=1) >= 6]
    if len(candidates) < MIN_ANALOG_CANDIDATE_ROWS:
        return {"Forecast Status": "NO FORECAST", "Historical Analog Count": 0, "Forecast Confidence": "INSUFFICIENT"}

    stds = candidates.std(ddof=0)
    used_cols = [c for c in FEATURE_COLS if pd.notna(current[c]) and pd.notna(stds.get(c)) and stds.get(c, 0) > 1e-9]
    if len(used_cols) < 6:
        return {"Forecast Status": "NO FORECAST", "Historical Analog Count": 0, "Forecast Confidence": "INSUFFICIENT"}

    z = pd.DataFrame(index=candidates.index)
    for c in used_cols:
        z[c] = (candidates[c] - current[c]).abs() / stds[c]
    distance = z.mean(axis=1, skipna=True)
    distance = distance.dropna()
    if distance.empty:
        return {"Forecast Status": "NO FORECAST", "Historical Analog Count": 0, "Forecast Confidence": "INSUFFICIENT"}

    nearest = distance.sort_values().head(int(analog_k))
    avg_similarity_distance = float(nearest.mean())

    mfe_list, mae_list, end_ret_list = [], [], []
    for idx in nearest.index:
        pos = frame.index.get_loc(idx)
        entry = float(frame["Close"].iloc[pos])
        if entry <= 0:
            continue
        end_pos = min(pos + 1 + LOOKAHEAD_DAYS, n_rows)
        if end_pos <= pos + 1:
            continue
        window_high = frame["High"].iloc[pos + 1:end_pos].dropna()
        window_low = frame["Low"].iloc[pos + 1:end_pos].dropna()
        window_close = frame["Close"].iloc[pos + 1:end_pos].dropna()
        if window_high.empty or window_low.empty or window_close.empty:
            continue
        mfe_list.append(float(window_high.max() / entry - 1) * 100)
        mae_list.append(float(window_low.min() / entry - 1) * 100)
        end_ret_list.append(float(window_close.iloc[-1] / entry - 1) * 100)

    n = len(mfe_list)
    if n < MIN_VALID_ANALOGS:
        return {"Forecast Status": "NO FORECAST", "Historical Analog Count": n, "Forecast Confidence": "INSUFFICIENT"}

    mfe = pd.Series(mfe_list)
    mae = pd.Series(mae_list)
    end_ret = pd.Series(end_ret_list)

    p3 = float((mfe >= 3).mean() * 100)
    p5 = float((mfe >= 5).mean() * 100)
    p10 = float((mfe >= 10).mean() * 100)
    pm3 = float((mae <= -3).mean() * 100)
    pm5 = float((mae <= -5).mean() * 100)

    median_mfe = float(mfe.median())
    q75_mfe = float(mfe.quantile(0.75))
    median_mae = float(mae.median())
    q75_abs_mae = float(mae.abs().quantile(0.75))

    positive_mfe = mfe[mfe > 0]
    expected_upside = float(positive_mfe.median()) if len(positive_mfe) >= 5 else max(median_mfe, 0.0)
    expected_downside = float(mae.abs().median())
    expected_return = float(end_ret.median())

    ratio = expected_upside / expected_downside if expected_downside > 1e-9 else np.nan

    # Confidence: sample count + similarity quality + outcome consistency.
    iqr_mfe = float(mfe.quantile(0.75) - mfe.quantile(0.25))
    consistent = iqr_mfe <= (abs(median_mfe) * 3 + 6)
    good_similarity = avg_similarity_distance <= 1.0

    if n >= 50 and good_similarity and consistent:
        confidence = "HIGH"
    elif n >= 25:
        confidence = "MEDIUM"
    elif n >= MIN_VALID_ANALOGS:
        confidence = "LOW"
    else:
        confidence = "INSUFFICIENT"

    return {
        "Forecast Status": "OK",
        "Forecast Horizon": "5D",
        "Expected 5D Upside %": round(expected_upside, 2),
        "Expected 5D Downside %": round(expected_downside, 2),
        "Expected 5D Return %": round(expected_return, 2),
        "P(+3%) 5D": round(p3, 1),
        "P(+5%) 5D": round(p5, 1),
        "P(+10%) 5D": round(p10, 1),
        "P(-3%) 5D": round(pm3, 1),
        "P(-5%) 5D": round(pm5, 1),
        "Median MFE 5D": round(median_mfe, 2),
        "75th Percentile MFE 5D": round(q75_mfe, 2),
        "Median MAE 5D": round(median_mae, 2),
        "75th Percentile MAE 5D": round(q75_abs_mae, 2),
        "Upside/Downside Ratio": round(ratio, 2) if pd.notna(ratio) else np.nan,
        "Historical Analog Count": n,
        "Forecast Confidence": confidence,
        "Analog Similarity (avg z-distance)": round(avg_similarity_distance, 3),
    }


def historical_forecast_score(row):
    """0-100 score built ONLY from the historical forecast statistics above — never a
    manually re-weighted technical score. A stock cannot score high just because it is
    far above its 200 EMA, has high RSI, or has a big 5D return; it must show an actual
    historical forward edge (high P(+5%), low P(-5%), real expected upside, real R:R)."""
    if row.get("Forecast Status") != "OK":
        return np.nan
    p5 = row.get("P(+5%) 5D", np.nan)
    p10 = row.get("P(+10%) 5D", np.nan)
    pm5 = row.get("P(-5%) 5D", np.nan)
    eu = row.get("Expected 5D Upside %", np.nan)
    rr = row.get("Upside/Downside Ratio", np.nan)
    n = row.get("Historical Analog Count", 0)
    if any(pd.isna(x) for x in [p5, p10, pm5, eu]):
        return np.nan

    s_p5 = min(100.0, max(0.0, p5))
    s_p10 = min(100.0, max(0.0, p10 * 1.5))
    s_down = min(100.0, max(0.0, 100.0 - pm5 * 2))
    s_upside = min(100.0, max(0.0, eu * 8))
    s_rr = min(100.0, max(0.0, (rr - 0.5) * 40)) if pd.notna(rr) else 0.0

    raw = 0.25 * s_p5 + 0.15 * s_p10 + 0.25 * s_down + 0.20 * s_upside + 0.15 * s_rr

    conf_mult = {"HIGH": 1.0, "MEDIUM": 0.80, "LOW": 0.55}.get(row.get("Forecast Confidence"), 0.35)
    sample_mult = min(1.0, n / 50.0) if n else 0.0
    score = raw * conf_mult * (0.6 + 0.4 * sample_mult)
    return round(min(100.0, max(0.0, score)), 1)


def classify_conviction(row, min_turnover_rs):
    if row.get("Technical Status") != "OK":
        return "⚫ NO DATA"
    if row.get("Forecast Status") != "OK":
        return "⚪ NO FORECAST"

    p5 = row.get("P(+5%) 5D", np.nan)
    pm5 = row.get("P(-5%) 5D", np.nan)
    eu = row.get("Expected 5D Upside %", np.nan)
    ed = row.get("Expected 5D Downside %", np.nan)
    rr = row.get("Upside/Downside Ratio", np.nan)
    n = row.get("Historical Analog Count", 0)
    conf = row.get("Forecast Confidence")
    above200 = bool(row.get("Above 200 EMA", False))
    above20 = bool(row.get("Above 20 EMA", False))
    turnover = row.get("20D Avg Turnover", np.nan)
    liquidity_ok = pd.notna(turnover) and turnover >= min_turnover_rs

    if any(pd.isna(x) for x in [p5, pm5, eu, ed]):
        return "⚪ NO FORECAST"

    # Conservative hard AVOID rules override everything else.
    if pm5 >= 60 or (pd.notna(rr) and ed >= eu) or (not above200 and conf == "LOW"):
        return "🔴 AVOID"

    if (conf == "HIGH" and p5 >= 65 and pm5 <= 25 and eu >= 6 and pd.notna(rr) and rr >= 1.8
            and above200 and above20 and liquidity_ok):
        return "🟢 VERY STRONG"

    if (conf in ("HIGH", "MEDIUM") and p5 >= 55 and pm5 <= 30 and eu >= 4
            and pd.notna(rr) and rr >= 1.4 and above200):
        return "🟢 STRONG"

    if p5 >= 45 and pm5 <= 40 and eu > 0 and n >= 25:
        return "🟡 MODERATE"

    if p5 >= 35 and eu > 0:
        return "🟡 WATCH"

    return "🔴 WEAK"


def swing_call_sentence(row):
    conv = row.get("Swing Conviction", "⚪ NO FORECAST")
    if conv == "⚫ NO DATA":
        return "⚫ NO DATA — insufficient Yahoo price history"
    if conv == "⚪ NO FORECAST":
        return "⚪ NO FORECAST — insufficient historical analogs"

    eu = row.get("Expected 5D Upside %", np.nan)
    ed = row.get("Expected 5D Downside %", np.nan)
    p5 = row.get("P(+5%) 5D", np.nan)
    pm5 = row.get("P(-5%) 5D", np.nan)
    if any(pd.isna(x) for x in [eu, ed, p5, pm5]):
        return "⚪ NO FORECAST — insufficient historical analogs"

    if conv in ("🟢 VERY STRONG", "🟢 STRONG"):
        head = "🟢 BUY BIAS"
    elif conv in ("🟡 MODERATE", "🟡 WATCH"):
        head = "🟡 WATCH"
    else:
        head = "🔴 AVOID"
    return f"{head} — Expected +{eu:.1f}% in 5D | Downside -{ed:.1f}% | P(+5%) {p5:.0f}% | P(-5%) {pm5:.0f}%"


def investment_priority(row):
    conv = row.get("Swing Conviction", "⚪ NO FORECAST")
    fp = bool(row.get("Fundamental Pass", False))
    if conv == "🟢 VERY STRONG":
        return (0, "1. VERY STRONG + Fund PASS") if fp else (2, "3. VERY STRONG + Fund FAIL")
    if conv == "🟢 STRONG":
        return (1, "2. STRONG + Fund PASS") if fp else (3, "4. STRONG + Fund FAIL")
    if conv == "🟡 MODERATE":
        return (4, "5. MODERATE + Fund PASS") if fp else (4, "5. MODERATE")
    if conv == "🟡 WATCH":
        return (5, "6. WATCH")
    if conv == "🔴 WEAK":
        return (6, "7. WEAK")
    if conv == "🔴 AVOID":
        return (7, "8. AVOID")
    if conv == "⚪ NO FORECAST":
        return (8, "9. NO FORECAST")
    return (9, "10. NO DATA")


# =====================================================================
# 5. PER-STOCK PIPELINE: one download feeds technicals + forecast
# =====================================================================

def build_row(sym, ticker, hist, intraday, analog_k, min_turnover_rs):
    frame = normalize_ohlcv(hist, ticker)
    base = {
        "NSE Symbol": sym, "Yahoo Ticker": ticker,
        "Exchange": "BSE" if ticker.endswith(".BO") else "NSE",
    }
    if frame is None or frame.empty:
        base.update({
            "Price": np.nan, "Historical Close": np.nan, "Price Source": "N/A", "Price Data Date": None,
            "Technical Status": "No Yahoo price data",
        })
        return base

    feats = compute_feature_frame(frame)
    historical_close = float(frame["Close"].iloc[-1])
    hist_date = frame.index[-1]

    intraday_price, intraday_ts = (np.nan, None)
    if intraday is not None:
        intraday_price, intraday_ts = intraday
    if pd.notna(intraday_price) and intraday_price > 0:
        last_price = float(intraday_price)
        price_source = "Intraday"
        price_date = intraday_ts
    else:
        last_price = historical_close
        price_source = "Delayed / Daily Close"
        price_date = hist_date

    base.update({
        "Price": last_price, "Historical Close": historical_close,
        "Price Source": price_source, "Price Data Date": price_date,
    })

    if len(frame) < MIN_TECH_BARS:
        base["Technical Status"] = "Insufficient history"
        return base

    cur = feats.iloc[-1]
    ema20, ema50, ema200 = float(cur["ema20"]), float(cur["ema50"]), float(cur["ema200"])
    base.update({
        "20 EMA": ema20, "50 EMA": ema50, "200 EMA": ema200,
        "Dist 20 EMA %": (last_price / ema20 - 1) * 100 if ema20 else np.nan,
        "Dist 50 EMA %": (last_price / ema50 - 1) * 100 if ema50 else np.nan,
        "Dist 200 EMA %": (last_price / ema200 - 1) * 100 if ema200 else np.nan,
        "EMA Distance %": (last_price / ema200 - 1) * 100 if ema200 else np.nan,  # legacy alias
        "Above 20 EMA": bool(last_price > ema20) if ema20 else False,
        "Above 50 EMA": bool(last_price > ema50) if ema50 else False,
        "Above 200 EMA": bool(last_price > ema200) if ema200 else False,
        "20 EMA Slope": float(cur["slope20"]) if pd.notna(cur["slope20"]) else np.nan,
        "50 EMA Slope": float(cur["slope50"]) if pd.notna(cur["slope50"]) else np.nan,
        "200 EMA Slope": float(cur["slope200"]) if pd.notna(cur["slope200"]) else np.nan,
        "20 EMA Trend": ema_slope_label(cur["slope20"]),
        "50 EMA Trend": ema_slope_label(cur["slope50"]),
        "200 EMA Trend": ema_slope_label(cur["slope200"]),
        "RSI 14": float(cur["rsi"]) if pd.notna(cur["rsi"]) else np.nan,
        "RSI Trend": rsi_trend_label(cur["rsi_slope"]),
        "ATR 14": float(cur["atr"]) if pd.notna(cur["atr"]) else np.nan,
        "ATR %": float(cur["atr_pct"]) if pd.notna(cur["atr_pct"]) else np.nan,
        "20D Avg Volume": float(cur["vol_avg20"]) if pd.notna(cur["vol_avg20"]) else np.nan,
        "Volume/20D": float(cur["volratio"]) if pd.notna(cur["volratio"]) else np.nan,
        "20D Avg Turnover": float(cur["turnover_avg20"]) if pd.notna(cur["turnover_avg20"]) else np.nan,
        "5D %": float(cur["ret5"]) if pd.notna(cur["ret5"]) else np.nan,
        "20D %": float(cur["ret20"]) if pd.notna(cur["ret20"]) else np.nan,
        "5D Volatility %": float(cur["vola5"]) if pd.notna(cur["vola5"]) else np.nan,
        "20D Volatility %": float(cur["vola20"]) if pd.notna(cur["vola20"]) else np.nan,
        "Position in 20D Range %": float(cur["pos20"]) if pd.notna(cur["pos20"]) else np.nan,
        "Position in 60D Range %": float(cur["pos60"]) if pd.notna(cur["pos60"]) else np.nan,
        "Dist from 20D High %": float(cur["dist_high20"]) if pd.notna(cur["dist_high20"]) else np.nan,
        "Dist from 20D Low %": float(cur["dist_low20"]) if pd.notna(cur["dist_low20"]) else np.nan,
        "Close Strength %": float(cur["close_strength"]) if pd.notna(cur["close_strength"]) else np.nan,
        "Technical Status": "OK",
    })

    forecast = historical_analog_forecast(frame, feats, analog_k, min_turnover_rs)
    base.update(forecast)

    atr14 = base.get("ATR 14", np.nan)
    if pd.notna(atr14) and last_price > 0:
        stop = last_price - 1.5 * atr14
        base["ATR Stop Distance %"] = round((1.5 * atr14 / last_price) * 100, 2)
        base["Suggested Stop Loss"] = round(stop, 2)
    eu = base.get("Expected 5D Upside %", np.nan)
    atr_pct = base.get("ATR %", np.nan)
    if pd.notna(eu) and pd.notna(atr_pct) and atr_pct > 1e-9:
        base["Expected Reward / ATR Risk"] = round(eu / atr_pct, 2)

    return base


def run_full_scan(symbols, years, batch_size, workers, analog_k, forecast_limit, min_turnover_rs, progress_cb=None):
    symbols = [str(x).strip().upper() for x in symbols]
    ticker_map = {sym: yahoo_candidates(sym)[0] for sym in symbols}
    first_tickers = list(ticker_map.values())

    hist_map = download_history_batch(first_tickers, years, batch_size, workers)
    intraday_map = latest_intraday_prices(first_tickers, max(25, min(int(batch_size), 75)), workers)

    rows = {}
    fallback_needed = []
    for sym in symbols:
        t = ticker_map[sym]
        if t in hist_map:
            rows[sym] = (t, hist_map[t], intraday_map.get(t))
        elif not sym.isdigit() and not t.endswith(".BO"):
            fallback_needed.append(sym)

    if fallback_needed:
        fallback_tickers = [s + ".BO" for s in fallback_needed]
        fb_hist = download_history_batch(fallback_tickers, years, batch_size, workers)
        fb_intraday = latest_intraday_prices(fallback_tickers, max(25, min(int(batch_size), 75)), workers)
        for sym in fallback_needed:
            t = sym + ".BO"
            if t in fb_hist:
                rows[sym] = (t, fb_hist[t], fb_intraday.get(t))

    # Decide which symbols actually get the (relatively expensive) analog forecast.
    forecast_set = set(rows.keys())
    if forecast_limit and forecast_limit > 0 and forecast_limit < len(rows):
        # Priority: most recent bars available first is arbitrary/fair since fundamentals
        # haven't been merged yet at this stage — simply forecast the first N discovered so
        # the limit is deterministic; nothing is silently dropped from the output table.
        forecast_set = set(list(rows.keys())[: int(forecast_limit)])

    records = []
    total = len(symbols)
    done = 0
    for sym in symbols:
        done += 1
        if progress_cb and done % 50 == 0:
            progress_cb(done, total)
        if sym in rows:
            t, hist, intraday = rows[sym]
            if sym in forecast_set:
                rec = build_row(sym, t, hist, intraday, analog_k, min_turnover_rs)
            else:
                rec = build_row(sym, t, hist, intraday, 0, min_turnover_rs)
                if rec.get("Technical Status") == "OK":
                    rec["Forecast Status"] = "NO FORECAST"
                    rec["Historical Analog Count"] = 0
                    rec["Forecast Confidence"] = "NOT SELECTED"
        else:
            t = ticker_map[sym]
            rec = {
                "NSE Symbol": sym, "Yahoo Ticker": t,
                "Exchange": "BSE" if t.endswith(".BO") else "NSE",
                "Price": np.nan, "Historical Close": np.nan, "Price Source": "N/A", "Price Data Date": None,
                "Technical Status": "No Yahoo price data",
            }
        records.append(rec)

    df = pd.DataFrame(records)
    for col in ["Forecast Status", "Historical Analog Count", "Forecast Confidence"]:
        if col not in df.columns:
            df[col] = np.nan
    df["Forecast Status"] = df["Forecast Status"].fillna("NO DATA")
    df["Historical Analog Count"] = df["Historical Analog Count"].fillna(0)
    df["Forecast Confidence"] = df["Forecast Confidence"].fillna("INSUFFICIENT")
    return df


# =====================================================================
# 6. LOAD SOURCES
# =====================================================================
st.subheader("1. Build complete NSE universe")
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

# Every fundamental candidate is checked even if it falls outside the broad universe list.
combined = pd.concat([universe, fundamentals], ignore_index=True)
combined["NSE Symbol"] = combined["NSE Symbol"].astype(str).str.upper().str.strip()
combined = combined[combined["NSE Symbol"].ne("") & combined["NSE Symbol"].ne("NAN")].drop_duplicates("NSE Symbol")
fund_syms = set(fundamentals["NSE Symbol"].astype(str).str.upper().str.strip())
combined["Fundamental Pass"] = combined["NSE Symbol"].isin(fund_syms)

universe_discovered = len(combined)
st.write(f"**Complete technical/forecast universe: {universe_discovered} stocks** (minimum broad universe {int(universe_size)} + any fundamental-qualified additions).")
st.caption("Every stock discovered here stays in the final table — missing data is marked NO DATA, it is never dropped or treated as a pass.")

if st.button("🚀 RUN V8 — FULL NSE HISTORICAL FORECAST SCAN", type="primary", use_container_width=True):
    prog = st.progress(0.0, text="Downloading price history...")

    def _cb(done, total):
        prog.progress(min(1.0, done / max(1, total)), text=f"Processing {done}/{total} stocks...")

    with st.spinner(f"Downloading {int(forecast_years)}y history and building technical + historical-analog forecasts for {universe_discovered} stocks..."):
        out = run_full_scan(
            combined["NSE Symbol"].tolist(), int(forecast_years), int(batch_size), int(max_workers),
            int(analog_count), int(forecast_limit), float(min_turnover),
            progress_cb=_cb,
        )
    prog.empty()

    out = combined.merge(out, on="NSE Symbol", how="left", suffixes=("", "_r"))
    out["Technical Status"] = out["Technical Status"].fillna("No Yahoo price data")

    # Requirement: ensure every dataframe column exists before sorting/display, even in the
    # edge case where not a single stock produced a given field this run.
    _ensure_cols = [
        "Price", "Historical Close", "Price Source", "Price Data Date",
        "20 EMA", "50 EMA", "200 EMA", "Dist 20 EMA %", "Dist 50 EMA %", "Dist 200 EMA %", "EMA Distance %",
        "Above 20 EMA", "Above 50 EMA", "Above 200 EMA",
        "20 EMA Slope", "50 EMA Slope", "200 EMA Slope", "20 EMA Trend", "50 EMA Trend", "200 EMA Trend",
        "RSI 14", "RSI Trend", "ATR 14", "ATR %", "20D Avg Volume", "Volume/20D", "20D Avg Turnover",
        "5D %", "20D %", "5D Volatility %", "20D Volatility %",
        "Position in 20D Range %", "Position in 60D Range %", "Dist from 20D High %", "Dist from 20D Low %",
        "Close Strength %", "Forecast Horizon",
        "Expected 5D Upside %", "Expected 5D Downside %", "Expected 5D Return %",
        "P(+3%) 5D", "P(+5%) 5D", "P(+10%) 5D", "P(-3%) 5D", "P(-5%) 5D",
        "Median MFE 5D", "75th Percentile MFE 5D", "Median MAE 5D", "75th Percentile MAE 5D",
        "Upside/Downside Ratio", "Historical Analog Count", "Forecast Confidence", "Forecast Status",
        "Analog Similarity (avg z-distance)", "ATR Stop Distance %", "Suggested Stop Loss",
        "Expected Reward / ATR Risk", "Technical Status",
    ]
    for _c in _ensure_cols:
        if _c not in out.columns:
            out[_c] = np.nan
    out["Forecast Status"] = out["Forecast Status"].fillna("NO DATA")
    out["Historical Analog Count"] = out["Historical Analog Count"].fillna(0)
    out["Forecast Confidence"] = out["Forecast Confidence"].fillna("INSUFFICIENT")

    out["Technical Pass"] = out["Technical Status"].eq("OK")
    if require_above_200:
        out["Technical Pass"] &= out["Above 200 EMA"].fillna(False)
    if min_rsi > 0:
        out["Technical Pass"] &= out["RSI 14"].ge(min_rsi).fillna(False)
    if max_rsi < 100:
        out["Technical Pass"] &= out["RSI 14"].le(max_rsi).fillna(False)
    if min_volume_ratio > 0:
        out["Technical Pass"] &= out["Volume/20D"].ge(min_volume_ratio).fillna(False)

    out["Final Pass"] = out["Fundamental Pass"] & out["Technical Pass"]

    out["Historical Forecast Score"] = out.apply(historical_forecast_score, axis=1)
    out["Swing Conviction"] = out.apply(lambda r: classify_conviction(r, float(min_turnover)), axis=1)
    out["Swing Call"] = out.apply(swing_call_sentence, axis=1)
    pr = out.apply(investment_priority, axis=1)
    out["Investment Priority Rank"] = pr.apply(lambda x: x[0])
    out["Investment Priority"] = pr.apply(lambda x: x[1])

    out = out.sort_values(
        ["Investment Priority Rank", "Historical Forecast Score", "P(+5%) 5D", "Upside/Downside Ratio", "Expected 5D Upside %"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )

    # ---------------- Diagnostics ----------------
    tech_ok = int(out["Technical Status"].eq("OK").sum())
    tech_missing = universe_discovered - tech_ok
    forecast_calc = int(out["Forecast Status"].eq("OK").sum())
    forecast_unavailable = universe_discovered - forecast_calc
    fund_count = int(out["Fundamental Pass"].sum())
    final_count = int(out["Final Pass"].sum())
    conv_counts = out["Swing Conviction"].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("NSE stocks scanned", universe_discovered)
    c2.metric("Forecasts available", forecast_calc)
    c3.metric("🟢 VERY STRONG", int(conv_counts.get("🟢 VERY STRONG", 0)))
    c4.metric("🟢 STRONG", int(conv_counts.get("🟢 STRONG", 0)))
    c5.metric("Fundamental pass", fund_count)

    st.subheader("TOP SWING CANDIDATES")
    top_cols = [c for c in [
        "Company", "NSE Symbol", "Fundamental Pass", "Price",
        "Expected 5D Upside %", "Expected 5D Downside %", "P(+5%) 5D", "P(-5%) 5D",
        "Upside/Downside Ratio", "Historical Forecast Score", "Forecast Confidence",
        "Swing Conviction", "Swing Call", "Suggested Stop Loss",
    ] if c in out.columns]
    top_table = out[out["Swing Conviction"].isin(["🟢 VERY STRONG", "🟢 STRONG", "🟡 MODERATE"])].copy()
    if top_table.empty:
        st.info("No stocks currently meet the VERY STRONG / STRONG / MODERATE conviction bar. Showing the best-ranked candidates instead.")
        top_table = out.head(50)
    st.dataframe(top_table[top_cols].head(200), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("FULL NSE SCAN")
    display_cols = [c for c in [
        "Company", "NSE Symbol", "Exchange", "Yahoo Ticker", "Fundamental Pass", "Technical Pass", "Final Pass",
        "Price", "Historical Close", "Price Source", "Price Data Date",
        "200 EMA", "50 EMA", "20 EMA", "Dist 20 EMA %", "Dist 50 EMA %", "Dist 200 EMA %",
        "20 EMA Trend", "50 EMA Trend", "200 EMA Trend",
        "Above 20 EMA", "Above 50 EMA", "Above 200 EMA",
        "RSI 14", "RSI Trend", "ATR 14", "ATR %",
        "20D Avg Volume", "Volume/20D", "20D Avg Turnover",
        "5D %", "20D %", "5D Volatility %", "20D Volatility %",
        "Position in 20D Range %", "Position in 60D Range %",
        "Dist from 20D High %", "Dist from 20D Low %", "Close Strength %",
        "Forecast Horizon", "Expected 5D Upside %", "Expected 5D Downside %", "Expected 5D Return %",
        "P(+3%) 5D", "P(+5%) 5D", "P(+10%) 5D", "P(-3%) 5D", "P(-5%) 5D",
        "Median MFE 5D", "75th Percentile MFE 5D", "Median MAE 5D", "75th Percentile MAE 5D",
        "Upside/Downside Ratio", "Historical Analog Count", "Forecast Confidence", "Forecast Status",
        "Historical Forecast Score", "Swing Conviction", "Swing Call",
        "ATR Stop Distance %", "Suggested Stop Loss", "Expected Reward / ATR Risk",
        "Investment Priority", "Technical Status",
    ] if c in out.columns]

    st.dataframe(out[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download V8 full scan CSV",
        out.to_csv(index=False).encode(),
        "smart_stock_scanner_v8_full_scan.csv",
        "text/csv",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Data quality / diagnostics")
    st.code(
        f"Universe discovered:        {universe_discovered}\n"
        f"Technical data available:   {tech_ok}\n"
        f"Technical data missing:     {tech_missing}\n"
        f"Forecast calculated:        {forecast_calc}\n"
        f"Forecast unavailable:       {forecast_unavailable}\n"
        f"Fundamental-qualified:      {fund_count}\n"
        f"Technical Pass:             {int(out['Technical Pass'].sum())}\n"
        f"Final strong candidates:    {final_count}\n"
        f"VERY STRONG:                {int(conv_counts.get('🟢 VERY STRONG', 0))}\n"
        f"STRONG:                     {int(conv_counts.get('🟢 STRONG', 0))}\n"
        f"MODERATE:                   {int(conv_counts.get('🟡 MODERATE', 0))}\n"
        f"WATCH:                      {int(conv_counts.get('🟡 WATCH', 0))}\n"
        f"WEAK:                       {int(conv_counts.get('🔴 WEAK', 0))}\n"
        f"AVOID:                      {int(conv_counts.get('🔴 AVOID', 0))}\n"
        f"NO FORECAST:                {int(conv_counts.get('⚪ NO FORECAST', 0))}\n"
        f"NO DATA:                    {int(conv_counts.get('⚫ NO DATA', 0))}\n"
    )
    if tech_missing:
        st.warning(f"{tech_missing} stocks had insufficient/missing Yahoo price history. They remain in the table marked NO DATA and are NEVER treated as a pass.")

    st.divider()
    st.subheader("How the V8 forecast engine works")
    st.write(
        "1) Build the complete broad NSE universe plus every exact-7-filter fundamental candidate — nothing is capped at 1000. "
        "2) Download daily OHLCV history once per stock (2–4 workers, batched, retried with backoff) and reuse it for both "
        "current indicators and the forecast — no per-analog web requests. "
        "3) Compute 20/50/200 EMA, Wilder RSI(14), Wilder ATR(14), slopes, volatility, range position and more, causally "
        "(each value only ever uses data up to its own date). "
        "4) For every stock with enough history, build a standardized feature snapshot for today and find its K nearest "
        "historical analogs using only data available on or before each historical date. "
        "5) For each analog, measure what ACTUALLY happened over the next 3–5 trading days (max favorable/adverse excursion, "
        "end-of-window return) — this is the walk-forward step, and it never uses information from after 'today'. "
        "6) Aggregate analogs into real historical hit-rates (P(+3/5/10%), P(-3/5%)) and robust expected upside/downside, "
        "then classify Forecast Confidence from sample size, analog similarity, and outcome consistency. "
        "7) Historical Forecast Score, Swing Conviction and Swing Call are all derived only from these historical statistics — "
        "never from raw technicals alone, and never from fewer than 10 valid analogs."
    )

st.warning(
    "This is a historical-pattern forecast, not a guaranteed prediction. Probabilities represent historical hit rates "
    "among comparable past setups. Market conditions can change and past performance does not guarantee future results."
)

st.caption(
    "Research tool only — not investment advice. Screener does not provide a public developer API; this app reads public screen pages. "
    "Yahoo Finance data can be incomplete or rate-limited. Verify final fundamentals and price data independently."
)
