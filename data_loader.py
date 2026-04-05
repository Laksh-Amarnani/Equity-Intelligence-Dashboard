import yfinance as yf
import pandas as pd
import numpy as np

SECTORS = {
    "Banking":  ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "IT":       ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "FMCG":     ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"],
    "Pharma":   ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS"],
    "Energy":   ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS"],
}

ALL_TICKERS = [t for tickers in SECTORS.values() for t in tickers]


def _extract_close(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Robustly extract Close prices regardless of yfinance column structure.
    Handles flat columns, (field, ticker) MultiIndex, and (ticker, field) MultiIndex.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        # flat columns — single ticker download
        col = next((c for c in raw.columns if str(c).lower() == "close"), None)
        if col is None:
            raise ValueError(f"No Close column found. Got: {list(raw.columns)}")
        prices = raw[[col]].copy()
        prices.columns = tickers
        return prices

    lvl0 = [str(c).lower() for c in raw.columns.get_level_values(0)]
    lvl1 = [str(c).lower() for c in raw.columns.get_level_values(1)]

    if "close" in lvl0:
        prices = raw.xs("Close", axis=1, level=0)
    elif "close" in lvl1:
        prices = raw.xs("Close", axis=1, level=1)
    else:
        raise ValueError(f"Cannot find Close in MultiIndex columns: {raw.columns.tolist()}")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        prices.columns = tickers

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)

    if len(tickers) == 1 and len(prices.columns) == 1:
        prices.columns = tickers

    return prices


def get_price_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    """
    Download adjusted closing prices for given tickers.
    Returns a DataFrame with dates as index, tickers as columns.
    """
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = _extract_close(raw, tickers)
    prices.dropna(how="all", inplace=True)
    return prices


def get_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from price DataFrame."""
    return prices.pct_change().dropna()


def get_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns — more useful for statistical tests."""
    return np.log(prices / prices.shift(1)).dropna()


def get_sector_returns(period: str = "2y") -> dict:
    """
    Returns a dict: { sector_name -> pd.Series of average daily returns }
    Useful for ANOVA and cross-sector comparisons.
    """
    sector_avg = {}
    for sector, tickers in SECTORS.items():
        prices  = get_price_data(tickers, period=period)
        returns = get_daily_returns(prices)
        sector_avg[sector] = returns.mean(axis=1)
    return sector_avg


def get_stock_info(ticker: str) -> dict:
    """Fetch basic metadata for a stock."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":       info.get("longName", ticker),
            "sector":     info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio":   info.get("trailingPE", "N/A"),
            "52w_high":   info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low":    info.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "market_cap": "N/A",
                "pe_ratio": "N/A", "52w_high": "N/A", "52w_low": "N/A"}


def engineer_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Build feature matrix for ML models (used by Laksh).
    Features: lagged returns, rolling stats, RSI, volume ratio.
    Target: next-day direction (1 = Up, 0 = Down)
    """
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    df = pd.DataFrame(index=raw.index)

    if isinstance(raw.columns, pd.MultiIndex):
        df["close"]  = raw.xs("Close",  axis=1, level=0).squeeze()
        df["volume"] = raw.xs("Volume", axis=1, level=0).squeeze()
    else:
        df["close"]  = raw["Close"].squeeze()
        df["volume"] = raw["Volume"].squeeze()

    df["return"]     = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    for lag in [1, 2, 3, 5]:
        df[f"lag_{lag}"] = df["return"].shift(lag)

    df["rolling_mean_5"]  = df["return"].rolling(5).mean()
    df["rolling_std_5"]   = df["return"].rolling(5).std()
    df["rolling_mean_20"] = df["return"].rolling(20).mean()
    df["rolling_std_20"]  = df["return"].rolling(20).std()

    delta     = df["close"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    df["ma_20"]     = df["close"].rolling(20).mean()
    df["ma_50"]     = df["close"].rolling(50).mean()
    df["dist_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["dist_ma50"] = (df["close"] - df["ma_50"]) / df["ma_50"]

    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    return df