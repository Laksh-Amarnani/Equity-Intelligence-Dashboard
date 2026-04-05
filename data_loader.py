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


def get_price_data(tickers: list, period: str = "2y") -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw["Close"].to_frame()
        prices.columns = tickers
    prices.dropna(how="all", inplace=True)
    return prices


def get_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from price DataFrame."""
    return prices.pct_change().dropna()


def get_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns — more useful for statistical tests."""
    return np.log(prices / prices.shift(1)).dropna()


def get_sector_returns(period: str = "2y") -> dict:
    sector_avg = {}
    for sector, tickers in SECTORS.items():
        prices = get_price_data(tickers, period=period)
        returns = get_daily_returns(prices)
        sector_avg[sector] = returns.mean(axis=1)   # avg return across sector
    return sector_avg


def get_stock_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":        info.get("longName", ticker),
            "sector":      info.get("sector", "N/A"),
            "market_cap":  info.get("marketCap", "N/A"),
            "pe_ratio":    info.get("trailingPE", "N/A"),
            "52w_high":    info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low":     info.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception:
        return {}


def engineer_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    prices = get_price_data([ticker], period=period)
    df = pd.DataFrame()
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df["close"] = raw["Close"].squeeze()
    df["volume"] = raw["Volume"].squeeze()

    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Lagged returns
    for lag in [1, 2, 3, 5]:
        df[f"lag_{lag}"] = df["return"].shift(lag)

    # Rolling statistics
    df["rolling_mean_5"]  = df["return"].rolling(5).mean()
    df["rolling_std_5"]   = df["return"].rolling(5).std()
    df["rolling_mean_20"] = df["return"].rolling(20).mean()
    df["rolling_std_20"]  = df["return"].rolling(20).std()

    # RSI (14-day)
    delta     = df["close"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Price distance from moving averages
    df["ma_20"]       = df["close"].rolling(20).mean()
    df["ma_50"]       = df["close"].rolling(50).mean()
    df["dist_ma20"]   = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["dist_ma50"]   = (df["close"] - df["ma_50"]) / df["ma_50"]

    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    return df