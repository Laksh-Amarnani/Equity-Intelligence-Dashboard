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
        prices = raw[["Close"]]
        prices.columns = tickers
    prices.dropna(how="all", inplace=True)
    return prices

def get_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:    
    return prices.pct_change().dropna()

def get_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def get_sector_returns(period: str = "2y") -> dict:
    sector_avg = {}
    for sector, tickers in SECTORS.items():
        prices = get_price_data(tickers, period=period)
        returns = get_daily_returns(prices)
        sector_avg[sector] = returns.mean(axis=1)
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