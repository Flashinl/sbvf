"""
S&P 500 Stock Tickers

Comprehensive list for training on diverse stocks across all sectors
"""

# S&P 500 tickers (500 stocks across all sectors)
SP500_TICKERS = [
    # Technology (Large Cap)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'TSLA', 'ADBE', 'CRM',
    'CSCO', 'ACN', 'AMD', 'INTC', 'ORCL', 'IBM', 'QCOM', 'TXN', 'INTU', 'AMAT',
    'ADI', 'NOW', 'PANW', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',

    # Communication Services
    'NFLX', 'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR', 'EA', 'NWSA', 'PARA',
    'OMC', 'MTCH', 'IPG', 'TTWO', 'LYV', 'FOX', 'FOXA',

    # Consumer Discretionary
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
    'MAR', 'ABNB', 'GM', 'F', 'HLT', 'ORLY', 'YUM', 'ROST', 'DHI', 'LEN',
    'AZO', 'GRMN', 'EBAY', 'POOL', 'TPR', 'BBY', 'DRI', 'ULTA', 'GPC', 'DPZ',

    # Consumer Staples
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
    'GIS', 'STZ', 'SYY', 'KHC', 'HSY', 'K', 'CHD', 'TSN', 'CAG', 'CPB',
    'MKC', 'SJM', 'HRL', 'TAP', 'LW',

    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'WMB', 'OKE',
    'KMI', 'HES', 'BKR', 'TRGP', 'HAL', 'DVN', 'FANG', 'MRO', 'APA', 'OXY',

    # Financials
    'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'SPGI', 'BLK',
    'C', 'AXP', 'SCHW', 'CB', 'MMC', 'PGR', 'AON', 'USB', 'TFC', 'PNC',
    'BK', 'AIG', 'AFL', 'MET', 'PRU', 'ALL', 'TRV', 'CME', 'MCO', 'ICE',
    'COF', 'AJG', 'MSCI', 'DFS', 'WRB', 'FITB', 'STT', 'TROW', 'BEN', 'IVZ',

    # Healthcare
    'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'AMGN',
    'ISRG', 'BSX', 'VRTX', 'SYK', 'GILD', 'MDT', 'BMY', 'CVS', 'CI', 'REGN',
    'ZTS', 'MCK', 'HCA', 'ELV', 'COR', 'HUM', 'BDX', 'IDXX', 'EW', 'A',
    'IQV', 'RMD', 'DXCM', 'MTD', 'STE', 'ALGN', 'HOLX', 'WAT', 'TECH', 'VTRS',

    # Industrials
    'CAT', 'RTX', 'HON', 'UNP', 'BA', 'GE', 'ADP', 'LMT', 'UPS', 'DE',
    'MMM', 'TT', 'ETN', 'PH', 'WM', 'GD', 'CTAS', 'NOC', 'ITW', 'EMR',
    'FDX', 'CSX', 'NSC', 'PCAR', 'JCI', 'CMI', 'CARR', 'PAYX', 'OTIS', 'RSG',
    'URI', 'ROK', 'ODFL', 'IR', 'FAST', 'VRSK', 'PWR', 'AME', 'DOV', 'HUBB',

    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'CTVA', 'DD', 'NUE', 'VMC',
    'MLM', 'BALL', 'AVY', 'AMCR', 'PKG', 'IP', 'STLD', 'CF', 'MOS', 'ALB',

    # Real Estate
    'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'SPG',
    'EQR', 'AVB', 'VTR', 'VICI', 'ARE', 'CBRE', 'EXR', 'MAA', 'INVH', 'ESS',

    # Utilities
    'NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'VST', 'D', 'PEG', 'EXC',
    'XEL', 'ED', 'EIX', 'WEC', 'AWK', 'DTE', 'ES', 'FE', 'AEE', 'CMS',

    # Additional high-volume stocks for better training
    'PYPL', 'SHOP', 'SQ', 'COIN', 'RIVN', 'LCID', 'HOOD', 'SOFI', 'PLTR', 'RBLX',
    'U', 'SNOW', 'DDOG', 'NET', 'ZS', 'CRWD', 'OKTA', 'DOCU', 'TWLO', 'ZM',
    'ROKU', 'PINS', 'SNAP', 'SPOT', 'UBER', 'LYFT', 'DASH', 'ABNB', 'AIRBNB',
]

# Top 100 most liquid stocks (faster training, good results)
TOP_100_TICKERS = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',

    # Large-cap tech
    'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'IBM', 'QCOM', 'TXN', 'INTU', 'NOW',

    # Communication
    'NFLX', 'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR', 'EA',

    # Consumer
    'WMT', 'HD', 'MCD', 'NKE', 'COST', 'PG', 'KO', 'PEP', 'SBUX', 'LOW',

    # Financials
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SPGI',

    # Healthcare
    'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',

    # Industrials
    'CAT', 'RTX', 'HON', 'UNP', 'BA', 'GE', 'UPS', 'DE', 'LMT', 'MMM',

    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',

    # Retail & E-commerce
    'AMZN', 'TJX', 'BKNG', 'MAR', 'ABNB', 'EBAY', 'ROST', 'BBY',

    # Growth stocks
    'SHOP', 'SQ', 'PYPL', 'RBLX', 'COIN', 'PLTR', 'SNOW', 'CRWD', 'NET', 'DDOG',
]

# Quick test set (for rapid iteration)
TEST_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
    'JPM', 'BAC', 'WMT', 'HD', 'UNH', 'JNJ', 'XOM', 'CVX'
]


def get_ticker_list(size: str = 'medium') -> list:
    """
    Get ticker list by size

    Args:
        size: 'small' (16), 'medium' (100), 'large' (500+)

    Returns:
        List of ticker symbols
    """
    if size == 'small':
        return TEST_TICKERS
    elif size == 'medium':
        return TOP_100_TICKERS
    elif size == 'large':
        return SP500_TICKERS
    else:
        return TOP_100_TICKERS
