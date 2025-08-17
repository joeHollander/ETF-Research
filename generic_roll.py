import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

FUTURE_MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

def _get_contract_year(row):
    contract_year_first_digit = int(row["symbol"][3])
    curr_year = row.name.year
    if contract_year_first_digit < curr_year % 10:
        contract_year = math.ceil(curr_year / 10) * 10 + contract_year_first_digit
    else:
        contract_year = math.floor(curr_year / 10) * 10 + contract_year_first_digit
    return contract_year

def _get_expiry_length(row):
    month = FUTURE_MONTH_MAP[row['symbol'][2]]
    year = row["contract_year"]
    # Calculate the difference in months
    return (year - row.name.year) * 12 + month - row.name.month

def combine_features(df: pd.DataFrame) -> pd.DataFrame:
    # filter for vanilla futures contracts
    single_contract_filter = (df["symbol"].str.len() == 4)
    fdf = df[single_contract_filter].copy()

    # adding necessary data
    fdf["contract_year"] = fdf.apply(_get_contract_year, axis=1)
    fdf["contract_month"] = fdf.apply(lambda row: FUTURE_MONTH_MAP[row['symbol'][2]], axis=1)
    fdf["expiry_length"] = fdf.apply(_get_expiry_length, axis=1)
    return fdf

def create_rolled_series(df: pd.DataFrame, length: int, verbose: bool = False) -> pd.DataFrame:
    """
    Creates a continuous, price-adjusted futures series by rolling contracts.
    """
    # filter for contracts with the target expiry length or the next one
    candidates = df[df['expiry_length'].isin([length, length + 1])].copy()
    
    # use a clean date column for daily selection
    candidates['date'] = pd.to_datetime(candidates.index.date)

    # for each day, prefer the contract with the shorter expiry length
    candidates.sort_values(by=['date', 'expiry_length'], inplace=True)
    ndf = candidates.drop_duplicates(subset='date', keep='first').set_index('date')

    if verbose:
        missing_days = ndf['close'].isna().sum()
        print(f"no valid contract found for {missing_days} business days before filling.")
    
    # identify the day before the contract roll to calculate the adjustment
    is_roll_day = ndf['symbol'] != ndf['symbol'].shift(-1)

    # calculate price adjustment using a forward-looking shift (-1)
    adjustment = np.where(
        is_roll_day,
        ndf['open'].shift(-1) - ndf['close'],
        0
    )
    ndf['adjustment'] = np.where(np.isnan(adjustment), 0, adjustment)
    
    # apply a reverse cumulative sum of adjustments to create a continuous series
    total_adjustment = ndf['adjustment'].iloc[::-1].cumsum().iloc[::-1]
    
    cols_to_adjust = ['open', 'high', 'low', 'close']
    ndf[cols_to_adjust] = ndf[cols_to_adjust].add(total_adjustment, axis=0)

    return ndf

if __name__ == "__main__":
    # sample usage
    df = pd.read_csv('Data/gold_futures_ohlcv.csv', parse_dates=['ts_event'])
    df["ts_event"] = pd.to_datetime(df["ts_event"]).dt.tz_convert('UTC') # ensure it's UTC
    df = df.set_index('ts_event', inplace=False)

    detailed_df = combine_features(df)

    for i in range(2, 13):
        gc = create_rolled_series(detailed_df, length=i, verbose=True)
        print(f"Number of missing values in 'close' column for LENGTH={i}: {gc['close'].isna().sum()}")