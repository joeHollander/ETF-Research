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

def _get_near_roll_date(row, DAYS_BEFORE_EXPIRY):
    year = row["contract_year"]
    month = row["contract_month"]
    # Get all business days in the contract month
    # Using a fixed day like 28 to be safe for all months
    start_of_month = pd.to_datetime(f"{year}-{month:02d}-01").tz_localize('America/Chicago')
    end_of_month = start_of_month + pd.offsets.MonthEnd(0)
    bdays = pd.bdate_range(start=start_of_month, end=end_of_month)
    # The roll date is 3 days before the 3rd to last business day
    roll_date = bdays[-3] - pd.DateOffset(days=DAYS_BEFORE_EXPIRY)
    if roll_date.weekday() == 5: # If it's a Saturday or Sunday
        roll_date = roll_date - pd.Timedelta(days=1)
    return roll_date


def combine_features(df: pd.DataFrame, near_roll: bool = False, DAYS_BEFORE_EXPIRY: int = 3) -> pd.DataFrame:
    # filter for vanilla futures contracts
    single_contract_filter = (df["symbol"].str.len() == 4)
    fdf = df[single_contract_filter].copy()

    # adding necessary data
    fdf["contract_year"] = fdf.apply(_get_contract_year, axis=1)
    fdf["contract_month"] = fdf.apply(lambda row: FUTURE_MONTH_MAP[row['symbol'][2]], axis=1)
    fdf["expiry_length"] = fdf.apply(_get_expiry_length, axis=1)

    if near_roll:
        fdf['contract_roll_date'] = fdf.apply(lambda x: _get_near_roll_date(x, DAYS_BEFORE_EXPIRY), axis=1)

    return fdf

def near_roll(df: pd.DataFrame, DAYS_BEFORE_EXPIRY: int = 3,verbose: bool = False) -> pd.DataFrame:
    # split into front month and second month
    frdf = df[df["expiry_length"] == 0].copy()
    sdf = df[df["expiry_length"] == 1].copy()

    # split into pre roll front month and post roll second month
    adj_frdf = frdf[frdf.index < frdf['contract_roll_date']].copy()

    # need to align because second month df doesn't have first month expiration date info
    common_idx = frdf.index.intersection(sdf.index)
    sdf.loc[common_idx, 'fm_roll_date'] = frdf.loc[common_idx, 'contract_roll_date'].copy()
    adj_sdf = sdf[sdf.index >= sdf['fm_roll_date']].copy()

    # combine adjusted dataframes
    ndf = pd.concat([adj_frdf, adj_sdf], axis=0).sort_index()

    # determine roll spread 
    is_roll_day = ndf.index.date == ndf['contract_roll_date'].dt.date if 'contract_roll_date' in ndf.columns else np.zeros(len(ndf), dtype=bool)

    # adj for roll
    adjustment = np.where(
        is_roll_day,
        ndf['open'].shift(-1) - ndf['close'],
        0
    )

    ndf['adjustment'] = np.where(np.isnan(adjustment), 0, adjustment)
    total_adjustment = ndf['adjustment'].iloc[::-1].cumsum().iloc[::-1].shift(-1)
    total_adjustment = np.where(np.isnan(total_adjustment), 0, total_adjustment)

    cols_to_adjust = ['open', 'high', 'low', 'close']
    ndf[cols_to_adjust] = ndf[cols_to_adjust].add(total_adjustment, axis=0)
    

    return ndf

def generic_roll(df: pd.DataFrame, length: int, verbose: bool = False) -> pd.DataFrame:
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
    df["ts_event"] = pd.to_datetime(df["ts_event"]).dt.tz_convert('America/Chicago')
    df = df.set_index('ts_event', inplace=False)

    detailed_df = combine_features(df, False, 3)

    for i in range(1, 13):
        test = generic_roll(detailed_df, i, False)
        if i == 1:
            compare = test.copy()
            print(test.iloc[-35:-25])
        default = detailed_df[(detailed_df['expiry_length'] == i) | (detailed_df['expiry_length'] == i + 1)]
        print(i)
        if not test.empty:
            print(test.index.nunique(), "/", default.index.nunique())
            print(test.index[0], test.index[-1])
            print(len(set(test.index).intersection(set(compare.index))))
            print("*"*60)
        else:
            print("test is empty")
