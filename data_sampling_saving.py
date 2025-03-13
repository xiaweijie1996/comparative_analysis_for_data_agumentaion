"""
This script provides functions for sampling time-series data with a 30-minute resolution, ensuring that only complete
days (48 samples per day) are retained before selection. The sampling can be performed in two ways:

Continuous Sampling: Selects the last n% of full days.
Distributed Sampling: Selects n% of full days evenly throughout the dataset.

Script uses train set from the dsets/ folder. It then creates a dsets/percentage subfolder to save resulting csv.
"""

import os
import loguru
import pandas as pd


def drop_incomplete_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops days that have fewer than 48 samples (assuming 30-minute resolution).

    :param df: Pandas DataFrame with a DatetimeIndex
    :return: Filtered DataFrame with only full days
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)

    # Count samples per day
    daily_counts = df.resample('D').count()

    # Identify full days (48 samples)
    full_days = daily_counts[daily_counts.iloc[:, 0] == 48].index

    loguru.logger.info(f"Number of full days: {full_days.shape[0]} of total {daily_counts.shape[0]} days.")
    # Filter original DataFrame to keep only full days
    return df[df.index.normalize().isin(full_days)]


def sampling(df: pd.DataFrame, mode: str = 'continuous', n_percent: int = 10) -> pd.DataFrame:
    """

    :param n_percent: % of full days to select from df
    :param df:
    :param mode: continuous or distributed. if continuous, last n% of days are selected.
                 if distributed, number of days representing n% of dataset is selected from different parts to cover
                 it evenly
    :return: shortened df
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)

    if n_percent > 1:
        fraction = n_percent / 100
    elif n_percent < 0:
        raise ValueError(f"n must be positive, provided {n_percent}")
    else:
        fraction = n_percent

    if mode == 'continuous':
        df = select_last_n_days(df, fraction)

    elif mode == 'distributed':
        df = select_every_nth_day(df, fraction)
    else:
        raise Exception("mode should be either 'continuous' or 'distributed'")

    return df


def select_last_n_days(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    # Get unique days from the index
    unique_days = df.index.normalize().unique()
    n_full_last_days = int(len(unique_days) * fraction)

    # Select the last N days
    last_n_days = unique_days[-n_full_last_days:]
    loguru.logger.info(f"{int(fraction * 100)} percent is {len(last_n_days)} days.")

    # Filter DataFrame to include only the selected full days
    filtered_df = df[df.index.normalize().isin(last_n_days)].copy()
    return filtered_df


def select_every_nth_day(df: pd.DataFrame, fraction: float) -> pd.DataFrame:

    # Compute k as the inverse of n%
    if fraction > 0.5 and fraction < 1:
        # drop every kth day
        k = int(100 / (100-int(fraction * 100)))
        unique_days = df.index.normalize().unique()
        selected_days = unique_days[::k]
        loguru.logger.info(f"{int(fraction * 100)} percent is {len(unique_days)-len(selected_days)} days.")
        filtered_df = df[~df.index.normalize().isin(selected_days)]

    else:
        # select every kth day
        k = int(100 / int(fraction * 100))
        unique_days = df.index.normalize().unique()
        selected_days = unique_days[::k]

        # reassure +- equal amount of days in continuous and distributed modes
        max_len = int(len(unique_days)*fraction)
        selected_days = selected_days[:max_len]
        loguru.logger.info(f"{int(fraction * 100)} percent is {len(selected_days)} days.")

        # Filter the DataFrame to include only the selected full days
        filtered_df = df[df.index.normalize().isin(selected_days)]

    return filtered_df


if __name__ == '__main__':

    orig_df = pd.read_csv(os.path.join('dsets', 'train_set_wind.csv'), index_col='Unnamed: 0')
    orig_df = drop_incomplete_days(orig_df)

    for mode in ['continuous', 'distributed']:
        loguru.logger.info(f"Preparing {mode} sets")
        for n_percent in [5, 10, 30, 50, 80, 100]:
            save_path = os.path.join('dsets', 'percentage', mode)
            os.makedirs(save_path, exist_ok=True)
            smaller_df = sampling(orig_df, mode, n_percent)
            smaller_df.to_csv(os.path.join(save_path, f'{n_percent}percent_dataset.csv'))
        loguru.logger.info("Done!")
