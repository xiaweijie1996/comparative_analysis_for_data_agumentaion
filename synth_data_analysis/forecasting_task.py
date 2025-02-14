"""
Simple forecasting task: predict variable for target_horizon steps ahead.
Uses LightGBM regressor.
Saves graphs and summary dataframes with predictions to a ../outputs_pres/ folder that is created automatically.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import random
import os

random.seed(22)
np.random.seed = 22


def return_real_df(path_dset: str = './data', solar: bool = False) -> pd.DataFrame:
    # read dataset and perform train-test split
    vertical_df = pd.read_csv(path_dset)
    vertical_df['Unnamed: 0'] = pd.to_datetime(vertical_df['Unnamed: 0'])

    cols_drop = [col for col in vertical_df.columns if
                 'calc' in col or 'lag' in col or 'lead' in col or 'rollmean' in col
                 or 'std' in col or 'PC' in col]

    vertical_df.drop(columns=['Unnamed: 0', 'year'] + cols_drop, inplace=True)

    if solar:
        cols_keep_sun = ['CloudCover_ref0_D1', 'SolarDownwardRadiation_ref0_D1', 'Temperature_ref0_D1',
                         'Solar_MWh', 'month', 'dayofyear', 'hour']
        keep = [col for col in vertical_df.columns if col in cols_keep_sun]
        vertical_df = vertical_df[keep]

    return vertical_df


def add_lags(df, target_column='Wind_MWh', lag_hours=24, min_lag_hours=6, time_step_per_hour=2):
    """
    Adds lagged features for all variables except the target variable.
    Lags cover the previous `lag_hours` hours but skip the last `min_lag_hours` hours.
    Assumes each row represents `time_step_per_hour` observations per hour.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The target variable to predict.
        lag_hours (int): Number of hours to create lags for.
        min_lag_hours (int): Minimum lag to include (skip recent `min_lag_hours`).
        time_step_per_hour (int): Number of timesteps per hour.

    Returns:
        pd.DataFrame: DataFrame with lagged features added.
    """

    # I assume that we can lag target variable
    features_to_lag = [col for col in df.columns if col not in ['month', 'hour', 'dayofyear', 'halfhour']]

    lags = [i for i in range(min_lag_hours * time_step_per_hour,
                             (lag_hours + min_lag_hours) * time_step_per_hour)
                             if i % 2 == 0]  # just to reduce number of columns

    for col in features_to_lag:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df.dropna()


def train_model(data: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42,
                params: dict = None,
                tuning: str = 'gen',
                solar: bool = False):
    """
    Trains a LightGBM model to forecast the target variable.

    Parameters:
    - data (pd.DataFrame): The processed dataframe with features and target.
    - test_size (float): The proportion of data to use for testing.
    - random_state (int): Random seed for reproducibility.
    - tuning (str): indicator on what dataset was used to tune the model. 'gen' is for synthetic data

    Returns:
    - model (LGBMRegressor): The trained LightGBM model.
    - pd.DataFrame: Dataframe containing true vs predicted values for test set.
    """

    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        shuffle=False)
    if params is None:
        params = {'subsample': 0.8, 'random_state': 22, 'num_leaves': 100, 'n_estimators': 50,
                  'max_depth': 3, 'learning_rate': 0.1, 'verbose': -1, 'colsample_bytree': 0.6}

        params_gen = {'subsample': 0.9, 'random_state': 22, 'num_leaves': 30,
                      'n_estimators': 200, 'max_depth': 13, 'learning_rate': 0.046, 'verbose': -1,
                      'colsample_bytree': 0.7}
        if tuning == 'gen':
            params = params_gen
        if solar:
            params = {'subsample': 0.9, 'random_state': 22, 'num_leaves': 20, 'n_estimators': 250,
                      'max_depth': 7, 'learning_rate': 0.02, 'verbose': -1, 'colsample_bytree': 0.9}
            params_gen = {'subsample': 0.89, 'random_state': 22, 'num_leaves': 130, 'n_estimators': 250,
                          'max_depth': 5, 'learning_rate': 0.046415888336127774, 'verbose': -1,
                          'colsample_bytree': 0.7}
            if tuning == 'gen':
                params = params_gen

    # Initialize and train the model
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    return model


def tune_lightgbm(data: pd.DataFrame, target: str, cv: int = 3):
    """
    Tunes the hyperparameters of a LightGBM model using GridSearchCV.

    Parameters:
    - data (pd.DataFrame): The processed dataframe with features and target.
    - param_grid (dict): The hyperparameters to tune.
    - cv (int): Number of cross-validation folds.

    Returns:
    - best_params (dict): The best parameters found in the grid search.
    """

    X = data.drop(columns=[target])
    y = data[target]
    param_distributions = {
        'num_leaves': np.arange(20, 150, 10),
        'max_depth': np.arange(3, 15, 1),
        'learning_rate': np.logspace(-3, 0, 10),
        'n_estimators': np.arange(50, 300, 50),
        'subsample': np.arange(0.6, 1.0, 0.1),
        'colsample_bytree': np.arange(0.6, 1.0, 0.1),
        'random_state': [22],
    }

    model = LGBMRegressor()
    grid_search = RandomizedSearchCV(model, param_distributions, cv=cv, scoring='neg_mean_squared_error',
                                     n_iter=100,
                                     verbose=0, n_jobs=3)
    grid_search.fit(X, y)

    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_params_


def perf_test_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    # print(f"Test RMSE: {rmse:.4f}")

    mae = mean_absolute_error(y_test, y_pred)
    # print(f"Test MAE: {rmse:.4f}")

    # Prepare a dataframe to visualize true vs predicted
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }, index=y_test.index)

    return results_df, mae, mse


def plot_results(results_df: pd.DataFrame, num_points: int = 100, figname: str='wind.png'):
    """
    Plots the actual vs predicted values for a quick evaluation of the model's performance.

    Parameters:
    - results_df (pd.DataFrame): Dataframe with 'Actual' and 'Predicted' columns.
    - num_points (int): Number of points to plot for visualization.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Actual'].iloc[:num_points], label='Actual', color='black')
    plt.plot(results_df['Predicted'].iloc[:num_points], label='Predicted Real', color='darkgrey', linestyle="--")
    plt.plot(results_df['Predicted_synth'].iloc[:num_points], label='Predicted Synthetic', color='red', linestyle="-")
    plt.xlabel('Time')
    plt.ylabel('Sum of Variables')
    plt.title('Actual vs Predicted data; ML model trained on generated data')
    plt.legend()
    plt.savefig(figname)
    plt.show()


def return_targets(solar: bool) -> tuple[str, str]:
    val_name = 'wind'
    target = 'Wind_MWh'
    if solar:
        val_name = 'solar'
        target = 'Solar_MWh'

    return val_name, target


def return_paths(val_name, gen_data_path, real_data_path, gen_data_filename, real_data_filename) -> tuple[str, str]:
    # Load the synthetic and real datasets
    if real_data_filename is None:
        real_data_filename = f'train_{val_name}_ref0.csv'
    if gen_data_filename is None:
        gen_data_filename = f'cpu_{val_name}_30000ep_lr0.0001_seq10_minmax.csv'
    real_data_path = os.path.join(real_data_path, real_data_filename)
    gen_data_path = os.path.join(gen_data_path, gen_data_filename)

    return real_data_path, gen_data_path


def main(solar: bool = False, tune: bool = False, target_horizon: int = 24,
         gen_data_path: str = '../outputs_pres/dsets/', gen_data_filename = None,
         real_data_path: str = '../data_pres/', real_data_filename=None ) -> None:

    """Main function to run the forecasting pipeline.
    Prints out errors, plots forecasting results.

    solar: a flag to switch between solar and wind dsets
    tune: if True, a lightGBM tuning will be performed
    target_horizon: how many steps to predict

    gen_data_path: generated data loc
    data_path: real data loc

    gen_filename: generated data loc
    real_filename: real data loc
    """

    val_name, target = return_targets(solar)
    real_data_path, gen_data_path = return_paths(val_name, gen_data_path, real_data_path,
                                                 gen_data_filename, real_data_filename)

    gen_data = pd.read_csv(gen_data_path)
    gen_data.drop(columns='Unnamed: 0', inplace=True)

    real_train_data = return_real_df(path_dset=real_data_path, solar=solar)
    real_test_data = return_real_df(path_dset=f"../data_pres/test_{val_name}_ref0.csv", solar=solar)

    # Prepare the data for forecasting
    prepared_gen_data = add_lags(gen_data, lag_hours=24, min_lag_hours=target_horizon, time_step_per_hour=2)
    prepared_real_train_data = add_lags(real_train_data, lag_hours=24, min_lag_hours=target_horizon, time_step_per_hour=2)
    prepared_real_test_data = add_lags(real_test_data, lag_hours=24, min_lag_hours=target_horizon, time_step_per_hour=2)

    # tune lgbm
    if tune:
        params = tune_lightgbm(prepared_gen_data, target=target)
        params['verbose'] = -1
    else:
        params = None

    summary_df = pd.DataFrame(columns=['MAE', 'MSE'],
                          index=['Real-Real-Real', 'Gen-Real-Real'])

    # Train the model and get predictions
    model = train_model(prepared_real_train_data, target, tuning='real', solar=solar, params=params)
    results_df, mae, mse = perf_test_model(model, prepared_real_test_data.drop(columns=[target]),
                                           prepared_real_test_data[target])
    print(f"Real-Real-Real: MAE {np.round(mae, 3)}, MSE {np.round(mse, 3)}")
    summary_df.loc['Real-Real-Real', 'MAE'] = mae
    summary_df.loc['Real-Real-Real', 'MSE'] = mse

    model = train_model(prepared_real_train_data, target, solar=solar, tuning='gen', params=params)
    results_df_real, mae, mse = perf_test_model(model, prepared_real_test_data.drop(columns=[target]),
                                                prepared_real_test_data[target])
    print(f"Gen-Real-Real: MAE {np.round(mae, 3)}, MSE {np.round(mse, 3)}")
    summary_df.loc['Gen-Real-Real', 'MAE'] = mae
    summary_df.loc['Gen-Real-Real', 'MSE'] = mse

    # Plot the results
    os.makedirs('../outputs_pres/pics/', exist_ok=True)
    results_df['Predicted_synth'] = results_df_real['Predicted'].copy()
    plot_results(results_df, num_points=results_df.shape[0] // 30,
                 figname=f'../outputs_pres/pics/{val_name}_for.png')
    summary_df.to_csv(f"../outputs_pres/{val_name}_errs_horizon{target_horizon}.csv")


if __name__ == "__main__":
    main(solar=True, tune=False)
    main(solar=False, tune=False)
