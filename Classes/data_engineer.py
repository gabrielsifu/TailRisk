import os

import pandas as pd
import numpy as np
import warnings
import joblib as joblib

class DataEngineer:
    def __init__(self, sett):
        self.sett = sett
        self.data = None
        self.xy_train, self.xy_test = None, None
        self.xy_train_windows, self.xy_test_windows = None, None

    def save_clean_data(self):
        print("save_clean_data")
        self.get_data()
        self.generate_features()
        self.split_periods()
        self.save()
        print("end")

    def get_data(self):
        self.data = pd.read_csv('./data/SP500.csv', sep=';')

    def generate_features(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
        self.data = self.data.loc[:, ['Date', 'Price']]
        self.data.set_index('Date', inplace=True, drop=True)
        self.data['Price'] = self.data['Price'].str.replace('.', '', regex=False)
        self.data['Price'] = self.data['Price'].str.replace(',', '.', regex=False)
        self.data['Price'] = self.data['Price'].astype(float)
        self.data['LogReturn'] = np.log(self.data['Price']/self.data['Price'].shift(1))
        self.data.drop(columns='Price', inplace=True)
        self.data.dropna(inplace=True)

        self.data = self.calculate_rolling(self.data, ['LogReturn'], [5, 10, 21],
                                      ['mean', 'median', 'kurt', 'skew', 'std', 'min', 'max'])

        self.data['y'] = (self.data['LogReturn'] < self.data['LogReturn'].expanding(
            min_periods=252).quantile(self.sett.DataEngineer.quantile_threshold)).astype(int)
        cols_to_shift = self.data.columns.difference(['y', 'LogReturn'])
        self.data[cols_to_shift] = self.data[cols_to_shift].shift(self.sett.DataEngineer.delay_x)
        self.data[cols_to_shift] = ((self.data[cols_to_shift]-self.data[cols_to_shift].expanding(min_periods=252).mean())/self.data[cols_to_shift].expanding(min_periods=252).std())
        self.data.dropna(how='any', inplace=True)

    @staticmethod
    def calculate_rolling(df, columns, window_sizes, operations):
        """
        Applies a rolling window on specific columns of a DataFrame with multi-level indices.

        Parameters:
        - df: DataFrame with 'DATE_REF' and 'TRADINGITEM_ID' as indices.
        - columns: List of columns on which to apply the rolling operations.
        - window_sizes: List of rolling window sizes.
        - operations: List of operations to be applied ('max', 'min', 'std', 'sum', 'mean', 'skew', 'median', 'kurt').

        Returns:
        - DataFrame with new calculated columns.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

            # Dictionary to map available rolling functions
            operations_map = {
                'max': 'max',
                'min': 'min',
                'std': 'std',
                'sum': 'sum',
                'mean': 'mean',
                'skew': 'skew',
                'median': 'median',
                'kurt': 'kurt'
            }

            # Check if all provided operations are valid
            for operation in operations:
                if operation not in operations_map:
                    raise ValueError(f"Invalid operation: {operation}. Choose from: {', '.join(operations_map.keys())}")

            # Initialize a DataFrame to store the new calculated columns
            df_result = df.copy()

            # Apply the operations for each column and each window size
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"Column {col} is not present in the DataFrame.")

                for window_size in window_sizes:
                    for operation in operations:
                        # Name for the new column
                        new_col_name = f"{col}_{operation}_rolling_{window_size}"

                        # Apply the correct rolling operation based on the operation type
                        if operation in ['max', 'min', 'std', 'sum', 'mean']:
                            df_result[new_col_name] = (df[col]
                                                       .rolling(window=window_size,
                                                                min_periods=window_size)
                                                       .agg(operations_map[operation]))
                        elif operation == 'skew':
                            df_result[new_col_name] = (df[col]
                                                       .rolling(window=window_size,
                                                                min_periods=window_size)
                                                       .skew())
                        elif operation == 'median':
                            df_result[new_col_name] = (df[col]
                                                       .rolling(window=window_size,
                                                                min_periods=window_size)
                                                       .median())
                        elif operation == 'kurt':
                            df_result[new_col_name] = (df[col]
                                                       .rolling(window=window_size,
                                                                min_periods=window_size)
                                                       .kurt())

            return df_result

    def split_periods(self):
        print("split_periods")
        first_test_year = self.data.index.min().year + self.sett.DataEngineer.first_test_year
        self.xy_train, self.xy_test = self.create_expanding_windows(self.data, first_test_year)
        self.xy_train_windows, self.xy_test_windows = self.create_expanding_lstm_windows(self.data, first_test_year, window_size=21)
        print("End")

    @staticmethod
    def create_expanding_windows(df, first_test_year):
        # Assuming the index is a MultiIndex, adjust if necessary
        start_year = df.index.min().year
        last_year = df.index.max().year
        train_windows = {}
        test_windows = {}
        # Iterate from the start year to the last available year in the DataFrame
        for end_year in range(first_test_year - 1, last_year):
            window_name = f"{start_year}_{end_year}"
            # Filtering the training window: Convert start_year to Timestamp
            filtered_df_train = df[(df.index.get_level_values(0) >= pd.Timestamp(f'{start_year}-01-01')) &
                                   (df.index.get_level_values(0) <= pd.Timestamp(f'{end_year}-12-31'))]
            train_windows[window_name] = filtered_df_train
            # Filtering the testing window
            filtered_df_test = df[(df.index.get_level_values(0) > pd.Timestamp(f'{end_year}-12-31')) &
                                  (df.index.get_level_values(0) <= pd.Timestamp(f'{end_year + 1}-12-31'))]
            test_windows[window_name] = filtered_df_test
        return train_windows, test_windows

    def save(self):
        os.makedirs('./data/xy_full/', exist_ok=True)
        os.makedirs('./data/xy_full_windows/', exist_ok=True)
        joblib.dump(self.xy_train, './data/xy_full/xy_train.joblib')
        joblib.dump(self.xy_test, './data/xy_full/xy_test.joblib')
        joblib.dump(self.xy_train_windows, './data/xy_full_windows/xy_train.joblib')
        joblib.dump(self.xy_test_windows, './data/xy_full_windows/xy_test.joblib')

    @staticmethod
    def create_lstm_windows(df, window_size=21):
        """
        Cria janelas de sequência para input em uma LSTM usando os últimos 'window_size' períodos.
        """
        X, y, index = [], [], []
        df = df.sort_index()
        for i in range(window_size, len(df)):
            X.append(df.iloc[i-window_size+1:i+1].values)
            y.append(df.iloc[i].values)
            index.append(df.index[i])  # Guarda o índice (timestamp) do target
        return np.array(X), np.array(y), index

    def create_expanding_lstm_windows(self, df, first_test_year, window_size=21):
        """
        Cria janelas expansivas de treino/teste para LSTM com base nos alvos (y) e seus anos.
        """
        df['ret'] = df['LogReturn']
        df = df.loc[:, ['LogReturn', 'ret', 'y']]
        df['ret'] = df['ret'].shift(self.sett.DataEngineer.delay_x)
        df = df.dropna()

        # Primeiro cria todas as janelas e guarda o índice temporal de y
        X_all, y_all, y_index = self.create_lstm_windows(df, window_size)

        start_year = pd.Timestamp(y_index[0]).year
        last_year = pd.Timestamp(y_index[-1]).year

        train_windows = {}
        test_windows = {}

        for end_year in range(first_test_year - 1, last_year):
            window_name = f"{start_year}_{end_year}"

            # Define treino: targets (y) com data até end_year
            train_mask = [ts.year <= end_year for ts in y_index]
            test_mask = [ts.year == end_year + 1 for ts in y_index]

            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]

            train_windows[window_name] = (X_train, y_train)
            test_windows[window_name] = (X_test, y_test)

        return train_windows, test_windows
