from collections import defaultdict
import time
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import sqlite3


class Model:
    def __init__(self, parameters, pairs):
        self.parameters = parameters
        self.data = {}
        self.versions = defaultdict(int)
        self.crypto_pairs = pairs
        
        # Create SQLite database for historical data
        self.db_path = 'historical_data.db'
        self.create_database()
        
        # Load historical data during initialization
        for pair in self.crypto_pairs:
            self.load_data(pair)
        
    def create_database(self):
        # Create a SQLite database if not exists
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create a table for historical data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS HistoricalData (
                    pair TEXT,
                    Time DATETIME,
                    close DECIMAL(20, 8),
                    PRIMARY KEY (pair, Time)
                )
            ''')

            conn.commit()
            conn.close()
            
    def save_to_database(self, pair, data):
        # Save historical data to the database
        conn = sqlite3.connect(self.db_path)
        data['pair'] = pair
        data.to_sql('HistoricalData', conn, index=False, if_exists='append', method='multi')
        conn.commit()
        conn.close()
        
    def save_to_temporary(self, pair, temp_df):
        # Save temporary data to the dictionary
        if pair not in self.data:
            self.data[pair] = temp_df
        else:
            self.data[pair] = pd.concat([self.data[pair], temp_df], ignore_index=True)


    # def retrain_strat(self, pair, temp_df):
    #     best_parameters= {}

    #     time_start = time.time()

    #     if pair not in self.data:
    #         self.load_data(pair)

    #     # Save data to the database and dictionary
    #     self.save_to_database(pair, temp_df)
    #     self.save_to_temporary(pair, temp_df)

    #     # Now proceed with retraining using the updated data
    #     data = self.data[pair]
        
    #     # Fetch data for each ticker
    #     data['Log Returns'] = np.log(data['close'] / data['close'].shift(1))
        
    #     # Initialize with a very low return
    #     best_return = -np.inf  

    #     for MAS in tqdm(range(100, 801, 15), desc=f"Processing MAS for {pair}", leave=False):
    #         for MAF in range(50, min(400, int(MAS * 0.75)), 6):
    #             data['MASlow'] = data['close'].rolling(MAS).mean()
    #             data['MAFast'] = data['close'].rolling(MAF).mean()
    #             data['Signal'] = np.where(data['MAFast'] > data['MASlow'], 1, 0)

    #             data['Strategy Log Returns'] = data['Log Returns'] * data['Signal'].shift(1)
    #             strategy_return = (np.exp(data['Strategy Log Returns'].sum()) - 1) * 100

    #             if strategy_return > best_return:
    #                 best_return = strategy_return
    #                 best_parameters[pair] = {'ma_slow': MAS, 'ma_fast': MAF}
        
    #     self.parameters[pair] = best_parameters[pair]

    #     time_end = time.time()
    #     print(f"Time taken to retrain model for {pair}: {time_end - time_start} seconds")

    #     return best_parameters[pair]
            
    def retrain_strat(self, pair, temp_df):
        best_parameters = {}

        time_start = time.time()

        if pair not in self.data:
            self.load_data(pair)

        # Save data to the database and dictionary
        self.save_to_database(pair, temp_df)
        self.save_to_temporary(pair, temp_df)

        # Now proceed with retraining using the updated data
        data = self.data[pair]

        # Fetch data for each ticker
        data['Log Returns'] = np.log(data['close'] / data['close'].shift(1))

        # Normalize data
        scaler = MinMaxScaler()
        data[['close', 'Log Returns']] = scaler.fit_transform(data[['close', 'Log Returns']])

        # Prepare data for LSTM
        X = []
        y = []
        for i in range(len(data) - self.lookback - 1):
            X.append(data['Log Returns'].values[i:(i + self.lookback)])
            y.append(data['Log Returns'].values[i + self.lookback])
        X, y = np.array(X), np.array(y)

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y, epochs=50, batch_size=32)

        # Make predictions
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        # Calculate returns based on predictions
        data['Strategy Log Returns'] = 0.0
        data['Strategy Log Returns'].iloc[self.lookback:len(predictions) + self.lookback] = predictions.flatten()

        strategy_return = (np.exp(data['Strategy Log Returns'].sum()) - 1) * 100

        print(f"Strategy return: {strategy_return}")

        # Update best parameters
        best_parameters[pair] = {'lstm_units': 50, 'epochs': 50, 'batch_size': 32}

        self.parameters[pair] = best_parameters[pair]

        time_end = time.time()
        print(f"Time taken to retrain model for {pair}: {time_end - time_start} seconds")

        return best_parameters[pair]
    
    
    def load_data(self, pair):
        # Load historical data from the database
        conn = sqlite3.connect(self.db_path)
        query = f'SELECT * FROM HistoricalData WHERE pair = "{pair}" ORDER BY Time DESC LIMIT 800'
        self.data[pair] = pd.read_sql(query, conn)
        conn.close()

    def delete_old_records(self, pair, num_to_keep=800):
        """
        Delete old records from the database, keeping only the last specified number of records.

        Parameters:
        - pair (str): Symbol of the asset pair (e.g., 'BTC/USD').
        - num_to_keep (int): Number of records to keep in the database. Default is 100,000.

        Notes:
        - This method is called after retraining to ensure that only recent data is kept.
        """
        conn = sqlite3.connect(self.db_path)

        try:
            # Get the total number of records for the pair
            query_count = f'SELECT COUNT(*) FROM HistoricalData WHERE pair = "{pair}"'
            total_records = conn.execute(query_count).fetchone()[0]
            print('number of records for the pair', total_records)

            # Calculate the number of records to delete
            num_to_delete = max(0, total_records - num_to_keep)

            if num_to_delete > 0:
                # Delete old records
                query_delete = f'''
                    DELETE FROM HistoricalData
                    WHERE pair = "{pair}" AND Time IN (
                        SELECT Time
                        FROM HistoricalData
                        WHERE pair = "{pair}"
                        ORDER BY Time
                        LIMIT {num_to_delete}
                    )
                '''
                conn.execute(query_delete)
                print(f"Deleted {num_to_delete} old records for {pair}")

            conn.commit()

        except Exception as e:
            print(f"Error deleting old records: {e}")

        finally:
            conn.close()

