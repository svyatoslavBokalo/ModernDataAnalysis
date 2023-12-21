import pandas as pd
import numpy as np
import json
import time
import websocket
from collections import deque, defaultdict
import alpaca_trade_api as tradeapi
from keys import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_URL, ALPACA_PAPER
from retrain_model import Model


class AlpacaAPI:
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url=base_url) # API Initialization
        self.order_queue = deque()  # Queue to track unfilled orders


    def place_order(self, action, qty, symbol):
        """
        Place a market order for a given asset symbol.

        Parameters:
        - action (str): Action to perform, either 'buy' or 'sell'.
        - qty (float): Quantity of the asset to buy or sell.
        - symbol (str): Symbol of the asset (e.g., 'AAPL', 'BTC/USD').

        Returns:
        - order_id (str): The ID of the submitted order.
        """
        try:
            self.order_queue.append(self.api.submit_order(symbol, qty, action, 'market', 'gtc').id)
        except Exception as e:
            print(f"Error submitting order: {e}")


    def get_position_size(self, symbol):
        """
        Get the current position size for a given asset symbol.

        Parameters:
        - symbol (str): Symbol of the asset (e.g., 'AAPL', 'BTC/USD').

        Returns:
        - position_size (float): Current position size. Returns 0 if no position is found or an error occurs.
        """
        try:
            return next((float(position.qty) for position in self.api.list_positions() if position.symbol == symbol), 0)
        except Exception as e:
            print(f"Error getting position size: {e}")
            return 0


    def check_order_status(self):
        """
        Check the status of orders in the order queue.

        Returns:
        - all_orders_filled (bool): True if all orders in the queue are filled, False otherwise.
        """
        while self.order_queue:
            order_id = self.order_queue[0]  # Get the first order in the queue
            order = self.api.get_order(order_id)
            if order.status == 'filled':
                self.order_queue.popleft()  # Remove the filled order from the queue
            else:
                return False  # Order is not filled, exit the function
        return True  # All orders in the queue are filled

class CryptoTradingStrategy:
    def __init__(self, alpaca_api, crypto_pairs, ws_url):
        self.alpaca_api = alpaca_api
        self.parameters = defaultdict(dict)
        self.crypto_pairs = crypto_pairs
        self.ws_url = ws_url
        self.model = Model(self.parameters, self.crypto_pairs)

    def connect_to_websocket(self):
        """
        Connect to the WebSocket and perform authentication and subscription.

        Args:
        - ws_url (str): WebSocket URL.
        - auth_message (dict): Authentication message.
        - subscription_message (dict): Subscription message.

        Returns:
        - ws (WebSocket): Connected WebSocket object.
        """
        ws = websocket.create_connection(self.ws_url)
        auth_message = {"action": "auth", "key": ALPACA_API_KEY, "secret": ALPACA_SECRET_KEY}
        subscription_message = {"action": "subscribe", "quotes": self.crypto_pairs}
        ws.send(json.dumps(auth_message))
        ws.send(json.dumps(subscription_message))
        return ws

    def calculate_and_print_moving_averages(self, pair, bid_price_data, ma_fast, ma_slow, parameters):
        """
        Calculate and print moving averages for a given trading pair.

        Parameters:
        - pair (str): The trading pair symbol (e.g., 'BTC/USD').
        - bid_price_data (dict): Dictionary containing bid price data for different trading pairs.
        - ma_fast (dict): Dictionary containing moving averages (fast) for different trading pairs.
        - ma_slow (dict): Dictionary containing moving averages (slow) for different trading pairs.
        - parameters (dict): Dictionary containing trading parameters for different trading pairs.
        """
        ma_fast[pair] = np.mean(bid_price_data[pair][-parameters[pair]['ma_fast']:])
        ma_slow[pair] = np.mean(bid_price_data[pair][-parameters[pair]['ma_slow']:])
        print(f"ma_fast: {ma_fast[pair]}\nma_slow: {ma_slow[pair]}")

    def evaluate_trading_signals(self, pair, ma_fast, ma_slow, ma_fast_prev, ma_slow_prev):
        """
        Evaluate trading signals based on moving averages and execute buy or sell orders.

        Parameters:
        - pair (str): The trading pair symbol (e.g., 'BTC/USD').
        - ma_fast (dict): Dictionary containing moving averages (fast) for different trading pairs.
        - ma_slow (dict): Dictionary containing moving averages (slow) for different trading pairs.
        - ma_fast_prev (dict): Dictionary containing previous moving averages (fast) for different trading pairs.
        - ma_slow_prev (dict): Dictionary containing previous moving averages (slow) for different trading pairs.
        """
        if ma_fast[pair] > ma_slow[pair] and ma_fast_prev[pair] < ma_slow_prev[pair]:
            self.execute_buy_order(pair)
        elif ma_fast[pair] < ma_slow[pair] and ma_fast_prev[pair] > ma_slow_prev[pair]:
            self.execute_sell_order(pair)
        else:
            print("Holding position")

    def execute_buy_order(self, pair):
        """
        Execute a buy order for a given trading pair if no position is currently held.

        Parameters:
        - pair (str): The trading pair symbol (e.g., 'BTC/USD').
        """
        if self.alpaca_api.get_position_size(pair) == 0:
            print("buy order")
            self.alpaca_api.place_order('buy', 0.01, pair)

    def execute_sell_order(self, pair):
        """
        Execute a sell order for a given trading pair if a position is currently held.

        Parameters:
        - pair (str): The trading pair symbol (e.g., 'BTC/USD').
        """
        position_size = self.alpaca_api.get_position_size(pair)
        if position_size > 0:
            print("sell order")
            self.alpaca_api.place_order('sell', position_size, pair)

    def retrain(self, pair, temp_data, model, parameters):
        """
        Retrain the trading model for a given pair using temporary data.

        Parameters:
        - pair (str): The trading pair symbol (e.g., 'BTC/USD').
        - temp_data (dict): Dictionary containing temporary data for different trading pairs.
        - model (Model): The trading model instance.
        - parameters (dict): Dictionary containing trading parameters for different trading pairs.
        """
        if len(temp_data[pair]) >= 5:  # parameter to be defined:
            temp_df = pd.DataFrame(temp_data[pair])
            print("pair", pair)
            params = model.retrain_strat(pair, temp_df)
            print("model data", len(model.data[pair]))
            print(f"parameters after retraining for {pair}: {params}")
            temp_data[pair] = []
            print(f"Retrained the model for {pair}")
            parameters[pair]['ma_fast'] = params['ma_fast']
            parameters[pair]['ma_slow'] = params['ma_slow']

            model.delete_old_records(pair, num_to_keep=800)

    def run_strategy(self):
        """
        Run the trading strategy.

        Connects to the Alpaca WebSocket, receives real-time price data for specified crypto pairs,
        and executes the trading strategy based on moving averages. It continuously updates temporary
        and historical data, evaluates trading signals, and re-trains the model.

        Raises:
        - Exception: If an error occurs during the execution of the strategy.

        Notes:
        - The strategy runs indefinitely until an exception is encountered.
        - The strategy uses moving averages to generate buy and sell signals.
        """
        ws = self.connect_to_websocket()

        temp_data = {pair: [] for pair in self.crypto_pairs}
        bid_price_data = defaultdict(list)
        ma_fast = defaultdict(float)
        ma_slow = defaultdict(float)
        ma_fast_prev = defaultdict(float)
        ma_slow_prev = defaultdict(float)

        ## These are the ideal parameters from backtesting
        ## We will start with them and retrain the modele
        # Ideal parameters from backtesting
        self.parameters = {
            'BTC/USD': {'ma_fast': 295, 'ma_slow': 200},
            'ETH/USD': {'ma_fast': 205, 'ma_slow': 140},
            'LTC/USD': {'ma_fast': 220, 'ma_slow': 128}
        }

        try:
            while True:
                response = ws.recv()
                data_dict = json.loads(response)[0]
                print("data", data_dict)
                if 'S' in data_dict and data_dict['S'] in self.crypto_pairs and 'bp' in data_dict and 't' in data_dict:
                    print(f"Received price for {data_dict['S'], data_dict['bp']}")
                    pair = data_dict['S']
                    ## We will also need to extract more data such as volume, etc.
                    ## This way we can check for data skewness
                    temp_data[pair].append({'Time': data_dict['t'], 'close': data_dict['bp']})
                    bid_price_data[pair].append(data_dict['bp'])

                    if ma_fast[pair] and ma_slow[pair]:
                        ma_fast_prev[pair], ma_slow_prev[pair] = ma_fast[pair], ma_slow[pair]
                        print(f"ma_fast_prev: {ma_fast_prev[pair]} \n ma_slow_prev: {ma_slow_prev[pair]}")

                    if len(bid_price_data[pair]) > self.parameters[pair]['ma_slow']:
                        self.calculate_and_print_moving_averages(pair, bid_price_data, ma_fast, ma_slow, self.parameters)

                    self.evaluate_trading_signals(pair, ma_fast, ma_slow, ma_fast_prev, ma_slow_prev)

                    self.retrain(pair, temp_data, self.model, self.parameters)

                    ## Model is trained on minute data, so we should receive at more or less minute freq.
                    time.sleep(0.1)


        except Exception as e:
            print(f"Exception in strategy: {e}")

        finally:
            if ws:
                ws.close()


if __name__ == "__main__":
    alpaca_api = AlpacaAPI(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER)
    crypto_pairs = ['BTC/USD', 'ETH/USD', 'LTC/USD']
    ws_url = ALPACA_URL

    crypto_strategy = CryptoTradingStrategy(alpaca_api, crypto_pairs, ws_url)

    failed_times = 0
    while True:
        try:
            crypto_strategy.run_strategy()
        except Exception as e:
            print(f"Exception in main loop: {e}")
            failed_times += 1
            if failed_times > 100:
                print("Too many failures, stopping.")
                break
            time.sleep(5)
            print(f"Failed times: {failed_times}")

