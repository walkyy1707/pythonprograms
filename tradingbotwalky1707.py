import asyncio
import ccxt.async_support as ccxt
import numpy as np
import talib
import logging
from dotenv import load_dotenv
import os

# Load API credentials from environment variables (optional for live trading)
load_dotenv()
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Retry decorator for API calls
def retry_on_failure(max_retries=3, delay=5):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                    else:
                        logging.error(f"All {max_retries} attempts failed: {e}")
                        raise
        return wrapper
    return decorator

# Data Handler with Precomputed Indicators
class DataHandler:
    def __init__(self, exchange, symbol, timeframe, batch_size=1000):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.batch_size = batch_size
        self.historical_data = np.array([])  # NumPy array for closing prices
        self.indicators = {
            'sma10': None,
            'sma50': None,
            'rsi': None,
            'macd': None,
            'macd_signal': None
        }
        self.running = False

    @retry_on_failure()
    async def fetch_initial_batch(self):
        """Fetch initial batch of historical data and precompute indicators."""
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.batch_size)
        self.historical_data = np.array([candle[4] for candle in ohlcv])  # Closing prices
        self.update_indicators()

    def update_indicators(self):
        """Precompute indicators for the entire batch."""
        if len(self.historical_data) >= 50:
            self.indicators['sma10'] = talib.SMA(self.historical_data, timeperiod=10)
            self.indicators['sma50'] = talib.SMA(self.historical_data, timeperiod=50)
            self.indicators['rsi'] = talib.RSI(self.historical_data, timeperiod=14)
            macd, macd_signal, _ = talib.MACD(self.historical_data, fastperiod=12, slowperiod=26, signalperiod=9)
            self.indicators['macd'] = macd
            self.indicators['macd_signal'] = macd_signal

    @retry_on_failure()
    async def update_latest(self):
        """Update with the latest candle and recompute indicators."""
        latest_ohlcv = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1)
        latest_close = latest_ohlcv[0][4]
        self.historical_data = np.append(self.historical_data[1:], latest_close)
        self.update_indicators()
        logging.debug(f"Updated historical_data: len={len(self.historical_data)}, latest={self.historical_data[-1]}")

    async def start(self):
        """Start the data stream."""
        self.running = True
        await self.fetch_initial_batch()
        while self.running:
            try:
                await self.update_latest()
                await asyncio.sleep(60)  # Adjust based on timeframe (e.g., 1 minute)
            except Exception as e:
                logging.error(f"Data update error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        self.running = False

    def get_data(self):
        return self.historical_data, self.indicators

# Trading Strategy
class TradingStrategy:
    def __init__(self, strategy_type):
        self.strategy_type = strategy_type

    def get_signal(self, indicators):
        """Generate trading signal based on the selected strategy."""
        signal = 'hold'
        if self.strategy_type == "MA_Crossover":
            sma10 = indicators['sma10'][-1]
            sma50 = indicators['sma50'][-1]
            signal = 'buy' if sma10 > sma50 else 'sell' if sma10 < sma50 else 'hold'
            logging.info(f"Strategy {self.strategy_type}: SMA10={sma10:.2f}, SMA50={sma50:.2f}, Signal={signal}")
        elif self.strategy_type == "RSI":
            rsi = indicators['rsi'][-1]
            signal = 'buy' if rsi < 30 else 'sell' if rsi > 70 else 'hold'
            logging.info(f"Strategy {self.strategy_type}: RSI={rsi:.2f}, Signal={signal}")
        elif self.strategy_type == "MACD":
            macd = indicators['macd'][-1]
            macd_signal = indicators['macd_signal'][-1]
            signal = 'buy' if macd > macd_signal else 'sell' if macd < macd_signal else 'hold'
            logging.info(f"Strategy {self.strategy_type}: MACD={macd:.2f}, MACD_Signal={macd_signal:.2f}, Signal={signal}")
        return signal

# Trading Bot with Asynchronous Execution
class TradingBot:
    def __init__(self, exchange, symbol, timeframe, strategy_type="MA_Crossover"):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_handler = DataHandler(exchange, symbol, timeframe)
        self.strategy = TradingStrategy(strategy_type)
        self.balance = 10000  # Initial balance in USD
        self.position = 0  # Position in asset units
        self.running = False

    async def run(self):
        """Main trading loop."""
        self.running = True
        data_task = asyncio.create_task(self.data_handler.start())
        while self.running:
            try:
                prices, indicators = self.data_handler.get_data()
                if len(prices) >= 50:  # Ensure enough data for indicators
                    signal = self.strategy.get_signal(indicators)
                    await self.execute_trade(signal, prices[-1])
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)
        data_task.cancel()
        await self.exchange.close()

    @retry_on_failure()
    async def execute_trade(self, signal, price):
        """Execute trades asynchronously with retries."""
        if signal == 'buy' and self.position == 0 and self.balance > 0:
            self.position = self.balance / price
            self.balance = 0
            logging.info(f"Buy {self.position:.6f} {self.symbol.split('/')[0]} at ${price:.2f}")
        elif signal == 'sell' and self.position > 0:
            self.balance = self.position * price
            self.position = 0
            logging.info(f"Sell at ${price:.2f}, Balance: ${self.balance:.2f}")

    def stop(self):
        self.running = False
        self.data_handler.stop()

# Main Function
async def main():
    exchange = ccxt.binance({
        'enableRateLimit': True,  # No API keys needed for public data
    })
    symbol = 'BTC/USDT'
    timeframe = '1m'
    strategy_type = "MA_Crossover"  # Options: "MA_Crossover", "RSI", "MACD"

    bot = TradingBot(exchange, symbol, timeframe, strategy_type)
    try:
        await bot.run()
    except KeyboardInterrupt:
        logging.info("Stopping bot due to KeyboardInterrupt")
        bot.stop()
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())