# Stock-Market-Analyzer
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pytz
from datetime import timedelta

def check_market_status():
    # Define the market open and close times in IST (Indian Standard Time)
    market_open_time = datetime.strptime("09:00:00", "%H:%M:%S").time()
    market_close_time = datetime.strptime("16:00:00", "%H:%M:%S").time()

    # Get the current time in IST (Indian Standard Time)
    india_timezone = pytz.timezone('Asia/Kolkata')
    india_time = datetime.now(india_timezone)

    # Get current time and current day (week day: 0 = Monday, 6 = Sunday)
    current_time = india_time.time()
    current_day = india_time.weekday()

    # Check if today is a weekday (Monday to Friday)
    if current_day >= 0 and current_day <= 4:  # Monday to Friday
        # Check if the current time is within market open and close times
        if market_open_time <= current_time <= market_close_time:
            return "Market Open"
        else:
            return "Market Closed"
    else:
        return "Market Closed (Weekend)"

# Get and print the market status
market_status = check_market_status()
print(f"Market Status: {market_status}")

# Function to check if the market is open or closed
def is_market_open():
    current_time = datetime.datetime.now()
    current_day = current_time.weekday()  # 0 = Monday, 6 = Sunday

    # Market hours: 9:30 AM - 4:00 PM EST (for NYSE and NASDAQ)
    market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

    # If it's a weekend (Saturday or Sunday), the market is closed
    if current_day >= 5:
        return "The market is closed for the weekend."

    # If it's a weekday, check if the current time is within market hours
    if market_open_time <= current_time <= market_close_time:
        return "The market is open."
    else:
        return "The market is closed."

print("\033[1;32m" + market_status + "\033[0m")

# List of stocks
tickers = {
    "AAPL": "Apple Inc.",
    "GOOG": "Alphabet Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NFLX": "Netflix Inc.",
    "NVDA": "NVIDIA Corporation",
    "INTC": "Intel Corporation",
    "BABA": "Alibaba Group Holding Limited",
    "RELIANCE.NS": "Reliance Industries Limited",
    "TCS.NS": "Tata Consultancy Services Limited",
    "HDFCBANK.NS": "HDFC Bank Limited",
    "AIRTEL.NS": "Bharti Airtel Limited",
    "ICICIBANK.NS": "ICICI Bank Limited",
    "INFY.NS": "Infosys Limited",
    "SBIN.NS": "State Bank of India",
    "HINDUNILVR.NS": "Hindustan Unilever Limited",
    "ITC.NS": "ITC Limited",
    "HCLTECH.NS": "HCL Technologies Limited",
    "LT.NS": "Larsen & Toubro Limited",
    "BAJFINANCE.NS": "Bajaj Finance Limited",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited",
    "MARUTI.NS": "Maruti Suzuki India Limited",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Limited",
    "ONGC.NS": "Oil and Natural Gas Corporation Limited",
    "AXISBANK.NS": "Axis Bank Limited",
    "WIPRO.NS": "Wipro Limited",
    "ULTRACEMCO.NS": "UltraTech Cement Limited",
    "TITAN.NS": "Titan Company Limited",
    "NTPC.NS": "NTPC Limited",
    "TATAMOTORS.NS": "Tata Motors Limited",
    "POWERGRID.NS": "Power Grid Corporation of India Limited",
    "ADANIENT.NS": "Adani Enterprises Limited",
    "BAJAJFINSV.NS": "Bajaj Finserv Ltd.",
    "HAL.NS": "Hindustan Aeronautics Limited",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone Limited",
    "DMART.NS": "Avenue Supermarts Limited",
    "TRENT.NS": "Trent Limited",
    "COALINDIA.NS": "Coal India Limited",
    "ASIANPAINT.NS": "Asian Paints Limited",
    "ZOMATO.NS": "Zomato Limited",
    "JSWSTEEL.NS": "JSW Steel Limited",
    "SIEMENS.NS": "Siemens Limited",
    "NESTLEIND.NS": "Nestlé India Limited",
    "VBL.NS": "Varun Beverages Limited",
    "BEL.NS": "Bharat Electronics Limited",
    "DLF.NS": "DLF Limited",
    "ADANIPOWER.NS": "Adani Power Limited",
    "BHEL.NS": "Bharat Heavy Electricals Limited",
    "INDUSINDBK.NS": "IndusInd Bank",
    "M&M.NS": "Mahindra & Mahindra",
    "TATAMOTORS.NS": "Tata Motors",
    "HCLTECH.NS": "HCL Technologies",
    "SHREECEM.NS": "Shree Cement",
    "NESTLEIND.NS": "Nestlé India",
    "BANKBARODA.NS": "Bank of Baroda",
    "MOTHERSUMI.NS": "Motherson Sumi",
    "TATASTEEL.NS": "Tata Steel Limited",
    "LUPIN.NS": "Lupin Limited",
    "BIOCON.NS": "Biocon Limited",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "ICICIGI.NS": "ICICI Lombard General Insurance",
    "ICICIPRULI.NS": "ICICI Prudential Life Insurance",
    "ADANIGREEN.NS": "Adani Green Energy",
    "ADANIPORTS.NS": "Adani Ports",
    "COALINDIA.NS": "Coal India",
    "HINDALCO.NS": "Hindalco Industries",
    "GAIL.NS": "GAIL India",
    "NTPC.NS": "NTPC Limited",
    "TATACONSUM.NS": "Tata Consumer Products",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "DIVISLAB.NS": "Divi's Laboratories",
    "TECHM.NS": "Tech Mahindra",
    "YESBANK.NS": "Yes Bank",
    "RECLTD.NS": "Rural Electrification Corporation",
    "BANKINDIA.NS": "Bank of India",
    "MOTHERSUMI.NS": "Motherson Sumi",
    "UPL.NS": "UPL Limited",
    "BIOCON.NS": "Biocon",
    "ZOMATO.NS": "Zomato",
    "DMART.NS": "Avenue Supermarts",
    "TRENT.NS": "Trent",
    "PGHH.NS": "Procter & Gamble Hygiene and Health Care",
    "GODREJCP.NS": "Godrej Consumer Products",
    "BOSCHLTD.NS": "Bosch Limited",
    "BALKRISHNA.NS": "Balkrishna Industries",
    "AJANTPHARM.NS": "Ajanta Pharma",
    "HAVELLS.NS": "Havells India",
    "HINDZINC.NS": "Hindustan Zinc",
    "TITAN.NS": "Titan Company",
    "JSWSTEEL.NS": "JSW Steel",
    "ACC.NS": "ACC Limited",
    "AMBUJACEM.NS": "Ambuja Cements",
    "GAIL.NS": "GAIL India",
    "HDFC.NS": "HDFC Limited",
    "BHEL.NS": "Bharat Heavy Electricals",
    "MARICO.NS": "Marico Limited",
    "TATAELXSI.NS": "Tata Elxsi",
    "MCDOWELL-N.NS": "McDowell's No. 1",
    "CIPLA.NS": "Cipla Limited",
    "BATAINDIA.NS": "Bata India",
    "BAJFINANCE.NS": "Bajaj Finance",
    "SBI.NS": "State Bank of India",
    "BRITANNIA.NS": "Britannia Industries",
    "MOTHERSON.NS": "Motherson Sumi",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "JINDALSTEL.NS": "Jindal Steel & Power",
    "OIL.NS": "Oil India Limited",
    "HDFC.NS": "HDFC Limited",
    "TCS.NS": "Tata Consultancy Services",
    "RELIANCE.NS": "Reliance Industries",
    "INFY.NS": "Infosys",
    "SBIN.NS": "State Bank of India",
    "ICICIBANK.NS": "ICICI Bank",
    "AXISBANK.NS": "Axis Bank",
    "WIPRO.NS": "Wipro",
    "HCLTECH.NS": "HCL Technologies",
    "BAJFINANCE.NS": "Bajaj Finance",
    "SUNPHARMA.NS": "Sun Pharmaceuticals",
    "MARUTI.NS": "Maruti Suzuki",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ONGC.NS": "Oil and Natural Gas Corporation",
    "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra & Mahindra",
    "ITC.NS": "ITC Limited",
    "NESTLEIND.NS": "Nestlé India",
    "ASIANPAINT.NS": "Asian Paints",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "BHEL.NS": "Bharat Heavy Electricals",
    "COALINDIA.NS": "Coal India",
    "NTPC.NS": "NTPC Limited",
    "TITAN.NS": "Titan Company",
    "JSWSTEEL.NS": "JSW Steel",
    "TATACONSUM.NS": "Tata Consumer Products",
    "GAIL.NS": "GAIL India",
    "HINDALCO.NS": "Hindalco Industries",
    "ADANIGREEN.NS": "Adani Green Energy",
    "ZOMATO.NS": "Zomato",
    "DMART.NS": "Avenue Supermarts",
    "TRENT.NS": "Trent Limited",
    "MOTHERSUMI.NS": "Motherson Sumi",
    "UPL.NS": "UPL Limited",
    "LUPIN.NS": "Lupin Limited",
    "CIPLA.NS": "Cipla Limited",
    "DIVISLAB.NS": "Divi's Laboratories",
    "RELIANCE.NS": "Reliance Industries",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "TECHM.NS": "Tech Mahindra",
    "YESBANK.NS": "Yes Bank",
    "BANKINDIA.NS": "Bank of India",
    "ICICIGI.NS": "ICICI Lombard General Insurance",
    "ICICIPRULI.NS": "ICICI Prudential Life Insurance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "ADANIPORTS.NS": "Adani Ports",
    "HAL.NS": "Hindustan Aeronautics",
    "TATELXSI.NS": "Tata Elxsi",
    "MCDOWELL-N.NS": "McDowell's No. 1",
    "BOSCHLTD.NS": "Bosch Limited",
    "BALKRISHNA.NS": "Balkrishna Industries",
    "AJANTPHARM.NS": "Ajanta Pharma",
    "HAVELLS.NS": "Havells India",
    "BRITANNIA.NS": "Britannia Industries",
    "HINDZINC.NS": "Hindustan Zinc",
    "SHREECEM.NS": "Shree Cement",
    "AMBUJACEM.NS": "Ambuja Cements",
    "JINDALSTEL.NS": "Jindal Steel & Power",
    "SBI.NS": "State Bank of India",
    "MOTHERSUMI.NS": "Motherson Sumi",
    "RECLTD.NS": "Rural Electrification Corporation",
    "OIL.NS": "Oil India Limited",
    "TCS.NS": "Tata Consultancy Services",
    "L&T.NS": "Larsen & Toubro",
    "RELIANCE.NS": "Reliance Industries",
    "TITAN.NS": "Titan Company Limited",
    "HDFC.NS": "HDFC Limited",
    "ICICIBANK.NS": "ICICI Bank",
    "HCLTECH.NS": "HCL Technologies",
    "INFY.NS": "Infosys Limited"
}

# Display available stocks
print("Available stocks:")
for i, (ticker, company) in enumerate(tickers.items(), start=1):
    print(f"{i}. {ticker}: {company}")

# User selects a stock
stock_number = int(input("Enter the number of the stock you want to analyze: "))
ticker = list(tickers.keys())[stock_number - 1]

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    data["SMA200"] = data["Close"].rolling(window=200).mean()
    return data

# Determine market condition
def determine_market_condition(data):
    if data["SMA50"].iloc[-1] > data["SMA200"].iloc[-1]:
        return "bullish"
    elif data["SMA50"].iloc[-1] < data["SMA200"].iloc[-1]:
        return "bearish"
    else:
        return "neutral"

# Investment advice based on market condition
def investment_advice(market_condition):
    advice = {
        "bullish": "The market is looking bullish. It might be a good time to invest.",
        "bearish": "The market is bearish. It might be wise to hold off on investments.",
        "neutral": "The market is neutral. Consider observing for more signals before making decisions."
    }
    return advice.get(market_condition, "Unable to determine market condition.")

# Prepare historical data
def prepare_historical_data(ticker):
    print(f"Downloading historical data for {ticker}...")
    data = yf.download(ticker, start="2020-01-01", end="2024-12-31")
    data = calculate_technical_indicators(data)
    return data

# Function to fetch live data
def fetch_live_data(ticker):
    try:
        live_data = yf.download(ticker, period="1d", interval="1m")
        live_data = calculate_technical_indicators(live_data)
        return live_data
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None

# Function to train LSTM model
def train_model(data):
    seq_length = 100
    scaler = MinMaxScaler(feature_range=(0, 1))
    data["Close"] = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i+seq_length])
            y.append(dataset[i+seq_length])
        return np.array(X), np.array(y)

    train_size = int(len(data) * 0.8)
    train_data = data["Close"].values[:train_size]
    X_train, y_train = create_sequences(train_data, seq_length)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model, scaler

# Function to determine if a stock is Indian or international
def is_indian_stock(ticker):
    return ticker.endswith(".NS")

# Function to format prices with the correct currency symbol
def format_price(price, is_indian):
    currency_symbol = "₹" if is_indian else "$"
    return f"{currency_symbol}{price:.2f}"

# Fetch company description and metrics
def fetch_company_details(ticker):
    try:
        stock_info = yf.Ticker(ticker).info
        details = {
            "description": stock_info.get("longBusinessSummary", "Description not available."),
            "all_time_high": stock_info.get("fiftyTwoWeekHigh", "N/A"),
            "all_time_low": stock_info.get("fiftyTwoWeekLow", "N/A"),
            "market_cap": stock_info.get("marketCap", "N/A"),
            "pe_ratio": stock_info.get("trailingPE", "N/A"),
            "avg_volume": stock_info.get("averageVolume", "N/A")
        }
        return details
    except Exception as e:
        print(f"Error fetching company details: {e}")
        return None

# Function to plot real-time graph with company details and metrics
def plot_real_time(data, live_data, model, scaler, seq_length):
    plt.figure(figsize=(14, 12))  # Adjusted size to accommodate extra details
    plt.ion()  # Enable interactive mode

    indian_stock = is_indian_stock(ticker)  # Check if the stock is Indian
    company_details = fetch_company_details(ticker)  # Fetch company details once

    description_displayed = False  # Track if the description has been displayed

    while True:
        live_data = fetch_live_data(ticker)
        if live_data is None or live_data.empty:
            continue

        scaled_data = scaler.transform(live_data["Close"].values.reshape(-1, 1))
        X_live = np.array([scaled_data[-seq_length:]])
        prediction = model.predict(X_live)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        # Update live data with prediction
        live_data["Prediction"] = np.nan
        live_data.iloc[-1, live_data.columns.get_loc("Prediction")] = predicted_price

        market_condition = determine_market_condition(live_data)
        advice = investment_advice(market_condition)

        # Clear and redraw the plot
        plt.clf()
        plt.subplot(2, 1, 1)  # First subplot for the graph
        plt.plot(live_data.index, live_data["Close"], label="Actual Close", color='blue')
        plt.plot(live_data.index, live_data["SMA50"], label="SMA50", linestyle='--', color='green')
        plt.plot(live_data.index, live_data["SMA200"], label="SMA200", linestyle='--', color='purple')
        plt.scatter(live_data.index[-1], predicted_price, color='red', label="Predicted Price")
        plt.title(f"Live Stock Analysis for {ticker}\nMarket: {market_condition.capitalize()} - {advice}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend(loc="best")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Second subplot for company details and metrics
        plt.subplot(2, 1, 2)
        plt.axis("off")

        latest_price = live_data["Close"].iloc[-1]
        if isinstance(latest_price, pd.Series):  # Handle cases with duplicate timestamps
            latest_price = latest_price.values[0]

        # Format prices with appropriate currency symbol
        latest_price_formatted = format_price(latest_price, indian_stock)
        predicted_price_formatted = format_price(predicted_price, indian_stock)

        # Display key metrics and description
        plt.text(0.01, 0.9, f"Latest Price: {latest_price_formatted}", fontsize=12)
        plt.text(0.01, 0.8, f"Predicted Price: {predicted_price_formatted}", fontsize=12)
        plt.text(0.01, 0.7, f"Market Condition: {market_condition.capitalize()}", fontsize=12)
        plt.text(0.01, 0.6, f"Advice: {advice}", fontsize=12)

        if company_details and not description_displayed:
            plt.text(0.01, 0.5, f"All-Time High: ₹{company_details['all_time_high']}" if indian_stock else f"${company_details['all_time_high']}", fontsize=12)
            plt.text(0.01, 0.4, f"All-Time Low: ₹{company_details['all_time_low']}" if indian_stock else f"${company_details['all_time_low']}", fontsize=12)
            plt.text(0.01, 0.3, f"Market Cap: ₹{company_details['market_cap']:,}" if indian_stock else f"${company_details['market_cap']:,}", fontsize=12)
            plt.text(0.01, 0.2, f"P/E Ratio: {company_details['pe_ratio']}", fontsize=12)
            plt.text(0.01, 0.1, f"Avg Volume: {company_details['avg_volume']:,}", fontsize=12)

            # Long description text at the bottom
            description = company_details['description']
            plt.text(0.01, -0.1, f"Company Description: {description[:500]}...", fontsize=12, wrap=True)

            description_displayed = True  # Mark description as displayed

        plt.pause(60)  # Update every 60 seconds

# Function to plot historical data with date range selection
def plot_historical_data(ticker):
    # Input validation for dates
    while True:
        start_date_input = input("Enter the start date (YYYY-MM-DD): ")
        end_date_input = input("Enter the end date (YYYY-MM-DD): ")
        try:
            start_date = datetime.strptime(start_date_input, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_input, "%Y-%m-%d")
            if start_date >= end_date:
                print("Start date must be before end date. Please try again.")
            else:
                break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # Fetch historical data
    historical_data = yf.download(ticker, start=start_date, end=end_date)
    if historical_data.empty:
        print("No data available for the selected date range.")
        return

    # Calculate technical indicators
    historical_data = calculate_technical_indicators(historical_data)

    # Plotting
    plt.figure(figsize=(14, 10))

    # Candlestick Chart
    plt.subplot(3, 1, 1)
    plt.plot(historical_data.index, historical_data["Close"], label="Close Price", color='blue')
    plt.plot(historical_data.index, historical_data["SMA50"], label="SMA50", linestyle='--', color='green')
    plt.plot(historical_data.index, historical_data["SMA200"], label="SMA200", linestyle='--', color='purple')
    plt.title(f"Historical Stock Analysis for {ticker} from {start_date.date()} to {end_date.date()}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.xticks(rotation=45)

    # RSI Plot
    plt.subplot(3, 1, 2)
    plt.plot(historical_data.index, historical_data["RSI"], label="RSI", color='orange')
    plt.axhline(70, linestyle='--', color='red', label="Overbought (70)")
    plt.axhline(30, linestyle='--', color='green', label="Oversold (30)")
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend(loc="best")
    plt.xticks(rotation=45)

    # Volume Plot
    plt.subplot(3, 1, 3)
    try:
        plt.bar(historical_data.index, historical_data["Volume"], color='gray', label="Volume")
    except TypeError as e:
        print(f"Error plotting volume: {e}")
        print("Volume data:", historical_data["Volume"].head())
    plt.title("Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend(loc="best")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Calculate summary metrics
    close_prices = historical_data["Close"]
    if len(close_prices) > 0:
        # Calculate percentage change as a scalar value
        percentage_change = ((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]) * 100

        # Calculate highest and lowest prices
        highest_price = close_prices.max()
        lowest_price = close_prices.min()

        # Convert highest_price and lowest_price to single numerical values
        if isinstance(highest_price, pd.Series):
            highest_price = highest_price.iloc[0]
        if isinstance(lowest_price, pd.Series):
            lowest_price = lowest_price.iloc[0]

        # Calculate average daily return and volatility
        daily_returns = close_prices.pct_change().dropna()  # Drop NaN values from daily returns
        if not daily_returns.empty:
            average_daily_return = daily_returns.mean() * 100  # Scalar value
            volatility = daily_returns.std() * 100  # Scalar value
        else:
            average_daily_return = 0
            volatility = 0

        # Handle edge cases for Sharpe Ratio calculation
        if isinstance(average_daily_return, pd.Series):
            average_daily_return = average_daily_return.iloc[0]
        if isinstance(volatility, pd.Series):
            volatility = volatility.iloc[0]

        if np.isinf(average_daily_return) or np.isnan(average_daily_return):
            average_daily_return = 0
        if np.isnan(volatility):
            volatility = 0

        # Calculate Sharpe Ratio
        if volatility != 0 and not np.isnan(average_daily_return) and not np.isnan(volatility):
            sharpe_ratio = average_daily_return / volatility
        else:
            sharpe_ratio = 0

        # Determine market condition
        market_condition = determine_market_condition(historical_data)
        sma_crossover = "50-day SMA crossed above 200-day SMA" if historical_data["SMA50"].iloc[-1] > historical_data["SMA200"].iloc[-1] else "50-day SMA did not cross above 200-day SMA"

        # Display historical analysis summary
        print("\n--- Historical Analysis Summary ---\n")
        print("\033[1mStock:\033[0m", ticker)
        print("\033[1mDate Range:\033[0m", f"{start_date.date()} to {end_date.date()}")
        print("\033[1mHighest Price:\033[0m", format_price(highest_price, is_indian_stock(ticker)))
        print("\033[1mLowest Price:\033[0m", format_price(lowest_price, is_indian_stock(ticker)))
        print("\033[1mPercentage Change:\033[0m", f"{percentage_change.iloc[0]:.2f}%")
        print("\033[1mAverage Daily Return:\033[0m", f"{average_daily_return:.2f}%")
        print("\033[1mVolatility:\033[0m", f"{volatility:.2f}%")
        print("\033[1mSharpe Ratio:\033[0m", f"{sharpe_ratio:.2f}")
        print("\033[1mMarket Trend:\033[0m", market_condition.capitalize())
        print("\033[1m{}\033[0m during this period.".format(sma_crossover))
        print("\033[1mMarket Situation:\033[0m", 'Stable' if market_condition == 'neutral' else 'Volatile')
        print("\n----------------------------------\n")
    else:
        print("No data available for the selected date range.")

# Helper function to calculate technical indicators
def calculate_technical_indicators(data):
    # Calculate SMA50 and SMA200
    data['SMA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['SMA200'] = data['Close'].rolling(window=200, min_periods=1).mean()

    # Calculate RSI
    data['RSI'] = calculate_rsi(data['Close'], window=14)

    return data

# Helper function to calculate RSI
def calculate_rsi(series, window=14):
    delta = series.diff()  # Calculate price changes
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()  # Average gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()  # Average loss
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

# Helper function to determine market condition
def determine_market_condition(data):
    # Ensure SMA50, SMA200, and RSI are scalars
    sma50 = data["SMA50"].iloc[-1]
    sma200 = data["SMA200"].iloc[-1]
    rsi = data["RSI"].iloc[-1]

    if sma50 > sma200 and rsi > 50:
        return "bullish"
    elif sma50 < sma200 and rsi < 50:
        return "bearish"
    else:
        return "neutral"

# Helper function to format price based on whether it's an Indian stock
def format_price(price, is_indian_stock):
    if is_indian_stock:
        return f"₹{price:.2f}"  # Format price in Indian Rupees
    else:
        return f"${price:.2f}"  # Format price in US Dollars

# Helper function to check if the stock is Indian
def is_indian_stock(ticker):
    # Indian stocks typically have ".NS" suffix on Yahoo Finance
    return ticker.endswith(".NS")
# Main execution
if __name__ == "__main__":
    historical_data = prepare_historical_data(ticker)
    if historical_data is None or historical_data.empty:
        print("No historical data available. Exiting.")
    else:
        lstm_model, data_scaler = train_model(historical_data)
        print("Model trained. Starting live analysis...")

        # Ask user if they want to see historical data or live analysis
        user_choice = input("Do you want to see historical data (H) or live analysis (L)? ").strip().upper()

        if user_choice == "H":
            plot_historical_data(ticker)
        elif user_choice == "L":
            try:
                plot_real_time(historical_data, None, lstm_model, data_scaler, seq_length=100)
            except KeyboardInterrupt:
                print("\nExiting program.")
        else:
            print("Invalid choice. Exiting.")
