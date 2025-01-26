import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Config
st.set_page_config(layout="wide")  # Set wide mode as default

# forex 1 minute
def usdjpy():
    return pd.read_csv('USDJPY_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

def eurgbp():
    return pd.read_csv('EURGBP_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

def eurusd():
    return pd.read_csv('EURUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

def gbpusd():
    return pd.read_csv('GBPUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

# Stocks
def nvidia():
    return pd.read_csv('NVDA.USUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

# metals
def xauusd():
    return pd.read_csv('XAUUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

# crypto
def ethusd():
    return pd.read_csv('ETHUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

def btcusd():
    return pd.read_csv('BTCUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

# indexes
def dollar_index():
    return pd.read_csv('USA30.IDXUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

def dollar_volatility():
    return pd.read_csv('VOL.IDXUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

def us500():
    return pd.read_csv('USA500.IDXUSD_Candlestick_1_Hour_BID_01.01.2024-01.01.2025.csv')

def us30():
    return pd.read_csv('')

def show_data(data):
    st.dataframe(data)

data_options = {
    usdjpy: "USDJPY 1 Hour",
    eurgbp: "EURGBP 1 Hour",
    nvidia: "Nvidia 1 Hour", 
    eurusd:"EURUSD 1 Hour",
    gbpusd:"GBPUSD 1 Hour",
    xauusd:"XAUUSD 1 Hour",
    ethusd:"ETHUSD 1 Hour",
    btcusd:"BTCUSD 1 Hour",
    dollar_index:"DXY 1 Hour",
    dollar_volatility:"Dollar Volatility 1 Hour",
    us500:"US500 1 Hour",
    us30:"US30 1 Hour",
    # ... add other pairs here ... 
}

selected_function = st.sidebar.selectbox("Data", list(data_options.values())) 

# Get the corresponding function from the dictionary
selected_data = list(data_options.keys())[list(data_options.values()).index(selected_function)] 

# Call the selected function to get the data
data = selected_data() 

data['Local time'] = pd.to_datetime(data['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')

def SMA(array, period):
    return array.rolling(period).mean()

st.title("Backtesting Framework")


# sidebar
with st.sidebar:
    # fast = st.slider('fast sma', min_value=0,max_value=100, value=9)
    # long = st.slider('long sma', min_value=0,max_value=100, value=30)

    # data["SMA fast"] = SMA(data["Close"], fast)
    # data["SMA long"] = SMA(data["Close"], long)

    # initial_balance = st.sidebar.number_input("Initial Balance", value=10000.0, min_value=0.0)
    # trade_size_type = st.sidebar.radio("Trade Size Type", ("Percent", "Value"))

    # if trade_size_type == "Percent":
    #     trade_size_percent = st.sidebar.slider("Trade Size (%)", 1, 100, 10, 1)
    #     trade_size = lambda balance: balance * trade_size_percent / 100
    # else:
    #     trade_size_value = st.sidebar.number_input("Trade Size (Value)", value=100.0, min_value=0.0)
    #     trade_size = lambda balance: trade_size_value

    # --- Strategy Selection ---
    strategy_choice = st.sidebar.selectbox("Select Strategy", 
                                      ["SMA Crossover", "Bollinger Bands", "MACD", "Custom"])

    # --- Strategy Parameters ---
    if strategy_choice == "SMA Crossover":
        short_window = st.sidebar.slider("Short Window", 5, 250, 50, 5)
        long_window = st.sidebar.slider("Long Window", 50, 500, 200, 5)
        data['SMA fast'] = data['Close'].rolling(window=short_window).mean()
        data['SMA long'] = data['Close'].rolling(window=long_window).mean()
        data['Signal'] = 0.0
        data['Signal'] = np.where(data['SMA fast'] > data['SMA long'], 1.0, 0.0)
        data['Position'] = data['Signal'].diff()

    elif strategy_choice == "Bollinger Bands":
        period = st.sidebar.slider("Period", 20, 60, 20, 1)
        std_dev = st.sidebar.slider("Standard Deviations", 1, 3, 2, 1)
        data['SMA'] = data['Close'].rolling(window=period).mean()
        data['STD'] = data['Close'].rolling(window=period).std()
        data['BB_upper'] = data['SMA'] + (std_dev * data['STD'])
        data['BB_lower'] = data['SMA'] - (std_dev * data['STD'])
        data['Signal'] = 0.0
        data['Signal'] = np.where(data['Close'] < data['BB_lower'], 1.0, 0.0)  # Buy when price crosses below lower band
        data['Signal'] = np.where(data['Close'] > data['BB_upper'], -1.0, data['Signal'])  # Sell when price crosses above upper band
        data['Position'] = data['Signal'].diff()

    elif strategy_choice == "MACD":
        fast_period = st.sidebar.slider("Fast Period", 12, 26, 12, 1)
        slow_period = st.sidebar.slider("Slow Period", 26, 52, 26, 1)
        signal_period = st.sidebar.slider("Signal Period", 9, 15, 9, 1)
        data['EMA_fast'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['MACD_Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
        data['Signal'] = 0.0
        data['Signal'] = np.where(data['MACD'] > data['MACD_Signal'], 1.0, 0.0)  # Buy when MACD crosses above Signal line
        data['Signal'] = np.where(data['MACD'] < data['MACD_Signal'], -1.0, data['Signal'])  # Sell when MACD crosses below Signal line
        data['Position'] = data['Signal'].diff()

    elif strategy_choice == "Custom":
        # Allow users to define custom logic here
        # (e.g., using code blocks or a custom configuration interface)
        st.sidebar.write("Custom strategy development is currently under construction.")

# --- Trading Parameters ---

st.sidebar.header("Account")

initial_balance = st.sidebar.number_input("Initial Balance", value=10000.0, min_value=0.0)
trade_size_type = st.sidebar.radio("Trade Size Type", ("Percent", "Lot Size"))

if trade_size_type == "Percent":
    trade_size_percent = st.sidebar.slider("Trade Size (%)", 1, 50, 10, 1)
    trade_size = lambda balance: balance * trade_size_percent / 100
else:
    lot_size = st.sidebar.number_input("Lot Size", value=0.1, min_value=0.01, step=0.01)
    trade_size = lambda balance: lot_size * 100000  # Assuming 1 lot = 100,000 units

# --- Backtesting Logic (Generic Framework) ---

data['Balance'] = initial_balance
data['Position_Size'] = 0.0

for i in range(1, len(data)):
    if data['Position'].iloc[i] == 1:  # Buy Signal
        trade_amount = trade_size(data['Balance'].iloc[i - 1])
        try:
            data['Position_Size'].iloc[i] = trade_amount / data['Close'].iloc[i]  # Calculate position size in units
            data['Balance'].iloc[i] = data['Balance'].iloc[i - 1] 
        except ZeroDivisionError:
            st.warning("Encountered potential division by zero. Skipping this trade.")
            data['Position_Size'].iloc[i] = 0
            data['Balance'].iloc[i] = data['Balance'].iloc[i - 1]
    elif data['Position'].iloc[i] == -1:  # Sell Signal
        if data['Position_Size'].iloc[i - 1] > 0: 
            profit_loss = data['Position_Size'].iloc[i - 1] * (data['Close'].iloc[i] - data['Close'].iloc[i - 1])
            data['Balance'].iloc[i] = data['Balance'].iloc[i - 1] + profit_loss
            data['Position_Size'].iloc[i] = 0
        else:
            data['Balance'].iloc[i] = data['Balance'].iloc[i - 1]
    else:
        data['Balance'].iloc[i] = data['Balance'].iloc[i - 1]


# --- Calculate Percentage Change ---
data['Percentage_Change'] = ((data['Balance'] - initial_balance) / initial_balance) * 100

# --- Calculate Daily P/L ---
data['Daily_Pnl'] = data['Balance'].diff()
data['Cumulative_Pnl'] = data['Daily_Pnl'].cumsum()

# --- Create Data for Chart 1 Shading ---
data['Zero_Line'] = 0.0 
data['Positive_Area'] = np.where(data['Cumulative_Pnl'] > 0, data['Cumulative_Pnl'], 0)
data['Negative_Area'] = np.where(data['Cumulative_Pnl'] < 0, data['Cumulative_Pnl'], 0)

# --- Calculate Win Rate ---
data['Trade_Result'] = np.where(data['Daily_Pnl'] > 0, 1, 0)  # 1 for win, 0 for loss or no trade
win_rate = data['Trade_Result'].sum() / len(data) * 100

# last 24h % change
latest_date = data.index[-1]
previous_date = data.index[-2]
latest_close = data.loc[latest_date, "Close"]
previous_close = data.loc[previous_date, "Close"]
price_change = latest_close - previous_close
percent_change = (price_change / previous_close) * 100

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
# Price change 24h
    if price_change > 0:
        st.metric(
            label=f"24h verandering in %",
            value=f"${latest_close:.2f}",
            delta=f"{percent_change:.2f}% ↑",
        )
    else:
        st.metric(
            label=f"24h verandering in %",
            value=f"${latest_close:.2f}",
            delta=f"{percent_change:.2f}% ↓",
        )

with metric2:
    # Backtest performance (e.g., Final Balance)
    st.metric(
        label=f"Backtest Performance",
        value=f"${data['Balance'].iloc[-1]:.2f}", 
        delta=f"${data['Balance'].iloc[-1] - initial_balance:.2f}" 
    )

with metric3:
    # Return on Investment (ROI)
    roi = ((data['Balance'].iloc[-1] - initial_balance) / initial_balance) * 100
    st.metric(
        label=f"Return on Investment (ROI)",
        value=f"{roi:.2f}%",
    )

with metric4:
    st.metric(
        label=f"Win Rate",
        value=f"{win_rate:.2f}%", 
    )

# Create a candlestick trace
candle = go.Candlestick(
    x=data.index,  # X-axis data (timestamps)
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick"
)

# # Create SMA traces
# trace_sma_fast = go.Scatter(x=data.index, y=data['SMA fast'], line_color='gray', opacity=0.7, name="Fast SMA")
# trace_sma_long = go.Scatter(x=data.index, y=data['SMA long'], line_color='blue', opacity=0.7, name="Slow SMA")

# Create scatter points for Buy Signals
buy = go.Scatter(
    x=data[data['Position'] == 1].index, 
    y=data['Close'][data['Position'] == 1], 
    mode='markers', 
    name='Buy Signal', 
    marker=dict(color='green', size=10, symbol='triangle-up')
)

# Create scatter points for Sell Signals
sell = go.Scatter(
    x=data[data['Position'] == -1].index, 
    y=data['Close'][data['Position'] == -1], 
    mode='markers', 
    name='Sell Signal', 
    marker=dict(color='red', size=10, symbol='triangle-down')
)

# Create the figure
fig = go.Figure(data=[candle, buy, sell])

fig.update_layout(width=1200, height=600)

st.plotly_chart(fig)

# --- Display Charts ---

chart1, chart2 = st.columns(2)

with chart1:
    fig1 = go.Figure()

    # Add the zero line
    fig1.add_trace(go.Scatter(x=data.index, y=data['Zero_Line'], line=dict(color='black', dash='dash')))

    # Add the positive area fill
    fig1.add_trace(go.Scatter(
        x=data.index,
        y=data['Positive_Area'],
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)',  # Green with transparency
        line=dict(color='rgba(0, 0, 0, 0)'),  # Transparent line
        name='profit'
    ))

    # Add the negative area fill
    fig1.add_trace(go.Scatter(
        x=data.index,
        y=data['Negative_Area'],
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)',  # Red with transparency
        line=dict(color='rgba(0, 0, 0, 0)'),  # Transparent line
        name='loss'
    ))

    # Add the cumulative profit/loss line on top of the fills
    fig1.add_trace(go.Scatter(
        x=data.index,
        y=data['Cumulative_Pnl'],
        line=dict(color='gray'),
        name='Cumulative P/L'
    ))

    # Calculate y-axis range based on initial balance and percentage
    y_range_min = -0.125 * initial_balance
    y_range_max = 0.125 * initial_balance
    fig1.update_layout(
        title='Cumulative Profit/Loss', 
        yaxis_title='Cumulative P/L', 
        yaxis_range=[y_range_min, y_range_max]  # Set y-axis range dynamically
    )
    st.plotly_chart(fig1)

with chart2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=data.index,
        y=data['Percentage_Change'], 
        line=dict(color='green'),
        name='Percentage Change'
    ))
    fig2.update_layout(title='Percentage Change', yaxis_title='Percentage Change (%)')
    st.plotly_chart(fig2)
