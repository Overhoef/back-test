import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
    # --- Strategy Selection ---
    strategy_choice = st.sidebar.selectbox("Select Strategy", 
                                      ["SMA Crossover", "Bollinger Bands", "MACD", "Custom"], index=1)

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
        period = st.sidebar.slider("Period", 20, 60, 20, 1)
        std_dev = st.sidebar.slider("Standard Deviations", 1, 3, 2, 1)
        data['SMA'] = data['Close'].rolling(window=period).mean()
        data['STD'] = data['Close'].rolling(window=period).std()
        data['BB_upper'] = data['SMA'] + (std_dev * data['STD'])
        data['BB_lower'] = data['SMA'] - (std_dev * data['STD'])
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

# --- Backtesting Logic ---

data['Balance'] = initial_balance
data['Position_Size'] = 0.0
data['Position'] = 0.0  # Initialize position

for i in range(1, len(data)):
    if data['Signal'].iloc[i] == 1:  # Buy Signal
        if data['Position'].iloc[i - 1] != 1:  # Enter long position if not already long
            trade_amount = trade_size(data['Balance'].iloc[i - 1])
            try:
                data['Position_Size'].iloc[i] = trade_amount / data['Close'].iloc[i]  # Calculate position size in units
                data['Balance'].iloc[i] = data['Balance'].iloc[i - 1] 
            except ZeroDivisionError:
                st.warning("Encountered potential division by zero. Skipping this trade.")
                data['Position_Size'].iloc[i] = 0
                data['Balance'].iloc[i] = data['Balance'].iloc[i - 1]
        else:  # Maintain long position
            data['Position_Size'].iloc[i] = data['Position_Size'].iloc[i - 1]
            data['Balance'].iloc[i] = data['Balance'].iloc[i - 1] 
    elif data['Signal'].iloc[i] == -1:  # Sell Signal
        if data['Position'].iloc[i - 1] != -1:  # Enter short position if not already short
            trade_amount = trade_size(data['Balance'].iloc[i - 1])
            try:
                data['Position_Size'].iloc[i] = -trade_amount / data['Close'].iloc[i]  # Calculate position size for short
                data['Balance'].iloc[i] = data['Balance'].iloc[i - 1] 
            except ZeroDivisionError:
                st.warning("Encountered potential division by zero. Skipping this trade.")
                data['Position_Size'].iloc[i] = 0
                data['Balance'].iloc[i] = data['Balance'].iloc[i - 1]
        else:  # Maintain short position
            data['Position_Size'].iloc[i] = data['Position_Size'].iloc[i - 1]
            data['Balance'].iloc[i] = data['Balance'].iloc[i - 1] 
    else:  # No signal
        if data['Position_Size'].iloc[i - 1] != 0:  # Exit position if any
            profit_loss = data['Position_Size'].iloc[i - 1] * (data['Close'].iloc[i] - data['Close'].iloc[i - 1])
            data['Balance'].iloc[i] = data['Balance'].iloc[i - 1] + profit_loss
            data['Position_Size'].iloc[i] = 0
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


# --- Calculate Number of Trades ---
num_trades = data['Position'].diff().abs().sum()
# --- Calculate Number of Long Trades ---
num_long_trades = (data['Position'] == 1).sum()
# --- Calculate Number of Short Trades ---
num_short_trades = (data['Position'] == -1).sum() 

# last 24h % change
latest_date = data.index[-1]
previous_date = data.index[-2]
latest_close = data.loc[latest_date, "Close"]
previous_close = data.loc[previous_date, "Close"]
price_change = latest_close - previous_close
percent_change = (price_change / previous_close) * 100

metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(6)

with metric1:
# Price change 24h
    if price_change > 0:
        st.metric(
            label=f"24h verandering in %",
            value=f"{latest_close:.2f}",
            delta=f"{percent_change:.2f}% ↑",
        )
    else:
        st.metric(
            label=f"24h verandering in %",
            value=f"{latest_close:.2f}",
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

with metric5:
    st.metric(
        label=f"Long Trades",
        value=f"{int(num_long_trades)}", 
    )

with metric6:
    st.metric(
        label=f"Short Trades",
        value=f"{int(num_short_trades)}", 
    )

# Create subplots and adjust layout
# Create subplots and adjust layout
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                   vertical_spacing=0.05, 
                   subplot_titles=("", "Volume"), 
                   row_width=[0.3, 0.7]) 

# Add Candlestick trace
candle = go.Candlestick(
    x=data.index, 
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick"
)
fig.add_trace(candle, row=1, col=1)

# Add SMA 9 line
data['SMA_9'] = data['Close'].rolling(window=9).mean()
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_9'], line_color='orange', opacity=0.9, name="SMA 9"), row=1, col=1)

# Add SMA 9 line
data['SMA_30'] = data['Close'].rolling(window=30).mean()
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_30'], line_color='blue', opacity=0.9, name="SMA 30"), row=1, col=1)

# Add Buy Signals
fig.add_trace(go.Scatter(
    x=data[data['Signal'] == 1].index, 
    y=data['Close'][data['Signal'] == 1], 
    mode='markers', 
    name='Buy Signal', 
    marker=dict(color='green', size=6, symbol='triangle-up')
), row=1, col=1)

# Add Sell Signals
fig.add_trace(go.Scatter(
    x=data[data['Signal'] == -1].index, 
    y=data['Close'][data['Signal'] == -1], 
    mode='markers', 
    name='Sell Signal', 
    marker=dict(color='red', size=6, symbol='triangle-down')
), row=1, col=1)

# Add Volume Bar trace
fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", opacity=0.7), row=2, col=1)

# Hide range slider for OHLC chart
fig.update(layout_xaxis_rangeslider_visible=False)

fig.update_layout(width=1250, height=600)

st.subheader(selected_function)
st.plotly_chart(fig)

# --- Display Charts ---

chart1, chart2 = st.columns(2)

with chart1:
    fig1 = go.Figure()

    # Add the zero line
    fig1.add_trace(go.Scatter(x=data.index, y=data['Zero_Line'], line=dict(color='#101010', dash='dot'), name='0 line'))

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
        name='P/L'
    ))

    # Calculate y-axis range based on initial balance and percentage
    y_range_min = -1 * initial_balance
    y_range_max = 1 * initial_balance
    fig1.update_layout(
        title='Cumulative Profit/Loss', 
        yaxis_title='Cumulative P/L', 
        yaxis_range=[y_range_min, y_range_max],  # Set y-axis range dynamically
        legend=dict(x=0, y=1, xanchor='left', yanchor='top')
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
    fig2.update_layout(title='Percentage Change', yaxis_title='Percentage Change (%)', legend=dict(x=0, y=1, xanchor='left', yanchor='top'))
    st.plotly_chart(fig2)
