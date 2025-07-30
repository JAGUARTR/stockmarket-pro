import plotly.graph_objects as go
import dateutil
import pandas_ta as pta
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

def plotly_table(dataframe):
    headerColor = 'grey'
    rowEvenColor = '#f0fafd'
    rowOddColor = '#e1efff'

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f'<b>{i}</b>' for i in dataframe.columns],
            line_color='#0078ff', fill_color='#0078ff',
            align='center', font=dict(color='white', size=15), height=35,
        ),
        cells=dict(
            # The logic in the image for `values` was slightly incorrect for Plotly.
            # The corrected version to display the index and columns is below.
            values=[dataframe.index] + [dataframe[col] for col in dataframe.columns],
            line_color='white',
            # Alternating row colors
            fill_color=[[rowOddColor, rowEvenColor] * (len(dataframe)//2)],
            align='left', font=dict(color="black", size=15)
        ))
    ])

    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def filter_data(dataframe, num_period):
    """
    Filters a dataframe based on a specified time period string.
    """
    if num_period == '1mo':
        # Calculate start date for 1 month ago
        date = dataframe.index[-1] - relativedelta(months=-1)
    elif num_period == '5d':
        # Calculate start date for 5 days ago
        date = dataframe.index[-1] - relativedelta(days=-5)
    elif num_period == '6mo':
        # Calculate start date for 6 months ago
        date = dataframe.index[-1] - relativedelta(months=-6)
    elif num_period == '1y':
        # Calculate start date for 1 year ago
        date = dataframe.index[-1] - relativedelta(years=-1)
    elif num_period == '5y':
        # Calculate start date for 5 years ago
        date = dataframe.index[-1] - relativedelta(years=-5)
    elif num_period == 'ytd':
        # Calculate start date for Year to Date (Jan 1st of the current year)
        date = datetime.datetime(dataframe.index[-1].year, 1, 1).strftime('%Y-%m-%d')
    else: # Default case, e.g., 'max'
        # Use the earliest date in the dataframe
        date = dataframe.index[0]
        
    # Filter the dataframe to return only the rows after the calculated start date
    return dataframe[dataframe.index > date]


# --- Updated Chart Generation Functions ---

def close_chart(dataframe, num_period=False):
    """
    Creates a line chart of the closing price, with optional date filtering.
    """
    if num_period:
        # If a period is provided, filter the data first
        dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Open'],
        mode='lines', name='Open', line=dict(width=2, color='#5ab7ff')))
    
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Close'],
        mode='lines', name='Close', line=dict(width=2, color='black')))

    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['High'],
        mode='lines', name='High', line=dict(width=2, color='#007bff')))

    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Low'],
        mode='lines', name='Low', line=dict(width=2, color='red')))

    # Update axes and layout for a professional look
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(height=500, margin=dict(l=0, r=20, t=20, b=0),
                      plot_bgcolor='white', paper_bgcolor='#e1e1ff',
                      legend=dict(yanchor="top", xanchor="right"))
    
    return fig

def candlestick(dataframe, num_period=False):
    """
    Creates a candlestick chart, with optional date filtering.
    """
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    fig = go.Figure(data=[go.Candlestick(x=dataframe.index,
                open=dataframe['Open'], high=dataframe['High'],
                low=dataframe['Low'], close=dataframe['Close'])])
    fig.update_layout(title_text=f'Candlestick Chart ({num_period or "Max"})', xaxis_rangeslider_visible=False)
    return fig

def RSI(dataframe, num_period=False):
    """
    Creates an RSI indicator chart, with optional date filtering.
    """
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    # Placeholder for actual RSI calculation
    fig.add_trace(go.Scatter(x=dataframe.index, y=[70]*len(dataframe), line=dict(dash='dash'), name='Overbought'))
    fig.add_trace(go.Scatter(x=dataframe.index, y=[30]*len(dataframe), line=dict(dash='dash'), name='Oversold'))
    fig.update_layout(title_text=f'RSI Indicator ({num_period or "Max"})')
    return fig

def RSI(dataframe, num_period):
    # The image shows pta.rsi(dataframe['Close']) but it likely should use num_period
    dataframe['RSI'] = pta.rsi(dataframe['Close'], length=num_period)
    dataframe = filter_data(dataframe,num_period)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe.RSI, name = 'RSI', marker_color='orange',line = dict( width=2, color = 'orange'),
    ))

    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[70]*len(dataframe), name = 'Overbought', marker_color='red',line = dict( width=2, color = 'red',dash='dash'),
    ))

    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[30]*len(dataframe), fill='tonexty', name = 'Oversold', marker_color='#79da84',line = dict( width=2, color = '#79da84',dash='dash')
    ))

    fig.update_layout(yaxis_range=[0,100],
    height=200,plot_bgcolor = 'white', paper_bgcolor = '#e1efff',margin=dict(l=0, r=0, t=0, b=0),legend=dict(orientation="h",
    yanchor="top",
    y=1.02,
    xanchor="right",
    x=1),
    )
    
    # To display the figure
    fig.show()  

def MACD(dataframe, num_period=False):
    """
    Creates a MACD indicator chart, with optional date filtering.
    """
    if num_period:
        dataframe = filter_data(dataframe, num_period)
    fig = go.Figure()
    # Placeholder for actual MACD calculation
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Close'].rolling(window=26).mean() - dataframe['Close'].rolling(window=12).mean(), name='MACD'))
    fig.update_layout(title_text=f'MACD Indicator ({num_period or "Max"})')
    return fig

def MACD(dataframe, num_period):
    # Calculate MACD components (optimized to one call)
    macd_df = pta.macd(dataframe['Close'])
    macd = macd_df.iloc[:, 0]
    macd_signal = macd_df.iloc[:, 1]
    macd_hist = macd_df.iloc[:, 2]
    
    dataframe['MACD'] = macd
    dataframe['MACD Signal'] = macd_signal
    dataframe['MACD Hist'] = macd_hist
    dataframe = filter_data(dataframe, num_period)
    
    fig = go.Figure()
    
    # Add the MACD line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD'], name = 'MACD', # Corrected name
        marker_color='orange', line = dict(width=2, color = 'orange'),
    ))

    # Add the Signal line
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD Signal'], name = 'Signal', # Corrected name
        marker_color='blue', line = dict(width=2, color = 'blue', dash='dash'), # Changed to blue for clarity
    ))

    # Prepare colors and add the Histogram bars (this part was missing)
    colors = ['green' if val >= 0 else 'red' for val in dataframe['MACD Hist']]
    fig.add_trace(go.Bar(
        x=dataframe['Date'],
        y=dataframe['MACD Hist'],
        name='Histogram',
        marker_color=colors
    ))

    # The update_layout call from the image is incomplete, but would look like this:
    fig.update_layout(
        title='MACD Indicator',
        plot_bgcolor='white',
        paper_bgcolor='#e1efff'
    )
    
    fig.show()

def Moving_average_forecast(forecast):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast.index[:-30], 
        y=forecast['Close'].iloc[:-30],
        mode='lines',
        name='Close Price',
        line=dict(width=2, color='black')
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index[-31:], 
        y=forecast['Close'].iloc[-31:],
        mode='lines',
        name='Future Close Price',
        line=dict(width=2, color='red')
    ))

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(
            yanchor="top",
            xanchor="right"
        )
    )

    return fig