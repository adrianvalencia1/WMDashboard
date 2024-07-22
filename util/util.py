import colorsys
import math
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fredapi import Fred
from scipy.optimize import minimize

# Function to get the x-axis values as an array
def get_x_axis_values(fig):
    x_values = []
    for trace in fig.data:
        x_values.extend(trace['x'])
    return x_values

def threshold_line(figure, threshold_input, values, threshold_type, threshold_direction):
    values_rounded = np.round(values, 2)
    if threshold_input is not None:
        try:
            if threshold_input >= values[0]:
                if threshold_direction == 'up':
                    threshold = np.where(values_rounded >= threshold_input)[-1][0]
                elif threshold_direction == 'down':
                    threshold = np.where(values_rounded >= threshold_input)[-1][-1]
            else:
                if threshold_direction == 'up':
                    threshold = np.where(values_rounded <= threshold_input)[0][0]
                elif threshold_direction == 'down':
                    threshold = np.where(values_rounded <= threshold_input)[0][-1]
            match threshold_type:
                case "mean":
                    line_color = '#636EFA'
                    annotation_text = f"On average, this portfolio will reach ${threshold_input} in {get_x_axis_values(figure)[threshold]}"
                case "median":
                    line_color = '#EF553B'
                    annotation_text = f"At the median, this portfolio will reach ${threshold_input} in {get_x_axis_values(figure)[threshold]}"
                case "95":
                    line_color = '#00CC96'
                    annotation_text = f"In a 'best case' scenario, this portfolio will reach ${threshold_input} in {get_x_axis_values(figure)[threshold]}"
                case "5":
                    line_color = '#AB63FA'
                    annotation_text = f"In a 'worst case' scenario, this portfolio will reach ${threshold_input} in {get_x_axis_values(figure)[threshold]}"
                case _:
                    line_color = 'black'
                    annotation_text = f"This portfolio will reach ${threshold_input} in {get_x_axis_values(figure)[threshold]} "
            
            x_index = get_x_axis_values(figure)[threshold]
            if type(x_index) is str:
                x_index = datetime.strptime(x_index, "%Y-%m").timestamp() * 1000 # bug in plotly, using seconds since epoch as workaround
            figure.add_vline(
                x=x_index, line_width=3, 
                line_dash="dash", 
                line_color=line_color, 
                annotation_text=annotation_text, 
                annotation_position="right",
                annotation_font_size=15,
            )

        except:
            print('')

def get_vlines_from_figure(figure):
    vlines = []

    if 'shapes' in figure['layout']:
        for shape in figure['layout']['shapes']:
            if shape['type'] == 'line' and shape['x0'] == shape['x1']:
                vlines.append({
                    'x': shape['x0'],
                    'y0': shape['y0'],
                    'y1': shape['y1']
                })
    
    return vlines

def update_paragraph(figure, threshold, threshold_type, threshold_direction):
    

    paragraph = f"On average, the portfolio is expected to grow to ${max(figure['data'][-4]['y']):.2f} over {len(figure['data'][-4]['x'])-1} years. The median outcome is slightly lower at ${max(figure['data'][-3]['y']):.2f}, as the average is skewed by a few high-performing simulations. The analysis also shows that the portfolio is likely to be within the range of ${max(figure['data'][-1]['y']):.2f} to ${max(figure['data'][-2]['y']):.2f} with 95% confidence (within two standard deviations)."


    if threshold_direction == "up":
        direction = "reach"
    else:
        direction = "drop to"
    if threshold_type == "5":
        threshold_type = "5th percentile"
    elif threshold_type == "95":
        threshold_type = "95th percentile"

    vlines = get_vlines_from_figure(figure)
    if vlines:
        paragraph += f" This portfolio's {threshold_type} will {direction} ${threshold} in {vlines[0]['x']} years."
    elif threshold is not None:
        if direction == "reach":
            if threshold > max(max(figure['data'], key=lambda x: x['y'])['y']):
                paragraph += f" This portfolio will never {direction} ${threshold} in {max(figure['data'][0]['x'])} years."
            else:
                paragraph += f" This portfolio's {threshold_type} will not {direction} ${threshold} in {max(figure['data'][0]['x'])} years."
        else:
            if threshold < min(min(figure['data'], key=lambda x: x['y'])['y']):
                paragraph += f" This portfolio will never {direction} ${threshold} in {max(figure['data'][0]['x'])} years."
            else:
                paragraph += f" This portfolio's {threshold_type} will not {direction} ${threshold} in {max(figure['data'][0]['x'])} years."

    if len(figure['data'])-4 < 500:
        paragraph += f" Using too few simulations (currently {len(figure['data'])-4}) can create variable results. Try increasing the number of simulations to increase accuracy."
    
    return paragraph

# TODO: Move to util file
def lighten_hex_color(hex_color, percent):
    # Convert hex to RGB
    hex_color = hex_color.strip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))
    
    # Convert RGB to HLS
    hls = colorsys.rgb_to_hls(*[x/255.0 for x in rgb])
    
    # Lighten the color
    lightness = hls[1] + percent/100.0
    lightness = min(max(lightness, 0.0), 1.0)
    
    # Convert HLS back to RGB
    rgb = tuple(int(round(x * 255.0)) for x in colorsys.hls_to_rgb(hls[0], lightness, hls[2]))
    
    # Convert RGB to hex
    hex_color = '#%02x%02x%02x' % rgb
    
    return hex_color

def get_adj_close(tickers, years) -> pd.DataFrame:
    tickers = tickers#['DIA', 'DIS', 'EFA', 'GOOGL', 'IJR', 'META', 'NVDA', 'QQQ', 'VNQ']#["SPY", "QQQ", "GLD", "NDX", "^GSPC", "^DJI", "CL=F", "GC=F", "EURUSD=X", "^N225", "BTC-USD", "AAPL", "RXO", "NVDA", "AMD", "TSM", "IVV", "AMZN", "META", "MSFT", "GOOG", "BRK-B"]

    end_date = datetime.today()
    start_date = end_date - timedelta(days = years * 365)
    #print(f"Fetching stock data from {start_date} to {end_date}:")

    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']

    return adj_close_df

def get_log_returns(df):

    log_returns = np.log(df / df.shift(1))
    log_returns = log_returns.dropna()

    return log_returns

def get_cov_matrix(log_returns, days=252):
    return log_returns.cov()*days # annual

# std dev
def std_dev(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights # transpose
    return np.sqrt(variance)

# expected return
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights) * 252

# sharpe ratio
def sharpe(weights, log_returns, cov_matrix, risk_free_rate):
    risk_premium = expected_return(weights, log_returns) - risk_free_rate
    stdev = std_dev(weights, cov_matrix)
    return risk_premium / stdev

def neg_sharpe(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe(weights, log_returns, cov_matrix, risk_free_rate)

def get_risk_free_rate():
    # 10 yr treasury rate as risk-free rate
    fred = Fred(api_key='57de28e694044d0d0c74470fb48e14ec')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100

    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    return risk_free_rate

def get_optimal_weights(tickers, tickers_df, low_bound=0, high_bound=1, method='SLSQP') -> list:
    # constriants and bounds
    # normalize
    constraints = {
        'type': 'eq', 
        'fun': lambda weights: np.sum(weights) - 1
    }
    bounds = [(low_bound,high_bound) for _ in range(len(tickers))] # max bound is max % of portfolio on asset

    # initial weights
    initial_weights = np.array([1/len(tickers)] * len(tickers))

    log_returns = get_log_returns(tickers_df)
    cov_matrix = get_cov_matrix(log_returns)
    risk_free_rate = get_risk_free_rate()

    # optimize the weights to maximize sharpe ratio, sequential least squares quadratic programming method
    optimized_results = minimize(neg_sharpe, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method=method, constraints=constraints, bounds=bounds)

    optimal_weights = optimized_results.x

    return optimal_weights

def build_portfolio(tickers, low_bound=0, high_bound=1, years=10, weights=None, adj_close_df=None) -> pd.DataFrame:
    if adj_close_df is None or adj_close_df.empty:
        adj_close_df = get_adj_close(tickers, years)

    # add more options in the future
    if weights == 'optimize':
        w = get_optimal_weights(tickers, adj_close_df, low_bound=low_bound, high_bound=high_bound)
    elif type(weights) is list:
        w = weights
    else: # equal weights
        w = np.array([1/len(tickers)] * len(tickers))

    portfolio = {}
    for ticker, weight in zip(tickers, w):
        portfolio.update({ticker: round(weight, 2)})

    df = pd.DataFrame.from_dict(portfolio, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index':'Ticker',0:'Weight'}, inplace=True)

    return df

def actual_return(weights, filtered_df):
    change = (filtered_df.iloc[-1] - filtered_df.iloc[0]) / filtered_df.iloc[0]
    return np.sum(change.mean()*weights)

def monthly_returns(adj_close_df, optimal_weights):

    adj_close_df_copy = adj_close_df.copy()
    adj_close_yyyy_mm =  pd.to_datetime(adj_close_df_copy.index).strftime('%Y-%m')
    adj_close_df_copy.index = adj_close_yyyy_mm

    returns_by_date = pd.DataFrame(index=adj_close_df_copy.index, columns=['Actual Return', 'Ending'])
    d = pd.to_datetime(adj_close_yyyy_mm)
    for year in range(d.year[0], d.year[-1] + 1):
        d_month = d[(pd.to_datetime(d).year == year)]
        for month in d_month.month.unique():

            if len(str(month)) < 2:
                month = '0' + str(month)
            
            df_index_monthly = adj_close_df.index.to_period('M')
            selected_rows = adj_close_df[df_index_monthly == pd.Period(f'{year}-{month}')]
            
            date = f'{year}-{month}'

            actual_ret = actual_return(optimal_weights, selected_rows)
            returns_by_date.loc[date, 'Actual Return'] = actual_ret
            
            ending_date = max(selected_rows.index)
            returns_by_date.loc[date, 'Ending'] = ending_date.strftime('%Y/%m/%d')

    returns_by_date = returns_by_date[~returns_by_date.index.duplicated()]
    return returns_by_date

def get_weights_monthly_returns(tickers, years=30, low=0, high=1, weights=None):

    adj_close_df = get_adj_close(tickers, years)
    weights_df = build_portfolio(tickers,low_bound=low, high_bound=high, years=years, weights=weights, adj_close_df=adj_close_df)
    
    returns_df = monthly_returns(adj_close_df, weights_df['Weight'])

    return weights_df, returns_df, adj_close_df

# analyze portfolio
def analyze_portfolio(tickers, optimal_weights) -> pd.DataFrame:

    portfolio = {}
    for ticker, weight in zip(tickers, optimal_weights):
        portfolio.update({ticker: weight})

    df = pd.DataFrame.from_dict(portfolio, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index':'Ticker',0:'Weight'}, inplace=True)

    return df

def portfolio_statistics(optimal_weights, adj_close_df):
    optimal_weights = optimal_weights['Weight'].values

    log_returns = get_log_returns(adj_close_df)
    cov_matrix = get_cov_matrix(log_returns, days=252)
    risk_free_rate = get_risk_free_rate()

    optimal_return = expected_return(optimal_weights, log_returns)
    optimal_volatility = std_dev(optimal_weights, cov_matrix)
    optimal_sharpe = sharpe(optimal_weights, log_returns, cov_matrix, risk_free_rate)

    return optimal_return, optimal_volatility, optimal_sharpe

def read_all_tickers():
    filepath = r"C:\Users\Adrian(SatovskyAsset\Desktop\Projects\IFA\Yahoo Ticker Symbols - September 2017.xlsx"

    stock = pd.read_excel(filepath, sheet_name='Stock', header=3)
    stock = stock['Ticker'].values
    
    etf = pd.read_excel(filepath, sheet_name='ETF', header=3)
    etf = etf['Ticker'].values

    index = pd.read_excel(filepath, sheet_name='Index', header=3)
    index = index['Ticker'].values

    return np.concatenate((stock,etf,index))

# convert ticker strings to ticker objects
def create_ticker_obj(tickers) -> list:
    """
    Converts a list of str tickers to a list of yFinance Tickers.

    Parameters:
    - tickers (list): List of str with yFinance tickers.

    Returns: 
    list: List of yFinance tickers.
    """
    ticker_objs = []

    for ticker in tickers:
        ticker_objs.append(yf.Ticker(ticker))

    return ticker_objs

def yield_in_period(ticker_obj, start_date, end_date) -> float:
    """
    Returns the sum of dividends in period.

    Parameters:
    - ticker_obj (yfinance.ticker.Ticker): Asset's ticker on Yahoo Finance.
    - start_date (str): Start of period.
    - end_date (str): End of period.

    Returns: 
    float: Dollar amount of dividends paid out in period.
    """
    dividends = ticker_obj.dividends
    dividends_in_period = dividends[(dividends.index >= start_date) & (dividends.index <= end_date)].sum()

    return dividends_in_period

def get_dividends_percent_weighted(ticker_obj, start_date, end_date, weight=1):
    start_date += "-01"
    end_date += "-01"
    hst = ticker_obj.history(start=start_date, end=end_date,interval="1mo")
    hst['yield'] = hst['Dividends'] / hst['Close']
    #return hst['yield'].tolist() * weight
    return [item * weight for item in hst['yield'].tolist()]

def normalize_to_percentage(arr):
    total = np.sum(arr)
    if total == 0:
        return np.zeros_like(arr)  # To handle cases where the sum is zero
    percentage_arr = (arr / total) * 100
    return percentage_arr

def weighted_average(values, weights):
    if len(values) != len(weights):
        raise ValueError("The length of values and weights must be the same")
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("The sum of weights must not be zero")
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight

def get_arstd_children(children):

    ratios = []
    avg_returns = []
    std_devs = []

    for input_group in children[1:]:
        if hasattr(input_group, 'get'):
            for input_group_attribute in input_group.get('props').items(): # looping through inputgroups' inputs
                if input_group_attribute[0] == 'children':
                    for item in input_group_attribute[1]:
                        if 'id' in item.get('props'):
                            if item.get('props').get('id').get('type') == 'monte-carlo-ratio':
                                ratios.append(item.get('props').get('value'))
                            if item.get('props').get('id').get('type') == 'monte-carlo-average-return':
                                avg_returns.append(item.get('props').get('value'))
                            if item.get('props').get('id').get('type') == 'monte-carlo-standard-deviation':
                                std_devs.append(item.get('props').get('value'))
    
    ratios_norm = normalize_to_percentage(ratios)

    #df = pd.DataFrame({
    #    'Average Return':avg_returns,
    #    'Standard Deviation':std_devs,
    #    'Ratio': ratios_norm
    #})

    weighted_average_average_returns = weighted_average(avg_returns, ratios_norm)
    weighted_average_standard_deviations = math.sqrt(weighted_average(np.array(std_devs)*np.array(std_devs), ratios_norm))

    return weighted_average_average_returns, weighted_average_standard_deviations