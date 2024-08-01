from util.util import *
from dateutil.parser import parse
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import util.inflation as i

updatemenus = [
    dict(
        type="buttons",
        direction="left",
        buttons=list([
            dict(
                args=[{'yaxis.type': 'linear'}],
                label="Linear",
                method="relayout"
            ),
            dict(
                args=[{'yaxis.type': 'log'}],
                label="Log",
                method="relayout"
            )
        ]),
        yanchor='top',
        y=1.13,
        x=1,
    ),
]  


def calculate_growth(starting_amount, returns, name, management_fee, inflation_data=False,contributions=0, dividends=[]):
    if type(inflation_data) == pd.DataFrame:
        growth_ot = []
        for i in returns.index:
            dividends_month = 0
            if len(dividends) != 0:
                if i in dividends.index:
                    dividends_month = dividends.loc[i] * starting_amount

            starting_amount += contributions
            starting_amount *= (1 + returns.loc[i, 'Actual Return'])
                
            starting_amount *= (1 - returns.loc[i, 'Inflation Rate'])
            fee = starting_amount * management_fee
            starting_amount *= (1 - management_fee)

            if len(dividends) == 0:
                dividends_month = 0
            dividends_month = dividends.loc[i]
            growth_ot.append([i, starting_amount, name, returns.loc[i, 'Ending'], fee, returns.loc[i, 'Inflation Rate'], contributions, dividends_month])
    
    else:
        growth_ot = []
        for i in returns.index:
            dividends_month = 0
            if len(dividends) != 0:
                if i in dividends.index:
                    dividends_month = dividends.loc[i] * starting_amount

            starting_amount += contributions
            starting_amount *= (1 + returns.loc[i, 'Actual Return'])
                
            fee = starting_amount * management_fee
            starting_amount *= (1 - management_fee)
            
            growth_ot.append([i, starting_amount, name, returns.loc[i, 'Ending'], fee, contributions, dividends_month])
    
    return growth_ot

def actual_return_hovertext(data, growth_df, title) -> dict:
    hovertext = []
    for date in data.index:
        row = data.loc[date]
        temp = []
        temp.append(row.iloc[0]*100)
        temp.append(title)
        temp.append(growth_df.loc[date, 'Ending'])
        hovertext.append(temp)
    return hovertext

def create_plot(starting_amount, data, title, management_fee=0, inflation_data=False, contributions_type='number', contributions=0, dividends=[]): 

    contributions /= 12

    if contributions_type == 'percent':
        contributions = starting_amount * (contributions/100)

    # Adjust for inflation
    if type(inflation_data) == pd.DataFrame:
        inflation_data = i.calculate_inflation(inflation_data, starting_amount)
        data = data.join(inflation_data)
        growth_ot = calculate_growth(starting_amount, data, title, management_fee, inflation_data, contributions=contributions, dividends=dividends)
        growth_df = pd.DataFrame(data=growth_ot, columns=['Date', 'Amount', title, 'Ending', 'Management Fees', 'Inflation Rate', 'Contribution', 'Dividends'])
        hover = "<br>".join([
            "Portfolio: %{customdata[2]}",
            "Ending: %{customdata[3]}",
            "$ Amount: $%{customdata[1]:.2f}",
            "Management Fees: $%{customdata[4]:.2f}",
            "Inflation Rate (MoM): %{customdata[5]:.5f}%",
            "Contribution: $%{customdata[6]:.2f}",
            "Dividends: $%{customdata[7]:.2f}",
            "<extra></extra>",
        ])
    else:
        # called after dataset loop to use same namespace
        growth_ot = calculate_growth(starting_amount, data, title, management_fee, contributions=contributions, dividends=dividends)
        growth_df = pd.DataFrame(data=growth_ot, columns=['Date', 'Amount', title, 'Ending', 'Management Fees', 'Contribution', 'Dividends'])
        hover = "<br>".join([
            "Portfolio: %{customdata[2]}",
            "Ending: %{customdata[3]}",
            "$ Amount: $%{customdata[1]:.2f}",
            "Management Fees: $%{customdata[4]:.2f}",
            "Contribution: $%{customdata[5]:.2f}",
            "Dividends: $%{customdata[6]:.2f}",
            "<extra></extra>",
        ])

    growth_df.set_index(['Date'], inplace=True)

    # Tracer 2: Growth
    fig = go.Figure(go.Bar(
        x=data.index, 
        y=growth_df['Amount'], 
        customdata=growth_ot,
        base="markers+lines", 
        hovertemplate=hover,
        showlegend=False,

    ))

    fig['layout'].update(
        title_text=f'{title}',
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{'yaxis.type': 'linear'}],
                    label="Linear",
                    method="relayout"
                ),
                dict(
                    args=[{'yaxis.type': 'log'}],
                    label="Log",
                    method="relayout"
                )
            ]),
            yanchor='top',
            xanchor='right',
            y=1.15,
            x=1,
        ),],
        xaxis = dict(
            title = 'Date',
            tickformatstops=[
            dict(
                enabled=True,
                dtickrange= [0, (31 * 86400000.0)],
                value='%b'
            )],
            rangemode='tozero'
        ),
        yaxis = dict(
            title = f'Growth of ${starting_amount:,}'
        )
    )
    fig.add_annotation(x=growth_df.index[-1], y=growth_df['Amount'][-1], text=f"<b>${growth_df['Amount'][-1]:,.2f}",showarrow=True, arrowhead=2)
    return fig


def avg_return(data):
    return data['Actual Return'].mean()

def create_plot_percent(starting_amount, data, title, management_fee=0): 

    # called after dataset loop to use same namespace
    growth_ot = calculate_growth(starting_amount, data, title, management_fee)
    growth_df = pd.DataFrame(data=growth_ot, columns=['Date', 'Amount', title, 'Ending', 'Management Fees', "contributions", 'dividends'])
    growth_df.set_index(['Date'], inplace=True)

    actual_return_h = actual_return_hovertext(data, growth_df, title)
    
    # Tracer 1: percent return
    fig = go.Figure(go.Bar(
        x=data.index, 
        y=data['Actual Return'], 
        customdata=actual_return_h,
        base="markers+lines", 
        hovertemplate="<br>".join([
            "Portfolio: %{customdata[1]}",
            "Ending: %{customdata[2]}",
            "Return: %{customdata[0]:.2f}%",
            "<extra></extra>",
        ]),
        showlegend=False,
    ))

    fig['layout'].update(
        margin = dict(t=0),
        xaxis = dict(
            title = 'Date',
            tickformatstops=[
            dict(
                enabled=True,
                dtickrange= [0, (31 * 86400000.0)],
                value="%b %Y"    
            )],
            rangemode='tozero',
        ),
        yaxis = dict(
            tickformat = '.1%',
            title = 'Return %'
        )
    )

    # Average return
    avg = avg_return(data)*12
    
    # Volatility
    deviations = np.array([data['Actual Return']]) - avg
    sqr_deviations = deviations ** 2
    variance = np.mean(sqr_deviations)
    volatility = np.sqrt(variance) * 100

    text = "<br>".join([f"Average Return: {avg*100:.2f}%", 
                        f"Volatility: {volatility:.2f}%",])

    fig.add_annotation(text=text,
                       xref="paper", yref="paper",
                        x=0, y=1,
                        showarrow=False)

    return fig

def read_returns_data_market_index(filepath, datasets) -> dict:
    crsp = pd.read_excel(filepath)

    for i in range(1,11):
        crsp_df = crsp
        crsp_df.columns = crsp_df.iloc[5]
        crsp_df = crsp_df.iloc[7:-10]
        crsp_df['Date'] = crsp_df['Date'].apply(lambda x: pd.to_datetime(x))
        crsp_df.set_index(['Date'], inplace=True)

        crsp_df2 = pd.DataFrame(columns=['Actual Return', 'Ending'])
        crsp_title = f'CRSP Decile {i} Index'
        crsp_df2['Actual Return'] = crsp_df[crsp_title]
        crsp_df2['Ending'] = crsp_df2.index.strftime('%Y/%m/%d')
        crsp_df2.index = crsp_df.index.strftime('%Y-%m')

        datasets.update({crsp_title: crsp_df2})

    return datasets

def read_returns_data(filepath, title, datasets) -> dict:
    rw = pd.read_excel(filepath)
    rw.columns = rw.iloc[5]
    rw = rw.iloc[7:-10]

    rw['Date'] = rw['Date'].apply(lambda x: pd.to_datetime(x))
    rw.set_index(['Date'], inplace=True)

    df = pd.DataFrame(columns=['Actual Return', 'Ending'])
    df['Actual Return'] = rw[title]
    df['Ending'] = df.index.strftime('%Y/%m/%d')
    df.index = rw.index.strftime('%Y-%m')

    datasets.update({title: df})

    return datasets

def portfolio_breakdown(weights, adj_close_df):
    fig = go.Figure([go.Pie(labels=weights['Ticker'], values=weights['Weight'])], layout=go.Layout(title='Portfolio Weights'))
    
    r, v, s = portfolio_statistics(weights, adj_close_df)
    r*=100
    text = "<br>".join([f"Expected Annual Return: {r:.2f}%",
                        f"Expected Volatility: {v:.4f}",
                        f"Sharpe Ratio: {s:.4f}"])
    
    fig.add_annotation(text=text,
                       xref="paper", yref="paper",
                        x=-0.1, y=1,
                        showarrow=False)
    return fig

def calculate_growth_test(starting_amount, returns, name, management_fee=0, contributions=0, dividends=[]):
    
    growth_ot = []
    for i in returns.index:
        
        # Dividends/yield
        dividends_month = 0
        if (len(dividends) != 0) and (i in dividends.index):
            dividends_month = dividends.loc[i] * starting_amount
            #dividends_month = dividends_month['Dividends']
        if dividends_month < 0: # prevent negative dividends
            dividends_month = 0
            
        # Contributions
        starting_amount += contributions

        # Return
        starting_amount *= (1 + returns.loc[i, 'Actual Return'])

        # Inflation
        starting_amount *= (1 - returns.loc[i, 'Inflation Rate'])

        # Management Fee
        fee = starting_amount * management_fee
        if fee < 0: # prevent positive fees
            fee = -fee
        starting_amount *= (1 - management_fee)

        growth_ot.append([i, name, returns.loc[i, 'Ending'], starting_amount, -fee, returns.loc[i, 'Inflation Rate'], contributions, dividends_month])
        
    return growth_ot

def get_marker_color(value, mode='discrete', **kwargs):
    if mode == 'discrete':
        if value > 0:
            return '#488f31'
        elif value < 0:
            return '#de425b'
        else:
            return '#ffffff'
    elif mode == 'continuous':
        if value > 0:
            min_value = kwargs.get('min_value', 0)
            max_value = kwargs.get('max_value', 0)
            color_sequence = px.colors.sequential.Greens
            
            norm_value = (value - min_value) / (max_value - min_value)  # Normalize value between 0 and 1
            norm_value = min(max(norm_value, 0), 1)
        else:
            neg_min_value = kwargs.get('neg_min_value', 0)
            neg_max_value = kwargs.get('neg_max_value', 0)
            color_sequence = px.colors.sequential.Reds_r
            
            norm_value = (value - neg_min_value) / (neg_max_value - neg_min_value)  # Normalize value between 0 and 1
            norm_value = min(max(norm_value, 0), 1)
        return color_sequence[int(norm_value * (len(color_sequence) - 1))]

def create_plot_test(starting_amount, data, title, management_fee=0, inflation_data=False, contributions_type='number', contributions=0, dividends=[], color_mode='discrete'): 
    
    # Contributions/withdrawals adjustment
    contributions /= 12
    if contributions_type == 'percent':
        contributions = starting_amount * (contributions/100)

    # Adjust for inflation
    if type(inflation_data) == pd.DataFrame:
        inflation_data = i.calculate_inflation(inflation_data, starting_amount)
        data = data.join(inflation_data)
    else:
        inflation_data = pd.DataFrame(index=data.index, data=[0] * len(data), columns=['Inflation Rate'])
        data = data.join(inflation_data)

    growth_ot = calculate_growth_test(starting_amount, data, title, management_fee=management_fee, contributions=contributions, dividends=dividends)
    growth_df = pd.DataFrame(data=growth_ot, columns=['Date', title, 'Ending', 'Amount', 'Management Fees', 'Inflation Rate', 'Contribution', 'Dividends'])
    growth_df.set_index(['Date'], inplace=True)
    
    

    # Hovertext
    hovertext_arr = [
        "<b>%{customdata[1]}</b>",
        "Ending: %{customdata[2]}",
        "$ Amount: $%{customdata[3]:.2f}",
        "Management Fees: $%{customdata[4]:.2f}",
        "Inflation Rate (MoM): %{customdata[5]:.5f}%",
        "Contribution: $%{customdata[6]:.2f}",
        "Dividends: $%{customdata[7]:.2f}",
        "<extra></extra>",
    ]
    hovertext_arr2 = []
    for itempos in range(len(hovertext_arr)):
        if growth_ot[0][itempos] != 0:
            hovertext_arr2.append(hovertext_arr[itempos-1])
    hover = "<br>".join(hovertext_arr2)
    
    # Sorting columns for barmode=stack
    inflation_deduction = -(growth_df['Inflation Rate'] * (growth_df['Amount']+growth_df['Contribution']+growth_df['Dividends']))
    growth_df['Inflation'] = inflation_deduction
    growth_df.drop(columns=['Inflation Rate', 'Ending', title], inplace=True)
    growth_df.reset_index(inplace=True)
    df_sorted = growth_df.sort_values(by='Amount')
    
    # Drop empty columns from legend
    for col in df_sorted.columns[1:]:
        if df_sorted[col].isna().all() or (df_sorted[col] == 0).all():
            df_sorted.drop(columns=col, inplace=True)

    # Color ranges for continuous marker colors
    all_values = pd.concat([growth_df[col] for col in growth_df.columns[2:]])
    neg_min_value = all_values[all_values < 0].min()
    neg_max_value = all_values[all_values < 0].max() 
    pos_min_value = all_values[all_values > 0].min()
    pos_max_value = all_values[all_values > 0].max() 
    #print(f"negmin: {neg_min_value} negmax: {neg_max_value} posmin: {pos_min_value} posmax: {pos_max_value}")
    #min_value_amount = -(abs(growth_df["Amount"].min()) * 2)
    #max_value_amount = growth_df["Amount"].max()   

    fig = go.Figure()
    # Add traces for each value column in sorted order
    
    for col in df_sorted.columns[1:]:
        name = col
        if col == "Contribution" and contributions < 0:
            name = "Withdrawal"
        try:
            color = [get_marker_color(v, color_mode, min_value=pos_min_value, max_value=pos_max_value, neg_min_value=neg_min_value, neg_max_value=neg_max_value) for v in df_sorted[col]]
        except:
            color = [get_marker_color(v) for v in df_sorted[col]]
        if col == 'Amount':
            #color = [get_marker_color(v, color_mode, min_value=min_value_amount, max_value=max_value_amount, color_sequence=px.colors.sequential.Blues) for v in df_sorted[col]]
            color = '#636EFA'
            #hovertemplate = hover
            name = "Portfolio Value"
        
        hovertemplate = '<br>'.join([f'<b>{name}</b>','$%{y:,.2f}'+'<extra></extra>'])
        fig.add_trace(go.Bar(
            x=df_sorted["Date"],
            y=df_sorted[col],
            name=name,
            marker_color=color,
            customdata=growth_ot,
            hovertemplate=hovertemplate,
        ))

    # Update layout for stacked bar chart
    fig.update_layout(
        hovermode='x',
        bargap=0.02,
        barmode='relative',
        xaxis=dict(
            title = 'Date',
            categoryorder = 'category ascending',
            tickformatstops=[
            dict(
                enabled=True,
                dtickrange= [0, (31 * 86400000.0)],
                value="%b"    
            )],
            rangemode='tozero',
        ),
        yaxis = dict(
            title = f'Growth of ${starting_amount:,}'
        ),
        title_text=f'{title}',
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{'yaxis.type': 'linear'}],
                    label="Linear",
                    method="relayout"
                ),
                dict(
                    args=[{'yaxis.type': 'log'}],
                    label="Log",
                    method="relayout"
                )
            ]),
            yanchor='top',
            xanchor='right',
            y=1.15,
            x=1,
        ),],
    )

    #fig.add_annotation(x=growth_df.iloc[-1]['Date'], y=growth_df.iloc[-1]['Amount'], text=f'<b>${growth_df.iloc[-1]['Amount']:,.2f}',showarrow=True, arrowhead=2)
    return fig

def create_plot_test_date(starting_amount, data, title, management_fee=0, inflation_data=False, contributions_type='number', contributions=0, dividends=[], color_mode='discrete', annual=False): 
    
    # Contributions/withdrawals adjustment
    contributions /= 12
    if contributions_type == 'percent':
        contributions = starting_amount * (contributions/100)

    # Adjust for inflation
    if type(inflation_data) == pd.DataFrame:
        inflation_data = i.calculate_inflation(inflation_data, starting_amount)
        data = data.join(inflation_data)
    else:
        inflation_data = pd.DataFrame(index=data.index, data=[0] * len(data), columns=['Inflation Rate'])
        data = data.join(inflation_data)

    growth_ot = calculate_growth_test(starting_amount, data, title, management_fee=management_fee, contributions=contributions, dividends=dividends)
    growth_df = pd.DataFrame(data=growth_ot, columns=['Date', title, 'Ending', 'Amount', 'Management Fees', 'Inflation Rate', 'Contribution', 'Dividends'])
    growth_df.set_index(['Date'], inplace=True)
    
    if annual:
        growth_df['date'] = pd.to_datetime(growth_df.index, utc=True)
        growth_df['Date'] = growth_df['date'].dt.to_period('M')
        growth_df.set_index(['Date'], inplace=True)
        growth_df.index = growth_df.index.to_timestamp() 
        growth_df.drop(columns=['date'], inplace=True)

        for row in growth_ot:
            row[0] = row[0][:-3]

        

    # Hovertext
    hovertext_arr = [
        "<b>%{customdata[1]}</b>",
        "Ending: %{customdata[2]}",
        "$ Amount: $%{customdata[3]:.2f}",
        "Management Fees: $%{customdata[4]:.2f}",
        "Inflation Rate (MoM): %{customdata[5]:.5f}%",
        "Contribution: $%{customdata[6]:.2f}",
        "Dividends: $%{customdata[7]:.2f}",
        "<extra></extra>",
    ]
    hovertext_arr2 = []
    for itempos in range(len(hovertext_arr)):
        if growth_ot[0][itempos] != 0:
            hovertext_arr2.append(hovertext_arr[itempos-1])
    hover = "<br>".join(hovertext_arr2)
    
    # Sorting columns for barmode=stack
    inflation_deduction = -(growth_df['Inflation Rate'] * (growth_df['Amount']+growth_df['Contribution']+growth_df['Dividends']))
    growth_df['Inflation'] = inflation_deduction
    growth_df.drop(columns=['Inflation Rate', 'Ending', title], inplace=True)
    growth_df.reset_index(inplace=True)
    df_sorted = growth_df.sort_values(by='Amount')
    
    # Drop empty columns from legend
    for col in df_sorted.columns[1:]:
        if df_sorted[col].isna().all() or (df_sorted[col] == 0).all():
            df_sorted.drop(columns=col, inplace=True)

    # Color ranges for continuous marker colors
    all_values = pd.concat([growth_df[col] for col in growth_df.columns[2:]])
    neg_min_value = all_values[all_values < 0].min()
    neg_max_value = all_values[all_values < 0].max() 
    pos_min_value = all_values[all_values > 0].min()
    pos_max_value = all_values[all_values > 0].max() 

    fig = go.Figure()
    # Add traces for each value column in sorted order
    
    for col in df_sorted.columns[1:]:
        name = col
        if col == "Contribution" and contributions < 0:
            name = "Withdrawal"
        try:
            color = [get_marker_color(v, color_mode, min_value=pos_min_value, max_value=pos_max_value, neg_min_value=neg_min_value, neg_max_value=neg_max_value) for v in df_sorted[col]]
        except:
            color = [get_marker_color(v) for v in df_sorted[col]]
        if col == 'Amount':
            color = '#636EFA'
            name = "Portfolio Value"
        
        hovertemplate = '<br>'.join([f'<b>{name}</b>','$%{y:,.2f}'+'<extra></extra>'])
        fig.add_trace(go.Bar(
            x=df_sorted["Date"],
            y=df_sorted[col],
            name=name,
            marker_color=color,
            customdata=growth_ot,
            hovertemplate=hovertemplate,
        ))

    # Update layout for stacked bar chart
    fig.update_layout(
        hovermode='x',
        bargap=0.02,
        barmode='relative',
        xaxis=dict(
            title = 'Date',
            categoryorder = 'category ascending',
            #tickformatstops=[
            #dict(
            #    enabled=True,
            #    dtickrange= [0, (31 * 86400000.0)],
            #    value="%b"    
            #)],
            rangemode='tozero',
        ),
        yaxis = dict(
            title = f'Growth of ${starting_amount:,}'
        ),
        title_text=f'{title}',
        #updatemenus=[dict(
        #    type="buttons",
        #    direction="left",
        #    buttons=list([
        #        dict(
        #            args=[{'yaxis.type': 'linear'}],
        #            label="Linear",
        #            method="relayout"
        #        ),
        #        dict(
        #            args=[{'yaxis.type': 'log'}],
        #            label="Log",
        #            method="relayout"
        #        )
        #    ]),
        #    yanchor='top',
        #    xanchor='right',
        #    y=1.15,
        #    x=1,
        #),],
    )

    return fig