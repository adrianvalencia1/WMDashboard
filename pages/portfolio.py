"""from dash import html, register_page
import dash
from dash import Dash, dcc, html, ctx
import numpy as np
import plotly.express as px
import util.portfoliographs as g
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
import dash_bootstrap_components as dbc
import copy
from dash.exceptions import PreventUpdate
import util.inflation as i
from util.util import create_ticker_obj, get_dividends_percent_weighted, get_weights_monthly_returns, read_all_tickers
import util.util as u
from dash import dcc, html, dash_table, ctx, callback, register_page

pd.options.mode.chained_assignment = None

LOAD_TICKERS = False

today_date = datetime.today().strftime('%Y-%m')
dates = pd.date_range(start='1927-06', end=today_date, freq='MS').strftime('%Y-%m')
initial_amount = 1

# Inflation data
# TODO: relative file paths
pd.set_option('future.no_silent_downcasting', True)
inflation_filepath = r"C:\Users\Adrian(SatovskyAsset\Desktop\Projects\IFA\SeriesReport-20240610103427_33cb36.xlsx"
datasets = {}
filepath = r"C:\\Users\\Adrian(SatovskyAsset\\Desktop\\Projects\\IFA\\Returns Data\\DFA_PeriodicReturns_20240604135057.xlsx"
g.read_returns_data(filepath, "Dimensional US Small Cap Index", datasets)
filepath = r"C:\\Users\\Adrian(SatovskyAsset\\Desktop\\Projects\\IFA\\Returns Data\\DFA_PeriodicReturns_20240613153633.xlsx"
g.read_returns_data(filepath, "Russell 3000 Index", datasets)
filepath = r"C:\Users\Adrian(SatovskyAsset\Desktop\Projects\IFA\Returns Data\DFA_PeriodicReturns_20240613153909.xlsx"
g.read_returns_data(filepath, "S&P 500 Index", datasets)
filepath = r"C:\Users\Adrian(SatovskyAsset\Desktop\Projects\IFA\Returns Data\DFA_PeriodicReturns_20240613154022.xlsx"
g.read_returns_data(filepath, "US High Relative Profitability Portfolio", datasets)
filepath = r"C:\\Users\\Adrian(SatovskyAsset\\Desktop\\Projects\\IFA\\Returns Data\\DFA_PeriodicReturns_20240604115746.xlsx"
g.read_returns_data_market_index(filepath, datasets)
fig = g.create_plot(1, list(datasets.values())[0], list(datasets.keys())[0], 0, dividends=[])
percent_fig = g.create_plot_percent(1, list(datasets.values())[0], list(datasets.keys())[0], 0)
min_date=list(datasets.values())[0].iloc[[0]].index[0]
max_date=list(datasets.values())[0].iloc[[-1]].index[0]

if LOAD_TICKERS:
    all_tickers = read_all_tickers()
    all_tickers = all_tickers[[("." not in x) for x in all_tickers]]
    np.save('tickers_array.npy', all_tickers)
else:
    all_tickers = np.load('tickers_array.npy', allow_pickle=True)

register_page(
    __name__,
    name='Portfolio Optimization',
    top_nav=True,
    path='/portfolio-optimization'
)

def layout():

    layout = html.Div([
        dcc.RadioItems(
            [{
                "label": "Build Portfolio",
                "value": "build",
            },
            {
                "label": "Use Existing Datasets",
                "value": "index",
            }],
            id='type'
        ),
        html.Div([ # Build portfolio
            html.Div([
                dcc.Store( id='portfolio-df', ),
                dcc.Dropdown(
                    all_tickers,
                    [],
                    id='portfolio-dropdown',
                    multi=True,
                ),
                dcc.Store(id='portfolio-weights', data=None),
                dcc.RadioItems(
                    ['Optimize Weights', 'Even Weights', 'Manually Input Weights'],
                    id='weights-options'
                ),
                html.Div([
                    html.Div(["Low Bound:"]),
                    dcc.Input(
                        id='input-low',
                        type='number',
                        value=0,
                        min=0,
                        max=1,
                        step=0.01,
                        placeholder="Low Bound",
                    ),
                    html.Div(["High Bound:"]),
                    dcc.Input(
                        id='input-high',
                        type='number',
                        value=1,
                        min=0,
                        max=1,
                        step=0.01,
                        placeholder="High Bound",
                    ),
                ], id='optimize-options-div'),
                html.Div([
                    dcc.Store(id='manual-weights'),
                    html.Button(
                        'Submit Weights', 
                        id='submit-weights', 
                        n_clicks=0
                    ),
                    html.Div([], id='manual-options-weights-div'),
                ], id='manual-options-div'),
                html.Button(
                    'Fetch Data', 
                    id='build-portfolio-button', 
                    n_clicks=0
                ),
            html.Div([
                dcc.Graph(id='portfolio-weights-figure', figure=go.Figure(data=None)),
            ],id='portfolio-data')
            ],id='use-portfolio'),
        ]),
        html.Div([
            dcc.Dropdown(
                list(datasets.keys()),
                list(datasets.keys())[0],
                id='dropdown'
            ),
        ],
        id='use-index',),
        html.Div([
            html.Div('Enter $:'),
            dcc.Input(
                id='input-number',
                type='number',
                value=initial_amount,
                min=0,
                placeholder="Enter $",
            ),
            html.Div('Management Fees (%):'),
            dcc.Input(
                id='input-management-fees',
                type='number',
                value=0,
                min=0,
                max=100,
                step=0.01,
                placeholder="Enter %",
            ),
            dcc.Checklist(
                ['Adjust for Inflation', 'Yearly Contributions'],
                [],
                id='checklist'
            ),
            html.Div([
            dcc.Dropdown(
                [{"label": "$", "value": "number"},
                 {"label": "%", "value": "percent"}],
                'number',
                clearable=False,
                id='contributions-dropdown',
            ),
            dcc.Input(
                id='contributions',
                type='number',
                value=0,
                placeholder="",
            ),
            ],
            style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'},)
        ]),
        html.Div(id='dd-output-container'),
        dcc.Graph(id='graph', figure=fig, style={'height': '90vh'}),
        dcc.Graph(id='percent-graph', figure=percent_fig),
        html.Div([
            dcc.RangeSlider(
                0,
                len(dates)-1,
                value=[0,len(dates)-1],
                step=6,
                marks={str(year): str(dates[year]) for year in range(0, len(dates), 60)},
                id='year-slider',
                allowCross=False,
                pushable=6,
            ), 
            html.Button('Reset', id='reset-rangeslider-button', n_clicks=0),], 
        id='rangeslider-div'),
    ], style={'padding':'10px'})

    return layout

# Rangeslider
@callback(
    Output('year-slider', 'value', allow_duplicate=True),
    [Input('reset-rangeslider-button', 'n_clicks')],
    prevent_initial_call=True
)
def reset_range_slider(n_clicks):
    if n_clicks > 0:
        return [0, len(dates)-1]  # Reset to default range
    else:
        return dash.no_update
    
# Optimize weights options
@callback(
    Output('optimize-options-div', 'style'),
    Input('weights-options', 'value')
)
def update_optimize_weights_options(value):
    if value == 'Optimize Weights':
        return {'display': 'flex'}
    else:
        return {'display': 'none'}
    
@callback(
    [Output('manual-options-div', 'style'),
     Output('manual-options-weights-div', 'children'),],
    [Input('weights-options', 'value'),
     Input('portfolio-dropdown', 'value'),],
    [State('manual-options-weights-div', 'children')]
)
def update_manual_weights_options(value, tickers, existing_children):
    if value == 'Manually Input Weights':
        existing_children = []
        for ticker in tickers:
            new_label = html.Div([f"{ticker}:"],
                id={
                    'type': 'dynamic-label', 
                    'index': tickers.index(ticker)
                })
            new_input = dcc.Input(
                id={
                    'type': 'dynamic-input', 
                    'index': tickers.index(ticker)
                },
                type='number',
                value=0,
                min=0,
                max=1,
                step=0.01,
            )
            existing_children += [new_label, new_input]
        return {'display': 'block'}, existing_children
    else:
        return {'display': 'none'}, []

@callback(
    Output('manual-weights', 'data'),
    [Input('submit-weights', 'n_clicks'),],
    [State('manual-options-weights-div', 'children')]
)
def submit_weights(n_clicks, input_weights):
    if n_clicks is None:
        n_clicks = 0
    if n_clicks != 0:
        input_weights = [input_weights[i]['props']['value'] for i in range(len(input_weights))]
        return [weight / sum(input_weights) for weight in input_weights]

@callback(
    [Output('year-slider', 'min'),
     Output('year-slider', 'max'),
     Output('year-slider', 'value'),
     Output('year-slider', 'marks'),],
    Input('dropdown', 'value'),
)
def update_rangeslider(value):
    df = datasets.get(value)
    curr_dates = df.index
    marks = {str(year): str(curr_dates[year]) for year in range(0, len(curr_dates), 60)}
    return 0, len(curr_dates)-1, [0,len(curr_dates)-1], marks

# build portfolio button
@callback(
    [Output('portfolio-data', 'style'),
    Output('build-portfolio-button', 'disabled'),
    Output('dropdown', 'value'),
    Output('portfolio-weights-figure', 'figure'),
    ],
    [Input('build-portfolio-button', 'n_clicks'),
     Input('portfolio-dropdown', 'value'),
     Input('weights-options', 'value'),
     Input('manual-weights', 'data'),
     Input('input-low', 'value'),
     Input('input-high', 'value'),],
    [State('build-portfolio-button', 'n_clicks')]
)
def build_portfolio(n_clicks, tickers, weights_option, manual_weights, low, high, prev_n_clicks):
    if n_clicks is None:
        n_clicks = 0
    if n_clicks == 0:
        return {'display': 'none'}, False, list(datasets.keys())[0], go.Figure(data=None)
    if n_clicks >= prev_n_clicks:
        if weights_option == 'Optimize Weights':
            weights, df, adj_close_df = get_weights_monthly_returns(tickers, low=low, high=high, weights='optimize')
        elif weights_option == 'Even Weights':
            even_weight = 1/len(tickers)
            weights, df, adj_close_df = get_weights_monthly_returns(tickers, low=even_weight, high=even_weight, weights='even')
        elif weights_option == 'Manually Input Weights':
            weights, df, adj_close_df = get_weights_monthly_returns(tickers, low=low, high=high, weights=manual_weights)
            print(f"Weights: {weights}, \n Type: {type(weights)}")
        weights_figure = g.portfolio_breakdown(weights, adj_close_df)
        datasets.update({'Custom Portfolio':df})
        return {'display': 'block'}, True, 'Custom Portfolio', weights_figure
    

# hides/shows dropdowns for indexes and portfolio
@callback(
    [Output('dropdown', 'disabled'),
     Output('use-portfolio', 'style'),
     Output('checklist','options'),
     Output('checklist','value')],
    Input('type','value'),
    State('checklist', 'options')
)
def select_type(value, checklist_options):
    if value is None:
        return [True,{'display': 'none'}, checklist_options, []]
    if value == 'build':
        if 'Dividends/Yield' not in checklist_options:
            checklist_options += ['Dividends/Yield']
        return [True,{'display': 'block'}, checklist_options, []]
    if value == 'index':
        if 'Dividends/Yield' in checklist_options:
            checklist_options.remove('Dividends/Yield')
        return [False,{'display': 'none'}, checklist_options, []]

# enable/disable contributions
@callback(
    [Output('contributions', 'disabled'),
     Output('contributions-dropdown', 'disabled')],
    Input('checklist', 'value')    
)
def update_contributions(value):
    if 'Yearly Contributions' in value:
        return [False, False]
    else:
        return [True, True]

@callback(
    [Output('graph', 'figure'),
     Output('percent-graph', 'figure')],
    [Input('input-number', 'value'),
     Input('year-slider', 'value'),
     Input('dropdown', 'value'),
     Input('input-management-fees', 'value'),
     Input('checklist', 'value'),
     Input('contributions-dropdown', 'value'),
     Input('contributions', 'value'),
     Input('portfolio-dropdown', 'value'),
     Input('portfolio-weights-figure', 'figure')],
)
def update_graph(value, date_value, dataset_name, management_fee, checklist, contributions_type, contribution, tickers, weights_figure):
    if dataset_name is None:
        raise PreventUpdate
    if value is None:
        raise PreventUpdate
    if management_fee is None:
        raise PreventUpdate
    if contribution is None:
        contribution = 0

    if management_fee != 0:
        management_fee /= 100
        management_fee /= 12
    else:
        management_fee = 0

    datasets_2 = copy.deepcopy(datasets)
    for key, df in datasets_2.items():
        df = df.iloc[date_value[0] : date_value[1]]
        
        datasets_2.update({key: df})

    data = datasets_2.get(dataset_name)

    dividends = []
    if 'Dividends/Yield' in checklist:
        ticker_objs = create_ticker_obj(tickers) 
        prev = None
        for ticker_obj in ticker_objs:
            weights = dict(zip(weights_figure['data'][0]['labels'], weights_figure['data'][0]['values']))
            curr = get_dividends_percent_weighted(ticker_obj, data.index[0], data.index[-1], weights[ticker_obj.ticker])
            if prev is None:
                prev = curr
            else:
                prev = [sum(pair) for pair in zip(prev, curr)]
        
        length_adjust = abs(len(data.index) - len(prev))
        data_index = data.index
        if len(prev) > len(data_index):
            prev = prev[length_adjust:]
        elif len(prev) < len(data_index):
            data_index = data_index[length_adjust:]

        dividends = pd.Series(data=prev, index=data_index)


    if 'Adjust for Inflation' in checklist and 'Yearly Contributions' in checklist:
        inflation_df = i.read_inflation_data(inflation_filepath)
        fig = g.create_plot_test(value, data, dataset_name, management_fee=management_fee, inflation_data=inflation_df, contributions_type=contributions_type, contributions=contribution, dividends=dividends, color_mode='continuous')
        percent_fig = g.create_plot_percent(1, data, dataset_name, management_fee=0)
    elif 'Adjust for Inflation' in checklist:
        inflation_df = i.read_inflation_data(inflation_filepath)
        fig = g.create_plot_test(value, data, dataset_name, management_fee=management_fee, inflation_data=inflation_df, dividends=dividends, color_mode='continuous')
        percent_fig = g.create_plot_percent(1, data, dataset_name, management_fee=0)
    elif 'Yearly Contributions' in checklist:
        fig = g.create_plot_test(value, data, dataset_name, management_fee=management_fee, contributions_type=contributions_type, contributions=contribution, dividends=dividends, color_mode='continuous')
        percent_fig = g.create_plot_percent(1, data, dataset_name, management_fee=0)
    else:
        fig = g.create_plot_test(value, data, dataset_name, management_fee=management_fee, dividends=dividends, color_mode='continuous')
        percent_fig = g.create_plot_percent(1, data, dataset_name, management_fee=0)
    return fig, percent_fig"""