from datetime import datetime
import math
import re
import time
import util.util as u
from dash import dcc, html, dash_table, ctx, callback, register_page, callback_context, MATCH, ALL
import numpy as np
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
from dateutil.relativedelta import relativedelta
import util.cwutil as cw
import util.datautil as du

MAX_SIMULATIONS = 100000
MAX_TIME_PERIODS = 100
START_DATE = datetime.today().strftime('%m/%Y')
RUN_UNTIL_DATE = (datetime.today() + relativedelta(years=10)).strftime('%m/%Y')

# Load in index presets from data folder
dropdown_menu_items = du.load_index_presets('dropdown')
index_presets = du.load_index_presets('data')

register_page(
    __name__,
    name='Monte Carlo Simulation',
    top_nav=True,
    path='/monte-carlo-simulation'
)

def layout():

    
    

    layout = html.Div([
    # TITLE
    html.Div([
        html.H1("Monte Carlo Simulation"),
        html.P("A Monte Carlo simulation models the probability of different outcomes in financial planning by running numerous simulations with random variables. By simulating a wide range of possible outcomes, this method provides a probabilistic forecast of an investment portfolio's performance, allowing for better risk assessment and decision-making. The results presented are a distribution of possible future portfolio values, illustrating the likelihood of achieving different financial goals and enabling more informed strategic planning.",
               style={
            'textIndent': '30px',
            'margin': '20px'
        }),
        
    ], style={'padding':'1vh 20vh 1vh'}),
    
    # GRAPH AND INPUTS
    html.Div([ # monte carlo
         
        # INPUTS
        html.Div([

            # Portfolio Settings
            html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.H4(['Portfolio']),

                        # INITIAL INVESTMENT
                        dbc.InputGroup([
                            dbc.InputGroupText("Initial Investment"), 
                            dbc.InputGroupText("$"), 
                            dbc.Input(
                                id='initial-investment',
                                value=10000,
                                min=0,
                                type='number',
                                placeholder="Initial Investment"
                            ),
                        ]),

                        # AVERAGE RETURN / STANDARD DEVIATION
                        dbc.Card(
                            dbc.CardBody([
                                html.H5(['Assets']),
                                html.Div(
                                    id='monte-carlo-arstd-container', 
                                    children=[
                                        html.Div([
                                            dbc.Button(
                                                "Add Asset",
                                                id='monte-carlo-add-arstd',
                                                n_clicks=0,
                                                color='secondary'
                                            ),
                                        ], style={'display':'flex', 'gap':'1vh'}),
                                        html.Div([
                                            dcc.Dropdown(
                                                dropdown_menu_items, 
                                                id={'type':'monte-carlo-arstd-dropdown', 'index':1},
                                                style={'min-width':'20vh'},
                                            ),
                                            dbc.InputGroup([
                                                dbc.InputGroupText("Average Return", id='monte-carlo-average-return'), 
                                                dbc.Input(
                                                    id={'type':'monte-carlo-average-return', 'index':1},
                                                    value=10,
                                                    type='number',
                                                    required=True,
                                                    placeholder="Enter Average Return"
                                                ),
                                                dbc.InputGroupText("%"),
                                                dbc.InputGroupText("Standard Deviation", id="monte-carlo-standard-deviation"), 
                                                dbc.Input(
                                                    id={'type':'monte-carlo-standard-deviation', 'index':1},
                                                    value=15,
                                                    type='number',
                                                    required=True,
                                                    placeholder="Enter Standard Deviation"
                                                ),
                                                dbc.InputGroupText("%"),
                                                dbc.InputGroupText("Allocation", id="monte-carlo-ratio"), 
                                                dbc.Input(
                                                    id={'type':'monte-carlo-ratio', 'index':1},
                                                    value=100,
                                                    min=0,
                                                    type='number',
                                                    required=True,
                                                    placeholder="Enter Weight"
                                                ),
                                                dbc.InputGroupText("%"),
                                            ])
                                        ], style={'display':'flex', 'gap':'1vh'})
                                    ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'}
                                ),

                                # ASSET ALLOCATION SUM
                                dbc.InputGroup([
                                    dbc.InputGroupText("Total"),
                                    dbc.Input(id='monte-carlo-allocation-sum', disabled=True, value=100)
                                ], style={'width':'20vh', 'margin-left':'auto', 'margin-right':'0'})

                            ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'})
                        ),
                        
                        # CONTRIBUTION AND WITHDRAWAL
                        html.Div([
                            dcc.Store(id='monte-carlo-cw-data')
                        ]),
                        dbc.Card(
                            dbc.CardBody([
                                html.H5(['Contributions and Withdrawals']),
                                html.Div(
                                    id='monte-carlo-cw-container', 
                                    children=[
                                        html.Div([
                                            dbc.Button(
                                                "Add Contribution Source",
                                                id='monte-carlo-add-c',
                                                n_clicks=0,
                                                color='secondary'
                                            ),
                                            dbc.Button(
                                                "Add Withdrawal Source",
                                                id='monte-carlo-add-w',
                                                n_clicks=0,
                                                color='secondary'
                                            ),
                                        ], style={'display':'flex', 'gap':'1vh'}),
                                        # CONTRIBUTION
                                        dbc.InputGroup(
                                            id={'type':'monte-carlo-contribution-igcontainer', 'index': 1},
                                            children=[ # Contribution inputs
                                            dbc.InputGroupText("Contribution", id='monte-carlo-contribution-input'),
                                            dbc.InputGroupText("$"), 
                                            dbc.Input(
                                                id={'type':'monte-carlo-contribution-input', 'index': 1},
                                                value=0,
                                                min=0,
                                                type='number',
                                                placeholder="Contribution"
                                            ),
                                            dbc.InputGroupText("Start", id='monte-carlo-contribution-start'), 
                                            dbc.Input(
                                                id={'type':'monte-carlo-contribution-start', 'index': 1},
                                                value=START_DATE,
                                                type='text',
                                                placeholder="Start Interval",
                                                pattern=r"(?<![0-9/])(0?[1-9]|1[0-2])/(\d{4})\b"
                                            ),
                                            dbc.InputGroupText("End", id='monte-carlo-contribution-end'), 
                                            dbc.Input(
                                                id={'type':'monte-carlo-contribution-end', 'index': 1},
                                                value=START_DATE,
                                                type='text',
                                                placeholder="End Interval",
                                                pattern=r"(?<![0-9/])(0?[1-9]|1[0-2])/(\d{4})\b"
                                            )]
                                        ),
                                        # WITHDRAWAL
                                        dbc.InputGroup(
                                            id={'type':'monte-carlo-withdrawal-igcontainer', 'index':2},
                                            children=[ # Withdrawal inputs
                                            dbc.InputGroupText("Withdrawal", id='monte-carlo-withdrawal-input'),
                                            dbc.InputGroupText("$"), 
                                            dbc.Input(
                                                id={'type':'monte-carlo-withdrawal-input', 'index': 2},
                                                value=0,
                                                min=0,
                                                type='number',
                                                placeholder="Withdrawal"
                                            ),
                                            dbc.InputGroupText("Start", id='monte-carlo-withdrawal-start'), 
                                            dbc.Input(
                                                id={'type':'monte-carlo-withdrawal-start', 'index': 2},
                                                value=START_DATE,
                                                type='text',
                                                placeholder="Start Interval",
                                                pattern=r"(?<![0-9/])(0?[1-9]|1[0-2])/(\d{4})\b"
                                            ),
                                            dbc.InputGroupText("End", id='monte-carlo-withdrawal-end'), 
                                            dbc.Input(
                                                id={'type':'monte-carlo-withdrawal-end', 'index': 2},
                                                value=START_DATE,
                                                type='text',
                                                placeholder="End Interval",
                                                pattern=r"(?<![0-9/])(0?[1-9]|1[0-2])/(\d{4})\b"
                                            )]
                                        )
                                    ], 
                                    style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'}
                                ),
                            ])
                        ),
                    ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'}),
                ),
            ], style={'max-width':'none'}),
            
            html.Div(id='monte-carlo-update-prevention'),
            # Simulation Settings
            html.Div([
                dcc.Store(id='monte-carlo-intervals-data'),
                dbc.Card(
                    dbc.CardBody([
                        html.H4(['Simulation'],),
                        dbc.InputGroup([
                            dbc.Select(
                                options=[
                                    {"label": "Months", "value": "m"},
                                    {"label": "Years", "value": "y"}
                                ],
                                value="m",
                                id="monte-carlo-time-interval"
                            ),
                            dbc.InputGroupText("Start", id="monte-carlo-time-start"),
                            dbc.Input(
                                value=f"{START_DATE}",
                                disabled=True
                            ),
                            dbc.InputGroupText("End", id="monte-carlo-time-end"),
                            # TIME PERIODS
                            dbc.Input(
                                id='monte-carlo-time-periods',
                                value=RUN_UNTIL_DATE,
                                invalid=False,
                                type='text',
                                placeholder="MM/YYYY",
                                pattern=r"(?<![0-9/])(0?[1-9]|1[0-2])/(\d{4})\b",
                                debounce=True
                            ),

                            # SIMULATIONS
                            dbc.InputGroupText("Simulations", id='monte-carlo-simulations-label'), 
                            dbc.Input(
                                id='monte-carlo-simulations',
                                value=1000,
                                min=1,
                                max=10000,
                                type='number',
                                placeholder="Simulations",
                                disabled=True
                            ),
                        ]),                  
                    ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'})
                ),
            ]),

            # Threshold
            html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.H4(['Threshold'],),
                        
                        dbc.InputGroup([
                            dbc.Select(
                                options=[
                                    {'label': 'Reaches', 'value': 'up'},
                                    {'label': 'Drops to', 'value': 'down'},
                                ],
                                value='up',
                                id='monte-carlo-threshold-direction'
                            ), 
                            dbc.InputGroupText("$"), 
                            dbc.Input(id='monte-carlo-threshold',
                                value=None,
                                min=0,
                                type='number',
                            ),
                            dbc.Select(
                                options=[
                                    {'label': 'Average', 'value': 'mean'},
                                    {'label': 'Median', 'value': 'median'},
                                    {'label': '95th Percentile', 'value': '95'},
                                    {'label': '5th Percentile', 'value': '5'},
                                ],
                                value='median',
                                id='monte-carlo-threshold-type'
                            ),  
                        ])
                    ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'})
                ),   
            ]),
           
        dbc.Button(
            'Run Simulation', 
            id='monte-carlo-button', 
            n_clicks=0,
            color="secondary",
            disabled=False
        ),
        ], style={'display':'flex', 'gap':'1vh', 'flex-direction':'column'}),
        
        dcc.Loading(
            id='load-monte-carlo',
            children=dcc.Graph(id='monte-carlo'),
            overlay_style={"visibility":"visible", "opacity": .5, "backgroundColor": "white"},
            custom_spinner=html.H2(["Running Simulation  ", dbc.Spinner(color="primary")]),
        ),
        
        dash_table.DataTable(id='data-table', columns=None, data=None, style_cell={
            'font-family': 'Arial, sans-serif',  # Set font family
            'fontSize': '16px',  # Set font size
            'textAlign': 'center',  # Set text alignment
        },),
        dcc.Loading(
            id='load-monte-carlo-graph-info',
            children=html.Div([
                html.H2(id='monte-carlo-graph-info-header', style={'margin-top': '20px'}),
                html.P(
                    id='monte-carlo-graph-info',
                    style={'textIndent': '30px','margin': '20px'}
                )
            ]),
            custom_spinner=dbc.Placeholder(animation='glow')
        ) 
        ],style={'padding':'0px 20vh 30px'}),
        
        # TOOLTIPS
        html.Div([
        
        dbc.Tooltip(
            "Initial investment in $.",
            target="initial-investment",
            placement="top"
        ),

        # ASSET TOOLTIPS
        dbc.Tooltip(
            "The average return of an initial investment.",
            target="monte-carlo-average-return",
            placement="top"
        ),
        dbc.Tooltip(
            "Determines by how much returns can deviate from the average. A higher standard deviation means more volatility and risk.",
            target="monte-carlo-standard-deviation",
            placement="top"
        ),
        dbc.Tooltip(
            "The % of the portfolio allocated to an asset.",
            target="monte-carlo-ratio",
            placement="top"
        ),

        # CONTRIBUTION/WITHDRAWAL TOOLTIPS
        dbc.Tooltip(
            "Monthly/yearly contribution in $.",
            target="monte-carlo-contribution-input",
            placement="top"
        ),
        dbc.Tooltip(
            "Contribution source start date. (MM/YYYY).",
            target="monte-carlo-contribution-start",
            placement="top"
        ),
        dbc.Tooltip(
            "Contribution source end date. (MM/YYYY)",
            target="monte-carlo-contribution-end",
            placement="top"
        ),

        dbc.Tooltip(
            "Monthly/yearly withdrawal in $.",
            target="monte-carlo-withdrawal-input",
            placement="bottom"
        ),
        dbc.Tooltip(
            "Withdrawal source start date. (MM/YYYY).",
            target="monte-carlo-withdrawal-start",
            placement="bottom"
        ),
        dbc.Tooltip(
            "Withdrawal source end date. (MM/YYYY)",
            target="monte-carlo-withdrawal-end",
            placement="bottom"
        ),

        # SIMULATION TOOLTIPS
        dbc.Tooltip(
            "Simulation start date. Set to today's month and year.",
            target="monte-carlo-time-start",
            placement="bottom"
        ),
        dbc.Tooltip(
            "Simulation end date. (MM/YYYY)",
            target="monte-carlo-time-end",
            placement="bottom"
        ),
        dbc.Tooltip(
            "A higher number of simulations increases the accuracy of the model.",
            target="monte-carlo-simulations-label",
            placement="bottom"
        ),

        # THRESHOLD TOOLTIPS
        dbc.Tooltip(
            "'Reaches' will label the first occurence of the threshold value from left to right. 'Drops to' will label the first occurence of the threshold value from right to left.",
            target='monte-carlo-threshold-direction',
            placement="top"
        ),
        dbc.Tooltip(
            "The value to be marked on the simulation. Useful for knowing when a portfolio might reach a certain $ value.",
            target='monte-carlo-threshold',
            placement="top"
        ),
        dbc.Tooltip(
            "Line to mark the threshold on.",
            target='monte-carlo-threshold-type',
            placement="top"
        ),])
    ])
    return layout

#################################
# ASSETS (AVG. RETURN / STDDEV) #
#################################

# add asset button
@callback(
    Output('monte-carlo-arstd-container', 'children'),
    Input('monte-carlo-add-arstd', 'n_clicks'),
    State('monte-carlo-arstd-container', 'children')
)
def update_cw_container(arstd_clicks, children):
    if arstd_clicks == 0:
        raise PreventUpdate
    
    num_children = len(children) # used to name the ids
    inputgroup = []

    if 'monte-carlo-add-arstd' == ctx.triggered_id:
        inputgroup = cw.create_asset_source(num_children, dropdown_menu_items)
    children += [inputgroup]
        
    return children

# Preset indexes
@callback(
    [Output({'type':'monte-carlo-average-return', 'index':MATCH}, 'value'),
     Output({'type':'monte-carlo-standard-deviation', 'index':MATCH}, 'value'),
     Output({'type':'monte-carlo-average-return', 'index':MATCH}, 'disabled'),
     Output({'type':'monte-carlo-standard-deviation', 'index':MATCH}, 'disabled')],
    Input({'type':'monte-carlo-arstd-dropdown', 'index':MATCH}, 'value'),
    State({'type':'monte-carlo-arstd-dropdown', 'index':MATCH}, 'id'),
)
def update_preset_index(value, id):
    if value is None: 
        raise PreventUpdate

    return index_presets.get(value)[0], index_presets.get(value)[1], True, True

# sum asset allocations
@callback(
    Output('monte-carlo-allocation-sum', 'value'),
    Input({'type':'monte-carlo-ratio', 'index':ALL}, 'value')
)
def sum_asset_allocations(asset_allocations):
    sum_allocations = 0
    for allocation in asset_allocations:
        if allocation is None:
            allocation = 0
        sum_allocations += allocation
    return sum_allocations

# check if asset allocation sum is 100
@callback(
    Output('monte-carlo-allocation-sum', 'invalid'),
    Input('monte-carlo-allocation-sum', 'value')
)
def asset_allocation_validity(value):
    if value != 100:
        return True
    return False

###############################
# CONTRIBUTIONS / WITHDRAWALS #
###############################

# validate contribution input
@callback(
    Output({'type':'monte-carlo-contribution-input', 'index': MATCH}, 'invalid'),
    Input({'type':'monte-carlo-contribution-input', 'index': MATCH}, 'value')
)
def validate_contribution_input(value):
    if value is None:
        return True
    
# validate withdrawal input
@callback(
    Output({'type':'monte-carlo-withdrawal-input', 'index': MATCH}, 'invalid'),
    Input({'type':'monte-carlo-withdrawal-input', 'index': MATCH}, 'value')
)
def validate_withdrawal_input(value):
    if value is None:
        return True
    
# store contributions and withdrawal data
@callback(
    [Output('monte-carlo-cw-data', 'data')],
    [Input('monte-carlo-intervals-data', 'data'),
     Input('monte-carlo-time-interval', 'value'),
     Input({'type':'monte-carlo-contribution-input', 'index': ALL}, 'value'),
     Input({'type':'monte-carlo-contribution-start', 'index': ALL}, 'value'),
     Input({'type':'monte-carlo-contribution-end', 'index': ALL}, 'value'),
     Input({'type':'monte-carlo-withdrawal-input', 'index': ALL}, 'value'),
     Input({'type':'monte-carlo-withdrawal-start', 'index': ALL}, 'value'), 
     Input({'type':'monte-carlo-withdrawal-end', 'index': ALL}, 'value'),
     Input({'type':'monte-carlo-contribution-input', 'index': ALL}, 'invalid'),
     Input({'type':'monte-carlo-contribution-start', 'index': ALL}, 'invalid'),
     Input({'type':'monte-carlo-contribution-end', 'index': ALL}, 'invalid'),
     Input({'type':'monte-carlo-withdrawal-input', 'index': ALL}, 'invalid'),
     Input({'type':'monte-carlo-withdrawal-start', 'index': ALL}, 'invalid'),
     Input({'type':'monte-carlo-withdrawal-end', 'index': ALL}, 'invalid'),]
)
def store_intervals_data(intervals, interval_type,
                         contribution_arr, contribution_start_arr, contribution_end_arr, 
                         withdrawal_arr, withdrawal_start_arr, withdrawal_end_arr, 
                         contribution_invalid, contribution_start_invalid, contribution_end_invalid,
                         withdrawal_invalid, withdrawal_start_invalid, withdrawal_end_invalid):
    
    if any([any(contribution_start_invalid), any(contribution_end_invalid), any(withdrawal_start_invalid), any(withdrawal_end_invalid)]):
        raise PreventUpdate

    if any([any(contribution_invalid), any(withdrawal_invalid)]):
        raise PreventUpdate
    cw_array = [0] * intervals

    start_date = datetime.strptime(START_DATE, "%m/%Y")

    for contribution in range(len(contribution_arr)):
        
        contribution_start = datetime.strptime(contribution_start_arr[contribution], "%m/%Y")
        contribution_end = datetime.strptime(contribution_end_arr[contribution], "%m/%Y")
        
        # TODO: move to own function
        if interval_type == "y":
            contribution_difference_start = math.ceil(contribution_start.year - start_date.year)
            contribution_difference_end = math.ceil(contribution_end.year - start_date.year)
        elif interval_type == "m":
            contribution_difference_start = math.ceil((contribution_start.year - start_date.year) * 12 + contribution_start.month - start_date.month)
            contribution_difference_end = math.ceil((contribution_end.year - start_date.year) * 12 + contribution_end.month - start_date.month)
        for interval in range(len(cw_array)): 
            if contribution_difference_start <= interval <= contribution_difference_end:
                cw_array[interval] += contribution_arr[contribution]

    for withdrawal in range(len(withdrawal_arr)):

        withdrawal_start = datetime.strptime(withdrawal_start_arr[withdrawal], "%m/%Y")
        withdrawal_end = datetime.strptime(withdrawal_end_arr[withdrawal], "%m/%Y")

        # TODO: move to own function
        if interval_type == "y":
            withdrawal_difference_start = math.ceil(withdrawal_start.year - start_date.year)
            withdrawal_difference_end = math.ceil(withdrawal_end.year - start_date.year)
        elif interval_type == "m":
            withdrawal_difference_start = math.ceil((withdrawal_start.year - start_date.year) * 12 + withdrawal_start.month - start_date.month)
            withdrawal_difference_end = math.ceil((withdrawal_end.year - start_date.year) * 12 + withdrawal_end.month - start_date.month)

        for interval in range(len(cw_array)):
            if withdrawal_difference_start <= interval <= withdrawal_difference_end:
                cw_array[interval] -= withdrawal_arr[withdrawal]

    return [cw_array]


# validate withdrawal end inputs
@callback(
    Output({'type':'monte-carlo-withdrawal-end', 'index': MATCH}, 'invalid'),
    [Input({'type':'monte-carlo-withdrawal-start', 'index': MATCH}, 'invalid'),
     Input({'type':'monte-carlo-withdrawal-start', 'index': MATCH}, 'value'),
     Input('monte-carlo-time-periods', 'value'),
     Input({'type':'monte-carlo-withdrawal-end', 'index': MATCH}, 'value'), 
     Input({'type':'monte-carlo-withdrawal-end', 'index': MATCH}, 'pattern')]
)
def update_time_period_validity(start_invalid, min_value, max_value, value, pattern):
    if start_invalid:
        raise PreventUpdate
    if not value or not re.match(pattern, value): #TODO: revise regex, causing errors on \\ after text
        return True
    elif datetime.strptime(min_value, '%m/%Y') > datetime.strptime(value, '%m/%Y') > datetime.strptime(max_value, '%m/%Y'):
        return True
    else:
        return False
    
# validate withdrawal start inputs
@callback(
    Output({'type':'monte-carlo-withdrawal-start', 'index': MATCH}, 'invalid'),
    [Input('monte-carlo-time-periods', 'value'),
     Input({'type':'monte-carlo-withdrawal-start', 'index': MATCH}, 'value'), 
     Input({'type':'monte-carlo-withdrawal-start', 'index': MATCH}, 'pattern')]
)
def update_time_period_validity(max_value, value, pattern):
    if not value or not re.match(pattern, value):
        return True
    elif datetime.strptime(START_DATE, '%m/%Y') > datetime.strptime(value, '%m/%Y') > datetime.strptime(max_value, '%m/%Y'): # must be greater than or equal to start date
        return True
    else:
        return False

# validate contribution end inputs
@callback(
    Output({'type':'monte-carlo-contribution-end', 'index': MATCH}, 'invalid'),
    [Input({'type':'monte-carlo-contribution-start', 'index': MATCH}, 'invalid'),
     Input({'type':'monte-carlo-contribution-start', 'index': MATCH}, 'value'),
     Input('monte-carlo-time-periods', 'value'),
     Input({'type':'monte-carlo-contribution-end', 'index': MATCH}, 'value'), 
     Input({'type':'monte-carlo-contribution-end', 'index': MATCH}, 'pattern')]
)
def update_time_period_validity(start_invalid, min_value, max_value, value, pattern):
    if start_invalid:
        raise PreventUpdate
    if not value or not re.match(pattern, value):
        return True
    elif datetime.strptime(min_value, '%m/%Y') > datetime.strptime(value, '%m/%Y') > datetime.strptime(max_value, '%m/%Y'):
        return True
    else:
        return False
    
# validate contribution start inputs
@callback(
    Output({'type':'monte-carlo-contribution-start', 'index': MATCH}, 'invalid'),
    [Input('monte-carlo-time-periods', 'value'),
     Input({'type':'monte-carlo-contribution-start', 'index': MATCH}, 'value'), 
     Input({'type':'monte-carlo-contribution-start', 'index': MATCH}, 'pattern')]
)
def update_time_period_validity(max_value, value, pattern):
    if not value or not re.match(pattern, value):
        return True
    elif datetime.strptime(START_DATE, '%m/%Y') > datetime.strptime(value, '%m/%Y') > datetime.strptime(max_value, '%m/%Y'): # must be greater than or equal to start date
        return True
    else:
        return False

# add contributions and withdrawals source button
@callback(
    Output('monte-carlo-cw-container', 'children'),
    [Input('monte-carlo-add-c', 'n_clicks'),
     Input('monte-carlo-add-w', 'n_clicks'),
     Input('monte-carlo-time-periods', 'value'),],
    State('monte-carlo-cw-container', 'children')
)
def update_cw_container(nc, nw, time_periods, children):
    if nw == 0 and nc == 0:
        raise PreventUpdate
    
    num_children = len(children) # used to name the ids
    inputgroup = []

    if 'monte-carlo-add-c' == ctx.triggered_id:
        inputgroup = cw.create_cw_source(num_children, time_periods, "Contribution")
    elif 'monte-carlo-add-w' == ctx.triggered_id:
        inputgroup = cw.create_cw_source(num_children, time_periods, "Withdrawal")
    children += [inputgroup]

    return children

##############
# SIMULATION #
##############

# convert time periods input to intervals
@callback(
    Output('monte-carlo-intervals-data', 'data'),
    [Input('monte-carlo-time-interval', 'value'),
     Input('monte-carlo-time-periods', 'value'),
     Input('monte-carlo-time-periods', 'invalid'),]
)
def store_intervals_data(interval_type, value, invalid):
    if invalid:
        raise PreventUpdate
    
    input_date = datetime.strptime(value, "%m/%Y")
    today = datetime.today()

    if interval_type == "y":
        years_difference = input_date.year - today.year
        return math.ceil(years_difference)
    
    elif interval_type == "m":
        months_difference = (input_date.year - today.year) * 12 + input_date.month - today.month
        return math.ceil(months_difference)

# validate time periods input
@callback(
    Output('monte-carlo-time-periods', 'invalid'),
    [Input('monte-carlo-time-periods', 'value'), 
     Input('monte-carlo-time-periods', 'pattern')]
)
def update_time_period_validity(value, pattern):
    if not value or not re.match(pattern, value):
        return True
    else:
        return False

# Paragraph explaining the results of the graph
@callback(
    [Output('monte-carlo-graph-info-header', 'children'),
     Output('monte-carlo-graph-info', 'children')],
    [Input('monte-carlo', 'figure'),
     Input('monte-carlo-threshold', 'value'),
     Input('monte-carlo-threshold-type', 'value'),
     Input('monte-carlo-threshold-direction', 'value')]
)
def update_paragraph(figure, threshold, threshold_type, threshold_direction):
    if figure is None or not figure.get('data'):
        return None, None
    
    paragraph = u.update_paragraph(figure, threshold, threshold_type, threshold_direction)

    return "What do these results mean?", paragraph

# Monte Carlo Simulation
@callback(
    [Output('monte-carlo', 'figure'),
     Output('monte-carlo-button', 'n_clicks'),
     Output('data-table', 'data'),
     Output('data-table', 'columns'),
     Output('data-table','style_data_conditional')],
    [Input('monte-carlo-simulations', 'value'),
     Input('monte-carlo-intervals-data', 'data'),
     Input('monte-carlo-button', 'n_clicks'),
     Input('initial-investment', 'value'),
     Input('monte-carlo-threshold', 'value'),
     Input('monte-carlo-threshold-type', 'value'),
     Input('monte-carlo-cw-container', 'children'),
     Input('monte-carlo-cw-data', 'data'),
     Input('monte-carlo-threshold-direction', 'value'),
     Input('monte-carlo-time-interval', 'value'),
     Input({'type':'monte-carlo-ratio', 'index':ALL}, 'value'),
     Input({'type':'monte-carlo-average-return', 'index':ALL}, 'value'),
     Input({'type':'monte-carlo-standard-deviation', 'index':ALL}, 'value')],
    
)
def update_monte_carlo(simulations, years, n_clicks, initial_investment, threshold_input, threshold_type, cw_children, cw_array, threshold_direction, time_interval, ratios, avg_rets, std_devs):    
    df = None
    datatable_columns = None
    style_data_conditional =[]
    fig = go.Figure()
    fig.update_layout(margin=dict(t=30))
    colors = ['#00CC96','#636EFA','#EF553B','#AB63FA']

    if n_clicks is None:
        n_clicks = 0
    if n_clicks != 0:
        
        # Averaging out stddevs and avg returns
        avg_temp, standard_deviation_temp = u.get_arstd_children(ratios, avg_rets, std_devs)
        avg = avg_temp/100
        standard_deviation = standard_deviation_temp/100

        # Switch between months and years
        if time_interval == "m":
            current_year = datetime.now()
            end_period = (current_year+relativedelta(months=years)).strftime("%Y-%m")
            current_year = current_year.strftime("%Y-%m")
            date_range = pd.date_range(start=current_year, end=end_period, freq='MS')
            x_range = [date.strftime("%Y-%m") for date in date_range]
            avg /= 12
            standard_deviation /= math.sqrt(12)
        elif time_interval == "y":
            current_year = datetime.now().year
            x_range = [current_year+i for i in range(years+1)]

        portfolio_values = np.zeros((simulations, years + 1))
        portfolio_values_clone = np.zeros((simulations, years + 1))

        for simulation in range(simulations):
            returns = np.random.normal(avg, standard_deviation, years)
            returns_text = np.insert(returns, 0, 0) * 100

            portfolio_values[simulation][0] = initial_investment
            portfolio_values_clone[simulation][0] = initial_investment

            for year in range(years):
                portfolio_values[simulation][year+1] = (portfolio_values[simulation][year]) * (1 + returns[year])
                portfolio_values_clone[simulation][year+1] = (portfolio_values_clone[simulation][year]) * (1 + returns[year])
                
                portfolio_values[simulation][year+1] += cw_array[year]

            fig.add_trace(go.Scatter(x=x_range, y=portfolio_values[simulation], line=dict(width=3, color='gray'), text=returns_text, name=f"Simulation {simulation+1}", visible='legendonly'))

        # Mean
        mean_portfolio_values = np.mean(portfolio_values, axis=0)
        
        mean_portfolio_values_c = np.mean(portfolio_values_clone, axis=0)
        returns_text_average_c = np.diff(mean_portfolio_values_c) / mean_portfolio_values_c[:-1] * 100
        returns_text_average_c = np.insert(returns_text_average_c, 0, 0)

        fig.add_trace(go.Scatter(x=x_range, y=mean_portfolio_values, name='Average', text=returns_text_average_c, line=dict(width=3), legendrank=1, marker_color=colors[1]))

        # Median
        median_portfolio_values = np.median(portfolio_values, axis=0) 

        median_portfolio_values_c = np.median(portfolio_values_clone, axis=0)
        returns_text_median_c = np.diff(median_portfolio_values_c) / median_portfolio_values_c[:-1] * 100
        returns_text_median_c = np.insert(returns_text_median_c, 0, 0)

        fig.add_trace(go.Scatter(x=x_range, y=median_portfolio_values, name='Median', text=returns_text_median_c, line=dict(width=3), legendrank=2, marker_color=colors[2]))

        # 95th percentile
        percentile_95_values = np.percentile(portfolio_values, 95, axis=0)

        percentile_95_values_c = np.percentile(portfolio_values_clone, 95, axis=0)
        returns_text_percentile_95_c =  np.diff(percentile_95_values_c) / percentile_95_values_c[:-1] * 100
        returns_text_percentile_95_c = np.insert(returns_text_percentile_95_c, 0, 0)

        fig.add_trace(go.Scatter(x=x_range, y=percentile_95_values, name='95th Percentile', text=returns_text_percentile_95_c, line=dict(width=3), legendrank=3, marker_color=colors[0]))

        # 5th percentile
        percentile_5_values = np.percentile(portfolio_values, 5, axis=0)

        percentile_5_values_c = np.percentile(portfolio_values_clone, 5, axis=0)
        returns_text_percentile_5_c =  np.diff(percentile_5_values_c) / percentile_5_values_c[:-1] * 100
        returns_text_percentile_5_c = np.insert(returns_text_percentile_5_c, 0, 0)

        fig.add_trace(go.Scatter(x=x_range, y=percentile_5_values, name='5th Percentile', text=returns_text_percentile_5_c, line=dict(width=3), legendrank=4, marker_color=colors[3]))
        
        # Fig styling
        text = "<br>".join(["<b>Year</b>: %{x}","<b>Value</b>: $%{y:.2f}","<b>Return</b>: %{text:.2f}%"])
        fig.update_traces(showlegend=True, mode='lines', hovertemplate=text)
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text=f"Growth of ${initial_investment:,}")
        fig.update_layout(hovermode="x")
        fig.add_hline(y=initial_investment, line_dash="dot",
              annotation_text=f"${initial_investment}", 
              annotation_position="bottom left",
              annotation_font_size=15,
        )
        
        match threshold_type:
            case "mean":
                values = mean_portfolio_values
            case "median":
                values = median_portfolio_values
            case "95":
                values = percentile_95_values
            case "5":
                values = percentile_5_values
            case _:
                values = mean_portfolio_values
        
        u.threshold_line(fig,threshold_input, values, threshold_type, threshold_direction)

        # Populate Datatable
        columns = x_range
        if time_interval == "y":
            columns = [str(x) for x in x_range]
        columns.append("Total Return")
        rows = ['95th Percentile', 'Average', 'Median', '5th Percentile']
        df = pd.DataFrame(index=rows, columns=columns)
        
        df.loc['95th Percentile'] = np.char.add(np.char.mod('%.2f', np.append(returns_text_percentile_95_c, ((percentile_95_values_c[-1]/percentile_95_values_c[0])-1)*100)), '%')
        df.loc['Average'] = np.char.add(np.char.mod('%.2f', np.append(returns_text_average_c, ((mean_portfolio_values_c[-1]/mean_portfolio_values_c[0])-1)*100)), '%')
        df.loc['Median'] = np.char.add(np.char.mod('%.2f', np.append(returns_text_median_c, ((median_portfolio_values_c[-1]/median_portfolio_values_c[0])-1)*100)), '%')
        df.loc['5th Percentile'] = np.char.add(np.char.mod('%.2f', np.append(returns_text_percentile_5_c, ((percentile_5_values_c[-1]/percentile_5_values_c[0])-1)*100)), '%')
        if time_interval == "m":
            df.drop(current_year, axis=1, inplace=True)
        elif time_interval == "y":
            df.drop(str(current_year), axis=1, inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index':''}, inplace=True)

        #Datatable row colors
        colors2 = [u.lighten_hex_color(color, 10) for color in colors]
        style_data_conditional = [
            {
                'if': {'row_index': i},
                'backgroundColor': color
            } for i, color in enumerate(colors2)
        ]

        datatable_columns = [{"name": i, "id": i} for i in df.columns]
        df = df.to_dict('records')

    # Hide graph before running
    if not fig.data:
        fig.update_layout(
            xaxis = dict(visible = False),
            yaxis = dict(visible = False),
            annotations = [dict(text="Run the simulation to populate this graph.",xref="paper",yref="paper", showarrow=False)]
        )

    return fig, 0, df, datatable_columns, style_data_conditional


    