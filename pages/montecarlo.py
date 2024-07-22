from datetime import datetime
import math
import time
import util as u
from dash import dcc, html, dash_table, ctx, callback, register_page
import numpy as np
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
from dateutil.relativedelta import relativedelta
import cwutil as cw


MAX_SIMULATIONS = 100000
MAX_TIME_PERIODS = 100

dropdown_menu_avg_return = [
    dbc.DropdownMenuItem("Input", id={'type':'dropdown-menu-avg-return-input','index':1}),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("S&P 500 (10y)", id={'type':'dropdown-menu-avg-return-sp500','index':1}),
    dbc.DropdownMenuItem("DJIA (34y)", id={'type':'dropdown-menu-avg-return-djia','index':1}),
]
dropdown_menu_std_dev = [
    dbc.DropdownMenuItem("Input", id={'type':'dropdown-menu-std-dev-input','index':1}),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("S&P 500 (10y)", id={'type':'dropdown-menu-std-dev-sp500','index':1}),
    dbc.DropdownMenuItem("DJIA (34y)", id={'type':'dropdown-menu-std-dev-djia','index':1})
]


register_page(
    __name__,
    name='Monte Carlo Simulation',
    top_nav=True,
    path='/monte-carlo-simulation'
)

def layout():

    
    dbc.Tooltip(
        "The average return of an initial investment.",
        target="monte-carlo-average-return",
        placement="top"
    ),

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
                        html.H5(['Portfolio']),

                        # INITIAL INVESTMENT
                        dbc.InputGroup([
                            dbc.InputGroupText("Initial Investment"), 
                            dbc.InputGroupText("$"), 
                            dbc.Input(
                                id='initial-investment',
                                value=1,
                                min=0,
                                type='number',
                                placeholder="Initial Investment"
                            ),
                        ]),
                        dbc.Tooltip(
                            "Initial investment in $.",
                            target="initial-investment",
                            placement="top"
                        ),

                        # AVERAGE RETURN / STANDARD DEVIATION
                        html.Div(id='monte-carlo-arstd-container', children=[
                            html.Div([
                                dbc.Button(
                                    "Add Asset",
                                    id='monte-carlo-add-arstd',
                                    n_clicks=0
                                ),
                            ]),
                            dbc.InputGroup([
                                dbc.InputGroupText("Average Return"), 
                                dbc.DropdownMenu(dropdown_menu_avg_return, color="secondary"),
                                dbc.Input(
                                    id={'type':'monte-carlo-average-return', 'index':1},
                                    value=10,
                                    type='number',
                                    required=True,
                                    placeholder="Enter Average Return"
                                ),
                                dbc.InputGroupText("%"),
                                dbc.InputGroupText(" "), # Gap
                                dbc.InputGroupText("Standard Deviation"), 
                                dbc.DropdownMenu(dropdown_menu_std_dev, color="secondary"),
                                dbc.Input(
                                    id={'type':'monte-carlo-standard-deviation', 'index':1},
                                    value=15,
                                    type='number',
                                    required=True,
                                    placeholder="Enter Standard Deviation"
                                ),
                                dbc.InputGroupText("%"),
                                dbc.InputGroupText(" "), # Gap
                                dbc.InputGroupText("Ratio"), 
                                dbc.Input(
                                    id={'type':'monte-carlo-ratio', 'index':1},
                                    value=100,
                                    min=0,
                                    max=100,
                                    type='number',
                                    required=True,
                                    placeholder="Enter Ratio"
                                ),
                                dbc.InputGroupText("%"),
                            ])
                            ]
                        ),
                        dbc.Tooltip(
                            "Determines by how much returns can deviate from the average. A higher standard deviation means more volatility and risk.",
                            target="monte-carlo-standard-deviation",
                            placement="top"
                        ),
                        
                        html.Div(id='monte-carlo-cw-container', children=[
                            html.Div([
                                dbc.Button(
                                    "Add Contributions",
                                    id='monte-carlo-add-c',
                                    n_clicks=0
                                ),
                                dbc.Button(
                                    "Add Withdrawals",
                                    id='monte-carlo-add-w',
                                    n_clicks=0
                                ),
                            ]),
                            # CONTRIBUTION
                            dbc.InputGroup(
                                id={'type':'monte-carlo-contribution-igcontainer', 'index': 1},
                                children=[ # Contribution inputs
                                dbc.InputGroupText("Contribution"),
                                dbc.InputGroupText("$"), 
                                dbc.Input(
                                    id={'type':'monte-carlo-contribution-input', 'index': 1},
                                    value=0,
                                    min=0,
                                    type='number',
                                    placeholder="Contribution"
                                ),
                                dbc.InputGroupText("Start"), 
                                dbc.Input(
                                    id={'type':'monte-carlo-contribution-start', 'index': 1},
                                    value=0,
                                    min=0,
                                    max=10,
                                    type='number',
                                    placeholder="Start Interval"
                                ),
                                dbc.InputGroupText("End"), 
                                dbc.Input(
                                    id={'type':'monte-carlo-contribution-end', 'index': 1},
                                    value=0,
                                    min=0,
                                    max=10,
                                    type='number',
                                    placeholder="End Interval"
                                )]
                            ),
                            # WITHDRAWAL
                            dbc.InputGroup(
                                id={'type':'monte-carlo-withdrawal-igcontainer', 'index':2},
                                children=[ # Withdrawal inputs
                                dbc.InputGroupText("Withdrawal"),
                                dbc.InputGroupText("$"), 
                                dbc.Input(
                                    id={'type':'monte-carlo-withdrawal-input', 'index': 2},
                                    value=0,
                                    min=0,
                                    type='number',
                                    placeholder="Withdrawal"
                                ),
                                dbc.InputGroupText("Start"), 
                                dbc.Input(
                                    id={'type':'monte-carlo-withdrawal-start', 'index': 2},
                                    value=0,
                                    min=0,
                                    max=10,
                                    type='number',
                                    placeholder="Start Interval"
                                ),
                                dbc.InputGroupText("End"), 
                                dbc.Input(
                                    id={'type':'monte-carlo-withdrawal-end', 'index': 2},
                                    value=0,
                                    min=0,
                                    max=10,
                                    type='number',
                                    placeholder="End Interval"
                                )
                            ])
                        ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'}),
                    ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'}),
                ),
            ], style={'max-width':'none'}),
            
            # Simulation Settings
            html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.H5(['Simulation'],),
                        dbc.InputGroup([
                            dbc.Select(
                                options=[
                                    {"label": "Years", "value": "y"},
                                    {"label": "Months", "value": "m"},
                                ],
                                value="y",
                                id="monte-carlo-time-interval"
                            ),
                            # TIME PERIODS
                            dbc.Input(
                                id='monte-carlo-time-periods',
                                value=10,
                                min=1,
                                max=100,
                                type='number',
                                placeholder="",
                                debounce=True
                            ),
                            dbc.Tooltip(
                                "How many intervals the simulation will run for.",
                                target="monte-carlo-time-periods",
                                placement="bottom"
                            ),

                            # SIMULATIONS
                            dbc.InputGroupText("Simulations"), 
                            dbc.Input(
                                id='monte-carlo-simulations',
                                value=1000,
                                min=1,
                                max=10000,
                                type='number',
                                placeholder="Simulations"
                            ),
                            dbc.Tooltip(
                                "A higher number of simulations increases the accuracy of the model.",
                                target="monte-carlo-simulations",
                                placement="bottom"
                            ),
                        ]),

                        dbc.Tooltip(
                            "A higher number of simulations increases the accuracy of the model.",
                            target="monte-carlo-simulations",
                            placement="bottom"
                        ),
                    ], style={'display':'flex', 'flex-direction':'column', 'gap':'1vh'})
                ),
            ]),

            # Threshold
            html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.H5(['Threshold'],),
                        
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
        inputgroup = cw.create_asset_source(num_children)
    children += [inputgroup]
        
    return children

###############################
# CONTRIBUTIONS / WITHDRAWALS #
###############################

# add contributions source button
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
    
    # after 3 hours it works ok
    for input_group in children[1:]: # looping across inputgroups
        if hasattr(input_group, 'get'):
            for input_group_attribute in input_group.get('props').items(): # looping through inputgroups' inputs
                if input_group_attribute[0] == 'children':
                    for item in input_group_attribute[1]:
                        if 'id' in item.get('props'):
                            if item.get('props').get('id').get('type') == 'monte-carlo-contribution-end' or item.get('props').get('id').get('type') == 'monte-carlo-withdrawal-end':
                                if 'max' in item.get('props'):
                                    item.get('props')['max'] = time_periods
        
    return children
    
# Switch time period between months and years
@callback(
    Output("monte-carlo-time-periods", "max"),
    Input("monte-carlo-time-interval", "value")
)
def update_time_periods(time_interval):
    if time_interval == "m":
        return MAX_TIME_PERIODS * 12
    elif time_interval == "y":
        return MAX_TIME_PERIODS
"""
# disable run simulation button
@callback(
    Output("monte-carlo-button", "disabled"),
    [Input('monte-carlo-standard-deviation', 'value'),
     Input('monte-carlo-average-return', 'value'),
     Input('monte-carlo-time-periods', 'value'),
     Input('initial-investment', 'value')]
)
def disable_button(stddev, avg_return, time_periods, initial_investment):
    disable = False
    if stddev == "" or avg_return == "":
        disable = True
    if time_periods is None:
        disable = True
    if initial_investment is None:
        disable = True
    return disable

@callback(
    Output("monte-carlo-standard-deviation", "value"),
    [Input("dropdown-menu-std-dev-input", "n_clicks"),
     Input("dropdown-menu-std-dev-sp500", "n_clicks"),
     Input("dropdown-menu-std-dev-djia", "n_clicks")],
)
def std_dev_dropdown(n1, n2, n3):
    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "dropdown-menu-std-dev-input":
        return 0
    elif button_id == "dropdown-menu-std-dev-sp500":
        return 15.51
    elif button_id == "dropdown-menu-std-dev-djia":
        return 14.68

@callback(
    Output("monte-carlo-average-return", "value"),
    [Input("dropdown-menu-avg-return-input", "n_clicks"),
     Input("dropdown-menu-avg-return-sp500", "n_clicks"),
     Input("dropdown-menu-avg-return-djia", "n_clicks")],
)
def avg_return_dropdown(n1, n2, n3):

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "dropdown-menu-avg-return-input":
        return 0
    elif button_id == "dropdown-menu-avg-return-sp500":
        return 13.05
    elif button_id == "dropdown-menu-avg-return-djia":
        return 10.79
    """
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
     Input('monte-carlo-time-periods', 'value'),
     Input('monte-carlo-button', 'n_clicks'),
     Input('initial-investment', 'value'),
     Input('monte-carlo-arstd-container', 'children'),
     Input('monte-carlo-threshold', 'value'),
     Input('monte-carlo-threshold-type', 'value'),
     Input('monte-carlo-cw-container', 'children'),
     Input('monte-carlo-threshold-direction', 'value'),
     Input('monte-carlo-time-interval', 'value')],
    
)
def update_monte_carlo(simulations, years, n_clicks, initial_investment, arstd_children, threshold_input, threshold_type, cw_children, threshold_direction, time_interval):
    # prevent updates for null inputs
    #if not contribution_start:
    #    contribution_start = 0
    #if not contribution_end:
    #    contribution_end = 0
    #if not withdrawal_start:
    #    withdrawal_start = 0
    #if not withdrawal_end:
    #    withdrawal_end = 0
    
    df = None
    datatable_columns = None
    style_data_conditional =[]
    fig = go.Figure()
    fig.update_layout(margin=dict(t=30))
    colors = ['#00CC96','#636EFA','#EF553B','#AB63FA']

    if n_clicks is None:
        n_clicks = 0
    if n_clicks != 0:
        
        avg_temp, standard_deviation_temp = u.get_arstd_children(arstd_children)

        avg = avg_temp/100
        print(avg)
        standard_deviation = standard_deviation_temp/100
        print(standard_deviation)

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

                withdrawal = 0
                contribution = 0
                contribution_start = 0
                contribution_end = 0 
                withdrawal_start = 0
                withdrawal_end = 0

                for input_group in cw_children[1:]: # looping across inputgroups
                    if hasattr(input_group, 'get'):
                        for input_group_attribute in input_group.get('props').items(): # looping through inputgroups' inputs
                            if input_group_attribute[0] == 'children':
                                for item in input_group_attribute[1]:
                                    if 'id' in item.get('props'):
                                        if item.get('props').get('id').get('type') == 'monte-carlo-contribution-start':
                                            contribution_start = item.get('props').get('value')
                                            if contribution_start is None:
                                                contribution_start = 0
                                        if item.get('props').get('id').get('type') == 'monte-carlo-contribution-end':
                                            contribution_end = item.get('props').get('value')
                                            if contribution_end is None:
                                                contribution_end = 0
                                        if item.get('props').get('id').get('type') == 'monte-carlo-contribution-input':
                                            contribution = item.get('props').get('value')
                                        if item.get('props').get('id').get('type') == 'monte-carlo-withdrawal-start':
                                            withdrawal_start = item.get('props').get('value')
                                            if withdrawal_start is None:
                                                withdrawal_start = 0
                                        if item.get('props').get('id').get('type') == 'monte-carlo-withdrawal-end':
                                            withdrawal_end = item.get('props').get('value')
                                            if withdrawal_end is None:
                                                withdrawal_end = 0
                                        if item.get('props').get('id').get('type') == 'monte-carlo-withdrawal-input':
                                            withdrawal = item.get('props').get('value')
                            
                if contribution_start-1 <= year <= contribution_end-1:
                    portfolio_values[simulation][year+1] += contribution
                    print(f"Year:{year} | Contribution:{contribution}")
                if withdrawal_start-1 <= year <= withdrawal_end-1:
                    portfolio_values[simulation][year+1] -= withdrawal

                if portfolio_values[simulation][year+1] < 0:
                    portfolio_values[simulation][year+1] = 0.0000001

                if portfolio_values_clone[simulation][year+1] < 0:
                    portfolio_values_clone[simulation][year+1] = 0.0000001
                
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


    