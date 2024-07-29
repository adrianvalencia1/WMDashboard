from datetime import datetime
import dash_bootstrap_components as dbc
from dash import dcc, html
from dateutil.relativedelta import relativedelta

START_DATE = datetime.today().strftime('%m/%Y')

def create_asset_source(arstd_id, dropdown_menu_items):

    inputgroup = html.Div([
        dcc.Dropdown(
            dropdown_menu_items, 
            id={'type':'monte-carlo-arstd-dropdown', 'index':arstd_id},
            style={'min-width':'20vh'},
        ),
        dbc.InputGroup([
            dbc.InputGroupText("Average Return"), 
            dbc.Input(
                id={'type':'monte-carlo-average-return', 'index':arstd_id},
                value=10,
                type='number',
                required=True,
                placeholder="Enter Average Return"
            ),
            dbc.InputGroupText("%"),
            dbc.InputGroupText("Standard Deviation"), 
            dbc.Input(
                id={'type':'monte-carlo-standard-deviation', 'index':arstd_id},
                value=15,
                type='number',
                required=True,
                placeholder="Enter Standard Deviation"
            ),
            dbc.InputGroupText("%"),
            dbc.InputGroupText("Allocation"), 
            dbc.Input(
                id={'type':'monte-carlo-ratio', 'index':arstd_id},
                value=0,
                min=0,
                type='number',
                required=True,
                placeholder="Enter Allocation"
            ),
            dbc.InputGroupText("%"),
        ])
    ], style={'display':'flex', 'gap':'1vh'})

    return inputgroup

def create_cw_source(cw_id, max_interval, cw_type):
    inputgroup = dbc.InputGroup(
        id={'type':'monte-carlo-' + cw_type.lower() + '-igcontainer', 'index':cw_id},
        children=[
            
            dbc.InputGroupText(cw_type),
            dbc.InputGroupText("$"), 
            dbc.Input(
                id={'type':'monte-carlo-' + cw_type.lower() + '-input', 'index':cw_id},
                value=0,
                min=0,
                type='number',
                placeholder=cw_type.lower(),
            ),

            dbc.InputGroupText("Start"), 
            dbc.Input(
                id={'type':'monte-carlo-' + cw_type.lower() + '-start', 'index':cw_id},
                value=START_DATE,
                type='text',
                placeholder="Start Interval",
                pattern=r"(?<![0-9/])(0?[1-9]|1[0-2])/(\d{4})\b"
            ),

            dbc.InputGroupText("End"), 
            dbc.Input(
                id={'type':'monte-carlo-' + cw_type.lower() + '-end', 'index':cw_id},
                value=START_DATE,
                type='text',
                placeholder="End Interval",
                pattern=r"(?<![0-9/])(0?[1-9]|1[0-2])/(\d{4})\b"
            ),
        ]
    )

    return inputgroup
