import dash_bootstrap_components as dbc
from dash import dcc, html
def create_asset_source(arstd_id, dropdown_menu_items):

    #dropdown_menu_items = [
    #    {"label": "VTI", "value": "vti"},
    #    {"label": "ITOT", "value": "itot"},
    #    {"label": "S&P 500", "value": "sp500"},
#
    #]

    inputgroup = html.Div([
        dcc.Dropdown(
            dropdown_menu_items, 
            id={'type':'monte-carlo-arstd-dropdown', 'index':arstd_id},
            style={'min-width':'15vh'},
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
            dbc.InputGroupText(" "), # Gap
            dbc.InputGroupText("Standard Deviation"), 
            dbc.Input(
                id={'type':'monte-carlo-standard-deviation', 'index':arstd_id},
                value=15,
                type='number',
                required=True,
                placeholder="Enter Standard Deviation"
            ),
            dbc.InputGroupText("%"),
            dbc.InputGroupText(" "), # Gap
            dbc.InputGroupText("Ratio"), 
            dbc.Input(
                id={'type':'monte-carlo-ratio', 'index':arstd_id},
                value=100,
                min=0,
                max=100,
                type='number',
                required=True,
                placeholder="Enter Ratio"
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
                placeholder=cw_type
            ),

            dbc.InputGroupText("Start"), 
            dbc.Input(
                id={'type':'monte-carlo-' + cw_type.lower() + '-start', 'index':cw_id},
                value=0,
                min=0,
                max=10,
                type='number',
                placeholder="Start Interval"
            ),

            dbc.InputGroupText("End"), 
            dbc.Input(
                id={'type':'monte-carlo-' + cw_type.lower() + '-end', 'index':cw_id},
                value=0,
                min=0,
                max=max_interval,
                type='number',
                placeholder="End Interval"
            ),
        ]
    )

    return inputgroup
