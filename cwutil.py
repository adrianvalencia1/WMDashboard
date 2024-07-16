import dash_bootstrap_components as dbc

BASE_CW_NAME = 'monte-carlo-cw-'

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
