import dash_bootstrap_components as dbc

def create_asset_source(arstd_id):

    dropdown_menu_avg_return = [
    dbc.DropdownMenuItem("Input", id={'type':'dropdown-menu-avg-return-input','index':arstd_id}),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("S&P 500 (10y)", id={'type':'dropdown-menu-avg-return-sp500','index':arstd_id}),
    dbc.DropdownMenuItem("DJIA (34y)", id={'type':'dropdown-menu-avg-return-djia','index':arstd_id}),
    ]
    dropdown_menu_std_dev = [
        dbc.DropdownMenuItem("Input", id={'type':'dropdown-menu-std-dev-input','index':arstd_id}),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("S&P 500 (10y)", id={'type':'dropdown-menu-std-dev-sp500','index':arstd_id}),
        dbc.DropdownMenuItem("DJIA (34y)", id={'type':'dropdown-menu-std-dev-djia','index':arstd_id})
    ]


    inputgroup = dbc.InputGroup([
        dbc.InputGroupText("Average Return"), 
        dbc.DropdownMenu(dropdown_menu_avg_return, color="secondary"),
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
        dbc.DropdownMenu(dropdown_menu_std_dev, color="secondary"),
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
