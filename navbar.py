from dash import html
import dash_bootstrap_components as dbc

LOGO = "https://www.satovsky.com/wp-content/uploads/2020/10/satovsky-new-logo.png"

def create_navbar():
    navbar = [dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Monte Carlo Simulation", href="/monte-carlo-simulation")),
            dbc.NavItem(dbc.NavLink("Portfolio Optimization", href="/portfolio-optimization"))
        ],
        brand=dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.Img(src=LOGO, height="57px")),
                    ]),
                    href="https://satovsky.com",
                )], 
                style={'margin':'0'}
            ),
        color="dark",
        dark=True,
        style={'margin':'0'}
    )]
    
    return navbar