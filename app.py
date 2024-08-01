import os
from dash import html, dcc, Dash, page_container
import dash_bootstrap_components as dbc
from navbar import create_navbar

NAVBAR = create_navbar()
APP_TITLE = "Wealth Management Dashboard"

app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title=APP_TITLE,
    use_pages=True
)

app.layout = html.Div([
    NAVBAR,
    page_container
])


server = app.server

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5000))