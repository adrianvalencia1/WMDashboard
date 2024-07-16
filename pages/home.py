from dash import html, register_page

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)

def layout():
    layout = html.Div([
        html.H1([
            "home page"
        ])
    ])

    return layout