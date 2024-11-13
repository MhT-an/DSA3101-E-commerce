import pandas as pd
from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import threading
import webbrowser

from data_utils import load_data, data_loader, get_stockcode_options, update_forecast, generate_peak_order_figures, load_and_process_data, generate_insights_figures

# Load and prepare data
channel_conversion_rate = data_loader('https://drive.google.com/file/d/1H708cwxwYSl4FrYdxOiU6wUJQ3U4xr3W/view?usp=sharing')
channel_conversion_rate['main_category'] = channel_conversion_rate['main_category'].fillna('Other')
conversion_funnel = data_loader('https://drive.google.com/file/d/1hUIjB77ZUI_Vmn4lKyCEilfS6WkLNRRr/view?usp=sharing')
data = load_data()
data2 = load_and_process_data()


# Stylesheets for external fonts and styles
external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.title = "E-Commerce Analytics"

# Header for dashboard
header = html.Div(
    children=[
        html.H1(children="E-Commerce Optimisation Dashboard", className="header-title"),
        html.P(
            children="An all-in-one dashboard for e-commerce transactions based on different categories",
            className="header-description",
        ),
    ],
    className="header",
)

# Layout with all tabs
app.layout = html.Div([
    header,
    html.Div([
        dcc.Tabs(id="tabs", value="tab-1", vertical=True, children=[
            dcc.Tab(label="Conversion Funnel", value="tab-1"),
            dcc.Tab(label="Conversion Rate by Channel", value="tab-2"),
            dcc.Tab(label="Optimising Inventory Levels", value="tab-3"),
            dcc.Tab(label="Demand & Competition", value="tab-4"),
            dcc.Tab(label="Peak Order Placement Periods", value="tab-5")
        ], style={"display": "flex", "flexDirection": "column", "width": "15%", "height": "100vh"}),
    ], style={"float": "left"}),
    
    html.Div(id='content', style={'marginLeft': '15%', 'padding': '20px'})
])

# Callback to render content based on selected tab
@app.callback(
    Output("content", "children"),
    [Input("tabs", "value")]
)
def render_content(tab):
    if tab == "tab-1":
        unique_categories = conversion_funnel['category'].unique()
        color_map = {category: f'rgba({(i * 50) % 255}, {(i * 100) % 255}, {(i * 150) % 255}, 0.8)' for i, category in enumerate(unique_categories)}
        colors = [color_map[category] for category in conversion_funnel['category']]
        funnel_graph = go.Figure(go.Funnel(
            y=conversion_funnel['action'],
            x=conversion_funnel['users'],
            text=conversion_funnel['category'],
            marker_color=colors,
            textposition='inside',
            textinfo='text+value+percent initial'
        ))
        funnel_graph.update_layout(
            title_text='Google Merchandise Store Conversion Path',
            height=700,
            width=1200
        )
        return html.Div([dcc.Graph(figure=funnel_graph)])

    elif tab == "tab-2":
        fig = px.sunburst(
            channel_conversion_rate,
            path=['channel', 'main_category'],
            values='total_conversions',
            color='channel',
            title='Conversions by Channel & Category'
        )
        return html.Div([dcc.Graph(figure=fig)])
    
    elif tab == "tab-3":
        return html.Div([
            html.Label("Select StockCode:"),
            dcc.Dropdown(id="stockcode-dropdown", options=get_stockcode_options(), style={'width': '50%'}),
            html.Div(id="forecast-content")
        ])
    
    elif tab == "tab-4":
        category_options = [{'label': cat, 'value': cat} for cat in data2['Category'].unique() if pd.notnull(cat)]
        return html.Div([
            dcc.Dropdown(id="category-dropdown", options=category_options, placeholder="Select a Category"),
            html.Div(id="insights-content")
        ])
    
    elif tab == "tab-5":
        hour_fig, day_fig, month_fig = generate_peak_order_figures(data)
        return html.Div([
            dcc.Graph(figure=hour_fig),
            dcc.Graph(figure=day_fig),
            dcc.Graph(figure=month_fig)
        ])

# Update forecast plot with hover info (EOQ, Safety Stock, ROP included)
@app.callback(Output("forecast-content", "children"), Input("stockcode-dropdown", "value"))
def forecast_content(selected_stockcode):
    forecast_fig = update_forecast(selected_stockcode, data)
    
    # If `forecast_fig` is a message (e.g., error or no data)
    if isinstance(forecast_fig, str):
        return html.Div(forecast_fig)
    else:
        # Display forecast figure in a graph component
        return dcc.Graph(figure=forecast_fig)

# Update insights content based on selected category
@app.callback(
    Output("insights-content", "children"),
    Input("category-dropdown", "value")
)
def update_insights_content(selected_category):
    # Generate figures based on the selected category
    elasticity_fig, demand_fig, competition_fig, table_fig = generate_insights_figures(data2, selected_category)
    
    # Return the generated figures as Graph components
    return [
        dcc.Graph(figure=elasticity_fig),
        dcc.Graph(figure=demand_fig),
        dcc.Graph(figure=competition_fig),
        dcc.Graph(figure=table_fig)
    ]

# Open browser automatically and start server
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")

if __name__ == "__main__":
    threading.Timer(1, open_browser).run()
    app.run_server(debug=True)
