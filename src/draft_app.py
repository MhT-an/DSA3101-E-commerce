import pandas as pd
from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import threading
import webbrowser

from data_utils import load_data, get_stockcode_options, update_forecast, generate_peak_order_figures, load_and_process_data, generate_insights_figures

# Load and prepare data
conversion_funnel = pd.read_parquet('conversion_funnel.parquet')
channel_conversion_rate = pd.read_parquet('channel_conversion_rate.parquet')
data = load_data()
data2 = load_and_process_data()

# Function to process data for churn rate
def load_churn_data():
    online_retail = pd.read_csv('online_retail.csv')
    online_retail['InvoiceDate'] = pd.to_datetime(online_retail['InvoiceDate'])
    online_retail['Country'] = online_retail['Country'].fillna('Unknown')
    
    # Sort by customer and invoice date for churn calculations
    online_retail = online_retail.sort_values(by=['CustomerID', 'InvoiceDate'])
    online_retail['NextPurchaseDate'] = online_retail.groupby('CustomerID')['InvoiceDate'].shift(-1)
    online_retail['DaysSinceLastPurchase'] = (online_retail['NextPurchaseDate'] - online_retail['InvoiceDate']).dt.days
    online_retail['YearMonth'] = online_retail['InvoiceDate'].dt.to_period('M')
    
    # Define churned customers (no purchase within 30 days)
    churned = online_retail[online_retail['DaysSinceLastPurchase'] > 30]
    
    # Monthly churn rate for each country
    monthly_churn_country = (
        churned.groupby(['YearMonth', 'Country']).size() / online_retail.groupby(['YearMonth', 'Country']).size()
    ).reset_index().rename(columns={0: 'ChurnRate'})
    monthly_churn_country['YearMonth'] = monthly_churn_country['YearMonth'].dt.to_timestamp()
    
    return monthly_churn_country

# Load churn data
country_churn = load_churn_data()

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
            dcc.Tab(label="Monthly Churn Rate by Country", value="tab-3"),
            dcc.Tab(label="Optimising Inventory Levels", value="tab-4"),
            dcc.Tab(label="Demand & Competition", value="tab-5"),
            dcc.Tab(label="Peak Order Placement Periods", value="tab-6")
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
        fig = px.choropleth(
            country_churn,
            locations="Country",
            locationmode="country names",
            color="ChurnRate",
            title="Churn Rate by Country",
            color_continuous_scale="Reds"
        )
        return html.Div([dcc.Graph(figure=fig)])
    
    elif tab == "tab-4":
        return html.Div([
            html.Label("Select StockCode:"),
            dcc.Dropdown(id="stockcode-dropdown", options=get_stockcode_options(), style={'width': '50%'}),
            html.Div(id="forecast-content")
        ])
    
    elif tab == "tab-5":
        category_options = [{'label': cat, 'value': cat} for cat in data2['Category'].unique() if pd.notnull(cat)]
        return html.Div([
            dcc.Dropdown(id="category-dropdown", options=category_options, placeholder="Select a Category"),
            html.Div(id="insights-content")
        ])
    
    elif tab == "tab-6":
        hour_fig, day_fig, month_fig = generate_peak_order_figures(data)
        return html.Div([
            dcc.Graph(figure=hour_fig),
            dcc.Graph(figure=day_fig),
            dcc.Graph(figure=month_fig)
        ])

# Callback to update forecast content
@app.callback(Output("forecast-content", "children"), Input("stockcode-dropdown", "value"))
def forecast_content(selected_stockcode):
    return update_forecast(selected_stockcode, data)

# Callback to update insights content
@app.callback(
    Output("insights-content", "children"),
    Input("category-dropdown", "value")
)
def update_insights_content(selected_category):
    elasticity_fig, demand_fig, table_fig = generate_insights_figures(data2, selected_category)
    return [
        dcc.Graph(figure=elasticity_fig),
        dcc.Graph(figure=demand_fig),
        dcc.Graph(figure=table_fig)
    ]

# Open browser automatically and start server
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050")

if __name__ == "__main__":
    threading.Timer(1, open_browser).run()
    app.run_server(debug=True)
