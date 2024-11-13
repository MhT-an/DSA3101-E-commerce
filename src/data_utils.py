import pandas as pd
import numpy as np
from prophet import Prophet
import requests
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
import certifi
from dash import dcc, html

# Constants for EOQ, Safety Stock, and ROP calculations
order_cost = 50  
holding_cost = 2  
service_level_factor = 1.65  
lead_time_weeks = 2 

# EOQ formula
def calculate_eoq(demand, order_cost, holding_cost):
    if demand > 0:  # Ensure demand is positive
        return math.sqrt((2 * demand * order_cost) / holding_cost)
    return 0

# Safety Stock formula
def calculate_safety_stock(z, demand_std, lead_time):
    return z * demand_std * math.sqrt(lead_time)

# ROP formula
def calculate_rop(avg_daily_demand, lead_time, safety_stock):
    return (avg_daily_demand * lead_time) + safety_stock

def load_data():
    # Load and prepare the data
    csv_url = 'https://drive.google.com/file/d/1GB-if5xcQM64dV4O6_H-HjcgOK21Z9j2/view?usp=sharing'
    file_id = csv_url.split('/')[-2]
    dwn_url = f'https://drive.google.com/uc?id={file_id}'
    response = requests.get(dwn_url, verify=certifi.where()).text
    data = pd.read_csv(StringIO(response))
    # data['Invoice Date'] = pd.to_datetime(data['Invoice Date'])
    return data

# Define the specific StockCodes we're interested in
stockcodes = ['84879', '85123A', '84077', '72802A', '15036', '90214G', '22361', '21973']

def get_stockcode_options():
    return [{'label': code, 'value': code} for code in stockcodes]

def update_forecast(selected_stockcode, data):
    if not selected_stockcode:
        return "Please select a StockCode."

    # Prepare the data
    data['Invoice Date'] = pd.to_datetime(data['Invoice Date'])
    daily_demand = data.groupby(['Invoice Date', 'StockCode'])['Quantity'].sum().reset_index()
    product_data = daily_demand[daily_demand['StockCode'] == selected_stockcode]
    
    if product_data.empty:
        return "No data available for the selected StockCode."

    product_data.set_index('Invoice Date', inplace=True)
    weekly_sales = product_data['Quantity'].resample('W').sum().reset_index()
    weekly_sales.columns = ['ds', 'y']

    # Fit the Prophet model
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model_prophet.fit(weekly_sales)

    # Create future dates and make predictions
    future_dates = model_prophet.make_future_dataframe(periods=52, freq='W')
    forecast_prophet = model_prophet.predict(future_dates)
    
    observed_mean = weekly_sales['y'][-10:].mean()
    forecast_future_mean = forecast_prophet['yhat'][len(weekly_sales):].mean()
    bias = forecast_future_mean - observed_mean

    forecast_prophet['yhat_adjusted'] = (forecast_prophet['yhat'] - bias).clip(lower=0)
    forecast_prophet['yhat_lower_adjusted'] = (forecast_prophet['yhat_lower'] - bias).clip(lower=0)
    forecast_prophet['yhat_upper_adjusted'] = (forecast_prophet['yhat_upper'] - bias).clip(lower=0)

    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=weekly_sales['ds'], 
        y=weekly_sales['y'], 
        mode='lines', 
        name='Observed', 
        line=dict(color='blue')
    ))

    last_observed_date = weekly_sales['ds'].max()
    forecast_fig.add_trace(go.Scatter(
        x=forecast_prophet[forecast_prophet['ds'] > last_observed_date]['ds'],
        y=forecast_prophet[forecast_prophet['ds'] > last_observed_date]['yhat_adjusted'],
        mode='lines',
        name='Forecast',
        line=dict(color='orange')
    ))

    forecast_fig.add_trace(go.Scatter(
        x=forecast_prophet[forecast_prophet['ds'] > last_observed_date]['ds'],
        y=forecast_prophet[forecast_prophet['ds'] > last_observed_date]['yhat_upper_adjusted'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    forecast_fig.add_trace(go.Scatter(
        x=forecast_prophet[forecast_prophet['ds'] > last_observed_date]['ds'],
        y=forecast_prophet[forecast_prophet['ds'] > last_observed_date]['yhat_lower_adjusted'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.2)',
        showlegend=False
    ))

    forecast_fig.update_layout(
        title=f'Weekly Sales Forecast for Product {selected_stockcode}',
        xaxis_title='Date',
        yaxis_title='Quantity Sold',
        height=600,
        width=900
    )

    # Calculate EOQ, Safety Stock, and ROP
    # data['Invoice Date'] = pd.to_datetime(data['Invoice Date'])
    data['Week'] = data['Invoice Date'].dt.to_period('W')
    weekly_data = data.groupby(['StockCode', 'Week']).agg({'Quantity': 'sum'}).reset_index()

    product_demand = weekly_data.groupby('StockCode').agg({
        'Quantity': ['sum', 'mean', 'std']
    }).reset_index()
    product_demand.columns = ['StockCode', 'TotalDemand', 'AvgWeeklyDemand', 'DemandStdDev']

    filtered_product_demand = product_demand[product_demand['StockCode'] == selected_stockcode]
    total_demand = filtered_product_demand['TotalDemand'].values[0]
    avg_weekly_demand = filtered_product_demand['AvgWeeklyDemand'].values[0]
    demand_std_dev = filtered_product_demand['DemandStdDev'].values[0]

    eoq = calculate_eoq(total_demand, order_cost, holding_cost)
    safety_stock = calculate_safety_stock(service_level_factor, demand_std_dev, lead_time_weeks)
    rop = calculate_rop(avg_weekly_demand, lead_time_weeks, safety_stock)

    # Display results with the forecast graph
    return html.Div([
        dcc.Graph(figure=forecast_fig),
        html.Div([
            html.H4("Inventory Calculations"),
            html.P(f"Economic Order Quantity (EOQ): {eoq:.2f} units"),
            html.P(f"Safety Stock: {safety_stock:.2f} units"),
            html.P(f"Reorder Point (ROP): {rop:.2f} units")
        ], style={'marginTop': '20px'})
    ])





####################################################################################################

##shanlei's question 3

def generate_peak_order_figures(data):
    # Ensure Invoice_DateTime is available
    data['Invoice_DateTime'] = pd.to_datetime(data['Invoice Date'].astype(str) + ' ' + data['Invoice Time'].astype(str))

    # Hour of the day
    data['Invoice_Hour'] = data['Invoice_DateTime'].dt.hour
    order_by_hour = data.groupby('Invoice_Hour').size()
    hour_fig = px.bar(
        x=order_by_hour.index,
        y=order_by_hour.values,
        labels={'x': 'Hour of the Day', 'y': 'Number of Orders'},
        title='Order Placement by Hour of the Day'
    )

    # Day of the week
    data['Invoice_DayofWeek'] = data['Invoice_DateTime'].dt.dayofweek
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data['Invoice_DayofWeek'] = data['Invoice_DayofWeek'].apply(lambda x: day_names[x])
    data['Invoice_DayofWeek'] = pd.Categorical(data['Invoice_DayofWeek'], categories=day_names, ordered=True)
    order_by_dayofweek = data.groupby('Invoice_DayofWeek').size()
    day_fig = px.bar(
        x=order_by_dayofweek.index,
        y=order_by_dayofweek.values,
        labels={'x': 'Day of the Week', 'y': 'Number of Orders'},
        title='Order Placement by Day of the Week'
    )

    # Month of the year
    data['Invoice_Month'] = data['Invoice_DateTime'].dt.month
    order_by_month = data.groupby('Invoice_Month').size()
    month_fig = px.bar(
        x=order_by_month.index,
        y=order_by_month.values,
        labels={'x': 'Month of the Year', 'y': 'Number of Orders'},
        title='Order Placement by Month of the Year'
    )

    return hour_fig, day_fig, month_fig






####################################################################################################

##arushi's question 3


def load_and_process_data():
    # Load data
    csv_url = 'https://drive.google.com/file/d/1GB-if5xcQM64dV4O6_H-HjcgOK21Z9j2/view?usp=sharing'
    file_id = csv_url.split('/')[-2]
    dwn_url = f'https://drive.google.com/uc?id={file_id}'
    response = requests.get(dwn_url).text
    online_retail = pd.read_csv(StringIO(response))

    # Data Cleaning
    online_retail = online_retail[(online_retail['Quantity'] > 0) & (online_retail['UnitPrice'] > 0)]
    description_counts = online_retail['Description'].value_counts()
    valid_descriptions = description_counts[description_counts >= 10].index.tolist()
    online_retail = online_retail[online_retail['Description'].isin(valid_descriptions)]

    # Vectorize 'Description' for clustering
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    description_vectors = tfidf.fit_transform(online_retail['Description'])

    # KMeans Clustering and Category Assignment
    num_clusters = 8
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    online_retail['CategoryCluster'] = kmeans.fit_predict(description_vectors)
    cluster_to_category = {
        0: "Storage & Lunch Boxes",
        1: "Kitchen Items",
        2: "Funny Metal Signs",
        3: "Home Decor & Toys",
        4: "Heart & Holiday Decor",
        5: "Greeting Cards & Card Holders",
        6: "Hot Water Bottles",
        7: "Vintage Gift Items"
    }
    online_retail['Category'] = online_retail['CategoryCluster'].map(cluster_to_category)

    # Group by Description and UnitPrice, calculate Price Elasticity
    grouped = online_retail.groupby(['Description', 'UnitPrice']).agg({'Quantity': 'sum'}).reset_index()
    grouped["Price_change"] = grouped.groupby('Description')['UnitPrice'].pct_change()
    grouped["Quantity_change"] = grouped.groupby('Description')['Quantity'].pct_change()
    grouped["Price_elasticity"] = np.abs(grouped["Quantity_change"] / grouped["Price_change"])

    # Calculate mean Price Elasticity and merge Category data
    price_elasticity_df = (
        grouped[grouped["Price_elasticity"] < 10]
        .groupby('Description')
        .agg(Price_elasticity=('Price_elasticity', 'mean'))
        .reset_index()
    )
    temp_category_df = online_retail[['Description', 'Category']].drop_duplicates()
    price_elasticity_df = price_elasticity_df.merge(temp_category_df, on='Description', how='left')

    # Add TotalQuantity and AvgUnitPrice
    product_summary = (
        online_retail.groupby('Description')
        .agg(TotalQuantity=('Quantity', 'sum'), AvgUnitPrice=('UnitPrice', 'mean'))
        .reset_index()
    )
    price_elasticity_df = price_elasticity_df.merge(product_summary, on='Description', how='left')

    # Categorize Demand
    quantiles = price_elasticity_df['TotalQuantity'].quantile([0.33, 0.66]).values
    def categorize_demand(quantity):
        if quantity < quantiles[0]: return 'Low'
        elif quantity < quantiles[1]: return 'Medium'
        else: return 'High'
    price_elasticity_df['Demand_Category'] = price_elasticity_df['TotalQuantity'].apply(categorize_demand)

    # Calculate Competition Level
    median_quantity = price_elasticity_df['TotalQuantity'].median()
    def calculate_competition(row):
        price_elasticity_score = 0 if row['Price_elasticity'] < 1 else (1 if row['Price_elasticity'] == 1 else 2)
        quantity_score = 0 if row['TotalQuantity'] < median_quantity else (1 if row['TotalQuantity'] == median_quantity else 2)
        total_score = price_elasticity_score + quantity_score
        return 'Low' if total_score <= 1 else 'Medium' if total_score == 2 else 'High'
    price_elasticity_df['Competition'] = price_elasticity_df.apply(calculate_competition, axis=1)

    # Adjusted Price Calculation
    def adjust_price(row):
        base_price = row['AvgUnitPrice']
        elasticity = row['Price_elasticity']
        demand_category = row['Demand_Category']
        competition = row['Competition']
        adjustment_factor = (
            0.10 if demand_category == 'High' and competition == 'Low' else
            0.05 if demand_category == 'High' else
            0.03 if demand_category == 'Medium' and competition == 'Medium' else
            0.00
        )
        adjusted_price = base_price * (1 + adjustment_factor * elasticity)
        return min(adjusted_price, base_price * 1.2)
    price_elasticity_df['Adjusted_Price'] = price_elasticity_df.apply(adjust_price, axis=1)

    return price_elasticity_df

def generate_insights_figures(data, selected_category=None):
    # Filter data by category if a category is selected
    if selected_category:
        filtered_data = data[data['Category'] == selected_category]
    else:
        filtered_data = data

    # Price Elasticity vs Quantity Scatter Plot
    elasticity_fig = px.scatter(
        filtered_data,
        x='Price_elasticity',
        y='TotalQuantity',
        color='Competition',
        size='AvgUnitPrice',
        hover_data=['Description'],
        title='Price Elasticity vs Total Quantity Sold by Competition Level'
    )
    elasticity_fig.update_layout(xaxis_title='Price Elasticity', yaxis_title='Total Quantity Sold')

    # Demand Category Counts
    demand_counts = filtered_data['Demand_Category'].value_counts().reset_index()
    demand_counts.columns = ['Demand_Category', 'Count']
    demand_fig = px.bar(
        demand_counts,
        x='Demand_Category',
        y='Count',
        title='Product Demand Categories',
        labels={'Demand_Category': 'Demand Category', 'Count': 'Number of Products'}
    )

    # Adjusted Price Comparison Table
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=['Description', 'AvgUnitPrice', 'Adjusted_Price', 'Price_elasticity', 'Category', 'Demand_Category', 'Competition']),
        cells=dict(values=[filtered_data[col] for col in ['Description', 'AvgUnitPrice', 'Adjusted_Price', 'Price_elasticity', 'Category', 'Demand_Category', 'Competition']])
    )])
    table_fig.update_layout(title='Adjusted Price Comparison')

    return elasticity_fig, demand_fig, table_fig