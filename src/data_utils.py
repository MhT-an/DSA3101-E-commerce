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
stockcodes = ['84879', '85123A', '84077','22361', '21973', '15036']
# stockcodes = ['84879', '85123A', '84077', '72802A', '15036', '90214G', '22361', '21973']

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

    # Apply bias correction
    observed_mean = weekly_sales['y'][-10:].mean()
    forecast_future_mean = forecast_prophet['yhat'][len(weekly_sales):].mean()
    bias = forecast_future_mean - observed_mean

    forecast_prophet['yhat_adjusted'] = (forecast_prophet['yhat'] - bias).clip(lower=0)
    forecast_prophet['yhat_lower_adjusted'] = (forecast_prophet['yhat_lower'] - bias).clip(lower=0)
    forecast_prophet['yhat_upper_adjusted'] = (forecast_prophet['yhat_upper'] - bias).clip(lower=0)

    # Calculate EOQ, Safety Stock, and ROP values
    ordering_cost_per_order = 50  # Fixed cost per order
    holding_cost_per_unit_per_week = 0.5  # Holding cost per unit per week
    lead_time_weeks = 2  # Lead time in weeks

    eoq_list = []
    safety_stock_list = []
    rop_list = []

    for i in range(len(forecast_prophet)):
        weekly_forecast = forecast_prophet.loc[i, 'yhat']
        
        # EOQ calculation
        eoq = np.sqrt((2 * weekly_forecast * ordering_cost_per_order) / holding_cost_per_unit_per_week)
        eoq_list.append(eoq)

        # Safety Stock calculation
        forecast_std_dev = (forecast_prophet.loc[i, 'yhat_upper'] - forecast_prophet.loc[i, 'yhat_lower']) / 2
        safety_stock = forecast_std_dev * np.sqrt(lead_time_weeks)
        safety_stock_list.append(safety_stock)

        # ROP calculation
        lead_time_demand = weekly_forecast * lead_time_weeks
        rop = lead_time_demand + safety_stock
        rop_list.append(rop)

    # Append calculated inventory metrics to forecast data
    forecast_prophet['EOQ'] = eoq_list
    forecast_prophet['Safety_Stock'] = safety_stock_list
    forecast_prophet['ROP'] = rop_list

    # Initialize plot
    forecast_fig = go.Figure()

    # Observed data trace
    forecast_fig.add_trace(go.Scatter(
        x=weekly_sales['ds'], 
        y=weekly_sales['y'], 
        mode='lines', 
        name='Observed', 
        line=dict(color='blue')
    ))

    # Only plot forecast data after the observed data ends
    last_observed_date = weekly_sales['ds'].max()
    forecast_only = forecast_prophet[forecast_prophet['ds'] > last_observed_date]

    # Forecast trace with EOQ, Safety Stock, and ROP in hover text
    forecast_fig.add_trace(go.Scatter(
        x=forecast_only['ds'],
        y=forecast_only['yhat_adjusted'],
        mode='lines',
        name='Forecast',
        line=dict(color='orange'),
        customdata=np.stack([forecast_only['EOQ'], forecast_only['Safety_Stock'], forecast_only['ROP']], axis=-1),
        hovertemplate=(
            "<b>Date:</b> %{x}<br>"
            "<b>Forecasted Quantity:</b> %{y:.2f}<br>"
            "<b>EOQ:</b> %{customdata[0]:.2f}<br>"
            "<b>Safety Stock:</b> %{customdata[1]:.2f}<br>"
            "<b>ROP:</b> %{customdata[2]:.2f}<br>"
            "<extra></extra>"
        )
    ))

    # Confidence interval shading for forecast
    forecast_fig.add_trace(go.Scatter(
        x=forecast_only['ds'],
        y=forecast_only['yhat_upper_adjusted'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    forecast_fig.add_trace(go.Scatter(
        x=forecast_only['ds'],
        y=forecast_only['yhat_lower_adjusted'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.2)',
        showlegend=False
    ))

    # Update layout
    forecast_fig.update_layout(
        title=f'Weekly Sales Forecast for Product {selected_stockcode}',
        xaxis_title='Date',
        yaxis_title='Quantity Sold',
        height=600,
        width=900
    )

    return forecast_fig

####################################################################################################

##arushi's question 2


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

    # Calculate Final Demand using the new logic
    demand_df = price_elasticity_df.copy()

    # Adjusted Demand using Price Elasticity
    demand_df['AdjustedDemand'] = demand_df['TotalQuantity'] * (1 + demand_df['Price_elasticity'])

    # Price-Weighted Demand
    demand_df['PriceWeightedDemand'] = demand_df['TotalQuantity'] / demand_df['AvgUnitPrice']

    # Combine both for Final Demand
    demand_df['Demand'] = demand_df['PriceWeightedDemand'] * (1 + demand_df['Price_elasticity'])
    demand_df.drop(columns=['AdjustedDemand', 'PriceWeightedDemand'], inplace=True)

    # Categorize Demand
    quantiles = demand_df['Demand'].quantile([0.33, 0.66]).values
    def categorize_demand(quantity):
        if quantity < quantiles[0]:  
            return 'Low'
        elif quantity < quantiles[1]:  
            return 'Medium'
        else:  
            return 'High'
    demand_df['Demand_Category'] = demand_df['Demand'].apply(categorize_demand)

    # Calculate Competition Level using the new logic
    competiton_df = demand_df.copy()

    category_stats = (
        competiton_df.groupby('Category')
        .agg(
            Q1_Quantity=('TotalQuantity', lambda x: x.quantile(0.25)),
            Q3_Quantity=('TotalQuantity', lambda x: x.quantile(0.75)),
            Q1_Price=('AvgUnitPrice', lambda x: x.quantile(0.25)),
            Q3_Price=('AvgUnitPrice', lambda x: x.quantile(0.75))
        )
        .reset_index()
    )
    
    competiton_df = competiton_df.merge(category_stats, on='Category', how='left')

    def calculate_competition(row):
        # Price Elasticity Scoring
        price_elasticity_score = 2 if row['Price_elasticity'] > 1 else (1 if row['Price_elasticity'] == 1 else 0)

        # Total Quantity Scoring based on IQR
        if row['TotalQuantity'] < row['Q1_Quantity']:
            total_quantity_score = 0
        elif row['TotalQuantity'] <= row['Q3_Quantity']:
            total_quantity_score = 1
        else:
            total_quantity_score = 2

        # Revenue Scoring based on IQR
        if row['AvgUnitPrice'] < row['Q1_Price']:
            revenue_score = 0
        elif row['AvgUnitPrice'] <= row['Q3_Price']:
            revenue_score = 1
        else:
            revenue_score = 2

        total_score = price_elasticity_score + total_quantity_score + revenue_score

        if total_score <= 2:
            return 'Low'
        elif total_score <= 4:
            return 'Medium'
        else:
            return 'High'

    competiton_df['Competition'] = competiton_df.apply(calculate_competition, axis=1)

    # Drop unnecessary columns
    competiton_df.drop(columns=['Q1_Quantity', 'Q3_Quantity', 'Q1_Price', 'Q3_Price'], inplace=True)

    # Convert competition and demand categories to numerical scores
    competition_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    competiton_df['Competition_Score'] = competiton_df['Competition'].map(competition_mapping)

    demand_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    competiton_df['Demand_Score'] = competiton_df['Demand_Category'].map(demand_mapping)

    # Adjust prices based on new logic
    def adjust_price_using_categories(row):
        base_price = row['AvgUnitPrice']
        elasticity = row['Price_elasticity']
        demand_score = row['Demand_Score']
        competition_score = row['Competition_Score']
        
        # Base adjustment factor
        adjustment_factor = 0

        # Adjust price based on demand and competition categories
        if demand_score == 2:
            if competition_score == 2:
                adjustment_factor = -0.05
            elif competition_score == 1:
                adjustment_factor = 0.15
            else:
                adjustment_factor = 0.25
        elif demand_score == 1:
            if competition_score == 2:
                adjustment_factor = 0
            elif competition_score == 1:
                adjustment_factor = 0.02
            else:
                adjustment_factor = 0.09
        else:
            if competition_score == 2:
                adjustment_factor = -0.10
            elif competition_score == 1:
                adjustment_factor = -0.05
            else:
                adjustment_factor = 0

        # Apply elasticity adjustment
        adjusted_price = base_price * (1 + adjustment_factor * (elasticity ** 0.5))

        # Apply price cap
        max_price = base_price * 1.5
        min_price = base_price * 0.5
        adjusted_price = max(min_price, min(adjusted_price, max_price))

        return adjusted_price

    competiton_df['Adjusted_Price'] = competiton_df.apply(adjust_price_using_categories, axis=1)

    # Create final DataFrame with relevant columns
    adjusted_price_df = competiton_df[['Description', 'AvgUnitPrice', 'Adjusted_Price', 'TotalQuantity', 'Price_elasticity', 'Category', 'Demand', 'Competition', 'Demand_Category']]
    adjusted_price_df['Price_change'] = competiton_df['Adjusted_Price'] - competiton_df['AvgUnitPrice']

    return adjusted_price_df

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
        y='Price_change',
        color='Competition',
        size='TotalQuantity',
        hover_data=['Description'],
        title='Price Elasticity vs Price change by Competition Level'
    )
    elasticity_fig.update_layout(xaxis_title='Price Elasticity', yaxis_title='Price Change')

    # Demand Category Counts
    demand_counts = filtered_data['Demand_Category'].value_counts().reset_index()
    demand_counts.columns = ['Demand_Category', 'Count']
    demand_fig = px.bar(
        demand_counts,
        x='Demand_Category',
        y='Count',
        title='Demand Categories',
        labels={'Demand_Category': 'Demand Category', 'Count': 'Number of Products'}
    )

    # Competition Category Counts
    demand_counts = filtered_data['Competition'].value_counts().reset_index()
    demand_counts.columns = ['Competition', 'Count']
    competition_fig = px.bar(
        demand_counts,
        x='Competition',
        y='Count',
        title='Competition Categories',
        labels={'Competition': 'Competition Category', 'Count': 'Number of Products'}
    )

    # Adjusted Price Comparison Table
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=['Category', 'Description', 'Price_change', 'Demand_Category', 'Competition', 'Price_elasticity']),
        cells=dict(values=[filtered_data[col] for col in ['Category', 'Description', 'Price_change', 'Demand_Category', 'Competition', 'Price_elasticity']])
    )])
    table_fig.update_layout(title='Price Change Comparison with Other Variables')

    return elasticity_fig, demand_fig, competition_fig, table_fig


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

##################################

#subgrp A stuff

def data_loader(gd_link):
    # Define the Google Drive link and convert it to a direct download link
    csv_url = gd_link
    file_id = csv_url.split('/')[-2]
    dwn_url = f'https://drive.google.com/uc?id={file_id}'

    # Get the CSV data with SSL verification
    response = requests.get(dwn_url, verify=certifi.where()).text

    # Load the CSV content directly into a pandas DataFrame without saving it
    csv_raw = StringIO(response)
    data = pd.read_csv(csv_raw)
    return data
