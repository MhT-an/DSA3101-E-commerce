from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from dotenv import load_dotenv
import os
import pyarrow.parquet as pq
import time

load_dotenv()

key = {
  "type": "service_account",
  "project_id": os.getenv("PROJ_ID"),
  "private_key_id": os.getenv("PKEY_ID"),
  "private_key": os.getenv("GCP_PKEY"),
  "client_email": os.getenv("CLIENT_EMAIL"),
  "client_id": "106977152441456710656",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": os.getenv("client_x509_cert_url"),
  "universe_domain": "googleapis.com"
}

credentials = service_account.Credentials.from_service_account_info(key)
client = bigquery.Client(credentials= credentials,project=os.getenv("PROJ_ID"))

exact_replacements = {
    "Apparel//Men's-T-Shirts": "Apparel/Men's/Men's-T-Shirts",
    'Backpacks': 'Bags/Backpacks',
    'Bottles': 'Accessories/Drinkware/Water Bottles and Tumblers',
    'Drinkware/Bottles': 'Accessories/Drinkware/Water Bottles and Tumblers',
    'Drinkware': 'Accessories/Drinkware',
    'Drinkware/Mugs and Cups': 'Accessories/Drinkware/Mugs and Cups',
    'Drinkware/Water Bottles and Tumblers': 'Accessories/Drinkware/Water Bottles and Tumblers',
    'Electronics/Accessories/Drinkware': 'Accessories/Drinkware',
    'Drinkware/Mugs': 'Accessories/Drinkware/Mugs and Cups',
    'Mugs': 'Accessories/Drinkware/Mugs and Cups',
    'Clearance Sale': 'Sale/Clearance',
    'Spring Sale!': 'Sale/Spring',
    'Fun': 'Accessories/Fun',
    'Fruit Games': 'Accessories/Fun',
    'Lifestyle/Fun': 'Accessories/Fun',
    "Men's-Outerwear": "Apparel/Men's/Men's-Outerwear",
    "Men's/Men's-Performance Wear": "Apparel/Men's/Men's-Performance Wear",
    'Mens Outerwear': "Apparel/Men's/Men's-Outerwear",
    'More Bags': 'Bags/More Bags',
    'Notebooks & Journals': 'Office/Notebooks & Journals',
    'Office/Office Other': 'Office/Other',
    'Office/Writing Instruments': 'Office/Writing',
    'Shop by Brand': 'Brands',
    'Shop by Brand/Google': 'Brands/Google',
    'Shop by Brand/Waze': 'Brands/Waze',
    'Shop by Brand/YouTube': 'Brands/YouTube',
    'Shop by Brand/Android': 'Brands/Android',
    'Google': 'Brands/Google',
    'Housewares': 'Accessories/Housewares',
    'Headgear': 'Apparel/Headgear',
    'Headwear': 'Apparel/Headwear',
    'Home': '',
    'Tumblers': 'Accessories/Drinkware/Water Bottles and Tumblers',
    'Waze': 'Brands/Waze',
    'Wearables': 'Apparel',
    "Wearables/Men's T-Shirts": "Apparel/Men's/Men's-T-Shirts",
    'Writing': 'Office/Writing',
    'YouTube': 'Brands/Youtube',
    'Android': 'Brands/Android',
}

def clean_categories(df, cat_var):
    def clean_elementary(category):
        # Remove unwanted characters and trim whitespace
        category = category.replace('${escCatTitle}', 'Unavailable') \
                        .replace('${productitem.product.origCatName}', 'Unavailable') \
                        .replace('(not set)', 'Unavailable')
        # Remove trailing slashes
        if category.endswith('/'):
            category = category[:-1]  # Remove the last character (the slash)

        # Remove prefix 'Home/'
        if category.startswith('Home/'):
            category = category.replace('Home/', '', 1)  # Remove 'Home/' only once

        if category.startswith('/'):
            category = category.replace('/', '', 1)

        return category
    
    df[cat_var] = df[cat_var].apply(clean_elementary).replace(exact_replacements)
    df[['main_category', 'sub_category', 'subsub_category']] = df[cat_var].str.split('/', expand = True)
    df.drop(cat_var, axis = 1, inplace = True)
    columns_to_fill = ['main_category', 'sub_category', 'subsub_category']
    df[columns_to_fill] = df[columns_to_fill].fillna('Other')

    return df

if __name__ == "__main__":
    query = '''
    SELECT
        trafficSource.medium AS channel,
        product.v2ProductCategory AS product_category,
        COUNT(DISTINCT fullVisitorId) AS total_users,
        COUNT(*) AS total_sessions,
        COUNT(DISTINCT hits.transaction.transactionID) AS total_transactions,
        SUM(IFNULL(product.productRevenue / 1e6, 0)) AS total_revenue -- converting from micros
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
        UNNEST(hits) AS hits, -- Unnesting hits first
        UNNEST(hits.product) AS product -- Then unnesting product from hit
    WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20170801' -- example date range
    GROUP BY
        channel, product_category
    ORDER BY
        total_revenue DESC;
    '''
    result = client.query(query).result().to_dataframe()
    result = clean_categories(result, 'product_category')
    result.to_parquet('data/marketing_channel_metrics.parquet')