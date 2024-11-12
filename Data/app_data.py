from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import os
import pyarrow.parquet as pq
import time
from dotenv import load_dotenv

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

def write_csv(my_dictionary):
    for key in my_dictionary:
        path = f'/data/{key}.parquet'
        my_dictionary[key].to_parquet(path)
        print(f'Completed {path}')
        

if __name__ == "__main__":
    print("Project ID:", os.getenv("PROJ_ID"))

    query_dict = {}
    query = """
    SELECT
        CASE WHEN hits.eCommerceAction.action_type = '1' THEN 'Click through of product lists'
              WHEN hits.eCommerceAction.action_type = '2' THEN 'Product detail views'
              WHEN hits.eCommerceAction.action_type = '5' THEN 'Check out'
              WHEN hits.eCommerceAction.action_type = '6' THEN 'Completed purchase'
        END AS action,
        CASE WHEN product.v2ProductCategory LIKE '%Office%'
          OR product.v2ProductCategory IN ('Notebooks & Journals', 'Writing') THEN 'Office'
        WHEN product.v2ProductCategory LIKE '%Apparel%'
          OR product.v2ProductCategory LIKE '%Wearables%'
          OR product.v2ProductCategory LIKE '%Men%' THEN 'Apparel'
        WHEN product.v2ProductCategory LIKE '%Bags%'
          OR product.v2ProductCategory IN ('Backpacks', 'More Bags') THEN 'Bags'
        WHEN product.v2ProductCategory LIKE '%Brand%'
          OR product.v2ProductCategory IN ('Apple', 'YouTube', 'Waze', 'Google') THEN 'Brands'
        WHEN product.v2ProductCategory LIKE '%Drinkware%'
          OR product.v2ProductCategory LIKE '%Fun%'
          OR product.v2ProductCategory IN ('Mugs', 'Fruit Games',
            'Housewares', 'Headgear', 'Headwear', 'Tumblers') THEN 'Accessories'
        WHEN product.v2ProductCategory LIKE '%Lifestyle%' THEN 'Lifestyle'
        WHEN product.v2ProductCategory LIKE '%Electronics%' THEN 'Electronics'
        WHEN product.v2ProductCategory IN ('Home', '${escCatTitle}', '(not set)',
        '${productitem.product.origCatName}')
           THEN 'Unavailable'
        WHEN product.v2ProductCategory LIKE '%Sale%' THEN 'Sale'
        ELSE 'Other'
        END AS category,
        COUNT(fullVisitorID) AS users,
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
        UNNEST(hits) AS hits,
        UNNEST(hits.product) AS product
    WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
    AND
        (
        hits.eCommerceAction.action_type != '0'
        AND
        hits.eCommerceAction.action_type != '3'
        AND
        hits.eCommerceAction.action_type != '4'
        )
    GROUP BY
        category, action
    ORDER BY
        users DESC
    """
    conversion_funnel = client.query(query).result().to_dataframe()
    query_dict['conversion_funnel'] = conversion_funnel

    query = '''
    SELECT
      trafficSource.medium AS channel,
      product.v2ProductCategory AS category,
      COUNT(DISTINCT fullVisitorId) AS total_visitors,
      COUNTIF(totals.transactions > 0) AS total_conversions,
      SAFE_DIVIDE(COUNTIF(totals.transactions > 0), COUNT(DISTINCT fullVisitorId)) * 100 AS conversion_rate
    FROM
      `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
      UNNEST(hits) AS hits,
      UNNEST(hits.product) AS product
    WHERE
      _TABLE_SUFFIX BETWEEN '20160801' AND '20170630'  -- Filter for specific date range if needed
    GROUP BY
      channel, category
    ORDER BY
      conversion_rate DESC
    '''
    channel_conversion_rate = client.query(query).result().to_dataframe()
    channel_conversion_rate = clean_categories(channel_conversion_rate, 'category')
    query_dict['channel_conversion_rate'] = channel_conversion_rate

    write_csv(query_dict)
