from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
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

if __name__ == "__main__":
    query = '''
    SELECT
        *,
        total_revenue / total_transactions AS avg_order_value,
        total_transactions / total_sessions AS conversion_rate
    FROM (
        SELECT
            trafficSource.medium AS channel,
            COUNT(DISTINCT fullVisitorId) AS total_users,
            COUNT(*) AS total_sessions,
            SUM(totals.transactions) AS total_transactions,
            SUM(totals.transactionRevenue) / 1e6 AS total_revenue -- converting from micros
            , AVG(totals.pageviews) AS avg_pageviews,
            AVG(totals.timeOnSite) AS avg_time_on_site,
            AVG(IF(totals.bounces IS NOT NULL, totals.bounces, 0)) AS avg_bounces
        FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        WHERE
            _TABLE_SUFFIX BETWEEN '20160801' AND '20170801' -- example date range
        GROUP BY
            trafficSource.medium
        ORDER BY
            total_revenue DESC
        );
    '''
    result = client.query(query).result().to_arrow()
    pq.write_table(result, 'aov_conversion.parquet')

    query = '''
    SELECT
    *,
    IF(total_transactions IS NOT NULL, total_revenue / total_transactions, 0) AS avg_order_value,
    total_transactions / total_sessions AS conversion_rate
    FROM (
    SELECT
        trafficSource.campaign,
        COUNT(DISTINCT fullVisitorId) AS total_users,
        COUNT(*) AS total_sessions,
        IF(SUM(totals.transactions) IS NOT NULL, SUM(totals.transactions), 0) AS total_transactions,
        SUM(totals.transactionRevenue) / 1e6 AS total_revenue,
        AVG(IF(totals.pageviews IS NOT NULL, totals.pageviews, 0)) AS avg_pageviews,
        AVG(IF(totals.timeOnSite IS NOT NULL, totals.timeOnSite, 0)) AS avg_time_on_site,
        AVG(IF(totals.bounces IS NOT NULL, totals.bounces, 0)) AS avg_bounces
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20170801' -- example date range
    GROUP BY
        trafficSource.campaign
    ORDER BY
        total_revenue DESC
    )
    ORDER BY
    avg_order_value DESC;
    '''

    result = client.query(query).result().to_arrow()
    pq.write_table(result, 'marketing_metrics.parquet')

    query = '''
    SELECT
        trafficSource.campaign,
        totals.pageviews AS pageviews,
        totals.timeOnSite AS time_on_site
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE
        totals.pageviews IS NOT NULL AND totals.timeOnSite IS NOT NULL
        AND _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
    '''

    result = client.query(query).result().to_arrow()
    pq.write_table(result, 'campaign_distribution.parquet')

    query = '''
    SELECT
        trafficSource.campaign,
        date,
        COUNTIF(totals.newVisits IS NULL) / COUNT(*) AS return_rate
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
    GROUP BY
        trafficSource.campaign, date
    ORDER BY
        date
    '''

    res2 = client.query(query).result().to_dataframe()
    res2['date'] = pd.to_datetime(res2['date'])
    res2.to_parquet('return_rate_by_campaign.parquet')

    query = '''
    WITH FirstVisit AS (
        SELECT
            fullVisitorId,
            MIN(PARSE_DATE('%Y%m%d', date)) AS first_visit_date
        FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
        GROUP BY
            fullVisitorId
    ),

    CustomerTypeByCampaign AS (
        SELECT
            t.fullVisitorId,
            t.trafficSource.campaign,
            IF(PARSE_DATE('%Y%m%d', t.date) = f.first_visit_date, 'New Customer', 'Returning Customer') AS customer_type
        FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS t, FirstVisit AS f
        WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
        AND
            t.fullVisitorId = f.fullVisitorId
    )

    SELECT
        campaign,
        customer_type,
        COUNT(DISTINCT fullVisitorId) AS customer_count
    FROM
        CustomerTypeByCampaign
    WHERE
        campaign IS NOT NULL
        AND campaign != "(not set)"
    GROUP BY
        campaign, customer_type
    ORDER BY
        customer_count DESC;
    '''

    result = client.query(query).result().to_dataframe()
    result = result.pivot(index='campaign', columns='customer_type', values='customer_count').fillna(0)
    result['ratio'] = np.where(
        result['Returning Customer'] == 0,
        0,  # or 0 or another placeholder if you prefer
        result['New Customer'] / result['Returning Customer']
    )
    result = result.sort_values(by='ratio', ascending=False)
    result = result.reset_index()
    result.columns = ['campaign', 'new_customers', 'returning_customers', 'ratio']
    result.to_parquet('new_customer_by_campaign.parquet')

    query = '''
    WITH campaign_metrics AS (
        SELECT
            t.trafficSource.campaign,
            t.trafficSource.medium AS channel,
            COUNT(DISTINCT t.fullVisitorId) AS user_count,
            SUM(t.totals.transactionRevenue) / 1e6 AS total_revenue  -- Revenue in standard units
        FROM
            `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS t
        WHERE
            _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
        AND
            t.trafficSource.campaign IS NOT NULL
        AND t.trafficSource.campaign != "(not set)"
        GROUP BY
            t.trafficSource.campaign, t.trafficSource.medium
        ORDER BY
            total_revenue DESC
    ),
        campaign_dates AS (
        SELECT
        trafficSource.campaign AS campaign,
        MIN(PARSE_DATE('%Y%m%d', date)) AS first_visit_date,
        MAX(PARSE_DATE('%Y%m%d', date)) AS last_visit_date
        FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
        GROUP BY
        trafficSource.campaign
    )

    SELECT
        cm.campaign,
        cm.channel,
        cm.user_count,
        cm.total_revenue,
        cd.first_visit_date,
        cd.last_visit_date
    FROM
        campaign_metrics AS cm, campaign_dates AS cd
        WHERE cm.campaign = cd.campaign
    ORDER BY
        total_revenue DESC;
    '''

    result = client.query(query).result().to_arrow()
    pq.write_table(result, 'data/campaign_info.parquet')
