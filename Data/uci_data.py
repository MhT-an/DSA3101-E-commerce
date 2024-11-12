from ucimlrepo import fetch_ucirepo

online_retail = fetch_ucirepo(id=352)

online_retail.data.original.to_csv("/data/online_retail.csv", index=False)