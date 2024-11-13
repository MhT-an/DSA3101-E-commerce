import os
from dotenv import load_dotenv
import json

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

print("KAGGLE_USERNAME:", os.getenv("KAGGLE_USERNAME"))
print("KAGGLE_KEY:", os.getenv("KAGGLE_KEY"))

# Create the .kaggle directory in the home directory if it doesn’t exist
kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Define the path to the kaggle.json file
kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

# Write credentials to kaggle.json
with open(kaggle_json_path, "w") as f:
    json.dump({"username": os.getenv("KAGGLE_USERNAME"), "key": os.getenv("KAGGLE_KEY")}, f)

# Set permissions to read-only for security
os.chmod(kaggle_json_path, 0o600)

api = KaggleApi()
api.authenticate()

# Define the dataset name (replace with your dataset)
dataset_name = "karkavelrajaj/amazon-sales-dataset"  # Example dataset

# Define a target directory to save the downloaded dataset
my_dir = '/data'

# Download the dataset
api.dataset_download_files(dataset_name, path=my_dir, unzip=True)

# Delete any JSON files in the download directory
for file in os.listdir(my_dir):
    if file.endswith(".json"):
        os.remove(os.path.join(my_dir, file))
        print(f"Deleted JSON file: {file}")

# Check if any CSV files exist in the directory
csv_files = [file for file in os.listdir(my_dir) if file.endswith(".csv")]
if csv_files:
    print(f"Downloaded and extracted CSV files: {csv_files}")
else:
    print("No CSV files found in the dataset.")
