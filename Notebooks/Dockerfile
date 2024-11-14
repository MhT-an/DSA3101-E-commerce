# Base image with Jupyter
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /project

# Copy requirements.txt and install dependencies
COPY requirements.txt /project/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . /project

# Command to run all notebooks in Subgroup_A and Subgroup_B
CMD ["/bin/bash", "-c", 'find ./Data_Cleaning ./Customer_Behavior_and_Sales_Analysis ./Inventory_Management_And_Pricing_Optimization -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute --inplace {} \;']
