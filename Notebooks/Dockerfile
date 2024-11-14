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
CMD ["/bin/bash", "-c", "find ./Subgroup_A ./Subgroup_B -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute --inplace {} \;"]
