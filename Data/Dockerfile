# Use a lightweight Python base image
FROM python:3.12-slim

# Set environment variables to avoid prompts during installation
ENV PYTHONUNBUFFERED=1

# Set up working directory
WORKDIR /app

# Copy requirements file to the container and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all Python scripts from /Data into the container
COPY . /app/Data

# Set up a directory for output data
RUN mkdir -p /data

