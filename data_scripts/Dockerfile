# Use a lightweight Python base image
FROM python:3.12-slim

# Set environment variables to avoid prompts during installation
ENV PYTHONUNBUFFERED=1

# Set up working directory
WORKDIR /query

# Copy requirements file to the container and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your scripts
COPY . .

# Set up a directory for output data
RUN mkdir -p /data

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]


