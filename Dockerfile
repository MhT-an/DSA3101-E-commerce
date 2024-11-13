FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy notebooks into the container

COPY /Notebooks/ /app/Notebooks/

