FROM python:3.12-slim

# Set environment variables to avoid prompts during installation
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY /src /app/

EXPOSE 8050

CMD ["python", "draft_app.py"]