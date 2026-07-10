# Force platform to linux/amd64
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Environment variables will be supplied at container run time

ENTRYPOINT ["python", "-m", "app.main"]
