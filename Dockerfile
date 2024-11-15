FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY serveChatbot.py .env ./

EXPOSE 8000

CMD ["uvicorn", "serveChatbot:app", "--host", "0.0.0.0", "--port", "8000"] 