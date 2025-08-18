FROM python:3.11-slim

WORKDIR /app

# System packages (optional, can be minimal)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential ca-certificates && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# temp uploads live here on Render free (ephemeral)
RUN mkdir -p /tmp/uploads

CMD ["python", "tg_churn_bot.py"]
