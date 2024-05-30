FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "WAY_AI:app", "--host", "0.0.0.0", "--port", "8001"]