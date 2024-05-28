FROM python:3.9-slim

WORKDIR /app

COPY install.txt .
RUN pip install -r install.txt --no-cache-dir --progress-bar off --prefer-binary --use-feature=fast-deps

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]