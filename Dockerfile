FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN export HNSWLIB_NO_NATIVE=1 
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "WAY_AI:app", "--host", "0.0.0.0", "--port", "8001"]