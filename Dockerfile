FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN sudo apt-get update
RUN apt-get install --reinstall build-essential
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "WAY_AI:app", "--host", "0.0.0.0", "--port", "8001"]