FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY install.txt .
RUN pip install -r install.txt --no-cache-dir --progress-bar off --prefer-binary --use-feature=fast-deps

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]