# 베이스 이미지
FROM continuumio/miniconda3

# 작업 디렉터리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY install.txt .

# conda를 사용하여 가상환경 생성 및 패키지 설치
RUN conda create --name myenv python=3.9 && \
    conda run -n myenv pip install -r install.txt

# 소스 코드 복사
COPY . .

# 가상환경 활성화 및 애플리케이션 실행
CMD ["sh", "-c", "conda run -n myenv uvicorn WAY_AI:app --host 0.0.0.0 --port 8001"]