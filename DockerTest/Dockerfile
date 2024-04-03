FROM python:3.8.9

RUN mkdir -p /home/Docker
WORKDIR /home/Docker
COPY . .
RUN pip3 install fastapi unicorn

EXPOSE 8001
CMD python FastapiTest.py