
FROM ubuntu:jammy

RUN mkdir /app
RUN apt update && apt upgrade -y

RUN apt-get update && apt-get install libgl1 -y  libglib2.0-0 libsm6 libxrender1 libxext6

RUN apt install -y python3 python3-pip curl netcat

RUN dos2unix /app/run.sh

ADD requirements.txt run.sh main.py /app

RUN python3 -m pip install -r /app/requirements.txt

CMD PYTHONPATH=/app /app/run.sh

