FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /ml

RUN apt-get update && \
    apt-get install -y python3 python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN apt-get update && apt-get install libgl1 -y  libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000


CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]