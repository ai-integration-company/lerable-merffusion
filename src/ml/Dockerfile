FROM python:3.10-slim

WORKDIR /ml

COPY . .

RUN apt-get update && apt-get install libgl1 -y  libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000


CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]
