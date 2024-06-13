FROM python:3.10-slim

WORKDIR /ml

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000


CMD ["uvicorn", "server:app", "--reload", "--host", "0.0.0.0", "--port", "3000"]