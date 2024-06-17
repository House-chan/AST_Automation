FROM --platform=linux/amd64 python:3.9

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

CMD gunicorn -b :8080 model:app
