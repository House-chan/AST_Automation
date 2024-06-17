FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

CMD gunicorn -b 0.0.0.0 'model:app'
