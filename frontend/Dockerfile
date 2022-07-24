FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY credentials.toml /.streamlit/

COPY . .

ENTRYPOINT streamlit run app.py --server.port 8000
