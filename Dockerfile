FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501
EXPOSE $PORT

CMD ["streamlit","run", "main.py"]
