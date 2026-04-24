FROM python:3.10

WORKDIR /app

COPY docker_requirements.txt .
RUN pip install -r docker_requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]