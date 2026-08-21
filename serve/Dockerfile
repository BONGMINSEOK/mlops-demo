FROM python:3.11-slim
WORKDIR /app
COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve/app.py .
ENV MLFLOW_TRACKING_URI=http://mlflow.mlops.svc.cluster.local:5000
ENV MODEL_NAME=iris-classifier
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
