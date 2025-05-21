FROM python:3.9

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.22.0 \
    boto3==1.26.154 \
    scikit-learn==1.6.1 \
    pandas==2.2.3 \
    numpy==2.0.2 \
    cloudpickle==3.1.1 \
    flask==2.2.5 \
    gunicorn==20.1.0

ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=minio123
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

RUN echo 'from mlflow.pyfunc import load_model\n\
from flask import Flask, request, jsonify\n\
import os\n\n\
model = load_model(os.environ["MODEL_URI"])\n\
app = Flask(__name__)\n\n\
@app.route("/invocations", methods=["POST"])\n\
def predict():\n\
    try:\n\
        input_data = request.json["inputs"]\n\
        predictions = model.predict(input_data)\n\
        return jsonify({"predictions": predictions.tolist()})\n\
    except Exception as e:\n\
        return jsonify({"error": str(e)}), 500\n\n\
@app.route("/ping", methods=["GET"])\n\
def ping():\n\
    return "OK"\n\n\
if __name__ == "__main__":\n\
    app.run(host="0.0.0.0", port=1234)' > server.py

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:1234", "--timeout", "120", "server:app"]