import mlflow
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Iris_Dockerized")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    iris = load_iris()
    model.fit(iris.data, iris.target)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", model.score(iris.data, iris.target))
    
    input_example = iris.data[0:1]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    print("Модель успешно сохранена!")