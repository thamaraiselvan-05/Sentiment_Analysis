from prefect import flow, task
import subprocess

@task
def train_model():
    subprocess.run(["python", "model_training.py"])

@flow
def training_pipeline():
    train_model()

if __name__ == "__main__":
    training_pipeline()
