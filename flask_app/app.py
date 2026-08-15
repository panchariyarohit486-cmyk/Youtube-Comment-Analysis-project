import torch
import mlflow
import mlflow.pytorch
import dagshub

# Connect to DagsHub MLflow
dagshub.init(
    repo_owner="panchariyarohit486",
    repo_name="youtube-sentiment-analysis",
    mlflow=True
)

# Model registered in MLflow
MODEL_NAME = "bert_yt_comment_sentiment"
MODEL_STAGE = "Staging"

# Load model from MLflow Registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = mlflow.pytorch.load_model(
    model_uri,
    map_location=device
)

model.to(device)
model.eval()

print("✅ Model loaded successfully!")
print("Model:", model)
print("Device:", device)