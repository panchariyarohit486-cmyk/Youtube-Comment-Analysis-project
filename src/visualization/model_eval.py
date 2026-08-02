import os
import json
import logging
import numpy as np
import torch
import yaml
import mlflow
import mlflow.pytorch
import dagshub
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

FEATURES_PATH = "data/interim/bert"
MODEL_PATH = "model/bert"
REPORTS_PATH = "reports"
EXPERIMENT_NAME = "BERT_YT_Comment_Sentiment"


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_val_data(path: str = FEATURES_PATH):
    input_ids = torch.from_numpy(np.load(os.path.join(path, "val_input_ids.npy"))).long()
    attention_mask = torch.from_numpy(np.load(os.path.join(path, "val_attention_mask.npy"))).long()
    labels = torch.from_numpy(np.load(os.path.join(path, "val_labels.npy"))).long()
    logger.debug("Loaded val data: %s", input_ids.shape)
    return input_ids, attention_mask, labels


def load_model(model_name: str, num_labels: int, weights_path: str = MODEL_PATH):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    state_dict = torch.load(
        os.path.join(weights_path, "best_model.pth"),
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.load_state_dict(state_dict)
    logger.debug("Loaded best BERT checkpoint from %s", weights_path)
    return model


def evaluate(model, input_ids, attention_mask, labels, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(ds, batch_size=batch_size)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            all_preds.append(preds)
            all_labels.append(batch[2].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted"),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted"),
        "classification_report": classification_report(all_labels, all_preds, output_dict=True),
    }
    return metrics


def save_metrics(metrics: dict, out_path: str = REPORTS_PATH) -> str:
    os.makedirs(out_path, exist_ok=True)
    metrics_path = os.path.join(out_path, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.debug("Metrics saved to %s", metrics_path)
    return metrics_path


def save_run_info(run_id: str, model_uri: str, out_path: str = REPORTS_PATH) -> None:
    os.makedirs(out_path, exist_ok=True)
    info = {"run_id": run_id, "model_uri": model_uri}
    with open(os.path.join(out_path, "experiment_info.json"), "w") as f:
        json.dump(info, f, indent=4)
    logger.debug("Run info saved: %s", info)


def main():
    try:
        dagshub.init(
            repo_owner="panchariyarohit486",
            repo_name="youtube-sentiment-analysis",
            mlflow=True,
        )

        params = load_params()
        bert_params = params["train_bert"]

        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            input_ids, attention_mask, labels = load_val_data()
            model = load_model(bert_params["model_name"], bert_params["num_labels"])

            metrics = evaluate(model, input_ids, attention_mask, labels)

            mlflow.log_params(bert_params)
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
            mlflow.log_metric("precision_weighted", metrics["precision_weighted"])
            mlflow.log_metric("recall_weighted", metrics["recall_weighted"])

            metrics_path = save_metrics(metrics)
            mlflow.log_artifact(metrics_path)

            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                serialization_format="pickle",
            )
            model_uri = f"runs:/{run.info.run_id}/model"

            save_run_info(run.info.run_id, model_uri)

            print(f"Eval accuracy: {metrics['accuracy']:.4f} | f1: {metrics['f1_weighted']:.4f}")
            logger.debug("model_evaluation completed successfully. run_id: %s", run.info.run_id)

    except Exception as e:
        logger.error("Failed to evaluate model: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()