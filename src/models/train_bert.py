import os
import json
import logging
import random
import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, classification_report

logger = logging.getLogger("train_bert")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

FEATURES_PATH = "data/interim/bert"
MODEL_PATH = "model/bert"
REPORTS_PATH = "reports"


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tensors(path: str = FEATURES_PATH):
    train_ds = TensorDataset(
        torch.tensor(
            np.load(os.path.join(path, "train_input_ids.npy")),
            dtype=torch.long,
        ),
        torch.tensor(
            np.load(os.path.join(path, "train_attention_mask.npy")),
            dtype=torch.long,
        ),
        torch.tensor(
            np.load(os.path.join(path, "train_labels.npy")),
            dtype=torch.long,
        ),
    )

    val_ds = TensorDataset(
        torch.tensor(
            np.load(os.path.join(path, "val_input_ids.npy")),
            dtype=torch.long,
        ),
        torch.tensor(
            np.load(os.path.join(path, "val_attention_mask.npy")),
            dtype=torch.long,
        ),
        torch.tensor(
            np.load(os.path.join(path, "val_labels.npy")),
            dtype=torch.long,
        ),
    )

    return train_ds, val_ds


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")


def evaluate(model, dataloader_val, device):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


def main():
    try:
        params = load_params()["train_bert"]

        set_seed(params.get("seed", 8))

        train_ds, val_ds = load_tensors()

        batch_size = params["batch_size"]
        dataloader_train = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=batch_size)
        dataloader_val = DataLoader(val_ds, sampler=RandomSampler(val_ds), batch_size=batch_size * 2)

        model = BertForSequenceClassification.from_pretrained(
            params["model_name"],
            num_labels=params["num_labels"],
            output_attentions=False,
            output_hidden_states=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.debug("Using device: %s", device)

        optimizer = AdamW(model.parameters(), lr=params["learning_rate"], eps=params.get("eps", 1e-8))
        epochs = params["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs
        )

        os.makedirs(MODEL_PATH, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            loss_train_total = 0

            for batch in dataloader_train:
                model.zero_grad()
                batch = tuple(b.to(device) for b in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}

                outputs = model(**inputs)
                loss = outputs[0]
                loss_train_total += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            loss_train_avg = loss_train_total / len(dataloader_train)
            val_loss, predictions, true_vals = evaluate(model, dataloader_val, device)
            val_f1 = f1_score_func(predictions, true_vals)
            val_accuracy = accuracy_score(true_vals.flatten(), np.argmax(predictions, axis=1).flatten())

            logger.debug(
                "Epoch %d - train_loss: %.4f, val_loss: %.4f, val_f1: %.4f, val_acc: %.4f",
                epoch, loss_train_avg, val_loss, val_f1, val_accuracy,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(MODEL_PATH, "best_model.pth"))
                logger.debug("Checkpoint saved at epoch %d", epoch)

        # final eval + report using the best checkpoint
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "best_model.pth")))
        _, predictions, true_vals = evaluate(model, dataloader_val, device)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        true_flat = true_vals.flatten()

        report = classification_report(true_flat, preds_flat, output_dict=True)
        metrics = {
            "best_val_loss": best_val_loss,
            "val_f1_weighted": f1_score_func(predictions, true_vals),
            "val_accuracy": accuracy_score(true_flat, preds_flat),
            "classification_report": report,
        }

        os.makedirs(REPORTS_PATH, exist_ok=True)
        with open(os.path.join(REPORTS_PATH, "bert_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Best val loss: {best_val_loss:.4f} | val_f1: {metrics['val_f1_weighted']:.4f}")
        logger.debug("train_bert completed successfully")

    except Exception as e:
        logger.error("Failed to train BERT: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()