import io
import logging
import os

import requests
import torch

from flask import Flask, request, jsonify, send_file
from transformers import BertTokenizer, BertForSequenceClassification

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask_cors import CORS

from wordcloud import WordCloud


logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)


LOCAL_WEIGHTS_PATH = "model/bert/best_model.pth"
BASE_MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 256

DAGSHUB_USER = "panchariyarohit486"
DAGSHUB_REPO = "youtube-sentiment-analysis"
DAGSHUB_REV = "main"
WEIGHTS_REMOTE_PATH = "model/bert/best_model.pth"

# Model index -> sentiment label
INDEX_TO_LABEL = {0: -1, 1: 0, 2: 1}

# Sentiment label -> name
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    }
)
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Using device: %s", device)


def download_weights():
    if os.path.exists(LOCAL_WEIGHTS_PATH):
        logger.info("Weights already present, skipping download")
        return

    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        raise RuntimeError("DAGSHUB_TOKEN env var is not set")

    url = (
        f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}"
        f"/raw/{DAGSHUB_REV}/{WEIGHTS_REMOTE_PATH}"
    )
    os.makedirs(os.path.dirname(LOCAL_WEIGHTS_PATH), exist_ok=True)
    tmp_path = LOCAL_WEIGHTS_PATH + ".part"

    logger.info("Downloading weights from DagsHub...")
    with requests.get(url, auth=(DAGSHUB_USER, token), stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    if os.path.getsize(tmp_path) < 100 * 1024 * 1024:
        os.remove(tmp_path)
        raise RuntimeError("Downloaded file is too small, not the model weights")

    os.replace(tmp_path, LOCAL_WEIGHTS_PATH)
    logger.info("Weights downloaded")


def load_artifacts():
    global model, tokenizer

    download_weights()

    logger.info("Loading BERT base model...")
    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    logger.info("Loading trained weights from: %s", LOCAL_WEIGHTS_PATH)
    state_dict = torch.load(LOCAL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME, do_lower_case=True)

    logger.info("Model and tokenizer loaded successfully")


def predict_sentiment(texts: list) -> list:
    if not texts:
        return []

    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = torch.max(probs, dim=1).values.cpu().numpy()

    results = []
    for text, pred_idx, conf in zip(texts, preds, confidences):
        label = INDEX_TO_LABEL[int(pred_idx)]
        results.append({
            "text": text,
            "sentiment": LABEL_NAMES[label],
            "label": label,
            "confidence": round(float(conf), 4),
        })

    return results


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if not data or "text" not in data:
            return jsonify({"error": "Request body must include 'text' field"}), 400

        texts = data["text"]

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return jsonify({"error": "'text' must be a string or list of strings"}), 400

        return jsonify({"predictions": predict_sentiment(texts)})

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    try:
        data = request.get_json(force=True)

        if not data or "comments" not in data:
            return jsonify({"error": "Request body must include 'comments'"}), 400

        comments = data["comments"]

        if not isinstance(comments, list):
            return jsonify({"error": "'comments' must be a list"}), 400

        if len(comments) == 0:
            return jsonify([])

        texts = []
        for comment in comments:
            if not isinstance(comment, dict) or "text" not in comment:
                return jsonify({"error": "Each comment must contain 'text'"}), 400
            texts.append(comment["text"])

        logger.info("Received %d comments for prediction", len(texts))

        predictions = predict_sentiment(texts)

        results = []
        for comment, prediction in zip(comments, predictions):
            results.append({
                "comment": prediction["text"],
                "sentiment": prediction["label"],
                "sentiment_name": prediction["sentiment"],
                "confidence": prediction["confidence"],
                "timestamp": comment.get("timestamp"),
                "authorId": comment.get("authorId"),
            })

        logger.info("Successfully predicted %d comments", len(results))

        return jsonify(results)

    except Exception as e:
        logger.exception("Prediction with timestamps failed")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json(force=True)

        sentiment_counts = data.get("sentiment_counts", {})

        negative = int(sentiment_counts.get("-1", 0))
        neutral = int(sentiment_counts.get("0", 0))
        positive = int(sentiment_counts.get("1", 0))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [negative, neutral, positive],
            labels=["Negative", "Neutral", "Positive"],
            autopct="%1.1f%%",
        )
        ax.set_title("YouTube Comment Sentiment")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)

        return send_file(buffer, mimetype="image/png")

    except Exception as e:
        logger.exception("Chart generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json(force=True)

        comments = data.get("comments", [])
        text = " ".join(comments)

        if not text.strip():
            return jsonify({"error": "No comments provided"}), 400

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
        ).generate(text)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)

        return send_file(buffer, mimetype="image/png")

    except Exception as e:
        logger.exception("Word cloud generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json(force=True)

        sentiment_data = data.get("sentiment_data", [])

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        sentiments = [int(item["sentiment"]) for item in sentiment_data]
        x = list(range(1, len(sentiments) + 1))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, sentiments)
        ax.set_title("Sentiment Trend Over Time")
        ax.set_xlabel("Comment")
        ax.set_ylabel("Sentiment")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Negative", "Neutral", "Positive"])
        ax.grid(True)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)

        return send_file(buffer, mimetype="image/png")

    except Exception as e:
        logger.exception("Trend graph generation failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)