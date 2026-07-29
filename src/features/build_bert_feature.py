import os
import logging
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

logger = logging.getLogger("build_bert_features")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

RAW_PATH = "data/raw"
OUT_PATH = "data/interim/bert"

LABEL_DICT = {-1: 0, 0: 1, 1: 2}


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str = RAW_PATH):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    logger.debug("Loaded raw train/test: %s, %s", train_df.shape, test_df.shape)
    return train_df, test_df


def encode(df: pd.DataFrame, tokenizer: BertTokenizer, max_length: int):
    texts = df["clean_comment"].astype(str).tolist()
    labels = df["category"].map(LABEL_DICT).values

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="np",
    )
    return encoded["input_ids"], encoded["attention_mask"], np.array(labels)


def save_arrays(input_ids, attention_mask, labels, out_path: str, split: str):
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, f"{split}_input_ids.npy"), input_ids)
    np.save(os.path.join(out_path, f"{split}_attention_mask.npy"), attention_mask)
    np.save(os.path.join(out_path, f"{split}_labels.npy"), labels)
    logger.debug("Saved %s arrays to %s", split, out_path)


def main():
    try:
        params = load_params()
        model_name = params["build_bert_features"]["model_name"]
        max_length = params["build_bert_features"]["max_length"]

        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

        train_df, test_df = load_data()

        ids_train, mask_train, y_train = encode(train_df, tokenizer, max_length)
        ids_val, mask_val, y_val = encode(test_df, tokenizer, max_length)

        save_arrays(ids_train, mask_train, y_train, OUT_PATH, "train")
        save_arrays(ids_val, mask_val, y_val, OUT_PATH, "val")

        logger.debug("build_bert_features completed successfully")
    except Exception as e:
        logger.error("Failed to build BERT features: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()