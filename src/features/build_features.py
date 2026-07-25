# apply bow vectorization , imbalance techniques(random oversampling ) 
# and max_feature , n_gram  and save the processed data in train,test split


import os
import logging
import yaml
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler

# ------------------------- logger setup -------------------------
logger = logging.getLogger("build_features")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
# ------------------------------------------------------------------

PROCESSED_PATH = "data/processed"
INTERIM_PATH = "data/interim"
VECTORIZER_PATH = "model"


def load_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("params.yaml not found at %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Failed to parse params.yaml: %s", e)
        raise


def load_data(data_path: str = PROCESSED_PATH):
    try:
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
        logger.debug("Train/test data loaded. Shapes: %s, %s", train_df.shape, test_df.shape)
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error("Train/test data not found in %s: %s", data_path, e)
        raise


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int,
    ngram_range: tuple,
    random_state: int = 42,
):
    try:
        X_train_text = train_df["clean_comment"]
        y_train = train_df["category"]
        X_test_text = test_df["clean_comment"]
        y_test = test_df["category"]

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        logger.debug(
            "Vectorization completed. X_train shape: %s, X_test shape: %s",
            X_train.shape, X_test.shape,
        )

        sampler = RandomOverSampler(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)

        logger.debug("RandomOverSampler completed. X_train shape after resample: %s", X_train.shape)

        return X_train, X_test, y_train, y_test, vectorizer

    except KeyError as e:
        logger.error("Missing expected column while building features: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while building features: %s", e)
        raise


def save_features(X_train, X_test, y_train, y_test, out_path: str = INTERIM_PATH) -> None:
    try:
        import scipy.sparse as sp
        import numpy as np

        os.makedirs(out_path, exist_ok=True)

        sp.save_npz(os.path.join(out_path, "X_train.npz"), X_train)
        sp.save_npz(os.path.join(out_path, "X_test.npz"), X_test)
        np.save(os.path.join(out_path, "y_train.npy"), y_train)
        np.save(os.path.join(out_path, "y_test.npy"), y_test)

        logger.debug("Features saved to %s", out_path)
    except Exception as e:
        logger.error("Failed to save features: %s", e)
        raise


def save_vectorizer(vectorizer: CountVectorizer, out_path: str = VECTORIZER_PATH) -> None:
    try:
        import pickle

        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, "vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

        logger.debug("Vectorizer saved to %s", out_path)
    except Exception as e:
        logger.error("Failed to save vectorizer: %s", e)
        raise


def main():
    try:
        params = load_params()

        max_features = params["build_features"]["max_features"]
        ngram_range = tuple(params["build_features"]["ngram_range"])
        random_state = params["build_features"]["random_state"]

        train_df, test_df = load_data()

        X_train, X_test, y_train, y_test, vectorizer = build_features(
            train_df, test_df, max_features, ngram_range, random_state
        )

        save_features(X_train, X_test, y_train, y_test)
        save_vectorizer(vectorizer)

        logger.debug("build_features stage completed successfully")

    except Exception as e:
        logger.error("Failed to build features: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()