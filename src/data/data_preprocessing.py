import os
import re
import logging
import yaml
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


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


def load_data(data_path: str = RAW_PATH):
    try:
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
        logger.debug("Train/test data loaded. Shapes: %s, %s", train_df.shape, test_df.shape)
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error("Train/test data not found in %s: %s", data_path, e)
        raise


def _ensure_nltk_resources() -> None:
    for resource in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_text(
    df: pd.DataFrame,
    remove_stopwords: bool,
    keep_negations: list,
    lemmatize: bool,
) -> pd.DataFrame:
    """Apply the notebook's text-cleaning pipeline to the 'clean_comment' column."""
    try:
        df = df.copy()

        # lowercase
        df["clean_comment"] = df["clean_comment"].str.lower()

        # strip leading/trailing whitespace
        df["clean_comment"] = df["clean_comment"].str.strip()

        # remove URLs
        df["clean_comment"] = df["clean_comment"].str.replace(URL_PATTERN, " ", regex=True)

        # remove newline characters
        df["clean_comment"] = df["clean_comment"].str.replace("\n", " ", regex=True)

        # keep only standard English letters, digits, whitespace, and basic punctuation
        df["clean_comment"] = df["clean_comment"].apply(
            lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(x))
        )

        if remove_stopwords:
            stop_words = set(stopwords.words("english")) - set(keep_negations)
            df["clean_comment"] = df["clean_comment"].apply(
                lambda x: " ".join([w for w in x.split() if w.lower() not in stop_words])
            )

        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            df["clean_comment"] = df["clean_comment"].apply(
                lambda x: " ".join([lemmatizer.lemmatize(w) for w in x.split()])
            )

        # drop rows that became empty after cleaning
        df["clean_comment"] = df["clean_comment"].str.strip()
        df = df[df["clean_comment"] != ""]

        logger.debug("Text cleaning completed. Shape: %s", df.shape)
        return df
    except KeyError as e:
        logger.error("Missing expected column during text cleaning: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during text cleaning: %s", e)
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str = PROCESSED_PATH) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)

        logger.debug("Processed train/test data saved to %s", data_path)
    except Exception as e:
        logger.error("Failed to save processed data: %s", e)
        raise


def main():
    try:
        params = load_params()

        remove_stopwords = params["data_preprocessing"]["remove_stopwords"]
        keep_negations = params["data_preprocessing"]["keep_negations"]
        lemmatize = params["data_preprocessing"]["lemmatize"]

        _ensure_nltk_resources()

        train_df, test_df = load_data()

        train_df = clean_text(train_df, remove_stopwords, keep_negations, lemmatize)
        test_df = clean_text(test_df, remove_stopwords, keep_negations, lemmatize)

        save_data(train_df, test_df)

        logger.debug("data_preprocessing stage completed successfully")

    except Exception as e:
        logger.error("Failed to preprocess the data: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()