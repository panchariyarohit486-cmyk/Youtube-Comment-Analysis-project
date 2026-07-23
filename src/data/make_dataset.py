import os
import logging
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger("make_dataset")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


DATA_URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
RAW_PATH = "data/raw"


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


def preprocess_Data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: drop nulls/duplicates, normalize the comment column."""
    try:
        df = df.copy()

        df.dropna(subset=["clean_comment"], inplace=True)
        df["clean_comment"] = df["clean_comment"].astype(str).str.strip()
        df = df[df["clean_comment"] != ""]

        df.drop_duplicates(inplace=True)

        logger.debug("Data preprocessing completed. Shape: %s", df.shape)
        return df
    except KeyError as e:
        logger.error("Missing expected column during preprocessing: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


def save_Data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str = RAW_PATH) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)

        logger.debug("Train/test data saved to %s", data_path)
    except Exception as e:
        logger.error("Failed to save train/test data: %s", e)
        raise


def main():
    try:
        # load the parameters from params.yaml
        params = load_params()

        test_size = params["make_dataset"]["test_size"]
        random_state = params["make_dataset"]["random_state"]

        df = pd.read_csv(DATA_URL)
        logger.debug("Raw data loaded from URL. Shape: %s", df.shape)

        # preprocess the data
        final_df = preprocess_Data(df)

        train_df, test_df = train_test_split(
            final_df, test_size=test_size, random_state=random_state
        )

        # save the preprocessed train/test split into data/raw (not data/processed)
        save_Data(train_df, test_df)

        logger.debug("make_dataset stage completed successfully")

    except Exception as e:
        logger.error("Failed to make the dataset: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()