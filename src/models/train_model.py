import os
import json
import pickle
import logging
import yaml
import numpy as np
import scipy.sparse as sp

from sklearnex import patch_sklearn
patch_sklearn()  # must run BEFORE importing SVC — patches sklearn to use Intel oneDAL (multi-core)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ------------------------- logger setup -------------------------
logger = logging.getLogger("train_model")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
# ------------------------------------------------------------------

INTERIM_PATH = "data/interim"
MODEL_PATH = "model"
REPORTS_PATH = "reports"


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


def load_features(data_path: str = INTERIM_PATH):
    try:
        X_train = sp.load_npz(os.path.join(data_path, "X_train.npz"))
        X_test = sp.load_npz(os.path.join(data_path, "X_test.npz"))
        y_train = np.load(os.path.join(data_path, "y_train.npy"), allow_pickle=True)
        y_test = np.load(os.path.join(data_path, "y_test.npy"), allow_pickle=True)

        logger.debug(
            "Features loaded. X_train: %s, X_test: %s, y_train: %s, y_test: %s",
            X_train.shape, X_test.shape, y_train.shape, y_test.shape,
        )
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logger.error("Feature files not found in %s: %s", data_path, e)
        raise


def build_svc(train_params: dict) -> SVC:
    """Build an SVC from the flat train_model params block, resolving gamma_type
    the same way _best_params_to_svc_kwargs did in the tuning notebook."""
    try:
        kernel = train_params["kernel"]

        kwargs = {
            "C": train_params["C"],
            "kernel": kernel,
            "class_weight": train_params.get("class_weight"),
            "random_state": train_params.get("random_state", 42),
        }

        if kernel in ["rbf", "poly", "sigmoid"]:
            gamma_type = train_params.get("gamma_type", "scale")
            kwargs["gamma"] = train_params["gamma"] if gamma_type == "value" else gamma_type

        if kernel == "poly":
            kwargs["degree"] = train_params["degree"]

        if kernel in ["poly", "sigmoid"]:
            kwargs["coef0"] = train_params["coef0"]

        logger.debug("Building SVC with params: %s", kwargs)
        return SVC(**kwargs)
    except KeyError as e:
        logger.error("Missing expected key in train_model params: %s", e)
        raise


def train_model(model: SVC, X_train, y_train) -> SVC:
    try:
        model.fit(X_train, y_train)
        logger.debug("Model training completed")
        return model
    except Exception as e:
        logger.error("Failed to train model: %s", e)
        raise


def evaluate_model(model: SVC, X_test, y_test) -> dict:
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.debug("Evaluation completed. Accuracy: %.4f", accuracy)
        return {"accuracy": accuracy, "classification_report": report}
    except Exception as e:
        logger.error("Failed to evaluate model: %s", e)
        raise


def save_model(model: SVC, out_path: str = MODEL_PATH) -> None:
    try:
        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", out_path)
    except Exception as e:
        logger.error("Failed to save model: %s", e)
        raise


def save_metrics(metrics: dict, out_path: str = REPORTS_PATH) -> None:
    try:
        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved to %s", out_path)
    except Exception as e:
        logger.error("Failed to save metrics: %s", e)
        raise


def main():
    try:
        params = load_params()
        train_params = params["train_model"]

        X_train, X_test, y_train, y_test = load_features()

        model = build_svc(train_params)
        model = train_model(model, X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        print(f"Test accuracy: {metrics['accuracy']:.4f}")

        save_model(model)
        save_metrics(metrics)

        logger.debug("train_model stage completed successfully")

    except Exception as e:
        logger.error("Failed to train the model: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()