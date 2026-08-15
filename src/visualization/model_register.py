import json
import logging
import mlflow
import dagshub
from mlflow.tracking import MlflowClient

logger = logging.getLogger("register_model")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

REPORTS_PATH = "reports"
REGISTERED_MODEL_NAME = "bert_yt_comment_sentiment"


def load_run_info(path: str = REPORTS_PATH) -> dict:
    try:
        with open(f"{path}/experiment_info.json", "r") as f:
            info = json.load(f)
        logger.debug("Loaded run info: %s", info)
        return info
    except FileNotFoundError as e:
        logger.error("experiment_info.json not found — run model_evaluation.py first: %s", e)
        raise


def register_model(model_name: str, model_uri: str, run_id: str):
    """Register a model version directly via MlflowClient, bypassing
    mlflow.register_model()'s MLflow-3.x 'logged model' detection logic —
    that logic fails against DagsHub's artifact listing format, even though
    the model artifact itself was logged successfully."""
    try:
        client = MlflowClient()

        try:
            client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            client.create_registered_model(model_name)
            logger.debug("Created new registered model '%s'", model_name)

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )
        logger.debug(
            "Registered model '%s' version %s from %s",
            model_name, model_version.version, model_uri,
        )
        return model_version
    except Exception as e:
        logger.error("Failed to register model: %s", e)
        raise


def transition_stage(model_name: str, version: str, stage: str = "Staging"):
    """Move the newly registered version into a lifecycle stage
    (None / Staging / Production / Archived)."""
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False,
        )
        logger.debug("Transitioned model '%s' v%s to stage '%s'", model_name, version, stage)
    except Exception as e:
        logger.error("Failed to transition model stage: %s", e)
        raise


def main():
    try:
        # connect to DagsHub-hosted MLflow via browser auth
        dagshub.init(
            repo_owner="panchariyarohit486",
            repo_name="youtube-sentiment-analysis",
            mlflow=True,
        )

        run_info = load_run_info()
        model_uri = run_info["model_uri"]
        run_id = run_info["run_id"]

        model_version = register_model(REGISTERED_MODEL_NAME, model_uri, run_id)
        transition_stage(REGISTERED_MODEL_NAME, model_version.version, stage="Staging")

        print(f"Registered '{REGISTERED_MODEL_NAME}' v{model_version.version} → Staging")
        logger.debug("register_model completed successfully")

    except Exception as e:
        logger.error("Failed to run model registration: %s", e)
        print(f"error : {e}")


if __name__ == "__main__":
    main()