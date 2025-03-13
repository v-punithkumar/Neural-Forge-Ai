import os

import numpy as np
import requests
from sklearn import metrics


BINARY_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1",
    "eval_auc",
    "eval_precision",
    "eval_recall",
)

MULTI_CLASS_CLASSIFICATION_EVAL_METRICS = (
    "eval_loss",
    "eval_accuracy",
    "eval_f1_macro",
    "eval_f1_micro",
    "eval_f1_weighted",
    "eval_precision_macro",
    "eval_precision_micro",
    "eval_precision_weighted",
    "eval_recall_macro",
    "eval_recall_micro",
    "eval_recall_weighted",
)

MODEL_CARD = """
---
library_name: transformers
tags:
- autotrain
- text-classification{base_model}
widget:
- text: "I love AutoTrain"{dataset_tag}
---

# Model Trained Using AutoTrain

- Problem type: Text Classification

## Validation Metrics
{validation_metrics}
"""


def _binary_classification_metrics(pred):
    """
    Calculate various binary classification metrics.

    Args:
        pred (tuple): A tuple containing raw predictions and true labels.
                      - raw_predictions (numpy.ndarray): The raw prediction scores from the model.
                      - labels (numpy.ndarray): The true labels.

    Returns:
        dict: A dictionary containing the following metrics:
              - "f1" (float): The F1 score.
              - "precision" (float): The precision score.
              - "recall" (float): The recall score.
              - "auc" (float): The Area Under the ROC Curve (AUC) score.
              - "accuracy" (float): The accuracy score.
    """
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    result = {
        "f1": metrics.f1_score(labels, predictions),
        "precision": metrics.precision_score(labels, predictions),
        "recall": metrics.recall_score(labels, predictions),
        "auc": metrics.roc_auc_score(labels, raw_predictions[:, 1]),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return result


def _multi_class_classification_metrics(pred):
    """
    Compute various classification metrics for multi-class classification.

    Args:
        pred (tuple): A tuple containing raw predictions and true labels.
                      - raw_predictions (numpy.ndarray): The raw prediction scores for each class.
                      - labels (numpy.ndarray): The true labels.

    Returns:
        dict: A dictionary containing the following metrics:
              - "f1_macro": F1 score with macro averaging.
              - "f1_micro": F1 score with micro averaging.
              - "f1_weighted": F1 score with weighted averaging.
              - "precision_macro": Precision score with macro averaging.
              - "precision_micro": Precision score with micro averaging.
              - "precision_weighted": Precision score with weighted averaging.
              - "recall_macro": Recall score with macro averaging.
              - "recall_micro": Recall score with micro averaging.
              - "recall_weighted": Recall score with weighted averaging.
              - "accuracy": Accuracy score.
    """
    raw_predictions, labels = pred
    predictions = np.argmax(raw_predictions, axis=1)
    results = {
        "f1_macro": metrics.f1_score(labels, predictions, average="macro"),
        "f1_micro": metrics.f1_score(labels, predictions, average="micro"),
        "f1_weighted": metrics.f1_score(labels, predictions, average="weighted"),
        "precision_macro": metrics.precision_score(labels, predictions, average="macro"),
        "precision_micro": metrics.precision_score(labels, predictions, average="micro"),
        "precision_weighted": metrics.precision_score(labels, predictions, average="weighted"),
        "recall_macro": metrics.recall_score(labels, predictions, average="macro"),
        "recall_micro": metrics.recall_score(labels, predictions, average="micro"),
        "recall_weighted": metrics.recall_score(labels, predictions, average="weighted"),
        "accuracy": metrics.accuracy_score(labels, predictions),
    }
    return results


def create_model_card(config, trainer, num_classes):
    """
    Generates a model card for a text classification model.

    Args:
        config (object): Configuration object containing various settings and paths.
        trainer (object): Trainer object used for evaluating the model.
        num_classes (int): Number of classes in the classification task.

    Returns:
        str: A formatted string representing the model card.
    """
    if config.valid_split is not None:
        eval_scores = trainer.evaluate()
        valid_metrics = (
            BINARY_CLASSIFICATION_EVAL_METRICS if num_classes == 2 else MULTI_CLASS_CLASSIFICATION_EVAL_METRICS
        )
        eval_scores = [f"{k[len('eval_'):]}: {v}" for k, v in eval_scores.items() if k in valid_metrics]
        eval_scores = "\n\n".join(eval_scores)

    else:
        eval_scores = "No validation metrics available"

    if config.data_path == f"{config.project_name}/autotrain-data" or os.path.isdir(config.data_path):
        dataset_tag = ""
    else:
        dataset_tag = f"\ndatasets:\n- {config.data_path}"

    if os.path.isdir(config.model):
        base_model = ""
    else:
        base_model = f"\nbase_model: {config.model}"

    model_card = MODEL_CARD.format(
        dataset_tag=dataset_tag,
        validation_metrics=eval_scores,
        base_model=base_model,
    )
    return model_card


def pause_endpoint(params):
    """
    Pauses a Hugging Face endpoint using the provided parameters.

    This function constructs an API URL using the endpoint ID from the environment
    variables, and sends a POST request to pause the specified endpoint.

    Args:
        params (object): An object containing the following attribute:
            - token (str): The authorization token required to authenticate the API request.

    Returns:
        dict: The JSON response from the API call.
    """
    endpoint_id = os.environ["ENDPOINT_ID"]
    username = endpoint_id.split("/")[0]
    project_name = endpoint_id.split("/")[1]
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{username}/{project_name}/pause"
    headers = {"Authorization": f"Bearer {params.token}"}
    r = requests.post(api_url, headers=headers)
    return r.json()
