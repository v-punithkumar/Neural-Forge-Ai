import os
from dataclasses import dataclass
from typing import Optional
import requests
from neural_forge_ai import logger

NEURAL_FORGE_AI_API = os.environ.get("NEURAL_FORGE_AI_API", "https://neural-forge-ai.hf.space/")

BACKENDS = {
    "spaces-a10g-large": "a10g-large",
    "spaces-a10g-small": "a10g-small",
    "spaces-a100-large": "a100-large",
    "spaces-t4-medium": "t4-medium",
    "spaces-t4-small": "t4-small",
    "spaces-cpu-upgrade": "cpu-upgrade",
    "spaces-cpu-basic": "cpu-basic",
    "spaces-l4x1": "l4x1",
    "spaces-l4x4": "l4x4",
    "spaces-l40sx1": "l40sx1",
    "spaces-l40sx4": "l40sx4",
    "spaces-l40sx8": "l40sx8",
    "spaces-a10g-largex2": "a10g-largex2",
    "spaces-a10g-largex4": "a10g-largex4",
}

PARAMS = {
    "llm": {
        "target_modules": "all-linear",
        "log": "tensorboard",
        "mixed_precision": "fp16",
        "quantization": "int4",
        "peft": True,
        "block_size": 1024,
        "epochs": 3,
        "padding": "right",
        "chat_template": "none",
        "max_completion_length": 128,
        "distributed_backend": "ddp",
        "scheduler": "linear",
        "merge_adapter": True,
    },
    "text-classification": {"mixed_precision": "fp16", "log": "tensorboard"},
    "st": {"mixed_precision": "fp16", "log": "tensorboard"},
    "image-classification": {"mixed_precision": "fp16", "log": "tensorboard"},
    "image-object-detection": {"mixed_precision": "fp16", "log": "tensorboard"},
    "seq2seq": {"mixed_precision": "fp16", "target_modules": "all-linear", "log": "tensorboard"},
    "tabular": {
        "categorical_imputer": "most_frequent",
        "numerical_imputer": "median",
        "numeric_scaler": "robust",
    },
    "token-classification": {"mixed_precision": "fp16", "log": "tensorboard"},
    "text-regression": {"mixed_precision": "fp16", "log": "tensorboard"},
    "image-regression": {"mixed_precision": "fp16", "log": "tensorboard"},
    "vlm": {
        "mixed_precision": "fp16",
        "target_modules": "all-linear",
        "log": "tensorboard",
        "quantization": "int4",
        "peft": True,
        "epochs": 3,
    },
    "extractive-qa": {
        "mixed_precision": "fp16",
        "log": "tensorboard",
        "max_seq_length": 512,
        "max_doc_stride": 128,
    },
}

DEFAULT_COLUMN_MAPPING = {
    "llm:sft": {"text_column": "text"},
    "llm:generic": {"text_column": "text"},
    "llm:default": {"text_column": "text"},
    "llm:dpo": {"prompt_column": "prompt", "text_column": "chosen", "rejected_text_column": "rejected"},
    "llm:orpo": {"prompt_column": "prompt", "text_column": "chosen", "rejected_text_column": "rejected"},
    "llm:reward": {"text_column": "chosen", "rejected_text_column": "rejected"},
    "vlm:captioning": {"image_column": "image", "text_column": "caption"},
    "vlm:vqa": {"image_column": "image", "prompt_text_column": "question", "text_column": "answer"},
    "text-classification": {"text_column": "text", "target_column": "target"},
    "text-regression": {"text_column": "text", "target_column": "target"},
    "image-classification": {"image_column": "image", "target_column": "label"},
    "image-regression": {"image_column": "image", "target_column": "target"},
    "image-object-detection": {"image_column": "image", "objects_column": "objects"},
    "extractive-qa": {"text_column": "context", "question_column": "question", "answer_column": "answers"},
}

VALID_TASKS = list(DEFAULT_COLUMN_MAPPING.keys())

@dataclass
class Client:
    """
    A client to interact with the Neural-Forge-Ai API.
    Attributes:
        host (Optional[str]): The host URL for the Neural-Forge-Ai API.
        token (Optional[str]): The authentication token for the API.
        username (Optional[str]): The username for the API.
    """

    host: Optional[str] = None
    token: Optional[str] = None
    username: Optional[str] = None

    def __post_init__(self):
        if self.host is None:
            self.host = NEURAL_FORGE_AI_API

        if self.token is None:
            self.token = os.environ.get("HF_TOKEN")

        if self.username is None:
            self.username = os.environ.get("HF_USERNAME")

        if self.token is None or self.username is None:
            raise ValueError("Please provide a valid username and token")

        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_project(self, project_name: str, task: str, base_model: str, backend: str, dataset: str, train_split: str):
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid task. Valid tasks are: {VALID_TASKS}")

        if backend not in BACKENDS:
            raise ValueError(f"Invalid backend. Valid backends are: {list(BACKENDS.keys())}")

        url = f"{self.host}/api/create_project"
        data = {
            "project_name": project_name,
            "task": task,
            "base_model": base_model,
            "hardware": backend,
            "params": PARAMS.get(task, {}),
            "username": self.username,
            "column_mapping": DEFAULT_COLUMN_MAPPING.get(task, {}),
            "hub_dataset": dataset,
            "train_split": train_split,
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()
