import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import ClassLabel, Features, Image, Sequence, Value, load_dataset
from sklearn.model_selection import train_test_split


ALLOWED_EXTENSIONS = ("jpeg", "png", "jpg", "JPG", "JPEG", "PNG")


@dataclass
class ImageClassificationPreprocessor:
    """
    A class used to preprocess image data for classification tasks.

    Attributes
    ----------
    train_data : str
        Path to the training data directory.
    username : str
        Username for the Hugging Face Hub.
    project_name : str
        Name of the project.
    token : str
        Authentication token for the Hugging Face Hub.
    valid_data : Optional[str], optional
        Path to the validation data directory, by default None.
    test_size : Optional[float], optional
        Proportion of the dataset to include in the validation split, by default 0.2.
    seed : Optional[int], optional
        Random seed for reproducibility, by default 42.
    local : Optional[bool], optional
        Whether to save the dataset locally or push to the Hugging Face Hub, by default False.

    Methods
    -------
    __post_init__():
        Validates the structure and contents of the training and validation data directories.
    split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        Splits the dataframe into training and validation sets.
    prepare() -> str:
        Prepares the dataset for training and either saves it locally or pushes it to the Hugging Face Hub.
    """

    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # Check if train data path contains at least 2 folders
        subfolders = [f.path for f in os.scandir(self.train_data) if f.is_dir()]
        # list subfolders
        if len(subfolders) < 2:
            raise ValueError(f"{self.train_data} should contain at least 2 subfolders.")

        # Check if each subfolder contains at least 5 image files in jpeg, png or jpg format only
        for subfolder in subfolders:
            image_files = [f for f in os.listdir(subfolder) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(image_files) < 5:
                raise ValueError(f"{subfolder} should contain at least 5 jpeg, png or jpg files.")
            # Check if there are no other files except image files in the subfolder
            if len(image_files) != len(os.listdir(subfolder)):
                raise ValueError(f"{subfolder} should not contain any other files except image files.")

            # Check if there are no subfolders inside subfolders
            subfolders_in_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir()]
            if len(subfolders_in_subfolder) > 0:
                raise ValueError(f"{subfolder} should not contain any subfolders.")

        if self.valid_data:
            # Check if valid data path exists
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # Check if valid data path contains at least 2 folders
            subfolders = [f.path for f in os.scandir(self.valid_data) if f.is_dir()]

            # make sure that the subfolders in train and valid data are the same
            train_subfolders = set(os.path.basename(f.path) for f in os.scandir(self.train_data) if f.is_dir())
            valid_subfolders = set(os.path.basename(f.path) for f in os.scandir(self.valid_data) if f.is_dir())
            if train_subfolders != valid_subfolders:
                raise ValueError(f"{self.valid_data} should have the same subfolders as {self.train_data}.")

            if len(subfolders) < 2:
                raise ValueError(f"{self.valid_data} should contain at least 2 subfolders.")

            # Check if each subfolder contains at least 5 image files in jpeg, png or jpg format only
            for subfolder in subfolders:
                image_files = [f for f in os.listdir(subfolder) if f.endswith(ALLOWED_EXTENSIONS)]
                if len(image_files) < 5:
                    raise ValueError(f"{subfolder} should contain at least 5 jpeg, png or jpg files.")

                # Check if there are no other files except image files in the subfolder
                if len(image_files) != len(os.listdir(subfolder)):
                    raise ValueError(f"{subfolder} should not contain any other files except image files.")

                # Check if there are no subfolders inside subfolders
                subfolders_in_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir()]
                if len(subfolders_in_subfolder) > 0:
                    raise ValueError(f"{subfolder} should not contain any subfolders.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df["subfolder"],
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns({"image": "autotrain_image", "label": "autotrain_label"})
            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        else:
            subfolders = [f.path for f in os.scandir(self.train_data) if f.is_dir()]

            image_filenames = []
            subfolder_names = []

            for subfolder in subfolders:
                for filename in os.listdir(subfolder):
                    if filename.endswith(("jpeg", "png", "jpg")):
                        image_filenames.append(filename)
                        subfolder_names.append(os.path.basename(subfolder))

            df = pd.DataFrame({"image_filename": image_filenames, "subfolder": subfolder_names})
            train_df, valid_df = self.split(df)

            for row in train_df.itertuples():
                os.makedirs(os.path.join(data_dir, "train", row.subfolder), exist_ok=True)
                shutil.copy(
                    os.path.join(self.train_data, row.subfolder, row.image_filename),
                    os.path.join(data_dir, "train", row.subfolder, row.image_filename),
                )

            for row in valid_df.itertuples():
                os.makedirs(os.path.join(data_dir, "validation", row.subfolder), exist_ok=True)
                shutil.copy(
                    os.path.join(self.train_data, row.subfolder, row.image_filename),
                    os.path.join(data_dir, "validation", row.subfolder, row.image_filename),
                )

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns({"image": "autotrain_image", "label": "autotrain_label"})
            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


@dataclass
class ObjectDetectionPreprocessor:
    """
    A class to preprocess data for object detection tasks.

    Attributes:
    -----------
    train_data : str
        Path to the training data directory.
    username : str
        Username for the Hugging Face Hub.
    project_name : str
        Name of the project.
    token : str
        Authentication token for the Hugging Face Hub.
    valid_data : Optional[str], default=None
        Path to the validation data directory.
    test_size : Optional[float], default=0.2
        Proportion of the dataset to include in the validation split.
    seed : Optional[int], default=42
        Random seed for reproducibility.
    local : Optional[bool], default=False
        Whether to save the dataset locally or push to the Hugging Face Hub.

    Methods:
    --------
    _process_metadata(data_path):
        Processes the metadata.jsonl file and extracts required columns and categories.
    __post_init__():
        Validates the existence and content of the training and validation data directories.
    split(df):
        Splits the dataframe into training and validation sets.
    prepare():
        Prepares the dataset for training by processing metadata, splitting data, and saving or pushing the dataset.
    """

    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    @staticmethod
    def _process_metadata(data_path):
        metadata = pd.read_json(os.path.join(data_path, "metadata.jsonl"), lines=True)
        # make sure that the metadata.jsonl file contains the required columns: file_name, objects
        if "file_name" not in metadata.columns or "objects" not in metadata.columns:
            raise ValueError(f"{data_path}/metadata.jsonl should contain 'file_name' and 'objects' columns.")

        # keeo only file_name and objects columns
        metadata = metadata[["file_name", "objects"]]
        # inside metadata objects column, values should be bbox, area and category
        # if area does not exist, it should be created by multiplying bbox width and height
        categories = []
        for _, row in metadata.iterrows():
            obj = row["objects"]
            if "bbox" not in obj or "category" not in obj:
                raise ValueError(f"{data_path}/metadata.jsonl should contain 'bbox' and 'category' keys in 'objects'.")
            # keep only bbox, area and category keys
            obj = {k: obj[k] for k in ["bbox", "category"]}
            categories.extend(obj["category"])

        categories = set(categories)

        return metadata, categories

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # check if self.train_data contains at least 5 image files in jpeg, png or jpg format only
        train_image_files = [f for f in os.listdir(self.train_data) if f.endswith(ALLOWED_EXTENSIONS)]
        if len(train_image_files) < 5:
            raise ValueError(f"{self.train_data} should contain at least 5 jpeg, png or jpg files.")

        # check if self.train_data contains a metadata.jsonl file
        if "metadata.jsonl" not in os.listdir(self.train_data):
            raise ValueError(f"{self.train_data} should contain a metadata.jsonl file.")

        # Check if valid data path exists
        if self.valid_data:
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # check if self.valid_data contains at least 5 image files in jpeg, png or jpg format only
            valid_image_files = [f for f in os.listdir(self.valid_data) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(valid_image_files) < 5:
                raise ValueError(f"{self.valid_data} should contain at least 5 jpeg, png or jpg files.")

            # check if self.valid_data contains a metadata.jsonl file
            if "metadata.jsonl" not in os.listdir(self.valid_data):
                raise ValueError(f"{self.valid_data} should contain a metadata.jsonl file.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            train_metadata, train_categories = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata, valid_categories = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            all_categories = train_categories.union(valid_categories)

            features = Features(
                {
                    "image": Image(),
                    "objects": Sequence(
                        {
                            "bbox": Sequence(Value("float32"), length=4),
                            "category": ClassLabel(names=list(all_categories)),
                        }
                    ),
                }
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir, features=features)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "objects": "autotrain_objects",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )
        else:
            metadata = pd.read_json(os.path.join(self.train_data, "metadata.jsonl"), lines=True)
            train_df, valid_df = self.split(metadata)

            # create train and validation folders
            os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)

            # move images to train and validation folders
            for row in train_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "train", row[1]["file_name"]),
                )

            for row in valid_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "validation", row[1]["file_name"]),
                )

            # save metadata.jsonl file to train and validation folders
            train_df.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_df.to_json(os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True)

            train_metadata, train_categories = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata, valid_categories = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            all_categories = train_categories.union(valid_categories)

            features = Features(
                {
                    "image": Image(),
                    "objects": Sequence(
                        {
                            "bbox": Sequence(Value("float32"), length=4),
                            "category": ClassLabel(names=list(all_categories)),
                        }
                    ),
                }
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir, features=features)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "objects": "autotrain_objects",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"


@dataclass
class ImageRegressionPreprocessor:
    train_data: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[str] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
    local: Optional[bool] = False

    @staticmethod
    def _process_metadata(data_path):
        metadata = pd.read_json(os.path.join(data_path, "metadata.jsonl"), lines=True)
        # make sure that the metadata.jsonl file contains the required columns: file_name, target
        if "file_name" not in metadata.columns or "target" not in metadata.columns:
            raise ValueError(f"{data_path}/metadata.jsonl should contain 'file_name' and 'target' columns.")

        # keep only file_name and target columns
        metadata = metadata[["file_name", "target"]]
        return metadata

    def __post_init__(self):
        # Check if train data path exists
        if not os.path.exists(self.train_data):
            raise ValueError(f"{self.train_data} does not exist.")

        # check if self.train_data contains at least 5 image files in jpeg, png or jpg format only
        train_image_files = [f for f in os.listdir(self.train_data) if f.endswith(ALLOWED_EXTENSIONS)]
        if len(train_image_files) < 5:
            raise ValueError(f"{self.train_data} should contain at least 5 jpeg, png or jpg files.")

        # check if self.train_data contains a metadata.jsonl file
        if "metadata.jsonl" not in os.listdir(self.train_data):
            raise ValueError(f"{self.train_data} should contain a metadata.jsonl file.")

        # Check if valid data path exists
        if self.valid_data:
            if not os.path.exists(self.valid_data):
                raise ValueError(f"{self.valid_data} does not exist.")

            # check if self.valid_data contains at least 5 image files in jpeg, png or jpg format only
            valid_image_files = [f for f in os.listdir(self.valid_data) if f.endswith(ALLOWED_EXTENSIONS)]
            if len(valid_image_files) < 5:
                raise ValueError(f"{self.valid_data} should contain at least 5 jpeg, png or jpg files.")

            # check if self.valid_data contains a metadata.jsonl file
            if "metadata.jsonl" not in os.listdir(self.valid_data):
                raise ValueError(f"{self.valid_data} should contain a metadata.jsonl file.")

    def split(self, df):
        train_df, valid_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        return train_df, valid_df

    def prepare(self):
        random_uuid = uuid.uuid4()
        cache_dir = os.environ.get("HF_HOME")
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        data_dir = os.path.join(cache_dir, "autotrain", str(random_uuid))

        if self.valid_data:
            shutil.copytree(self.train_data, os.path.join(data_dir, "train"))
            shutil.copytree(self.valid_data, os.path.join(data_dir, "validation"))

            train_metadata = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "target": "autotrain_label",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )
        else:
            metadata = pd.read_json(os.path.join(self.train_data, "metadata.jsonl"), lines=True)
            train_df, valid_df = self.split(metadata)

            # create train and validation folders
            os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)

            # move images to train and validation folders
            for row in train_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "train", row[1]["file_name"]),
                )

            for row in valid_df.iterrows():
                shutil.copy(
                    os.path.join(self.train_data, row[1]["file_name"]),
                    os.path.join(data_dir, "validation", row[1]["file_name"]),
                )

            # save metadata.jsonl file to train and validation folders
            train_df.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_df.to_json(os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True)

            train_metadata = self._process_metadata(os.path.join(data_dir, "train"))
            valid_metadata = self._process_metadata(os.path.join(data_dir, "validation"))

            train_metadata.to_json(os.path.join(data_dir, "train", "metadata.jsonl"), orient="records", lines=True)
            valid_metadata.to_json(
                os.path.join(data_dir, "validation", "metadata.jsonl"), orient="records", lines=True
            )

            dataset = load_dataset("imagefolder", data_dir=data_dir)
            dataset = dataset.rename_columns(
                {
                    "image": "autotrain_image",
                    "target": "autotrain_label",
                }
            )

            if self.local:
                dataset.save_to_disk(f"{self.project_name}/autotrain-data")
            else:
                dataset.push_to_hub(
                    f"{self.username}/autotrain-data-{self.project_name}",
                    private=True,
                    token=self.token,
                )

        if self.local:
            return f"{self.project_name}/autotrain-data"
        return f"{self.username}/autotrain-data-{self.project_name}"
