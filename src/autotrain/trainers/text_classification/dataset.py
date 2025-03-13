import torch


class TextClassificationDataset:
    """
    A dataset class for text classification tasks.

    Args:
        data (list): The dataset containing text and target columns.
        tokenizer (PreTrainedTokenizer): The tokenizer to preprocess the text data.
        config (object): Configuration object containing dataset parameters.

    Attributes:
        data (list): The dataset containing text and target columns.
        tokenizer (PreTrainedTokenizer): The tokenizer to preprocess the text data.
        config (object): Configuration object containing dataset parameters.
        text_column (str): The name of the column containing text data.
        target_column (str): The name of the column containing target labels.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Returns a dictionary containing tokenized input ids, attention mask, token type ids (if available), and target labels for the given item index.
    """

    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.text_column = self.config.text_column
        self.target_column = self.config.target_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item][self.text_column])
        target = self.data[item][self.target_column]
        target = int(target)
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            token_type_ids = None

        if token_type_ids is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "labels": torch.tensor(target, dtype=torch.long),
            }
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.long),
        }
