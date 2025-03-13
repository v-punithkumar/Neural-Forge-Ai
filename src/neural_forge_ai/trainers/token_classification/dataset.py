class TokenClassificationDataset:
    """
    A dataset class for token classification tasks.

    Args:
        data (Dataset): The dataset containing the text and tags.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenizing the text.
        config (Config): Configuration object containing necessary parameters.

    Attributes:
        data (Dataset): The dataset containing the text and tags.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenizing the text.
        config (Config): Configuration object containing necessary parameters.

    Methods:
        __len__():
            Returns the number of samples in the dataset.

        __getitem__(item):
            Retrieves a tokenized sample and its corresponding labels.

            Args:
                item (int): The index of the sample to retrieve.

            Returns:
                dict: A dictionary containing tokenized text and corresponding labels.
    """

    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item][self.config.tokens_column]
        tags = self.data[item][self.config.tags_column]

        label_list = self.data.features[self.config.tags_column].feature.names
        label_to_id = {i: i for i in range(len(label_list))}

        tokenized_text = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        word_ids = tokenized_text.word_ids(batch_index=0)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[tags[word_idx]])
            else:
                label_ids.append(label_to_id[tags[word_idx]])
            previous_word_idx = word_idx

        tokenized_text["labels"] = label_ids
        return tokenized_text
