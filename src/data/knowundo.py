import torch
from torch.utils.data import Dataset

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index

import datasets as hf_datasets


class KnowUnDoDataset(Dataset):
    """Dataset handler for zjunlp/KnowUnDo.

    The HF dataset has a nested structure: 1 row with 'train' and 'val' columns,
    each containing a list of {text, labels} dicts. This handler flattens it into
    a standard QA dataset compatible with the framework's chat preprocessing.

    Args:
        hf_args: HuggingFace dataset loading args (path, name, split, etc.)
        template_args: Chat template config for tokenization.
        tokenizer: Tokenizer instance.
        question_key: Key for the question/prompt field.
        answer_key: Key for the answer/completion field.
        subset: Which nested subset to load ("train" or "val").
        additional_splits: List of additional HF splits to merge (e.g., ["retention"]
            to load both unlearn+retention for fine-tuning).
        max_length: Max sequence length.
        predict_with_generate: If True, input_ids only contains the prompt.
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="text",
        answer_key="labels",
        subset="train",
        additional_splits=None,
        max_length=512,
        predict_with_generate=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_key = question_key
        self.answer_key = answer_key
        self.template_args = template_args
        self.predict_with_generate = predict_with_generate

        raw_ds = load_hf_dataset(**hf_args)
        nested_list = raw_ds[0][subset]

        if additional_splits:
            base_hf_args = dict(hf_args)
            for extra_split in additional_splits:
                base_hf_args_copy = dict(base_hf_args)
                base_hf_args_copy["split"] = extra_split
                extra_ds = load_hf_dataset(**base_hf_args_copy)
                nested_list = nested_list + extra_ds[0][subset]

        self.data = hf_datasets.Dataset.from_list(nested_list)
        self.data = self.data.add_column("index", list(range(len(self.data))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]

        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            [question],
            [answer],
            self.max_length,
            self.predict_with_generate,
        )
        return {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
