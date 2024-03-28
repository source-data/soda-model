# from typing import List
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from datasets import load_dataset
import numpy as np
from collections import Counter

import argparse

def _shift_label(label):
    # If the label is B-XXX we change it to I-XX
    if label % 2 == 1:
        label += 1
    return label

def _align_labels_with_tokens(labels, word_ids, do_shift=True):
    """
    Expands the NER tags once the sub-word tokenization is added.
    Arguments
    ---------
    labels list[int]:
    word_ids list[int]
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            # As far as word_id matches the index of the current word
            # We append the same label
            new_labels.append(labels[word_id])
        else:
            if do_shift:
                new_labels.append(_shift_label(labels[word_id]))
            else:
                new_labels.append(labels[word_id])

    return new_labels

def _tokenize_and_align_labels(self, examples) -> DatasetDict:
    """
    Tokenizes data split into words into sub-token tokenization parts.
    Args:
        examples: batch of data from a `datasets.DatasetDict`

    Returns:
        `datasets.DatasetDict` with entries tokenized to the `AutoTokenizer`
    """
    tokenized_inputs = self.tokenizer(
        examples['words'],
        truncation=self.truncation,
        padding=self.padding,
        is_split_into_words=True,
        max_length=self.max_length,
        # return_tensors="pt",
        )
    all_labels = examples['labels']
    all_flags = examples['only_in_test']

    new_labels = []
    tag_mask = []
    new_only_in_test_flag = []

    for i, (labels, flags) in enumerate(zip(all_labels, all_flags)):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(self._align_labels_with_tokens(labels, word_ids))
        new_only_in_test_flag.append(self._align_only_test_flag_with_tokens(flags, word_ids))
        tag_mask.append([0 if tag in [0, -100] else 1 for tag in new_labels[-1]])

    if self.is_category:
        all_cats = examples['is_category']
        is_category = []
        for i, cats in enumerate(all_cats):
            word_ids = tokenized_inputs.word_ids(i)
            is_category.append(self._align_labels_with_tokens(cats, word_ids, do_shift=False))
        tokenized_inputs['is_category'] = is_category

    tokenized_inputs['labels'] = new_labels
    tokenized_inputs['tag_mask'] = tag_mask
    tokenized_inputs['only_in_test'] = new_only_in_test_flag

    return tokenized_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(help="Error report for token classification.")
    parser.add_argument(
        "--model",
        help="Local or HuggingFace patht to the model to be tested."
        )
    parser.add_argument(
        "--dataset",
        help="Local or HuggingFace patht to the dataset to be tested."
        )
    parser.add_argument(
        "--split",
        default="test",
        help="Split of the dataset to be loaded."
        )

    args = parser.parse_args()

    model = AutoModelForTokenClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset(args.dataset, split=args.split)
    # Tokenize the test dataset
    tokenized_test_dataset = test_dataset.map(lambda x: tokenizer(x['words'], truncation=True, padding='max_length'), batched=True)

    model.eval()

    predictions = []
    labels = []

    for batch in tokenized_test_dataset:
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predictions.extend(logits.argmax(-1).cpu().numpy())
        labels.extend(batch["labels"].numpy())

    # Align predictions with actual tokens
    aligned_predictions, aligned_labels = align_predictions(predictions, labels, tokenizer)

    # Initialize dictionary to store errors
    errors = {f"class_{i}": Counter() for i in range(model.config.num_labels)}

    # Analyze errors
    for prediction, label in zip(aligned_predictions, aligned_labels):
        for p, l in zip(prediction, label):
            if p != l:  # Error made by the model
                errors[f"class_{l}"].update([f"class_{p}"])

    # errors now contains the count of most common errors for each class
    print(errors)


