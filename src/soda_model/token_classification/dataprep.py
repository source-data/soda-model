from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    )
from typing import Tuple, Union, List
from soda_model.logging import get_logger
import numpy as np

logger = get_logger(__name__)


class DataLoader:
    def __init__(
            self,
            dataset_id: str,
            task: str,
            version: str = "1.0.0",
            verification_mode: str = "no_checks",
            ner_labels: List[str] = ["all"],
            max_length: int = 512,
            padding: Union[bool, str, None] = False,
            truncation: bool = True,
            **kwargs
    ):
        self.dataset_id = dataset_id
        self.task = task
        self.version = version
        self.kwargs = kwargs
        self.verification_mode = verification_mode
        self.ner_labels = ner_labels
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def load_data(self) -> DatasetDict:
        logger.info(f"Obtaining data from the HuggingFace 🤗 Hub: load_dataset('{self.dataset_id}',' {self.task}', version='{self.version})")
        data = load_dataset(
            self.dataset_id,
            self.task,
            version=self.version,
            verification_mode=self.verification_mode,
            )
        logger.info("Generating the label2id mapping")
        self.id2label, self.label2id = self._get_data_labels(data["train"])
        self.dataset_id2label = self.id2label
        self.dataset_label2id = self.label2id

        logger.info("Storing training and test entities for memorization and generalization")
        self.training_entities = self._get_entity_list(data, splits=["train", "validation"])
        self.test_entities = self._get_entity_list(data, splits=["test"])
        self.test_only_entities = self.test_entities - self.training_entities
        only_in_test = self._get_test_only_mask(data)
        for split in ["train", "validation", "test"]:
            data[split] = data[split].add_column('only_in_test', only_in_test[split])

        return data

    def tokenize_data(self, data: DatasetDict, tokenizer) -> DatasetDict:
        logger.info("Tokenizing data")
        self.tokenizer = tokenizer
        tokenized_data = data.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=["words", "text"] if self.task != "PANELIZATION" else ["words"])
        return tokenized_data["train"], tokenized_data['validation'], tokenized_data['test']

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
        tokenized_inputs['labels'] = new_labels
        tokenized_inputs['tag_mask'] = tag_mask
        tokenized_inputs['only_in_test'] = new_only_in_test_flag

        return tokenized_inputs

    def _get_data_labels(self, data: Dataset) -> Tuple[dict, dict]:
        num_labels = data.info.features['labels'].feature.num_classes
        label_list = data.info.features['labels'].feature.names
        id2label, label2id = {}, {}
        for class_, label in zip(range(num_labels), label_list):
            id2label[class_] = label
            label2id[label] = class_
        logger.info(f"The data set has {num_labels} features: {label_list}")
        logger.info(f"\nTraining on {len(self.ner_labels)} features: {self.ner_labels}")
        return id2label, label2id

    def _get_entity_list(self, dataset, splits):
        words_in_training = []
        inside_word = []
        for split in splits:
            for example in dataset[split]:
                for word, label in zip(example["words"], example["labels"]):
                    if self.id2label[label].startswith("B-") and not inside_word:
                        inside_word = [word]
                    elif self.id2label[label].startswith("B-") and inside_word:
                        words_in_training.append(" ".join(inside_word))
                        inside_word = [word]
                    elif self.id2label[label].startswith("I-"):
                        inside_word.append(word)
                    elif (label in [0, "O", "0"]) and inside_word:
                        words_in_training.append(" ".join(inside_word))
                        inside_word = []
                    else:
                        continue

        return set(words_in_training)

    def _get_test_only_mask(self, data):
        output = {"train": [], "test": [], "validation": []}
        for split in ["train", "test", "validation"]:
            for _, example in enumerate(data[split]):
                example_flag = []
                if split in ["train", "validation"]:
                    output[split].append([0] * len(example["labels"]))
                else:
                    entity_idx_dict = self._get_entity_idx_dict(example["words"], example["labels"])

                    # do a zeros numpy array, the length of labels
                    example_flag = np.zeros_like(example["labels"])

                    # assign 1 to the idx values of entities not in training
                    for _, index in entity_idx_dict.items():
                        for idx in index:
                            example_flag[idx] = 1

                    output[split].append(list(example_flag))
                    assert len(example_flag) == len(example["words"]),  f"Labels: {example['labels']} \n Flags: {example_flag}"
        return output

    def _get_entity_idx_dict(self, words, labels, ):
        entities_in_example = {}
        entity_index = []
        inside_word = []
        for idx, (word, label) in enumerate(zip(words, labels)):
            if self.id2label[label].startswith("B-") and not inside_word:
                inside_word = [word]
                entity_index.append(idx)
            elif self.id2label[label].startswith("B-") and inside_word:
                if " ".join(inside_word) in self.test_only_entities:
                    if " ".join(inside_word) in list(entities_in_example.keys()):
                        for token in entity_index:
                            entities_in_example[" ".join(inside_word)].append(token)
                    else:
                        entities_in_example[" ".join(inside_word)] = entity_index
                inside_word = [word]
                entity_index = [idx]
            elif self.id2label[label].startswith("I-"):
                inside_word.append(word)
                entity_index.append(idx)
            elif (label in [0, "O", "0"]) and inside_word:
                if " ".join(inside_word) in self.test_only_entities:
                    if " ".join(inside_word) in list(entities_in_example.keys()):
                        for token in entity_index:
                            entities_in_example[" ".join(inside_word)].append(token)
                    else:
                        entities_in_example[" ".join(inside_word)] = entity_index
                inside_word = []
                entity_index = []

        for entity in list(entities_in_example.keys()):
            assert entity not in self.training_entities
        return entities_in_example

    @staticmethod
    def _shift_label(label):
        # If the label is B-XXX we change it to I-XX
        if label % 2 == 1:
            label += 1
        return label

    def _align_labels_with_tokens(self, labels, word_ids):
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
                new_labels.append(self._shift_label(labels[word_id]))

        return new_labels

    def _align_only_test_flag_with_tokens(self, labels, word_ids):
        """
        Expands the only test flag tags once the sub-word tokenization is added.
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
                new_labels.append(labels[word_id])

        return new_labels


class DataLoaderSelectAndFilter(DataLoader):
    def __init__(
            self,
            dataset_id,
            task,
            version,
            filter_empty: bool = True,
            ner_labels: List[str] = ["DISEASE"],
            verification_mode: str = "no_checks",
            max_length: int = 512,
            padding: Union[bool, str, None] = False,
            truncation: bool = True,
            ):
        super().__init__(dataset_id, task, version, ner_labels=ner_labels,
                         verification_mode=verification_mode,
                         max_length=max_length,
                         padding=padding,
                         truncation=truncation)
        self.filter_empty = filter_empty

    def _generate_new_label_dict(self):
        id2label, label2id = {}, {}
        new_labels = ["O"]
        for label in self.ner_labels:
            new_labels.append(f"B-{label}")
            new_labels.append(f"I-{label}")
        for i, label in enumerate(new_labels):
            id2label[i] = label
            label2id[label] = i
        return id2label, label2id

    def _substitute_training_labels(self, examples):

        all_labels = examples['labels']
        self.id2label, self.label2id = self._generate_new_label_dict()
        new_labels = []
        new_tag_mask = []
        for labels in all_labels:
            new_labels_sentence = []
            for label in labels:
                if label == -100:
                    new_labels_sentence.append(label)
                elif self.dataset_id2label[label] in list(self.id2label.values()):
                    new_labels_sentence.append(self.label2id[self.dataset_id2label[label]])
                else:
                    new_labels_sentence.append(0)
            new_labels.append(new_labels_sentence)
            new_tag_mask.append([0 if tag in [0, -100] else 1 for tag in new_labels[-1]])

        examples['labels'] = new_labels
        examples['tag_mask'] = new_tag_mask

        return examples

    def tokenize_data(self, data: DatasetDict, tokenizer) -> DatasetDict:
        logger.info("Tokenizing data")
        self.tokenizer = tokenizer
        tokenized_data = data.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=["words", "text"])
        tokenized_data = tokenized_data.map(
            self._substitute_training_labels,
            batched=True)

        if self.filter_empty:
            tokenized_data = tokenized_data.filter(lambda example: (1 in example["labels"]) or (2 in example["labels"]))

        return tokenized_data["train"], tokenized_data['validation'], tokenized_data['test']
