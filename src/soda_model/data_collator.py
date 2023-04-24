import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from transformers import (
    PreTrainedTokenizerBase,
)
from transformers.data.data_collator import (
    DataCollatorMixin,
)
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForMaskedTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (:obj:`str` or `None`, `optional`, defaults to :obj:`'pt'`): type of tensors to return.
            It accepts those supported by HuggingFace.
        masking_probability (float, defaults to .0): The probability of masking a token.
        replacement_probability (float, defaults to .0): The probability of replacing a masked token with a random token.
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        select_labels (bool, defaults to False):
            Whether use only the labels at the masked position to calculate the loss
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    masking_probability: float = .0
    replacement_probability: float = .0
    select_labels: bool = False

    def torch_call(self, features) -> Dict[str, torch.Tensor]:
        """
        In addition to input_ids, a feature 'tag_mask' needs to be provided to specify which token might be masked.
        """
        if 'tag_mask' in features[0].keys():
            tag_mask = [feature['tag_mask'] for feature in features]
        else:
            raise ValueError("A mask should be provided to indicate which input token class to mask.")
        label_name = "label" if "label" in features[0].keys() else "labels"
        if label_name in features[0].keys():
            labels = [feature[label_name] for feature in features]
        else:
            raise ValueError("A feature 'label' or 'labels' should be provided for token classification")
        if "is_category" in features[0].keys():
            categories = [feature["is_category"] for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        # batch['input_ids'] are now padded
        # we still need to 'manually' pad the labels and the tag mask

        sequence_length = len(batch["input_ids"][0])
        padding_side = self.tokenizer.padding_side

        if padding_side == "right":
            batch["tag_mask"] = [x + [0] * (sequence_length - len(x)) for x in tag_mask]
            batch["labels"] = [x + [self.label_pad_token_id] * (sequence_length - len(x)) for x in labels]
            if "is_category" in batch.keys():
                batch["is_category"] = [x + [0] * (sequence_length - len(x)) for x in categories]  # type: ignore
        else:
            batch["tag_mask"] = [[0] * (sequence_length - len(x)) + x for x in tag_mask]
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(x)) + x for x in labels]
            if "is_category" in batch.keys():
                batch["is_category"] = [[0] * (sequence_length - len(x)) + x for x in categories]  # type: ignore
        # convert dict of list of lists into ditc of tensors
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        # stochastically mask input ids according to tag_mask
        if "is_category" in batch.keys():
            batch["input_ids"], batch["labels"], batch["is_category"] = self.tag_mask_tokens(batch["input_ids"], batch["labels"], batch["tag_mask"], category=batch["is_category"])  # type: ignore
        else:
            batch["input_ids"], batch["labels"] = self.tag_mask_tokens(batch["input_ids"], batch["labels"], batch["tag_mask"])  # type: ignore
        # remove tak_mask from match as it would be rejected by model.forward()
        batch.pop("tag_mask")
        return batch

    def tag_mask_tokens(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, category: Union[None, torch.Tensor] = None) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Masks the input as specified by the tag mask prepared by the loader"""

        inputs = inputs.clone()  # not sure if necessary; might be safer to avoid changing input features when provided as tensor
        if self.select_labels:
            targets = targets.clone()
        # create the probability matrix for masking
        masking_probability_matrix = torch.full(inputs.size(), self.masking_probability)
        # use the probability matrix to draw whether to replace or not and intersect with the mask
        masked_indices = torch.bernoulli(masking_probability_matrix).bool() & mask.bool()
        # replace input_ids by the mask token id at position that need to be masked
        inputs[masked_indices] = self.tokenizer.mask_token_id
        # second probability matrix is to determin whether to randomize remaining marked tokens
        replacement_probability_matrix = torch.full(inputs.size(), self.replacement_probability)
        # indices of token to replace found by drawing from prob matric and intersecting with mask but exclusin alreayd masked positions
        replaced_indices = torch.bernoulli(replacement_probability_matrix).bool() & mask.bool() & ~masked_indices
        # draw random int from vocab size of tokenizer and fill tenzoer of shape like intput
        random_input_ids = torch.randint(len(self.tokenizer), inputs.size(), dtype=torch.long)
        # at the replacmenet indices, change to random token
        inputs[replaced_indices] = random_input_ids[replaced_indices]
        if self.select_labels:
            # only labels at the makred position (irrespective of whether they are masked) will be used for calculating the loss
            targets[~mask] = -100
        if category is not None:
            category = category.clone()
            return inputs, targets, category
        else:
            return inputs, targets


@dataclass
class MaskedDataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    masking_probability: float = 0.15
    return_tensors: str = "pt"

    def torch_call(self, features):

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        masked_tensor = (torch.rand_like(batch["input_ids"], dtype=torch.float) < self.masking_probability)
        masked_labels = (batch["labels"] == -100)
        batch["input_ids"][masked_tensor] = self.tokenizer.mask_token_id
        batch["input_ids"][masked_labels] = self.label_pad_token_id
        return batch

    def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()
