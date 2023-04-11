import torch
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
        else:
            batch["tag_mask"] = [[0] * (sequence_length - len(x)) + x for x in tag_mask]
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(x)) + x for x in labels]
        # convert dict of list of lists into ditc of tensors
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        # stochastically mask input ids according to tag_mask
        batch["input_ids"], batch["labels"] = self.tag_mask_tokens(batch["input_ids"], batch["labels"], batch["tag_mask"])
        # remove tak_mask from match as it would be rejected by model.forward()
        batch.pop("tag_mask")
        return batch

    def tag_mask_tokens(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return inputs, targets
