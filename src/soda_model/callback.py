from random import randrange
from typing import Dict

import torch
from transformers import TrainerCallback


class ShowExample(TrainerCallback):
    """Visualizes on the console the result of a prediction with the current state of the model.
    It uses a randomly picked input example and decodes input and output with the provided tokenizer.
    The predicted words are colored depending on whether the prediction is correct or not.
    If the prediction is incorrect, the expected word is displayed in square brackets.

    Args:

        tokenizer (RobertaTokenizer): the tokenizer used to generate the dataset.

    Class Attributes:

        COLOR_CHAR (Dict): terminal colors used to produced colored string
    """

    COLOR_CHAR = {}

    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer

    def on_evaluate(
        self,
        *args,
        model=None,
        eval_dataloader: torch.utils.data.DataLoader = None,
        **kwargs
    ):
        """Method called when evaluating the model. Only the needed kwargs are unpacked.

        Args:

            model: the current model being trained.
            eval_dataloader (torch.utils.data.DataLoader): the DataLoader used to produce the evaluation examples
        """
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred = model(inputs["input_ids"], labels=inputs["labels"], attention_mask=inputs["attention_mask"])  # type: ignore
            if isinstance(pred, dict):
                pred_idx = pred['logits'].argmax(-1)[0].cpu()
                self.to_console(inputs, pred_idx)
            else:
                pass

    def on_predict(
        self,
        *args,
        model=None,
        eval_dataloader: torch.utils.data.DataLoader = None,
        **kwargs
    ):
        """Method called when evaluating the model. Only the needed kwargs are unpacked.

        Args:

            model: the current model being trained.
            eval_dataloader (torch.utils.data.DataLoader): the DataLoader used to produce the evaluation examples
        """
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred = model(inputs["input_ids"], labels=inputs["labels"], attention_mask=inputs["attention_mask"])  # type: ignore
            if isinstance(pred, dict):
                pred_idx = pred['logits'].argmax(-1)[0].cpu()
                self.to_console(inputs, pred_idx)
            else:
                pass
        inputs = {k: v[0] for k, v in inputs.items()}
        self.to_console(inputs, pred_idx)

    def pick_random_example(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        L = len(dataloader.dataset)
        dataset = dataloader.dataset
        rand_example_idx = randrange(L)
        batch = dataloader.collate_fn([dataset[rand_example_idx]])  # batch with a single random example
        inputs = {}
        for k, v in batch.items():
            inputs[k] = v.cuda() if torch.cuda.is_available() else v
        return inputs

    def to_console(self, inputs: Dict[str, torch.Tensor], pred_idx):
        pred_idx = [e.item() for e in pred_idx]
        input_ids = [e.item() for e in inputs["input_ids"][0]]
        labels = [e.item() for e in inputs["labels"][0]]
        colored = ""
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            label_idx = pred_idx[i]
            true_label = labels[i]
            colored += self._correct_incorrect(input_id, label_idx, true_label) + " "
        print(f"\n\n{colored}\n\n")

    def _correct_incorrect(self, input_id: int, label_idx: int, true_label: int) -> str:
        raise NotImplementedError


class ShowExampleTOKCL(ShowExample):

    COLOR_CHAR = {
        "underscore": "\033[4m",
        "bold": "\033[1m",
        "close":  "\033[0m",
        "var_color": "\033[38;5;{color_idx}m",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _correct_incorrect(self, input_id, label_idx, true_label, **kwargs) -> str:
        colored = ""
        if input_id != self.tokenizer.pad_token_id:  # don't display padding
            decoded = self.tokenizer.decode(input_id)
            # indicate the true label with underline
            underscore = self.COLOR_CHAR["underscore"] if label_idx == true_label else ''
            if label_idx > 0:  # don't show default no_label
                colored = f"{self.COLOR_CHAR['bold']}{underscore}{self.COLOR_CHAR['var_color'].format(color_idx=label_idx)}{decoded}{self.COLOR_CHAR['close']}"
            else:
                colored = f"{decoded}"
        return colored
