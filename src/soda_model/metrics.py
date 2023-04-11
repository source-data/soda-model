from typing import List, Dict
from transformers import EvalPrediction
import numpy as np
from seqeval.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
from seqeval.scheme import IOB2


class MetricsTOKCL:
    """Computes metrics for token classifications. Assumes the labels follow the IOB2 scheme.

    Args:
        label_list: the list of IOB2 string labels.
    """
    def __init__(self, label_list: List = []):
        self.label_list = label_list

    def __call__(self, eval_pred: EvalPrediction) -> Dict:
        """Computes accuracy precision, recall and f1 based on the list of IOB2 labels.
        Positions with labels with a value of -100 will be filtered out both from true labela dn prediction.

        Args:
            eval_pred (EvalPrediction): the predictions and targets to be matched as np.ndarrays.

        Returns:
            (Dict): a dictionary with accuracy_score, precision, recall and f1.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        print("\n"+" " * 80)

        try:
            print(classification_report(true_labels, true_predictions, digits=4, mode="strict", scheme=IOB2, output_dict=True))
        except ValueError as e:
            print(e)

        return {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }


class MetricsCRFTOKCL:
    """Computes metrics for token classifications. Assumes the labels follow the IOB2 scheme.
    Args:
        label_list: the list of IOB2 string labels.
    """
    def __init__(self, label_list: List = []):
        self.label_list = label_list

    def __call__(self, eval_pred: EvalPrediction) -> Dict:
        """Computes accuracy precision, recall and f1 based on the list of IOB2 labels.
        Positions with labels with a value of -100 will be filtered out both from true labela dn prediction.
        Args:
            eval_pred (EvalPrediction): the predictions and targets to be matched as np.ndarrays.
        Returns:
            (Dict): a dictionary with accuracy_score, precision, recall and f1.
        """
        predictions, labels = eval_pred

        predictions = np.argmax(predictions[0], axis=-1)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        print("\n"+" " * 80)
        try:

            # print("*******Classical classification report*****")
            # print(classification_report(true_labels, true_predictions, digits=4))
            print("*******Strict classification report*****")
            print(classification_report(true_labels, true_predictions, digits=4, mode="strict", scheme=IOB2))
        except ValueError as e:
            print(e)

        return {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
