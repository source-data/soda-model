# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
from typing import List
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    DefaultFlowCallback,
    BertTokenizerFast,
    CanineConfig,
    HfArgumentParser,
)
from soda_model.metrics import MetricsTOKCL, MetricsCRFTOKCL
from soda_model.callback import ShowExampleTOKCL
from soda_model.data_classes import TrainingArgumentsTOKCL

from soda_model.data_collator import DataCollatorForMaskedTokenClassification
import numpy as np
from soda_model.token_classification.modelling import CRFforTokenClassification
from seqeval.metrics import (
    classification_report
)
from seqeval.scheme import IOB2
from soda_model import RESULTS_FOLDER
from soda_model.token_classification.dataprep import (
    DataLoader,
    DataLoaderSelectAndFilter,
    )
import logging
from soda_model.logging import get_logger
import os
import pickle

logger = get_logger()
logger.setLevel(logging.INFO)  # type: ignore


class TrainTokenClassification:
    def __init__(
        self,
        training_args: TrainingArgumentsTOKCL,
        from_pretrained: str,
        dataset_id: str = "EMBO/SourceData",
        task: str = "NER",
        version: str = "1.0.0",
        filter_empty: bool = True,
        ner_labels: List[str] = ["all"],
        add_prefix_space: bool = False,
        use_crf: bool = False,
        max_length: int = 512,
        padding: str = "true",
        truncation: bool = True,
    ):
        self.training_args = training_args
        self.from_pretrained = from_pretrained
        self.dataset_id = dataset_id
        self.task = task
        self.version = version
        self.ner_labels = ner_labels
        self.filter_empty = filter_empty
        self.add_prefix_space = add_prefix_space
        self.use_crf = use_crf
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        if padding == "true":
            self.padding = True
        elif padding == "false":
            self.padding = False
        else:
            self.padding = padding

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self):
        logger.info("Loading the tokenizer")
        self.tokenizer = self._get_tokenizer()

        logger.info("Loading the dataset")
        data_loader = DataLoader if "all" in self.ner_labels else DataLoaderSelectAndFilter
        data_loader = data_loader(
            self.dataset_id,
            self.task,
            self.version,
            ner_labels=self.ner_labels,
            filter_empty=self.filter_empty,
            padding=self.padding,
            truncation=self.truncation,
            )

        dataset = data_loader.load_data()

        self.train_dataset, self.eval_dataset, self.test_dataset = data_loader.tokenize_data(dataset, self.tokenizer)
        self.label2id = data_loader.label2id
        self.id2label = data_loader.id2label

        logger.info("Loading and preparing the data collator")
        self.data_collator = self._get_data_collator()

        logger.info("Defining the metrics computation class")
        self.compute_metrics = MetricsCRFTOKCL if self.use_crf else MetricsTOKCL
        self.compute_metrics = self.compute_metrics(label_list=list(self.label2id.keys()))

        logger.info("Defining the model")
        self.model = CRFforTokenClassification if self.use_crf else AutoModelForTokenClassification
        self.model = self.model.from_pretrained(
            self.from_pretrained,
            num_labels=len(list(self.label2id.keys())),
            max_position_embeddings=self._max_position_embeddings(),
            id2label=self.id2label,
            label2id=self.label2id,
            classifier_dropout=self.training_args.classifier_dropout,
            max_length=self.max_length,
            return_dict=False if self.use_crf else True,
            )

        logger.info("Defining the trainer class")
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_dataset.remove_columns(['only_in_test', 'token_type_ids', 'attention_mask']),
            eval_dataset=self.eval_dataset.remove_columns(['only_in_test', 'token_type_ids', 'attention_mask']),
            compute_metrics=self.compute_metrics,
            callbacks=[
                DefaultFlowCallback,
                ShowExampleTOKCL(self.tokenizer),
                ]
        )

        logger.info(f"Training model for token classification {self.from_pretrained}.")
        self.trainer.train()

        if self.training_args.push_to_hub:
            logger.info(f"Uploading the model {self.trainer.model} and tokenizer {self.trainer.tokenizer} to HuggingFace")
            self.trainer.push_to_hub(commit_message="End of training")

        if self.training_args.do_predict:
            logger.info(f"Testing on {len(self.test_dataset)}.")
            # self.trainer.args.prediction_loss_only = False
            if self.use_crf:
                (_, all_predictions), all_labels, _ = self.trainer.predict(self.test_dataset.remove_columns(['only_in_test', 'token_type_ids', 'attention_mask']), metric_key_prefix='test')
            else:
                all_predictions, all_labels, _ = self.trainer.predict(self.test_dataset.remove_columns(['only_in_test', 'token_type_ids', 'attention_mask']), metric_key_prefix='test')
                all_predictions = np.argmax(all_predictions, axis=-1)

            only_in_test_pad = []
            for idx, example in enumerate(self.test_dataset["only_in_test"]):
                only_in_test_pad.append(example)
                while len(only_in_test_pad[idx]) < all_predictions.shape[1]:
                    only_in_test_pad[idx].append(-100)

            only_in_test_pad = np.array(only_in_test_pad)

            generalized_predictions = all_predictions * only_in_test_pad
            memorized_predictions = all_predictions * ((only_in_test_pad - 1) * (-1))

            generalized_labels = all_labels * only_in_test_pad
            memorized_labels = all_labels * ((only_in_test_pad - 1) * (-1))

            # Here is were to prepare the results for memorization and not memorization
            total_file = os.path.join(RESULTS_FOLDER, f"{self.training_args.results_file}_{self.task}_{self.training_args.masking_probability}_all.pkl")
            print("******* All report classification report *****")
            total_metrics = self._get_metrics(all_predictions, all_labels)
            with open(total_file, 'wb') as handle:
                pickle.dump(total_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
            memo_file = os.path.join(RESULTS_FOLDER, f"{self.training_args.results_file}_{self.task}_{self.training_args.masking_probability}_memo.pkl")
            print("******* Memorization classification report *****")
            memorization_metrics = self._get_metrics(memorized_predictions, memorized_labels)
            with open(memo_file, 'wb') as handle:
                pickle.dump(memorization_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("******* Generalization classification report *****")
            gen_file = os.path.join(RESULTS_FOLDER, f"{self.training_args.results_file}_{self.task}_{self.training_args.masking_probability}_gen.pkl")
            generalization_metrics = self._get_metrics(generalized_predictions, generalized_labels)
            with open(gen_file, 'wb') as handle:
                pickle.dump(generalization_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save the results to a python file that I can load later to get the values and plot

    def _get_data_collator(self):
        """
        Loads the data collator for the training. The options are the typical
        `DataCollatorForTokenClassification` or a special `DataCollationForMaskedTokenClassification`.
        Deciding between both of them can be done by setting up the parameter `--masked_data_collator`.
        Returns:
            DataCollator
        """
        self.training_args.remove_unused_columns = False
        data_collator = DataCollatorForMaskedTokenClassification(
            tokenizer=self.tokenizer,
            return_tensors='pt',
            padding=self.padding,
            max_length=self.max_length,
            masking_probability=self.training_args.masking_probability,
            replacement_probability=self.training_args.replacement_probability,
            pad_to_multiple_of=None,
            select_labels=self.training_args.select_labels,
            )
        return data_collator

    def _get_tokenizer(self):
        if "Megatron" in self.from_pretrained:
            tokenizer = BertTokenizerFast.from_pretrained(
                self.from_pretrained,
                is_pretokenized=True
                )
            self.get_roberta = False
        else:
            logger.info(f"Loading the tokenizer: {self.from_pretrained}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.from_pretrained,
                is_pretokenized=True,
                add_prefix_space=self.add_prefix_space
                )
            if any(x in self.from_pretrained for x in ["roberta", "gpt2"]):
                self.get_roberta = True
            else:
                self.get_roberta = False
        return tokenizer

    def _max_position_embeddings(self) -> int:
        if any(x in self.from_pretrained for x in ["roberta", "gpt2"]) or self.get_roberta:
            return self.max_length + 2
        elif "canine" in self.from_pretrained:
            return CanineConfig.from_pretrained(self.from_pretrained).max_position_embeddings
        else:
            return self.max_length

    def _get_metrics(self, predictions, labels):

        true_predictions, true_labels = [], []

        for prediction, label in zip(predictions, labels):
            preds, labs = [], []
            for p, l in zip(prediction, label):
                preds.append(self.id2label.get(p, 'O'))
                labs.append(self.id2label.get(l, 'O'))
            true_predictions.append(preds)
            true_labels.append(labs)

        print("\n"+" " * 80)
        try:
            return classification_report(true_labels, true_predictions, digits=4, mode="strict", scheme=IOB2, output_dict=True)
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArgumentsTOKCL, description="Traing script.")
    parser.add_argument(
        "--dataset_id",
        help="Path of the loader."
        )
    parser.add_argument(
        "--task",
        choices=["NER", "ROLES_GP", "ROLES_SM", "ROLES_MULTI", "PANELIZATION"],
        help="The task for which we want to train the token classification."
        )
    parser.add_argument(
        "--version",
        default="latest",
        help="The version of the dataset."
        )
    parser.add_argument(
        "--from_pretrained",
        default="microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
        help="The pretrained model to fine tune."
        )
    parser.add_argument(
        "--add_prefix_space",
        action="store_true",
        help="Set to true if uing roberta with word splitted lists."
        )
    parser.add_argument(
        "--ner_labels",
        nargs="*",
        type=str,
        default="all" ,
        help="""Which NER entities are to be classify. Choose all or any combination of:
            [GENEPROD, TISSUE, ORGANISM, SMALL_MOLECULE, EXP_ASSAY, CELL_LINE, CELL_TYPE, DISEASE, SUBCELLULAR].
        """
        )
    parser.add_argument(
        "--use_crf",
        action="store_true",
        help="Adds a CRF to the classification layer."
        )

    parser.add_argument(
        "--filter_empty",
        action="store_true",
        help="If filtering examples that do not contain the selected labels."
        )

    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
        help="If filtering examples that do not contain the selected labels."
        )
    parser.add_argument(
        "--padding",
        default="max_length",
        type=str,
        help="If filtering examples that do not contain the selected labels."
        )
    parser.add_argument(
        "--truncation",
        action="store_true",
        help="If filtering examples that do not contain the selected labels."
        )

    training_args, args = parser.parse_args_into_dataclasses()
    dataset_id = args.dataset_id
    task = args.task
    version = args.version
    from_pretrained = args.from_pretrained
    add_prefix_space = args.add_prefix_space
    use_crf = args.use_crf
    filter_empty = args.filter_empty
    max_length = args.max_length
    ner_labels = ["all"] if args.ner_labels == "all" else args.ner_labels

    trainer = TrainTokenClassification(
        training_args=training_args,
        from_pretrained=from_pretrained,
        dataset_id=dataset_id,
        task=task,
        version=version,
        filter_empty=filter_empty,
        ner_labels=ner_labels,
        add_prefix_space=add_prefix_space,
        use_crf=use_crf,
        max_length=max_length
    )

    trainer()