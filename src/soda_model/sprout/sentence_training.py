"""
Training the sentence transformers model for SPROUT.

This model will generate specialized protein embeddings to be used
in downstream tasks.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import losses, models
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample
from soda_model.logging import get_logger
import os
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import torch
from typing import Tuple
import wandb
from soda_model.sprout.sentence_modelling import MySentenceTransformer
import matplotlib.pyplot as plt
import torch.nn.functional as F
from enum import Enum


logger = get_logger(__name__)

loss_names = ["contrastive_loss", "custom_loss"]


def log_callback_st(train_ix, training_steps, global_step, current_lr, loss_value, mode="train"):
    wandb.log(
        {
            f"{mode}/{loss_names[train_ix]}_loss": loss_value,
            f"{mode}/{loss_names[train_ix]}_lr": current_lr[0],
            "train/steps": training_steps
            }
    )
    logger.info(
        {
            f"{mode}/{loss_names[train_ix]}_loss": loss_value,
            f"{mode}/{loss_names[train_ix]}_lr": current_lr[0],
            "train/steps": training_steps
        }
    )
    print(
        {
            f"{mode}/{loss_names[train_ix]}_loss": loss_value,
            f"{mode}/{loss_names[train_ix]}_lr": current_lr[0],
            "train/steps": training_steps
        }
    )


class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    def cosine(self, x, y):
        return 1 - F.cosine_similarity(x, y)

    def euclidean(self, x, y):
        return F.pairwise_distance(x, y, p=2)

    def manhattan(self, x, y):
        return F.pairwise_distance(x, y, p=1)


@dataclass
class SentenceTransformerTrainingArgs:
    pooling_mode: str = field(default="mean", metadata={"help": "Pooling mode to be used. Choose from 'mean', 'cls', 'max'."})
    training_batch_size: int = field(default=8, metadata={"help": "Training batch size."})
    eval_batch_size: int = field(default=16, metadata={"help": "Validation batch size."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Percentage of steps to build up the learning rate."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    num_epochs: int = field(default=1, metadata={"help": "Number of epochs to train the dataset."})
    eval_steps: int = field(default=10000, metadata={"help": "Training steps at which an evaluation should be performed."})
    log_steps: int = field(default=10000, metadata={"help": "Training log.."})
    triplet_margin: float = field(default=5.0, metadata={"help": "Triplet margin."})
    distance_metric: str = field(default="euclidean", metadata={"help": "Distance metric to be used. Choose from 'euclidean' and 'cosine'."})
    max_length: int = field(default=512, metadata={"help": "Maximum size of embeddings. Truncating size."})
    embed_dim: int = field(default=256, metadata={"help": "Embedding size"})
    # HuggingFace Hub Parameters
    push_to_hub: bool = field(default=False, metadata={"help": "Push the model to the HuggingFace hub."})
    repo_name: str = field(default="", metadata={"help": "HuggingFace hub repo."})
    hf_token: str = field(default="", metadata={"help": "HuggingFace Token."})
    organization: str = field(default="EMBO", metadata={"help": "HuggingFace Organization."})
    commit_message: str = field(default="Add new SentenceTransformer model.", metadata={"help": "Commit message to the HF Hub."})
    overwrite: bool = field(default=False, metadata={"help": "If the model can overwrite an existing model"})
    replace_model_card: bool = field(default=True, metadata={"help": "Replace existing model card"})
    wandb_api_key: str = field(default="", metadata={"help": "Wandb API key."})
    wandb_project: str = field(default="sprout", metadata={"help": "Wandb project name."})
    wandb_run: str = field(default="sprout", metadata={"help": "Wandb project name."})


class SentenceTransformerTrainer:
    def __init__(
            self,
            training_args: SentenceTransformerTrainingArgs,
            checkpoint: str = "michiyasunaga/BioLinkBERT-base",
            output_dir: str = "/app/data/models/sentence_transformer_sprout_uniprot",
            train_dataset: str = "/app/data/sprout/sprout_triplets_train.jsonl",
            eval_dataset: str = "/appdata/sprout/sprout_triplets_eval.jsonl",
            test_dataset: str = "/appdata/sprout/sprout_triplets_train.jsonl",
            smoke_test: bool = False,
            ) -> None:
        self.training_args = training_args
        self.smoke_test = smoke_test
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = checkpoint
        self.model = self._load_model()
        self.output_dir = output_dir
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.triplet_margin = self.training_args.triplet_margin
        self.train_loss = losses.TripletLoss(model=self.model)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader, self.evaluation_list = self._data_loader()
        self.distance_metric = TripletDistanceMetric.EUCLIDEAN if self.training_args.distance_metric == "euclidean" else TripletDistanceMetric.COSINE

    def train(self):
        os.environ["WANDB_API_KEY"] = self.training_args.wandb_api_key
        os.environ["WANDB_PROJECT"] = self.training_args.wandb_project
        wandb.init(name=self.training_args.wandb_run)
        train_loss = losses.OnlineContrastiveLoss(
            model=self.model,
            margin=self.triplet_margin,
            distance_metric=self.distance_metric,
            )
        # train_loss = losses.TripletLoss(
        #     model=self.model,
        #     triplet_margin=self.triplet_margin,
        #     distance_metric=self.distance_metric,
        #     )

        train_objectives = [(self.train_dataloader, train_loss)]
        warmup_steps = math.ceil(len(self.train_dataloader) * self.training_args.num_epochs * self.training_args.warmup_ratio)
        # evaluator = MyTripletEvaluator(
        #     self.evaluation_list[0],
        #     self.evaluation_list[1],
        #     self.evaluation_list[2],
        #     main_distance_function=self.distance_metric,
        #     name=self.training_args.repo_name,
        #     show_progress_bar=True,
        #     write_csv=True,
        #     batch_size=self.training_args.eval_batch_size,
        #     triplet_margin=self.triplet_margin
        #     )
        # Train the model
        self._plot_histogram(prior=True)
        # self.model.start_multi_process_pool()
        self.model.fit(
            train_objectives=train_objectives,
            # evaluator=evaluator,
            epochs=self.training_args.num_epochs,
            # evaluation_steps=self.training_args.eval_steps,
            warmup_steps=warmup_steps,
            output_path=self.output_dir,
            log_callable=log_callback_st,
            log_steps=self.training_args.log_steps,
        )
        self.model.save(self.output_dir)
        self._plot_histogram(prior=False)

        if self.training_args.push_to_hub:
            self.model.save_to_hub(
                repo_name=self.training_args.repo_name,
                use_auth_token=self.training_args.hf_token,
                organization=self.training_args.organization,
                commit_message=self.training_args.commit_message,
                exist_ok=self.training_args.overwrite,
                replace_model_card=self.training_args.replace_model_card,
                )

    def _plot_histogram(self, prior=True):
        if self.distance_metric == TripletDistanceMetric.EUCLIDEAN:
            neg_distance = F.pairwise_distance(torch.Tensor(self.model.encode(self.evaluation_list[0])), torch.Tensor(self.model.encode(self.evaluation_list[2])), p=2).tolist()
            pos_distance = F.pairwise_distance(torch.Tensor(self.model.encode(self.evaluation_list[0])), torch.Tensor(self.model.encode(self.evaluation_list[1])), p=2).tolist()
        if self.distance_metric == TripletDistanceMetric.COSINE:
            neg_distance = F.cosine_similarity(torch.Tensor(self.model.encode(self.evaluation_list[0])), torch.Tensor(self.model.encode(self.evaluation_list[2]))).tolist()
            pos_distance = F.cosine_similarity(torch.Tensor(self.model.encode(self.evaluation_list[0])), torch.Tensor(self.model.encode(self.evaluation_list[1]))).tolist()

        if prior:
            plt.hist(neg_distance, bins=100, alpha=0.5, label='negative_prior', histtype='step')
            plt.hist(pos_distance, bins=100, alpha=0.5, label='positive_prior', histtype='step')
            plt.legend()
            plt.savefig("/app/data/results/distance_distribution_prior_cosine_distance.png")
        else:
            plt.hist(neg_distance, bins=100, alpha=0.5, label='negative_posterior', histtype='step')
            plt.hist(pos_distance, bins=100, alpha=0.5, label='positive_posterior', histtype='step')
            plt.legend()
            plt.savefig("/app/data/results/distance_distribution_posterior_cosine_distance.png")

    def _data_loader(self) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
        examples = {"train": [], "eval": [], "test": []}

        eval_anchor, eval_pos, eval_neg = [], [], []

        for dataset, split in zip([self.train_dataset, self.eval_dataset, self.test_dataset], "train eval test".split()):
            with open(dataset, 'r') as json_file:
                json_list = list(json_file)

            if self.smoke_test:
                json_list = json_list[:5000]

            for json_str in json_list:
                result = json.loads(json_str)

                # examples[split].append(InputExample(texts=[result["query"], result["pos"], result["neg"]]))
                examples[split].append(InputExample(texts=[result["query"], result["pos"]], label=1))
                examples[split].append(InputExample(texts=[result["query"], result["neg"]], label=0))

                if split == "eval":
                    eval_anchor.append(result["query"])
                    eval_pos.append(result["pos"])
                    eval_neg.append(result["neg"])

        return (
            DataLoader(examples["train"], shuffle=True, batch_size=self.training_args.training_batch_size),  # type: ignore
            DataLoader(examples["eval"], shuffle=True, batch_size=self.training_args.training_batch_size),  # type: ignore
            DataLoader(examples["test"], shuffle=True, batch_size=self.training_args.training_batch_size),  # type: ignore
            [eval_anchor, eval_pos, eval_neg]
            )

    def _load_model(self) -> SentenceTransformer:
        self.word_embedding_model = models.Transformer(
            self.checkpoint,
            tokenizer_args={
                "truncation": True,
                "max_length": self.training_args.max_length
            }
            )
        if self.training_args.pooling_mode == "mean":
            self.pooling_model = models.Pooling(
                self.training_args.embed_dim,
                pooling_mode_mean_tokens=True,
                )
        elif self.training_args.pooling_mode == "cls":
            self.pooling_model = models.Pooling(
                self.training_args.embed_dim,
                pooling_mode_cls_token=True,
                )
        elif self.training_args.pooling_mode == "max":
            self.pooling_model = models.Pooling(
                self.training_args.embed_dim,
                pooling_mode_max_tokens=True,
                )
        else:
            raise ValueError(f"Pooling mode {self.training_args.pooling_mode} not supported. Choose between 'mean', 'cls' and 'max'.")

        return MySentenceTransformer(
            modules=[self.word_embedding_model, self.pooling_model],
            device=self.device,
            )


if __name__ == "__main__":

    parser = HfArgumentParser(SentenceTransformerTrainingArgs, description="Trains a sentence transformer model.")
    parser.add_argument(
        "--output_dir",
        default="/app/data/models/sentence_transformer_sprout_uniprot",
        help="Uniprot file with the references.",
        )
    parser.add_argument(
        "--base_model",
        default="michiyasunaga/BioLinkBERT-base",
        help="HuggingFace compatible model to be used as base."
        )
    parser.add_argument(
        "--datasets",
        nargs='+',
        default="app/data/sprout/sprout_triplets_train.jsonl app/data/sprout/sprout_triplets_eval.jsonl app/data/sprout/sprout_triplets_test.jsonl",
        help="Three jsonl files containing the train, eval and test datasets."
        )
    parser.add_argument(
        "--smoke_test",
        action='store_true',
        help="Run a small model to test the code."
        )
    training_args, args = parser.parse_args_into_dataclasses()

    sprout_trainer = SentenceTransformerTrainer(
        training_args,
        checkpoint=args.base_model,
        output_dir=args.output_dir,
        train_dataset=args.datasets[0],
        eval_dataset=args.datasets[1],
        test_dataset=args.datasets[2],
        smoke_test=True if args.smoke_test else False,
    )

    sprout_trainer.train()
