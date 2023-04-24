from transformers import TrainingArguments, IntervalStrategy
from dataclasses import dataclass, field
from typing import List, Optional
from soda_model import MODELS_FOLDER


@dataclass
class TrainingArgumentsTOKCL(TrainingArguments):
    output_dir: str = field(
        default=f"{MODELS_FOLDER}/tokcl",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
        )
    # Main Hyperparameters to tune
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "The initial learning rate for Adam."},
        )
    lr_schedule: str = field(
        default='cosine',
        metadata={"help": "The learning rate schedule to use."},
        )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/CPU for training."},
        )
    per_device_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU/CPU for evaluation."},
        )
    num_train_epochs: float = field(
        default=3.,
        metadata={"help": "Total number of training epochs to perform."},
        )
    masking_probability: float = field(
        default=1.0,
        metadata={"help": "The probability of masking a token."},
        )
    classifier_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability for the classifier."},
        )
    replacement_probability: float = field(
        default=0.0,
        metadata={"help": "The probability of replacing a masked token with a random token."},
        )
    random_masking: bool = field(
        default=False,
        metadata={"help": "If true, does random masking."},
        )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
        )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "The ratio of warmup steps to total steps."},
        )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "The number of warmup steps for learning rate scheduler."},
        )

    # Logging and evaluation strategy
    evaluation_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to adopt during training."},
        )
    eval_steps: Optional[float] = field(
        default=0.5,
        metadata={"help": "Log and evaluate every X updates steps."},
        )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "Only log the loss."},
        )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps to accumulate the gradients on (only to increase batch size)"},
        )
    log_level: Optional[str] = field(
        default="passive",
        metadata={"help": "The level of logging to use."},
        )
    logging_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The directory where the logs will be saved."},
        )
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to adopt during training."},
        )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": "Log and evaluate the first global_step"},
        )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Log and evaluate every X steps."},
        )
    logging_nan_inf_filter: bool = field(
        default=True,
        metadata={"help": "Filter out all nan/inf values before logging."},
        )
    log_on_each_node: bool = field(
        default=True,
        metadata={"help": "Whether or not to log on each node."},
        )
    save_strategy: IntervalStrategy = field(
        default="epoch",
        metadata={"help": "The checkpoint save strategy to adopt during training."},
        )
    save_steps: Optional[float] = field(
        default=1,
        metadata={"help": "Save checkpoint every X updates steps."},
        )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default"},
        )
    seed: int = field(
        default=42,
        metadata={"help": "random seed for initialization"},
        )
    select_labels: bool = field(
        default=False,
        metadata={"help": "Whether or not to select labels for the training set."},
        )

    # Optimization parameters
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
        )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay if we apply some."},
        )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam optimizer"},
        )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for Adam optimizer"},
        )
    adam_epsilon: float = field(
        default=1e-10,
        metadata={"help": "Epsilon for Adam optimizer"},
        )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm."},
        )
    adafactor: bool = field(
        default=True,
        metadata={"help": "Whether or not to replace AdamW by Adafactor."},
        )

    # Folders and identifications
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory"},
        )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
        )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run eval on the dev set."},
        )
    do_predict: bool = field(
        default=True,
        metadata={"help": "Whether to run predictions on the test set."},
        )
    run_name: Optional[str] = field(
        default="excell-roberta-fine-tuned",
        metadata={"help": "The name of this run. Used for logging."},
        )

    # Other params
    disable_tqdm: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether or not to disable the tqdm progress bars."},
        )
    remove_unused_columns: Optional[bool] = field(
        default=True,
        metadata={"help": "Remove columns not required by the model when using a CSV/JSON dataset."},
        )
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
        )
    metric_for_best_model: Optional[str] = field(
        default='f1',
        metadata={"help": "The metric to use to compare two different models."},
        )
    greater_is_better: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."},
        )
    report_to: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The list of integrations to report the results and logs to."},
        )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder containing a checkpoint to be loaded."},
        )
    class_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use class weights."},
        )
    results_file: Optional[str] = field(
        default="",
        metadata={"help": "File to store the results."},
    )
