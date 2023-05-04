# Separating the args from the main to make it less busy

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to model
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    output_dir: str = field(
        default="output", metadata={"help": "Where to store model outputs"}
    )
    do_upload: bool = field(
        default=False, metadata={"help": "Whether model should be uploaded to wandb"}
    )

@dataclass
class TrainingArguments:
    """
    Arguments pertaining to training
    """
    group_name: str = field(
        metadata={"help": "Name of the group to group the runs on wandb"}
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Wether model should be trained or not"}
    )
    do_cl_train: bool = field(
        default=False,
        metadata={"help": "Wether model should be trained using curriculum "
                          "learning technique or not"}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Wether model should be evaluated or not"}
    )
    do_prediction: bool = field(
        default=False,
        metadata={"help": "Wether model should predict or not"}
    )
    do_sweep: bool = field(
        default=False,
        metadata={"help": "Perform wandb sweep "}
    )
    data_dir: str = field(
        default="train_valid_data",
        metadata={"help": "The input data artifact dir"},
    )
    data_version: str = field(
        default="v3",
        metadata={"help": "The version of input data artifact"},
    )
    data_name: str = field(
        default="train_valid_data",
        metadata={"help": "The input data artifact name."},
    )
    num_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of steps for the warmup in the lr scheduler"}
    )
    learning_rate: Optional[float] = field(
        default=3e-5,
        metadata={"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": "learning rate scheduler type to use."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        }
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "If the training should continue from a checkpoint folder."
        }
    )
    epochs: Optional[int] = field(default=10, metadata={"help": "Total number of training epochs to perform."})
    eval_epoch: Optional[int] = field(default=10, metadata={"help": "Epoch model to use for evaluation."})
    eval_shard: Optional[int] = field(default=0, metadata={"help": "Model shard to use for evaluation."})
    per_device_train_batch_size: Optional[int] = field(
        default=2,
        metadata={"help": "Batch size (per device) for the training dataloader."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size (per device) for the eval dataloader."}
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    log_interval: Optional[int] = field(default=100, metadata={"help": "wandb logging interval."})
    num_shards: Optional[int] = field(default=4, metadata={"help": "no of splits for training data during"
                                                                   "curriculum training"})







