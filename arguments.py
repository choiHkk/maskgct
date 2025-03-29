from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    asr_model_name_or_path: str = field(
        default="large-v2",
        metadata={
            "help": "Used to compute WER during evaluation. Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    wandb_project: str = field(
        default="maskgct",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_run_name: str = field(
        default=None,
        metadata={
            "help": "If specified, the name of the run. If not specified, wandb will give a random name to this run."
        },
    )
    data_config_path: str = field(
        default=None,
        metadata={
            "help": "If set, will save the dataset to this path if this is an empyt folder. If not empty, will load the datasets from it."
        },
    )
    add_audio_samples_to_wandb: bool = field(
        default=False,
        metadata={
            "help": "If set and if `wandb` in args.report_to, will add generated audio samples to wandb logs."
        },
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "The data type (dtype) in which to run training. One of `float32` (full-precision), "
                "`float16` or `bfloat16` (both half-precision)."
            )
        },
    )
